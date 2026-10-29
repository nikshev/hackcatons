"""Prompt-grouped out-of-fold baseline evaluation.

impl: FR-003-01, FR-003-02
impl: FR-003-03, FR-003-04
impl: FR-003-05, FR-003-09
impl: FR-003-06, FR-003-07
impl: FR-003-08
impl: FR-007-02
impl: FR-008-02, FR-008-03
impl: FR-009-02, FR-009-03
impl: FR-010-02, FR-010-03, FR-010-06
impl: FR-011-02, FR-011-03
impl: FR-012-02, FR-012-03
"""

from __future__ import annotations

import time
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as parquet
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ntllm.data import AuditDataError, inspect_csv


class TrainingError(ValueError):
    """Training or OOF evaluation inputs violate a project invariant."""


@dataclass(frozen=True)
class OOFBenchmark:
    """Predictions and metrics from one set of grouped folds."""

    predictions: tuple[dict[str, str | int], ...]
    metrics: dict[str, dict[str, object]]


def _validate_ngram_range(value: object, *, key: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) or part < 1 for part in value)
        or value[0] > value[1]
    ):
        raise TrainingError(f"model setting '{key}' must be a pair of positive n-gram sizes")
    return value[0], value[1]


def _make_vectorizer(
    *,
    analyzer: str,
    ngram_range: object,
    min_df: object,
    max_features: object,
    sublinear_tf: object,
    model_name: str,
) -> TfidfVectorizer:
    parsed_range = _validate_ngram_range(
        ngram_range,
        key=f"{model_name}.ngram_range",
    )
    if isinstance(min_df, bool) or not isinstance(min_df, (int, float)) or min_df <= 0:
        raise TrainingError(f"model setting '{model_name}.min_df' must be positive")
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1:
        raise TrainingError(f"model setting '{model_name}.max_features' must be positive")
    if not isinstance(sublinear_tf, bool):
        raise TrainingError(f"model setting '{model_name}.sublinear_tf' must be boolean")
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=parsed_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def _linear_classifier(
    model_config: Mapping[str, object], candidate_name: str, seed: int, *, dual: bool = True
):
    """Shared LinearSVC / logreg choice for TF-IDF and style estimators."""
    classifier_name = model_config.get("classifier", "svc")
    c_value = model_config.get("c")
    if isinstance(c_value, bool) or not isinstance(c_value, (int, float)) or c_value <= 0:
        raise TrainingError(f"model '{candidate_name}' requires a positive 'c' value")
    if classifier_name == "svc":
        max_iter = model_config.get("max_iter", 1000)
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise TrainingError(f"model '{candidate_name}' requires a positive integer 'max_iter'")
        return LinearSVC(C=float(c_value), dual=dual, max_iter=max_iter, random_state=seed)
    if classifier_name == "logreg":
        return LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        )
    raise TrainingError(
        f"model '{candidate_name}' has unknown classifier '{classifier_name}'"
    )


def build_estimator(
    candidate_name: str,
    model_config: Mapping[str, object],
    *,
    seed: int,
):
    """Build one configured estimator; vectorizers stay inside the fit pipeline."""
    model_type = model_config.get("type")
    if model_type == "dummy":
        if "svd_components" in model_config:
            raise TrainingError(f"model '{candidate_name}' cannot combine dummy with 'svd_components'")
        strategy = model_config.get("strategy", "most_frequent")
        if strategy not in {"most_frequent", "prior", "stratified", "uniform"}:
            raise TrainingError(f"model '{candidate_name}' has an unsupported dummy strategy")
        return DummyClassifier(strategy=strategy, random_state=seed)
    if model_type == "sbert_logreg":
        if "svd_components" in model_config:
            raise TrainingError(f"model '{candidate_name}' cannot combine sbert with 'svd_components'")
        c_value = model_config.get("c")
        if isinstance(c_value, bool) or not isinstance(c_value, (int, float)) or c_value <= 0:
            raise TrainingError(f"model '{candidate_name}' requires a positive 'c' value")
        max_iter = model_config.get("max_iter", 1000)
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise TrainingError(f"model '{candidate_name}' requires a positive integer 'max_iter'")
        return LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
    svd_components = model_config.get("svd_components")
    if svd_components is not None and (
        isinstance(svd_components, bool) or not isinstance(svd_components, int) or svd_components < 1
    ):
        raise TrainingError(f"model '{candidate_name}' requires a positive integer 'svd_components'")

    if model_type in {"style_svc", "style_char_tfidf"}:
        if svd_components is not None:
            raise TrainingError(f"model '{candidate_name}' cannot combine style features with 'svd_components'")
        from ntllm.stylometry import StyleTransformer

        from sklearn.preprocessing import StandardScaler

        style_branch: list[tuple[str, object]] = [
            ("extract", StyleTransformer()),
            ("scaler", StandardScaler()),
        ]
        if model_type == "style_svc":
            # Dense few-feature input: solve the primal problem, which converges
            # where the dual formulation stalls. Same model family, solver detail.
            classifier = _linear_classifier(model_config, candidate_name, seed, dual=False)
            return Pipeline(
                [
                    ("style", Pipeline(style_branch)),
                    ("classifier", classifier),
                ]
            )
        char_tfidf = _make_vectorizer(
            analyzer=model_config.get("analyzer", "char"),
            ngram_range=model_config.get("ngram_range"),
            min_df=model_config.get("min_df", 1),
            max_features=model_config.get("max_features"),
            sublinear_tf=model_config.get("sublinear_tf", True),
            model_name=candidate_name,
        )
        if model_config.get("analyzer", "char") not in {"char", "char_wb"}:
            raise TrainingError(f"model '{candidate_name}' has unsupported char analyzer")
        return Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [("char", char_tfidf), ("style", Pipeline(style_branch))]
                    ),
                ),
                ("classifier", _linear_classifier(model_config, candidate_name, seed)),
            ]
        )

    if model_type not in {"word_tfidf", "char_tfidf", "word_char_tfidf"}:
        raise TrainingError(f"model '{candidate_name}' has unknown type '{model_type}'")
    c_value = model_config.get("c")
    if isinstance(c_value, bool) or not isinstance(c_value, (int, float)) or c_value <= 0:
        raise TrainingError(f"model '{candidate_name}' requires a positive 'c' value")
    min_df = model_config.get("min_df", 1)
    sublinear_tf = model_config.get("sublinear_tf", True)

    def vectorizer(analyzer: str, ngrams: object, max_features: object) -> TfidfVectorizer:
        return _make_vectorizer(
            analyzer=analyzer,
            ngram_range=ngrams,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            model_name=candidate_name,
        )

    if model_type == "word_tfidf":
        features = vectorizer(
            "word",
            model_config.get("ngram_range"),
            model_config.get("max_features"),
        )
    elif model_type == "char_tfidf":
        analyzer = model_config.get("analyzer", "char")
        if analyzer not in {"char", "char_wb"}:
            raise TrainingError(f"model '{candidate_name}' has unsupported char analyzer '{analyzer}'")
        features = vectorizer(
            analyzer,
            model_config.get("ngram_range"),
            model_config.get("max_features"),
        )
    else:
        char_analyzer = model_config.get("char_analyzer", "char")
        if char_analyzer not in {"char", "char_wb"}:
            raise TrainingError(f"model '{candidate_name}' has unsupported char analyzer '{char_analyzer}'")
        features = FeatureUnion(
            [
                (
                    "word",
                    vectorizer(
                        "word",
                        model_config.get("word_ngram_range"),
                        model_config.get("word_max_features"),
                    ),
                ),
                (
                    "char",
                    vectorizer(
                        char_analyzer,
                        model_config.get("char_ngram_range"),
                        model_config.get("char_max_features"),
                    ),
                ),
            ]
        )
    classifier = _linear_classifier(model_config, candidate_name, seed)
    steps: list[tuple[str, object]] = [("features", features)]
    if svd_components is not None:
        steps.append(
            ("svd", TruncatedSVD(n_components=svd_components, random_state=seed))
        )
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _score_predictions(
    actual: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> dict[str, object]:
    recalls = recall_score(
        actual,
        predicted,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    return {
        "balanced_accuracy": float(mean(recalls)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "recall_by_label": {
            label: float(score) for label, score in zip(labels, recalls, strict=True)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(labels)).tolist(),
    }


def evaluate_oof(
    records: Iterable[Mapping[str, str]],
    assignments: Iterable[Mapping[str, object]],
    *,
    id_column: str,
    group_column: str,
    text_column: str,
    label_column: str,
    allowed_labels: Sequence[str],
    n_splits: int,
    seed: int,
    models: Mapping[str, Mapping[str, object]],
) -> OOFBenchmark:
    """Fit configured candidates on non-held-out folds and return OOF metrics."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise TrainingError("n_splits must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    if not allowed_labels or any(not isinstance(label, str) or not label.strip() for label in allowed_labels):
        raise TrainingError("official allowed_labels must contain nonblank strings")
    if len(set(allowed_labels)) != len(allowed_labels):
        raise TrainingError("official allowed_labels contains duplicates")
    if not models:
        raise TrainingError("at least one baseline model must be configured")

    rows = list(records)
    ids: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        identifier = row.get(id_column)
        group = row.get(group_column)
        text = row.get(text_column)
        label = row.get(label_column)
        for field_name, value in (
            ("id", identifier),
            ("prompt group", group),
            ("text", text),
            ("label", label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TrainingError(f"train row {row_number} has a blank {field_name}")
        if identifier in ids:
            raise TrainingError(f"train data has duplicate id '{identifier}'")
        if label not in allowed_labels:
            raise TrainingError(f"train data has unrecognized label '{label}'")
        ids[identifier] = row

    missing_labels = sorted(set(allowed_labels) - {row[label_column] for row in rows})
    if missing_labels:
        raise TrainingError(f"train data is missing allowed labels: {', '.join(missing_labels)}")

    assignment_by_id: dict[str, tuple[str, int]] = {}
    fold_groups: dict[str, set[int]] = {}
    for assignment in assignments:
        identifier = assignment.get("id")
        group = assignment.get("prompt_id")
        fold = assignment.get("fold")
        if not isinstance(identifier, str) or not isinstance(group, str):
            raise TrainingError("split assignments require string id and prompt_id fields")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < n_splits:
            raise TrainingError(f"split assignment for '{identifier}' has invalid fold")
        if identifier in assignment_by_id:
            raise TrainingError(f"split assignments repeat id '{identifier}'")
        if identifier not in ids:
            raise TrainingError(f"split assignments contain unknown id '{identifier}'")
        if ids[identifier][group_column] != group:
            raise TrainingError(f"split group does not match training row for id '{identifier}'")
        assignment_by_id[identifier] = (group, fold)
        fold_groups.setdefault(group, set()).add(fold)

    if set(assignment_by_id) != set(ids):
        raise TrainingError("split assignments have missing or extra IDs")
    if any(len(folds) != 1 for folds in fold_groups.values()):
        raise TrainingError("a prompt group appears in multiple validation folds")
    if {fold for _, fold in assignment_by_id.values()} != set(range(n_splits)):
        raise TrainingError("split assignments do not contain every configured fold")

    rows_by_fold: dict[int, list[Mapping[str, str]]] = {
        fold: [row for row in rows if assignment_by_id[row[id_column]][1] == fold]
        for fold in range(n_splits)
    }
    if any(not fold_rows for fold_rows in rows_by_fold.values()):
        raise TrainingError("every validation fold must contain at least one row")

    predictions: list[dict[str, str | int]] = []
    metrics: dict[str, dict[str, object]] = {}
    allowed_order = list(allowed_labels)
    for candidate_name, model_config in models.items():
        if not isinstance(candidate_name, str) or not candidate_name.strip() or not isinstance(model_config, Mapping):
            raise TrainingError("model definitions must be named mappings")
        started = time.perf_counter()
        candidate_predictions: list[dict[str, str | int]] = []
        fold_metrics: list[dict[str, object]] = []
        for fold in range(n_splits):
            validation_rows = rows_by_fold[fold]
            fit_rows = [row for row in rows if assignment_by_id[row[id_column]][1] != fold]
            estimator = build_estimator(candidate_name, model_config, seed=seed)
            try:
                estimator.fit(
                    [row[text_column] for row in fit_rows],
                    [row[label_column] for row in fit_rows],
                )
                fold_predictions = estimator.predict([row[text_column] for row in validation_rows])
            except Exception as exc:
                raise TrainingError(f"model '{candidate_name}' failed on fold {fold}: {exc}") from exc
            if len(fold_predictions) != len(validation_rows):
                raise TrainingError(f"model '{candidate_name}' returned an incomplete fold {fold}")
            fold_prediction_records: list[dict[str, str | int]] = []
            for row, predicted in zip(validation_rows, fold_predictions, strict=True):
                predicted_label = str(predicted)
                if predicted_label not in allowed_labels:
                    raise TrainingError(
                        f"model '{candidate_name}' predicted unrecognized label '{predicted_label}'"
                    )
                fold_prediction_records.append({
                    "id": row[id_column],
                    "prompt_id": row[group_column],
                    "fold": fold,
                    "candidate": candidate_name,
                    "actual": row[label_column],
                    "prediction": predicted_label,
                })
            candidate_predictions.extend(fold_prediction_records)
            fold_score = _score_predictions(
                [str(item["actual"]) for item in fold_prediction_records],
                [str(item["prediction"]) for item in fold_prediction_records],
                allowed_order,
            )
            fold_metrics.append({
                "fold": fold,
                "row_count": len(fold_prediction_records),
                "group_count": len({item["prompt_id"] for item in fold_prediction_records}),
                **fold_score,
            })

        candidate_predictions.sort(key=lambda item: str(item["id"]))
        if len(candidate_predictions) != len(ids) or {
            item["id"] for item in candidate_predictions
        } != set(ids):
            raise TrainingError(f"model '{candidate_name}' did not produce exactly one OOF prediction per ID")

        actual = [str(item["actual"]) for item in candidate_predictions]
        predicted = [str(item["prediction"]) for item in candidate_predictions]
        metrics[candidate_name] = {
            **_score_predictions(actual, predicted, allowed_order),
            "fold_metrics": fold_metrics,
            "runtime_seconds": time.perf_counter() - started,
        }
        predictions.extend(candidate_predictions)

    predictions.sort(key=lambda item: (str(item["candidate"]), str(item["id"])))
    return OOFBenchmark(predictions=tuple(predictions), metrics=metrics)


_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gpt", "openai"),
    ("gemini", "google"),
    ("claude", "anthropic"),
    ("qwen", "qwen"),
    ("glm", "zai"),
    ("kimi", "moonshot"),
)


def provider_of_label(label: str) -> str:
    """Map an official model name to its provider lab, deterministically."""
    if not isinstance(label, str) or not label.strip():
        raise TrainingError("cannot derive a provider from a blank label")
    lowered = label.strip().lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix):
            return provider
    raise TrainingError(f"label '{label}' has no recognized provider prefix")


def evaluate_hierarchical_oof(
    records: Iterable[Mapping[str, str]],
    assignments: Iterable[Mapping[str, object]],
    *,
    id_column: str,
    group_column: str,
    text_column: str,
    label_column: str,
    allowed_labels: Sequence[str],
    n_splits: int,
    seed: int,
    models: Mapping[str, Mapping[str, object]],
) -> OOFBenchmark:
    """Two-stage OOF evaluation: provider router, then within-provider specialist.

    Architecture is the only difference from `evaluate_oof`: features, folds,
    seed, and allowlist are identical. Both stages fit exclusively on
    non-held-out rows.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise TrainingError("n_splits must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    if not allowed_labels or any(not isinstance(label, str) or not label.strip() for label in allowed_labels):
        raise TrainingError("official allowed_labels must contain nonblank strings")
    if len(set(allowed_labels)) != len(allowed_labels):
        raise TrainingError("official allowed_labels contains duplicates")
    if not models:
        raise TrainingError("at least one baseline model must be configured")
    provider_of_allowed = {label: provider_of_label(label) for label in allowed_labels}

    rows = list(records)
    ids: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        identifier = row.get(id_column)
        group = row.get(group_column)
        text = row.get(text_column)
        label = row.get(label_column)
        for field_name, value in (
            ("id", identifier),
            ("prompt group", group),
            ("text", text),
            ("label", label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TrainingError(f"train row {row_number} has a blank {field_name}")
        if identifier in ids:
            raise TrainingError(f"train data has duplicate id '{identifier}'")
        if label not in allowed_labels:
            raise TrainingError(f"train data has unrecognized label '{label}'")
        ids[identifier] = row

    missing_labels = sorted(set(allowed_labels) - {row[label_column] for row in rows})
    if missing_labels:
        raise TrainingError(f"train data is missing allowed labels: {', '.join(missing_labels)}")

    assignment_by_id: dict[str, tuple[str, int]] = {}
    fold_groups: dict[str, set[int]] = {}
    for assignment in assignments:
        identifier = assignment.get("id")
        group = assignment.get("prompt_id")
        fold = assignment.get("fold")
        if not isinstance(identifier, str) or not isinstance(group, str):
            raise TrainingError("split assignments require string id and prompt_id fields")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < n_splits:
            raise TrainingError(f"split assignment for '{identifier}' has invalid fold")
        if identifier in assignment_by_id:
            raise TrainingError(f"split assignments repeat id '{identifier}'")
        if identifier not in ids:
            raise TrainingError(f"split assignments contain unknown id '{identifier}'")
        if ids[identifier][group_column] != group:
            raise TrainingError(f"split group does not match training row for id '{identifier}'")
        assignment_by_id[identifier] = (group, fold)
        fold_groups.setdefault(group, set()).add(fold)

    if set(assignment_by_id) != set(ids):
        raise TrainingError("split assignments have missing or extra IDs")
    if any(len(folds) != 1 for folds in fold_groups.values()):
        raise TrainingError("a prompt group appears in multiple validation folds")
    if {fold for _, fold in assignment_by_id.values()} != set(range(n_splits)):
        raise TrainingError("split assignments do not contain every configured fold")

    rows_by_fold: dict[int, list[Mapping[str, str]]] = {
        fold: [row for row in rows if assignment_by_id[row[id_column]][1] == fold]
        for fold in range(n_splits)
    }
    if any(not fold_rows for fold_rows in rows_by_fold.values()):
        raise TrainingError("every validation fold must contain at least one row")

    predictions: list[dict[str, str | int]] = []
    metrics: dict[str, dict[str, object]] = {}
    allowed_order = list(allowed_labels)
    for candidate_name, model_config in models.items():
        if not isinstance(candidate_name, str) or not candidate_name.strip() or not isinstance(model_config, Mapping):
            raise TrainingError("model definitions must be named mappings")
        architecture = model_config.get("architecture", "hierarchical")
        if architecture != "hierarchical":
            raise TrainingError(f"model '{candidate_name}' is not a hierarchical candidate")
        started = time.perf_counter()
        candidate_predictions: list[dict[str, str | int]] = []
        fold_metrics: list[dict[str, object]] = []
        for fold in range(n_splits):
            validation_rows = rows_by_fold[fold]
            fit_rows = [row for row in rows if assignment_by_id[row[id_column]][1] != fold]
            try:
                router = build_estimator(candidate_name, model_config, seed=seed)
                router.fit(
                    [row[text_column] for row in fit_rows],
                    [provider_of_allowed[row[label_column]] for row in fit_rows],
                )
                specialists: dict[str, object] = {}
                fallback_counts: Counter[str] = Counter()
                for provider in sorted({provider_of_allowed[row[label_column]] for row in fit_rows}):
                    provider_rows = [
                        row for row in fit_rows
                        if provider_of_allowed[row[label_column]] == provider
                    ]
                    provider_labels = [row[label_column] for row in provider_rows]
                    fallback_counts.update(provider_labels)
                    distinct = sorted(set(provider_labels))
                    if len(distinct) == 1:
                        specialists[provider] = distinct[0]
                    else:
                        specialist = build_estimator(candidate_name, model_config, seed=seed)
                        specialist.fit(
                            [row[text_column] for row in provider_rows],
                            provider_labels,
                        )
                        specialists[provider] = specialist
                global_fallback = fallback_counts.most_common(1)[0][0]
                routed = router.predict([row[text_column] for row in validation_rows])
            except TrainingError:
                raise
            except Exception as exc:
                raise TrainingError(f"model '{candidate_name}' failed on fold {fold}: {exc}") from exc
            if len(routed) != len(validation_rows):
                raise TrainingError(f"model '{candidate_name}' returned an incomplete fold {fold}")
            fold_prediction_records: list[dict[str, str | int]] = []
            for row, routed_provider in zip(validation_rows, routed, strict=True):
                specialist = specialists.get(str(routed_provider))
                if specialist is None:
                    predicted_label = global_fallback
                elif isinstance(specialist, str):
                    predicted_label = specialist
                else:
                    try:
                        predicted_label = str(specialist.predict([row[text_column]])[0])
                    except Exception as exc:
                        raise TrainingError(
                            f"model '{candidate_name}' specialist failed on fold {fold}: {exc}"
                        ) from exc
                if predicted_label not in allowed_labels:
                    raise TrainingError(
                        f"model '{candidate_name}' predicted unrecognized label '{predicted_label}'"
                    )
                fold_prediction_records.append({
                    "id": row[id_column],
                    "prompt_id": row[group_column],
                    "fold": fold,
                    "candidate": candidate_name,
                    "actual": row[label_column],
                    "prediction": predicted_label,
                })
            candidate_predictions.extend(fold_prediction_records)
            fold_score = _score_predictions(
                [str(item["actual"]) for item in fold_prediction_records],
                [str(item["prediction"]) for item in fold_prediction_records],
                allowed_order,
            )
            fold_metrics.append({
                "fold": fold,
                "row_count": len(fold_prediction_records),
                "group_count": len({item["prompt_id"] for item in fold_prediction_records}),
                **fold_score,
            })

        candidate_predictions.sort(key=lambda item: str(item["id"]))
        if len(candidate_predictions) != len(ids) or {
            item["id"] for item in candidate_predictions
        } != set(ids):
            raise TrainingError(f"model '{candidate_name}' did not produce exactly one OOF prediction per ID")

        actual = [str(item["actual"]) for item in candidate_predictions]
        predicted = [str(item["prediction"]) for item in candidate_predictions]
        metrics[candidate_name] = {
            **_score_predictions(actual, predicted, allowed_order),
            "fold_metrics": fold_metrics,
            "runtime_seconds": time.perf_counter() - started,
        }
        predictions.extend(candidate_predictions)

    predictions.sort(key=lambda item: (str(item["candidate"]), str(item["id"])))
    return OOFBenchmark(predictions=tuple(predictions), metrics=metrics)


GEMINI_TRIO: tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")


def evaluate_gemini_rerank_oof(
    records: Iterable[Mapping[str, str]],
    assignments: Iterable[Mapping[str, object]],
    *,
    id_column: str,
    group_column: str,
    text_column: str,
    label_column: str,
    allowed_labels: Sequence[str],
    n_splits: int,
    seed: int,
    models: Mapping[str, Mapping[str, object]],
) -> OOFBenchmark:
    """Flat pass plus a per-fold trio specialist rerank for Gemini-flash rows.

    Only validation rows the flat stage assigns to the trio are rescored, by a
    specialist fit exclusively on trio rows of the same fold-train set. All
    other predictions pass through unchanged.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise TrainingError("n_splits must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    if not allowed_labels or any(not isinstance(label, str) or not label.strip() for label in allowed_labels):
        raise TrainingError("official allowed_labels must contain nonblank strings")
    if len(set(allowed_labels)) != len(allowed_labels):
        raise TrainingError("official allowed_labels contains duplicates")
    if not models:
        raise TrainingError("at least one baseline model must be configured")
    if set(GEMINI_TRIO) - set(allowed_labels):
        raise TrainingError("official allowlist does not contain the full Gemini trio")

    rows = list(records)
    ids: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        identifier = row.get(id_column)
        group = row.get(group_column)
        text = row.get(text_column)
        label = row.get(label_column)
        for field_name, value in (
            ("id", identifier),
            ("prompt group", group),
            ("text", text),
            ("label", label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TrainingError(f"train row {row_number} has a blank {field_name}")
        if identifier in ids:
            raise TrainingError(f"train data has duplicate id '{identifier}'")
        if label not in allowed_labels:
            raise TrainingError(f"train data has unrecognized label '{label}'")
        ids[identifier] = row

    missing_labels = sorted(set(allowed_labels) - {row[label_column] for row in rows})
    if missing_labels:
        raise TrainingError(f"train data is missing allowed labels: {', '.join(missing_labels)}")

    assignment_by_id: dict[str, tuple[str, int]] = {}
    fold_groups: dict[str, set[int]] = {}
    for assignment in assignments:
        identifier = assignment.get("id")
        group = assignment.get("prompt_id")
        fold = assignment.get("fold")
        if not isinstance(identifier, str) or not isinstance(group, str):
            raise TrainingError("split assignments require string id and prompt_id fields")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < n_splits:
            raise TrainingError(f"split assignment for '{identifier}' has invalid fold")
        if identifier in assignment_by_id:
            raise TrainingError(f"split assignments repeat id '{identifier}'")
        if identifier not in ids:
            raise TrainingError(f"split assignments contain unknown id '{identifier}'")
        if ids[identifier][group_column] != group:
            raise TrainingError(f"split group does not match training row for id '{identifier}'")
        assignment_by_id[identifier] = (group, fold)
        fold_groups.setdefault(group, set()).add(fold)

    if set(assignment_by_id) != set(ids):
        raise TrainingError("split assignments have missing or extra IDs")
    if any(len(folds) != 1 for folds in fold_groups.values()):
        raise TrainingError("a prompt group appears in multiple validation folds")
    if {fold for _, fold in assignment_by_id.values()} != set(range(n_splits)):
        raise TrainingError("split assignments do not contain every configured fold")

    rows_by_fold: dict[int, list[Mapping[str, str]]] = {
        fold: [row for row in rows if assignment_by_id[row[id_column]][1] == fold]
        for fold in range(n_splits)
    }
    if any(not fold_rows for fold_rows in rows_by_fold.values()):
        raise TrainingError("every validation fold must contain at least one row")

    predictions: list[dict[str, str | int]] = []
    metrics: dict[str, dict[str, object]] = {}
    allowed_order = list(allowed_labels)
    trio_set = set(GEMINI_TRIO)
    for candidate_name, model_config in models.items():
        if not isinstance(candidate_name, str) or not candidate_name.strip() or not isinstance(model_config, Mapping):
            raise TrainingError("model definitions must be named mappings")
        architecture = model_config.get("architecture", "gemini_rerank")
        if architecture != "gemini_rerank":
            raise TrainingError(f"model '{candidate_name}' is not a Gemini-rerank candidate")
        started = time.perf_counter()
        candidate_predictions: list[dict[str, str | int]] = []
        fold_metrics: list[dict[str, object]] = []
        for fold in range(n_splits):
            validation_rows = rows_by_fold[fold]
            fit_rows = [row for row in rows if assignment_by_id[row[id_column]][1] != fold]
            try:
                flat = build_estimator(candidate_name, model_config, seed=seed)
                flat.fit(
                    [row[text_column] for row in fit_rows],
                    [row[label_column] for row in fit_rows],
                )
                flat_predictions = flat.predict([row[text_column] for row in validation_rows])
                trio_fit_rows = [row for row in fit_rows if row[label_column] in trio_set]
                trio_labels = sorted({row[label_column] for row in trio_fit_rows})
                specialist = None
                if len(trio_labels) == 1:
                    specialist = trio_labels[0]
                elif trio_labels:
                    specialist = build_estimator(candidate_name, model_config, seed=seed)
                    specialist.fit(
                        [row[text_column] for row in trio_fit_rows],
                        [row[label_column] for row in trio_fit_rows],
                    )
            except TrainingError:
                raise
            except Exception as exc:
                raise TrainingError(f"model '{candidate_name}' failed on fold {fold}: {exc}") from exc
            if len(flat_predictions) != len(validation_rows):
                raise TrainingError(f"model '{candidate_name}' returned an incomplete fold {fold}")
            fold_prediction_records: list[dict[str, str | int]] = []
            for row, flat_predicted in zip(validation_rows, flat_predictions, strict=True):
                flat_label = str(flat_predicted)
                if flat_label not in allowed_labels:
                    raise TrainingError(
                        f"model '{candidate_name}' predicted unrecognized label '{flat_label}'"
                    )
                if flat_label in trio_set and specialist is not None:
                    if isinstance(specialist, str):
                        predicted_label = specialist
                    else:
                        try:
                            predicted_label = str(specialist.predict([row[text_column]])[0])
                        except Exception as exc:
                            raise TrainingError(
                                f"model '{candidate_name}' specialist failed on fold {fold}: {exc}"
                            ) from exc
                    if predicted_label not in trio_set:
                        raise TrainingError(
                            f"model '{candidate_name}' specialist predicted outside the trio: '{predicted_label}'"
                        )
                else:
                    predicted_label = flat_label
                fold_prediction_records.append({
                    "id": row[id_column],
                    "prompt_id": row[group_column],
                    "fold": fold,
                    "candidate": candidate_name,
                    "actual": row[label_column],
                    "prediction": predicted_label,
                })
            candidate_predictions.extend(fold_prediction_records)
            fold_score = _score_predictions(
                [str(item["actual"]) for item in fold_prediction_records],
                [str(item["prediction"]) for item in fold_prediction_records],
                allowed_order,
            )
            fold_metrics.append({
                "fold": fold,
                "row_count": len(fold_prediction_records),
                "group_count": len({item["prompt_id"] for item in fold_prediction_records}),
                **fold_score,
            })

        candidate_predictions.sort(key=lambda item: str(item["id"]))
        if len(candidate_predictions) != len(ids) or {
            item["id"] for item in candidate_predictions
        } != set(ids):
            raise TrainingError(f"model '{candidate_name}' did not produce exactly one OOF prediction per ID")

        actual = [str(item["actual"]) for item in candidate_predictions]
        predicted = [str(item["prediction"]) for item in candidate_predictions]
        metrics[candidate_name] = {
            **_score_predictions(actual, predicted, allowed_order),
            "fold_metrics": fold_metrics,
            "runtime_seconds": time.perf_counter() - started,
        }
        predictions.extend(candidate_predictions)

    predictions.sort(key=lambda item: (str(item["candidate"]), str(item["id"])))
    return OOFBenchmark(predictions=tuple(predictions), metrics=metrics)


def evaluate_embedding_oof(
    records: Iterable[Mapping[str, str]],
    assignments: Iterable[Mapping[str, object]],
    *,
    id_column: str,
    group_column: str,
    text_column: str,
    label_column: str,
    allowed_labels: Sequence[str],
    n_splits: int,
    seed: int,
    models: Mapping[str, Mapping[str, object]],
    embedding_model_dir: str | Path,
) -> OOFBenchmark:
    """OOF evaluation on frozen precomputed embeddings.

    The encoder runs once over all training texts (it is frozen, so this leaks
    nothing); each candidate's classifier fits only on non-held-out fold rows.
    """
    from ntllm.embeddings import encode_texts

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise TrainingError("n_splits must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    if not allowed_labels or any(not isinstance(label, str) or not label.strip() for label in allowed_labels):
        raise TrainingError("official allowed_labels must contain nonblank strings")
    if len(set(allowed_labels)) != len(allowed_labels):
        raise TrainingError("official allowed_labels contains duplicates")
    if not models:
        raise TrainingError("at least one baseline model must be configured")

    rows = list(records)
    ids: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        identifier = row.get(id_column)
        group = row.get(group_column)
        text = row.get(text_column)
        label = row.get(label_column)
        for field_name, value in (
            ("id", identifier),
            ("prompt group", group),
            ("text", text),
            ("label", label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TrainingError(f"train row {row_number} has a blank {field_name}")
        if identifier in ids:
            raise TrainingError(f"train data has duplicate id '{identifier}'")
        if label not in allowed_labels:
            raise TrainingError(f"train data has unrecognized label '{label}'")
        ids[identifier] = row

    missing_labels = sorted(set(allowed_labels) - {row[label_column] for row in rows})
    if missing_labels:
        raise TrainingError(f"train data is missing allowed labels: {', '.join(missing_labels)}")

    assignment_by_id: dict[str, tuple[str, int]] = {}
    fold_groups: dict[str, set[int]] = {}
    for assignment in assignments:
        identifier = assignment.get("id")
        group = assignment.get("prompt_id")
        fold = assignment.get("fold")
        if not isinstance(identifier, str) or not isinstance(group, str):
            raise TrainingError("split assignments require string id and prompt_id fields")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < n_splits:
            raise TrainingError(f"split assignment for '{identifier}' has invalid fold")
        if identifier in assignment_by_id:
            raise TrainingError(f"split assignments repeat id '{identifier}'")
        if identifier not in ids:
            raise TrainingError(f"split assignments contain unknown id '{identifier}'")
        if ids[identifier][group_column] != group:
            raise TrainingError(f"split group does not match training row for id '{identifier}'")
        assignment_by_id[identifier] = (group, fold)
        fold_groups.setdefault(group, set()).add(fold)

    if set(assignment_by_id) != set(ids):
        raise TrainingError("split assignments have missing or extra IDs")
    if any(len(folds) != 1 for folds in fold_groups.values()):
        raise TrainingError("a prompt group appears in multiple validation folds")
    if {fold for _, fold in assignment_by_id.values()} != set(range(n_splits)):
        raise TrainingError("split assignments do not contain every configured fold")

    rows_by_fold: dict[int, list[int]] = {
        fold: [index for index, row in enumerate(rows) if assignment_by_id[row[id_column]][1] == fold]
        for fold in range(n_splits)
    }
    if any(not fold_rows for fold_rows in rows_by_fold.values()):
        raise TrainingError("every validation fold must contain at least one row")

    vectors = encode_texts([row[text_column] for row in rows], embedding_model_dir)
    if len(vectors) != len(rows):
        raise TrainingError("encoder returned a different number of vectors than train rows")

    predictions: list[dict[str, str | int]] = []
    metrics: dict[str, dict[str, object]] = {}
    allowed_order = list(allowed_labels)
    for candidate_name, model_config in models.items():
        if not isinstance(candidate_name, str) or not candidate_name.strip() or not isinstance(model_config, Mapping):
            raise TrainingError("model definitions must be named mappings")
        if model_config.get("type") != "sbert_logreg":
            raise TrainingError(f"model '{candidate_name}' is not an sbert embedding candidate")
        started = time.perf_counter()
        candidate_predictions: list[dict[str, str | int]] = []
        fold_metrics: list[dict[str, object]] = []
        for fold in range(n_splits):
            validation_idx = rows_by_fold[fold]
            fit_idx = [index for index in range(len(rows)) if assignment_by_id[rows[index][id_column]][1] != fold]
            estimator = build_estimator(candidate_name, model_config, seed=seed)
            try:
                estimator.fit(
                    vectors[fit_idx],
                    [rows[index][label_column] for index in fit_idx],
                )
                fold_predictions = estimator.predict(vectors[validation_idx])
            except Exception as exc:
                raise TrainingError(f"model '{candidate_name}' failed on fold {fold}: {exc}") from exc
            if len(fold_predictions) != len(validation_idx):
                raise TrainingError(f"model '{candidate_name}' returned an incomplete fold {fold}")
            fold_prediction_records: list[dict[str, str | int]] = []
            for index, predicted in zip(validation_idx, fold_predictions, strict=True):
                row = rows[index]
                predicted_label = str(predicted)
                if predicted_label not in allowed_labels:
                    raise TrainingError(
                        f"model '{candidate_name}' predicted unrecognized label '{predicted_label}'"
                    )
                fold_prediction_records.append({
                    "id": row[id_column],
                    "prompt_id": row[group_column],
                    "fold": fold,
                    "candidate": candidate_name,
                    "actual": row[label_column],
                    "prediction": predicted_label,
                })
            candidate_predictions.extend(fold_prediction_records)
            fold_score = _score_predictions(
                [str(item["actual"]) for item in fold_prediction_records],
                [str(item["prediction"]) for item in fold_prediction_records],
                allowed_order,
            )
            fold_metrics.append({
                "fold": fold,
                "row_count": len(fold_prediction_records),
                "group_count": len({item["prompt_id"] for item in fold_prediction_records}),
                **fold_score,
            })

        candidate_predictions.sort(key=lambda item: str(item["id"]))
        if len(candidate_predictions) != len(ids) or {
            item["id"] for item in candidate_predictions
        } != set(ids):
            raise TrainingError(f"model '{candidate_name}' did not produce exactly one OOF prediction per ID")

        actual = [str(item["actual"]) for item in candidate_predictions]
        predicted = [str(item["prediction"]) for item in candidate_predictions]
        metrics[candidate_name] = {
            **_score_predictions(actual, predicted, allowed_order),
            "fold_metrics": fold_metrics,
            "runtime_seconds": time.perf_counter() - started,
        }
        predictions.extend(candidate_predictions)

    predictions.sort(key=lambda item: (str(item["candidate"]), str(item["id"])))
    return OOFBenchmark(predictions=tuple(predictions), metrics=metrics)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(project_root: Path, value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"configuration field '{key}' must be a nonblank path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _git_provenance(project_root: Path) -> tuple[str | None, bool]:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if root.returncode != 0:
        return None, False
    repository_root = Path(root.stdout.strip()).resolve()
    try:
        project_relative = str(project_root.resolve().relative_to(repository_root))
    except ValueError:
        return None, False
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", project_relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (commit.stdout.strip() if commit.returncode == 0 else None, bool(status.stdout.strip()))


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_sha256(project_root: Path) -> str:
    source_root = project_root / "src" / "ntllm"
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def train_from_config(config_path: str | Path) -> dict[str, object]:
    """Run grouped OOF baselines, gated by the latest audit and split artifacts."""
    source_config = Path(config_path).resolve()
    if not source_config.is_file():
        raise TrainingError(f"configuration file does not exist: {source_config}")
    try:
        config_bytes = source_config.read_bytes()
        config = yaml.safe_load(config_bytes)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrainingError(f"cannot read YAML configuration {source_config}: {exc}") from exc
    if not isinstance(config, dict):
        raise TrainingError(f"{source_config}: configuration must be a YAML mapping")
    config_version = config.get("version")
    if not isinstance(config_version, (int, str)) or not str(config_version).strip():
        raise TrainingError(f"{source_config}: a nonblank configuration version is required")

    project_root = source_config.parent.parent
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    report_config = config.get("report")
    if not isinstance(report_config, dict):
        raise TrainingError(f"{source_config}: 'report' mapping is required")
    audit_report_path = _project_path(project_root, report_config.get("path"), key="report.path")
    if not audit_report_path.is_file():
        raise TrainingError("current audit report is missing; run `ntllm audit` first")
    try:
        audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read audit report {audit_report_path}: {exc}") from exc
    audit_config = audit_report.get("configuration", {})
    if audit_config.get("sha256") != config_hash or audit_config.get("version") != config_version:
        raise TrainingError("audit report is stale for the current configuration; rerun `ntllm audit`")
    if audit_report.get("audit_status") != "passed":
        raise TrainingError("data audit did not pass")
    readiness = audit_report.get("training_readiness", {})
    if readiness.get("status") != "ready":
        reasons = "; ".join(readiness.get("blockers", []))
        raise TrainingError(f"training readiness is blocked: {reasons}")

    data_config = config.get("data")
    if not isinstance(data_config, dict) or not isinstance(data_config.get("column_roles"), dict):
        raise TrainingError(f"{source_config}: 'data.column_roles' mapping is required")
    roles = data_config["column_roles"]
    files = data_config.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("train"), dict):
        raise TrainingError(f"{source_config}: 'data.files.train' mapping is required")
    train_path = _project_path(project_root, files["train"].get("path"), key="data.files.train.path")
    audit_train = audit_report.get("files", {}).get("train", {})
    train_sha = audit_train.get("sha256")
    if not isinstance(train_sha, str) or not train_sha:
        raise TrainingError("audit report does not contain the train data checksum")
    checksum_before = _sha256(train_path)
    if checksum_before != train_sha:
        raise TrainingError("train data checksum differs from the latest audit; rerun `ntllm audit`")

    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise TrainingError(f"{source_config}: 'validation' mapping is required")
    split_report_path = _project_path(
        project_root,
        validation.get("report_path"),
        key="validation.report_path",
    )
    assignments_path = _project_path(
        project_root,
        validation.get("assignments_path"),
        key="validation.assignments_path",
    )
    if not split_report_path.is_file() or not assignments_path.is_file():
        raise TrainingError("current prompt-group split artifacts are missing; run `ntllm split` first")
    try:
        split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
        assignments = parquet.read_table(assignments_path).to_pylist()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, pa.ArrowException) as exc:
        raise TrainingError(f"cannot read split artifacts: {exc}") from exc
    split_config = split_report.get("configuration", {})
    if (
        split_config.get("sha256") != config_hash
        or split_config.get("version") != config_version
        or split_report.get("dataset_sha256") != train_sha
        or split_report.get("n_splits") != validation.get("n_splits")
        or split_report.get("seed") != validation.get("seed")
    ):
        raise TrainingError("split artifacts are stale for current data/config; rerun `ntllm split`")

    group_role = validation.get("prompt_group_role")
    group_column = roles.get(group_role) if isinstance(group_role, str) else None
    id_column = roles.get("id")
    text_column = roles.get("text")
    label_column = roles.get("label")
    if not all(
        isinstance(column, str) and column.strip()
        for column in (id_column, group_column, text_column, label_column)
    ):
        raise TrainingError("data id/group/text/label column roles must be configured")
    try:
        inspect_csv(
            train_path,
            column_roles=roles,
            required_roles=("id", group_role, "text", "label"),
        )
        with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error, AuditDataError) as exc:
        raise TrainingError(f"cannot read audited training rows: {exc}") from exc
    if _sha256(train_path) != checksum_before:
        raise TrainingError("train data changed while training rows were being loaded")

    competition = config.get("competition")
    training = config.get("training")
    if not isinstance(competition, dict) or not isinstance(training, dict):
        raise TrainingError("competition and training configuration mappings are required")
    allowed_labels = competition.get("allowed_labels")
    models = training.get("models")
    if not isinstance(allowed_labels, list) or not isinstance(models, dict) or not models:
        raise TrainingError("official allowed_labels and configured training.models are required")

    result = evaluate_oof(
        records,
        assignments,
        id_column=id_column,
        group_column=group_column,
        text_column=text_column,
        label_column=label_column,
        allowed_labels=allowed_labels,
        n_splits=validation["n_splits"],
        seed=validation["seed"],
        models=models,
    )

    source_hash = _source_sha256(project_root)
    python_version = platform.python_version()
    library_versions = {
        package: _package_version(package)
        for package in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib")
    }
    commit, working_tree_dirty = _git_provenance(project_root)
    run_key = json.dumps(
        {
            "config_sha256": config_hash,
            "dataset_sha256": train_sha,
            "split_sha256": split_report["split_sha256"],
            "source_sha256": source_hash,
            "python_version": python_version,
            "library_versions": library_versions,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    run_id = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:16]
    artifact_root = _project_path(
        project_root,
        training.get("artifact_root"),
        key="training.artifact_root",
    )
    run_dir = artifact_root / run_id
    if run_dir.exists():
        try:
            existing_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            existing_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrainingError(f"existing run directory is incomplete: {run_dir}") from exc
        if (
            existing_manifest.get("run_id") != run_id
            or existing_manifest.get("source_sha256") != source_hash
            or not (run_dir / "predictions_oof.parquet").is_file()
        ):
            raise TrainingError(f"existing run directory is incomplete or stale: {run_dir}")
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": existing_manifest,
            "metrics": existing_metrics["models"],
            "reused": True,
        }

    manifest: dict[str, object] = {
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path.relative_to(project_root)),
        "train_sha256": train_sha,
        "config_path": str(source_config.relative_to(project_root)),
        "config_version": config_version,
        "config_sha256": config_hash,
        "split_sha256": split_report["split_sha256"],
        "source_sha256": source_hash,
        "seed": validation["seed"],
        "n_splits": validation["n_splits"],
        "git_commit": commit,
        "working_tree_dirty": working_tree_dirty,
        "python_version": python_version,
        "library_versions": library_versions,
        "models": models,
    }
    metrics_report = {
        "run_id": run_id,
        "split_sha256": split_report["split_sha256"],
        "models": result.metrics,
    }

    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=artifact_root))
    try:
        parquet.write_table(
            pa.Table.from_pylist([dict(row) for row in result.predictions]),
            temporary_dir / "predictions_oof.parquet",
        )
        (temporary_dir / "metrics.json").write_text(
            json.dumps(metrics_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, run_dir)
    except (OSError, pa.ArrowException) as exc:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise TrainingError(f"cannot write run artifacts: {exc}") from exc

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "metrics": result.metrics,
        "reused": False,
    }
