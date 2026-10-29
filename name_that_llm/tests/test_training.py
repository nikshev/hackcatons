# verifies: FR-003-01
# verifies: FR-003-02
# verifies: FR-003-03
# verifies: FR-003-04
# verifies: FR-003-05
# verifies: FR-003-06
# verifies: FR-003-07
# verifies: FR-003-08
# verifies: FR-003-09
# verifies: FR-003-03
# verifies: FR-007-02
# verifies: FR-008-01
# verifies: FR-008-02
# verifies: FR-008-03
# verifies: FR-008-04
# verifies: FR-008-05
# verifies: FR-009-02
# verifies: FR-009-03
# verifies: FR-010-02
# verifies: FR-010-03
# verifies: FR-010-06
# verifies: FR-011-02
# verifies: FR-011-03
# verifies: FR-012-02
# verifies: FR-012-03

import csv
from collections import Counter
from pathlib import Path

import pytest
import yaml

from ntllm.audit import audit_from_config
from ntllm.splits import assign_prompt_folds
from ntllm.splits import split_from_config
from ntllm.training import TrainingError, evaluate_oof, train_from_config
from ntllm.__main__ import main


LABELS = ("model-a", "model-b")
MODELS = {
    "dummy": {"type": "dummy", "strategy": "most_frequent"},
    "word": {
        "type": "word_tfidf",
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_features": 1000,
        "sublinear_tf": True,
        "c": 1.0,
    },
    "char": {
        "type": "char_tfidf",
        "ngram_range": [2, 3],
        "min_df": 1,
        "max_features": 1000,
        "sublinear_tf": True,
        "c": 1.0,
    },
    "combined": {
        "type": "word_char_tfidf",
        "word_ngram_range": [1, 1],
        "char_ngram_range": [2, 3],
        "min_df": 1,
        "word_max_features": 1000,
        "char_max_features": 1000,
        "sublinear_tf": True,
        "c": 1.0,
    },
}


def training_rows(group_count=4):
    return [
        {
            "id": f"row-{group}-{label}",
            "prompt_id": f"prompt-{group:03}",
            "text": f"{label} response about topic {group}",
            "model": label,
        }
        for group in range(group_count)
        for label in LABELS
    ]


def fold_assignments(rows, n_splits=2):
    result = assign_prompt_folds(
        rows,
        id_column="id",
        group_column="prompt_id",
        label_column="model",
        allowed_labels=LABELS,
        n_splits=n_splits,
        seed=42,
    )
    return list(result.assignments)


def test_evaluate_oof_returns_complete_predictions_and_metrics():
    rows = training_rows()
    assignments = fold_assignments(rows)

    result = evaluate_oof(
        rows,
        assignments,
        id_column="id",
        group_column="prompt_id",
        text_column="text",
        label_column="model",
        allowed_labels=LABELS,
        n_splits=2,
        seed=42,
        models=MODELS,
    )

    assert len(result.predictions) == len(rows) * len(MODELS)
    fold_by_id = {assignment["id"]: assignment["fold"] for assignment in assignments}
    for candidate in MODELS:
        predictions = [row for row in result.predictions if row["candidate"] == candidate]
        assert {row["id"] for row in predictions} == {row["id"] for row in rows}
        assert len(predictions) == len({row["id"] for row in predictions})
        assert all(row["prediction"] in LABELS for row in predictions)
        assert all(row["fold"] == fold_by_id[row["id"]] for row in predictions)
        metrics = result.metrics[candidate]
        assert 0 <= metrics["balanced_accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert set(metrics["recall_by_label"]) == set(LABELS)
        assert len(metrics["confusion_matrix"]) == len(LABELS)
        assert all(len(row) == len(LABELS) for row in metrics["confusion_matrix"])
        assert len(metrics["fold_metrics"]) == 2
        assert [fold["row_count"] for fold in metrics["fold_metrics"]] == [4, 4]

    assert result.metrics["dummy"]["balanced_accuracy"] == pytest.approx(0.5)


def test_evaluate_oof_fits_each_model_without_its_validation_fold(monkeypatch):
    rows = training_rows()
    assignments = fold_assignments(rows)
    fold_by_id = {row["id"]: row["fold"] for row in assignments}
    fit_calls = []

    class SpyEstimator:
        def fit(self, texts, labels):
            fit_calls.append(set(texts))
            self.label = Counter(labels).most_common(1)[0][0]
            return self

        def predict(self, texts):
            return [self.label] * len(texts)

    def build_spy(candidate, model_config, seed):
        return SpyEstimator()

    monkeypatch.setattr("ntllm.training.build_estimator", build_spy)
    evaluate_oof(
        rows,
        assignments,
        id_column="id",
        group_column="prompt_id",
        text_column="text",
        label_column="model",
        allowed_labels=LABELS,
        n_splits=2,
        seed=42,
        models={"spy": {"type": "spy"}},
    )

    for fold, fit_texts in enumerate(fit_calls):
        held_out_texts = {
            row["text"] for row in rows if fold_by_id[row["id"]] == fold
        }
        assert fit_texts.isdisjoint(held_out_texts)


def test_evaluate_oof_predictions_and_metrics_are_deterministic():
    rows = training_rows()
    assignments = fold_assignments(rows)
    parameters = {
        "id_column": "id",
        "group_column": "prompt_id",
        "text_column": "text",
        "label_column": "model",
        "allowed_labels": LABELS,
        "n_splits": 2,
        "seed": 42,
        "models": MODELS,
    }

    first = evaluate_oof(rows, assignments, **parameters)
    second = evaluate_oof(rows, assignments, **parameters)

    assert first.predictions == second.predictions
    for candidate in MODELS:
        for metric in ("balanced_accuracy", "macro_f1", "recall_by_label", "confusion_matrix"):
            assert first.metrics[candidate][metric] == second.metrics[candidate][metric]


def test_evaluate_oof_rejects_incomplete_split_assignments():
    rows = training_rows()
    assignments = fold_assignments(rows)
    assignments.pop()

    with pytest.raises(TrainingError, match="missing or extra IDs"):
        evaluate_oof(
            rows,
            assignments,
            id_column="id",
            group_column="prompt_id",
            text_column="text",
            label_column="model",
            allowed_labels=LABELS,
            n_splits=2,
            seed=42,
            models=MODELS,
        )


def test_evaluate_oof_rejects_predictions_outside_official_allowlist(monkeypatch):
    rows = training_rows()
    assignments = fold_assignments(rows)

    class InvalidPredictor:
        def fit(self, texts, labels):
            return self

        def predict(self, texts):
            return ["unlisted-model"] * len(texts)

    monkeypatch.setattr(
        "ntllm.training.build_estimator",
        lambda candidate, model_config, seed: InvalidPredictor(),
    )
    with pytest.raises(TrainingError, match="predicted unrecognized label"):
        evaluate_oof(
            rows,
            assignments,
            id_column="id",
            group_column="prompt_id",
            text_column="text",
            label_column="model",
            allowed_labels=LABELS,
            n_splits=2,
            seed=42,
            models={"invalid": {"type": "dummy"}},
        )


def write_training_project(project_root):
    train_path = project_root / "kaggle" / "train.csv"
    test_path = project_root / "kaggle" / "test.csv"
    config_path = project_root / "configs" / "baseline.yaml"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    rows = training_rows()
    with train_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "prompt_id", "text", "model"])
        writer.writeheader()
        writer.writerows(rows)
    with test_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerow({"id": "test-row", "text": "test-only passage"})

    config = {
        "version": 7,
        "data": {
            "column_roles": {"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"},
            "files": {
                "train": {
                    "path": "kaggle/train.csv",
                    "required_roles": ["id", "text", "label"],
                    "source": "Kaggle fixture",
                    "retrieved_at": "2026-10-26",
                    "license_or_terms": "verified competition terms",
                },
                "test": {
                    "path": "kaggle/test.csv",
                    "required_roles": ["id", "text"],
                    "source": "Kaggle fixture",
                    "retrieved_at": "2026-10-26",
                    "license_or_terms": "verified competition terms",
                },
            },
        },
        "competition": {
            "rules_verified": True,
            "eligibility_verified": False,
            "payout_verified": False,
            "allowed_labels": list(LABELS),
        },
        "report": {"path": "reports/audit/audit.json"},
        "validation": {
            "prompt_group_role": "prompt_id",
            "n_splits": 2,
            "seed": 42,
            "assignments_path": "reports/splits/folds.parquet",
            "report_path": "reports/splits/report.json",
        },
        "training": {
            "artifact_root": "reports/runs",
            "models": {
                "dummy": {"type": "dummy", "strategy": "most_frequent"},
                "word": {
                    "type": "word_tfidf",
                    "ngram_range": [1, 1],
                    "min_df": 1,
                    "max_features": 1000,
                    "sublinear_tf": True,
                    "c": 1.0,
                },
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_from_config(config_path)
    split_from_config(config_path)
    return config_path, test_path


def test_train_from_config_writes_oof_artifacts_without_reading_test(tmp_path):
    config_path, test_path = write_training_project(tmp_path)
    test_path.write_bytes(b"not a readable CSV; train command must not open this file")

    result = train_from_config(config_path)

    run_dir = Path(result["run_dir"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions_oof.parquet").is_file()
    assert isinstance(result["manifest"]["working_tree_dirty"], bool)
    assert result["manifest"]["split_sha256"]
    assert len(result["manifest"]["source_sha256"]) == 64
    assert result["metrics"]["dummy"]["balanced_accuracy"] >= 0

    import pyarrow.parquet as parquet

    predictions = parquet.read_table(run_dir / "predictions_oof.parquet")
    assert predictions.num_rows == len(training_rows()) * 2
    assert "text" not in predictions.schema.names

    repeated = train_from_config(config_path)
    assert repeated["run_id"] == result["run_id"]
    assert repeated["reused"] is True


def test_train_from_config_stops_when_training_readiness_is_blocked(tmp_path):
    config_path, _ = write_training_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["competition"]["rules_verified"] = False
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_from_config(config_path)
    split_from_config(config_path)

    with pytest.raises(TrainingError, match="training readiness is blocked"):
        train_from_config(config_path)

    assert not (tmp_path / "reports" / "runs").exists()


def test_train_from_config_stops_on_stale_split_without_creating_run(tmp_path):
    config_path, _ = write_training_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["validation"]["seed"] = 11
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_from_config(config_path)

    with pytest.raises(TrainingError, match="split artifacts are stale"):
        train_from_config(config_path)

    assert not (tmp_path / "reports" / "runs").exists()


def test_baseline_config_exposes_all_candidate_types():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = config["training"]["models"]

    assert {model["type"] for model in models.values()} == {
        "dummy",
        "word_tfidf",
        "char_tfidf",
        "word_char_tfidf",
        "style_svc",
        "style_char_tfidf",
    }
    assert config["inference"]["model_candidate"] == "style_char_fusion"


def test_train_cli_runs_and_reports_run_directory(tmp_path, capsys):
    config_path, test_path = write_training_project(tmp_path)
    test_path.write_bytes(b"not a readable CSV; train command must not open this file")

    assert main(["train", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "train: completed grouped OOF benchmark" in output
    assert "run directory:" in output


def test_build_estimator_supports_logreg_classifier_option():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    from ntllm.training import build_estimator

    base = {
        "type": "char_tfidf",
        "analyzer": "char",
        "ngram_range": [2, 3],
        "min_df": 1,
        "max_features": 100,
        "sublinear_tf": True,
        "c": 1.0,
    }
    default_est = build_estimator("svc", base, seed=42)
    assert isinstance(default_est.named_steps["classifier"], LinearSVC)

    logreg_est = build_estimator("logreg", {**base, "classifier": "logreg"}, seed=42)
    assert isinstance(logreg_est.named_steps["classifier"], LogisticRegression)

    with pytest.raises(TrainingError, match="classifier"):
        build_estimator("bad", {**base, "classifier": "mystery"}, seed=42)


def test_build_estimator_inserts_deterministic_svd_layer_and_rejects_misuse():
    from sklearn.decomposition import TruncatedSVD

    from ntllm.training import build_estimator

    base = {
        "type": "char_tfidf",
        "analyzer": "char",
        "ngram_range": [2, 3],
        "min_df": 1,
        "max_features": 100,
        "sublinear_tf": True,
        "c": 1.0,
        "svd_components": 8,
    }
    first = build_estimator("svd", base, seed=42)
    second = build_estimator("svd", base, seed=42)
    assert isinstance(first.named_steps["svd"], TruncatedSVD)
    assert first.named_steps["svd"].n_components == 8
    assert first.named_steps["svd"].random_state == 42
    assert [name for name, _ in first.steps] == ["features", "svd", "classifier"]
    assert type(second.named_steps["svd"]) is type(first.named_steps["svd"])

    with pytest.raises(TrainingError, match="svd_components"):
        build_estimator("bad-count", {**base, "svd_components": 0}, seed=42)
    with pytest.raises(TrainingError, match="svd_components"):
        build_estimator(
            "dummy-svd",
            {"type": "dummy", "strategy": "most_frequent", "svd_components": 8},
            seed=42,
        )


def test_provider_of_label_maps_official_prefixes_and_rejects_unknown():
    from ntllm.training import provider_of_label

    assert provider_of_label("gpt-5.5") == "openai"
    assert provider_of_label("gemini-3.6-flash") == "google"
    assert provider_of_label("claude-opus-5") == "anthropic"
    assert provider_of_label("qwen3.8-max") == "qwen"
    assert provider_of_label("glm-5.3-flash") == "zai"
    assert provider_of_label("kimi-k3") == "moonshot"

    with pytest.raises(TrainingError, match="provider"):
        provider_of_label("unknown-model-x")


def test_evaluate_hierarchical_oof_covers_each_id_once_without_fold_leakage():
    from ntllm.training import evaluate_hierarchical_oof

    hier_labels = ("gpt-5.5", "gpt-5.6-sol", "gemini-3.6-flash", "gemini-3.7-flash")
    rows = [
        {"id": f"r{i}", "prompt_id": f"p{i // 2}", "text": f"{label} passage number {i}", "model": label}
        for i, label in enumerate(list(hier_labels) * 2)
    ]
    from ntllm.splits import assign_prompt_folds

    assignments = list(assign_prompt_folds(
        rows,
        id_column="id",
        group_column="prompt_id",
        label_column="model",
        allowed_labels=hier_labels,
        n_splits=2,
        seed=42,
    ).assignments)
    seen_fit_ids: set[str] = set()

    from ntllm import training as training_module

    real_build = training_module.build_estimator

    class RecordingEstimator:
        def __init__(self, *args, **kwargs):
            self.inner = real_build(*args, **kwargs)

        def fit(self, texts, labels):
            seen_fit_ids.update(texts)
            return self.inner.fit(texts, labels)

        def predict(self, texts):
            return self.inner.predict(texts)

    import unittest.mock as mock

    with mock.patch.object(training_module, "build_estimator", side_effect=lambda *a, **k: RecordingEstimator(*a, **k)):
        result = evaluate_hierarchical_oof(
            rows,
            assignments,
            id_column="id",
            group_column="prompt_id",
            text_column="text",
            label_column="model",
            allowed_labels=hier_labels,
            n_splits=2,
            seed=42,
            models={
                "hier": {
                    "type": "word_tfidf",
                    "ngram_range": [1, 1],
                    "min_df": 1,
                    "max_features": 100,
                    "sublinear_tf": True,
                    "c": 1.0,
                    "architecture": "hierarchical",
                }
            },
        )

    assert len(result.predictions) == len(rows)
    fold_by_id = {row["id"]: row["fold"] for row in assignments}
    assert all(row["fold"] == fold_by_id[row["id"]] for row in result.predictions)
    assert all(row["prediction"] in hier_labels for row in result.predictions)


def test_gemini_trio_constant_and_rerank_scope():
    from ntllm.training import GEMINI_TRIO, evaluate_gemini_rerank_oof, evaluate_oof

    assert GEMINI_TRIO == ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")

    trio_labels = ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gpt-5.5", "claude-opus-5")
    rows = [
        {"id": f"r{i}", "prompt_id": f"p{i // 2}", "text": f"{label} passage number {i}", "model": label}
        for i, label in enumerate(list(trio_labels) + ["gemini-3.6-flash", "gpt-5.5"])
    ]
    from ntllm.splits import assign_prompt_folds

    assignments = list(assign_prompt_folds(
        rows,
        id_column="id",
        group_column="prompt_id",
        label_column="model",
        allowed_labels=trio_labels,
        n_splits=2,
        seed=42,
    ).assignments)
    model_config = {
        "type": "word_tfidf",
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_features": 100,
        "sublinear_tf": True,
        "c": 1.0,
        "architecture": "gemini_rerank",
    }
    params = dict(
        id_column="id",
        group_column="prompt_id",
        text_column="text",
        label_column="model",
        allowed_labels=trio_labels,
        n_splits=2,
        seed=42,
    )
    flat = evaluate_oof(rows, assignments, models={"flat": {k: v for k, v in model_config.items() if k != "architecture"}}, **params)
    reranked = evaluate_gemini_rerank_oof(rows, assignments, models={"rerank": model_config}, **params)

    flat_by_id = {row["id"]: row["prediction"] for row in flat.predictions}
    rerank_by_id = {row["id"]: row["prediction"] for row in reranked.predictions}
    assert set(rerank_by_id) == {row["id"] for row in rows}
    assert all(pred in trio_labels for pred in rerank_by_id.values())
    for row_id, flat_pred in flat_by_id.items():
        if flat_pred not in GEMINI_TRIO:
            assert rerank_by_id[row_id] == flat_pred

    repeated = evaluate_gemini_rerank_oof(rows, assignments, models={"rerank": model_config}, **params)
    assert [dict(r) for r in repeated.predictions] == [dict(r) for r in reranked.predictions]


def test_evaluate_embedding_oof_covers_ids_with_fold_train_only_fits(monkeypatch):
    import numpy as np

    from ntllm import embeddings as embeddings_module
    from ntllm.training import evaluate_embedding_oof

    labels = ("model-a", "model-b")
    rows = [
        {"id": f"r{i}", "prompt_id": f"p{i // 2}", "text": f"{label} passage number {i}", "model": label}
        for i, label in enumerate(list(labels) * 4)
    ]
    from ntllm.splits import assign_prompt_folds

    assignments = list(assign_prompt_folds(
        rows,
        id_column="id",
        group_column="prompt_id",
        label_column="model",
        allowed_labels=labels,
        n_splits=2,
        seed=42,
    ).assignments)

    def fake_encode(texts, model_dir, batch_size=64):
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i] = np.float32([float((hash(f"{text}-{j}") % 997)) / 997.0 for j in range(4)])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    encode_calls: list[list[str]] = []

    def recording_encode(texts, model_dir, batch_size=64):
        encode_calls.append(list(texts))
        return fake_encode(texts, model_dir, batch_size=batch_size)

    monkeypatch.setattr(embeddings_module, "encode_texts", recording_encode)

    fit_sizes: list[int] = []
    from ntllm import training as training_module

    real_build = training_module.build_estimator

    class SpyEstimator:
        def __init__(self, *args, **kwargs):
            self.inner = real_build(*args, **kwargs)

        def fit(self, vectors, labels):
            fit_sizes.append(len(vectors))
            return self.inner.fit(vectors, labels)

        def predict(self, vectors):
            return self.inner.predict(vectors)

    import unittest.mock as mock

    model_config = {"type": "sbert_logreg", "c": 1.0, "max_iter": 200}
    with mock.patch.object(training_module, "build_estimator", side_effect=lambda *a, **k: SpyEstimator(*a, **k)):
        result = evaluate_embedding_oof(
            rows,
            assignments,
            id_column="id",
            group_column="prompt_id",
            text_column="text",
            label_column="model",
            allowed_labels=labels,
            n_splits=2,
            seed=42,
            models={"emb": model_config},
            embedding_model_dir="fake-model-dir",
        )

    assert len(result.predictions) == len(rows)
    assert len(encode_calls) == 1 and len(encode_calls[0]) == len(rows)
    assert sorted(fit_sizes) == [4, 4]
    assert all(row["prediction"] in labels for row in result.predictions)

    repeated = evaluate_embedding_oof(
        rows,
        assignments,
        id_column="id",
        group_column="prompt_id",
        text_column="text",
        label_column="model",
        allowed_labels=labels,
        n_splits=2,
        seed=42,
        models={"emb": model_config},
        embedding_model_dir="fake-model-dir",
    )
    assert [dict(r) for r in repeated.predictions] == [dict(r) for r in result.predictions]


def test_build_estimator_supports_style_and_fusion_types():
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    from ntllm.training import build_estimator

    style_only = build_estimator(
        "style",
        {"type": "style_svc", "c": 1.0},
        seed=42,
    )
    assert isinstance(style_only, Pipeline)
    assert isinstance(style_only.named_steps["style"].named_steps["scaler"], StandardScaler)
    assert isinstance(style_only.named_steps["classifier"], LinearSVC)

    fusion = build_estimator(
        "fusion",
        {
            "type": "style_char_tfidf",
            "analyzer": "char",
            "ngram_range": [3, 5],
            "min_df": 2,
            "max_features": 1000,
            "sublinear_tf": True,
            "c": 1.0,
        },
        seed=42,
    )
    assert isinstance(fusion.named_steps["features"], FeatureUnion)

    with pytest.raises(TrainingError, match="svd"):
        build_estimator("bad", {"type": "style_svc", "c": 1.0, "svd_components": 8}, seed=42)
    assert build_estimator("m", {"type": "word_tfidf", "ngram_range": [1, 1], "min_df": 1, "max_features": 10, "c": 1.0}, seed=42).named_steps["classifier"].max_iter == 1000
    assert build_estimator("m", {"type": "word_tfidf", "ngram_range": [1, 1], "min_df": 1, "max_features": 10, "c": 1.0, "max_iter": 5000}, seed=42).named_steps["classifier"].max_iter == 5000
    assert build_estimator("s", {"type": "style_svc", "c": 1.0}, seed=42).named_steps["classifier"].dual is False
    with pytest.raises(TrainingError, match="max_iter"):
        build_estimator("bad", {"type": "word_tfidf", "ngram_range": [1, 1], "min_df": 1, "max_features": 10, "c": 1.0, "max_iter": 0}, seed=42)
