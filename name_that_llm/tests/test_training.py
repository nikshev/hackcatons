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


def test_baseline_config_exposes_all_four_candidate_types():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = config["training"]["models"]

    assert {model["type"] for model in models.values()} == {
        "dummy",
        "word_tfidf",
        "char_tfidf",
        "word_char_tfidf",
    }


def test_train_cli_runs_and_reports_run_directory(tmp_path, capsys):
    config_path, test_path = write_training_project(tmp_path)
    test_path.write_bytes(b"not a readable CSV; train command must not open this file")

    assert main(["train", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "train: completed grouped OOF benchmark" in output
    assert "run directory:" in output
