# verifies: FR-006-01
# verifies: FR-006-02
# verifies: FR-006-03
# verifies: FR-006-04
# verifies: FR-006-05
# verifies: FR-006-06
# verifies: FR-006-07
# verifies: FR-006-08
# verifies: FR-006-09
# verifies: FR-009-01
# verifies: FR-009-04
# verifies: FR-009-05
# verifies: FR-010-01
# verifies: FR-010-04
# verifies: FR-010-05
# verifies: FR-011-01
# verifies: FR-011-04
# verifies: FR-011-05
# verifies: FR-012-01
# verifies: FR-012-04
# verifies: FR-012-05
# verifies: FR-013-01
# verifies: FR-013-04
# verifies: FR-013-05
# verifies: FR-007-01
# verifies: FR-007-03
# verifies: FR-007-04

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import yaml

from ntllm.analysis import analyze_oof_run, tune_from_config
from ntllm.audit import audit_from_config
from ntllm.__main__ import main
from ntllm.splits import split_from_config


LABELS = ["model-a", "model-b"]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_analysis_project(project_root):
    rows = [
        {"id": "row-a1", "prompt_id": "prompt-a", "text": "private passage alpha", "model": "model-a"},
        {"id": "row-a2", "prompt_id": "prompt-a", "text": "private passage beta", "model": "model-b"},
        {"id": "row-b1", "prompt_id": "prompt-b", "text": "private passage gamma", "model": "model-a"},
        {"id": "row-b2", "prompt_id": "prompt-b", "text": "private passage delta", "model": "model-b"},
    ]
    train_path = project_root / "kaggle" / "train.csv"
    write_csv(train_path, ["id", "prompt_id", "text", "model"], rows)
    config_path = project_root / "configs" / "baseline.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 10,
        "data": {
            "column_roles": {"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"},
            "files": {
                "train": {
                    "path": "kaggle/train.csv",
                    "required_roles": ["id", "text", "label"],
                    "source": "fixture",
                    "retrieved_at": "2026-10-27",
                    "license_or_terms": "verified competition data terms",
                }
            },
        },
        "competition": {
            "rules_verified": True,
            "eligibility_verified": False,
            "payout_verified": False,
            "allowed_labels": LABELS,
        },
        "report": {"path": "reports/audit/audit.json"},
        "validation": {
            "prompt_group_role": "prompt_id",
            "n_splits": 2,
            "seed": 42,
            "assignments_path": "reports/splits/folds.parquet",
            "report_path": "reports/splits/report.json",
        },
        "training": {"artifact_root": "reports/runs", "models": {}},
        "inference": {"model_candidate": "char_tfidf_svc"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_from_config(config_path)
    split = split_from_config(config_path)

    assignments = parquet.read_table(project_root / "reports" / "splits" / "folds.parquet").to_pylist()
    fold_by_id = {row["id"]: row["fold"] for row in assignments}
    predictions = [
        {
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "fold": fold_by_id[row["id"]],
            "candidate": "char_tfidf_svc",
            "actual": row["model"],
            "prediction": (
                "model-b" if row["id"] == "row-b1" else
                "model-a" if row["id"] == "row-a2" else
                row["model"]
            ),
        }
        for row in rows
    ]
    run_dir = project_root / "reports" / "runs" / "fixture-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist(predictions), run_dir / "predictions_oof.parquet")
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "fixture-run",
            "train_sha256": split["dataset_sha256"],
            "split_sha256": split["split_sha256"],
            "n_splits": 2,
            "seed": 42,
            "models": {"char_tfidf_svc": {"type": "fixture"}},
        }),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"models": {"char_tfidf_svc": {"balanced_accuracy": 0.5}}}),
        encoding="utf-8",
    )
    return config_path, rows, run_dir


def test_error_analysis_reports_aggregates_without_identifiers_or_text(tmp_path):
    config_path, _, run_dir = make_analysis_project(tmp_path)

    result = analyze_oof_run("fixture-run", config_path, candidate_name="char_tfidf_svc")

    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert result["summary"]["class_metrics"]["model-a"]["recall"] == 0.5
    assert result["summary"]["class_metrics"]["model-b"]["recall"] == 0.5
    assert result["summary"]["top_confusions"]
    assert result["summary"]["fold_metrics"]
    assert result["summary"]["prompt_group_errors"]["groups_with_errors"] == 2
    assert "row-a1" not in report_text
    assert "prompt-a" not in report_text
    assert "private passage" not in report_text
    assert (run_dir / "error_analysis.md").is_file()


def test_error_analysis_rejects_stale_dataset_checksum(tmp_path):
    config_path, _, run_dir = make_analysis_project(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        analyze_oof_run("fixture-run", config_path, candidate_name="char_tfidf_svc")


def test_evaluate_cli_writes_aggregate_error_report(tmp_path, capsys):
    config_path, _, run_dir = make_analysis_project(tmp_path)

    assert main(["evaluate", "--run-id", "fixture-run", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "evaluate: char_tfidf_svc balanced_accuracy=" in output
    assert (run_dir / "error_analysis.md").is_file()


def test_tuning_uses_same_saved_split_and_writes_aggregate_summary(tmp_path):
    config_path, _, _ = make_analysis_project(tmp_path)
    experiment_path = tmp_path / "configs" / "experiments" / "mini.yaml"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment = {
        "version": 1,
        "name": "mini-char-grid",
        "hypothesis": "A tighter character range may improve generalization.",
        "output_path": "reports/tuning/mini-char-grid/summary.json",
        "candidates": {
            "char_base": {
                "type": "char_tfidf",
                "analyzer": "char",
                "ngram_range": [2, 4],
                "min_df": 1,
                "max_features": 1000,
                "sublinear_tf": True,
                "c": 1.0,
            },
            "char_wb": {
                "type": "char_tfidf",
                "analyzer": "char_wb",
                "ngram_range": [2, 4],
                "min_df": 1,
                "max_features": 1000,
                "sublinear_tf": True,
                "c": 1.0,
            },
        },
    }
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    result = tune_from_config(config_path, experiment_path)

    assert result["status"] == "passed"
    assert result["train_sha256"]
    assert result["split_sha256"]
    assert [candidate["rank"] for candidate in result["candidates"]] == [1, 2]
    assert all(len(candidate["fold_balanced_accuracy"]) == 2 for candidate in result["candidates"])
    output = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert output == result


def test_tuning_rejects_stale_data_before_writing_summary(tmp_path):
    config_path, _, _ = make_analysis_project(tmp_path)
    experiment_path = tmp_path / "configs" / "experiments" / "mini.yaml"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "name": "stale-grid",
            "output_path": "reports/tuning/stale-grid/summary.json",
            "candidates": {"dummy": {"type": "dummy", "strategy": "most_frequent"}},
        }),
        encoding="utf-8",
    )
    train_path = tmp_path / "kaggle" / "train.csv"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        tune_from_config(config_path, experiment_path)

    assert not (tmp_path / "reports" / "tuning" / "stale-grid").exists()


def test_tune_routes_embedding_candidate_on_identical_folds_without_test_data(tmp_path, monkeypatch):
    import numpy as np

    from ntllm import embeddings as embeddings_module

    config_path, _, _ = make_analysis_project(tmp_path)
    assert not (tmp_path / "kaggle" / "test.csv").exists()

    def fake_encode(texts, model_dir, batch_size=64):
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i] = np.float32([float((hash(f"{text}-{j}") % 997)) / 997.0 for j in range(4)])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    monkeypatch.setattr(embeddings_module, "encode_texts", fake_encode)
    fake_model = tmp_path / "fake-minilm"
    fake_model.mkdir()
    (fake_model / "pytorch_model.bin").write_bytes(b"fake-weights")
    (fake_model / "config.json").write_text("{}", encoding="utf-8")
    experiment_path = tmp_path / "configs" / "experiments" / "emb.yaml"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "name": "emb-grid",
            "embedding_model_dir": "fake-minilm",
            "output_path": "reports/tuning/emb-grid/summary.json",
            "candidates": {
                "flat_dummy": {"type": "dummy", "strategy": "most_frequent"},
                "emb_logreg": {"type": "sbert_logreg", "c": 1.0, "max_iter": 200},
            },
        }),
        encoding="utf-8",
    )

    result = tune_from_config(config_path, experiment_path)

    assert result["status"] == "passed"
    assert [candidate["rank"] for candidate in result["candidates"]] == [1, 2]
    assert all(len(candidate["fold_balanced_accuracy"]) == 2 for candidate in result["candidates"])
    assert len(result["embedding_model_sha256"]) == 64


def test_tune_supports_style_candidates_on_identical_folds(tmp_path):
    config_path, _, _ = make_analysis_project(tmp_path)
    experiment_path = tmp_path / "configs" / "experiments" / "style.yaml"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "name": "style-grid",
            "output_path": "reports/tuning/style-grid/summary.json",
            "candidates": {
                "flat_dummy": {"type": "dummy", "strategy": "most_frequent"},
                "style_dummy": {"type": "style_svc", "c": 1.0},
            },
        }),
        encoding="utf-8",
    )

    result = tune_from_config(config_path, experiment_path)

    assert result["status"] == "passed"
    assert [candidate["rank"] for candidate in result["candidates"]] == [1, 2]
    assert all(len(candidate["fold_balanced_accuracy"]) == 2 for candidate in result["candidates"])
    assert not (tmp_path / "kaggle" / "test.csv").exists()


def test_tune_cli_reports_candidate_ranking(tmp_path, capsys):
    config_path, _, _ = make_analysis_project(tmp_path)
    experiment_path = tmp_path / "configs" / "experiments" / "cli.yaml"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "name": "cli-grid",
            "output_path": "reports/tuning/cli-grid/summary.json",
            "candidates": {"dummy": {"type": "dummy", "strategy": "most_frequent"}},
        }),
        encoding="utf-8",
    )

    assert main(["tune", "--config", str(config_path), "--experiment", str(experiment_path)]) == 0

    output = capsys.readouterr().out
    assert "tune: passed — 1/1 candidates" in output
    assert (tmp_path / "reports" / "tuning" / "cli-grid" / "summary.json").is_file()


def test_tune_reports_rerank_candidate_failure_explicitly_without_test_file(tmp_path):
    config_path, _, _ = make_analysis_project(tmp_path)
    assert not (tmp_path / "kaggle" / "test.csv").exists()
    experiment_path = tmp_path / "configs" / "experiments" / "rerank.yaml"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "name": "rerank-guard",
            "output_path": "reports/tuning/rerank-guard/summary.json",
            "candidates": {
                "flat_dummy": {"type": "dummy", "strategy": "most_frequent"},
                "trio_rerank": {
                    "type": "word_tfidf",
                    "ngram_range": [1, 1],
                    "min_df": 1,
                    "max_features": 100,
                    "sublinear_tf": True,
                    "c": 1.0,
                    "architecture": "gemini_rerank",
                },
            },
        }),
        encoding="utf-8",
    )

    result = tune_from_config(config_path, experiment_path)

    assert result["status"] == "failed"
    assert [candidate["name"] for candidate in result["candidates"]] == ["flat_dummy"]
    assert any(failure["name"] == "trio_rerank" for failure in result["failures"])
