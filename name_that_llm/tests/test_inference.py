# verifies: FR-004-01
# verifies: FR-004-02
# verifies: FR-004-03
# verifies: FR-004-04
# verifies: FR-004-05
# verifies: FR-004-06
# verifies: FR-004-07
# verifies: FR-004-08
# verifies: FR-004-09

import csv
import json

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import yaml

from ntllm.audit import audit_from_config
from ntllm.inference import (
    InferenceError,
    fit_final_from_config,
    predict_from_artifact,
    validate_submission_file,
)
from ntllm.splits import split_from_config
from ntllm.__main__ import main


LABELS = ["model-a", "model-b"]
MODEL_CONFIG = {
    "type": "word_tfidf",
    "ngram_range": [1, 1],
    "min_df": 1,
    "max_features": 1000,
    "sublinear_tf": True,
    "c": 1.0,
}


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_inference_project(project_root):
    config_path = project_root / "configs" / "baseline.yaml"
    train_rows = [
        {
            "id": f"train-{group}-{label}",
            "prompt_id": f"prompt-{group}",
            "text": f"{label} answer about subject {group}",
            "model": label,
        }
        for group in range(4)
        for label in LABELS
    ]
    test_rows = [
        {"id": "test-b", "text": "model-b answer"},
        {"id": "test-a", "text": "model-a answer"},
    ]
    write_csv(project_root / "kaggle" / "train.csv", ["id", "prompt_id", "text", "model"], train_rows)
    write_csv(project_root / "kaggle" / "test.csv", ["id", "text"], test_rows)

    config = {
        "version": 8,
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
        "training": {
            "artifact_root": "reports/runs",
            "models": {"word": MODEL_CONFIG},
        },
        "inference": {
            "model_candidate": "word",
            "selected_from_run": "selected-oof-run",
            "artifact_root": "artifacts/final_models",
            "submission_output": "reports/submissions/submission.parquet",
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit = audit_from_config(config_path)
    split = split_from_config(config_path)

    selected_run = project_root / "reports" / "runs" / "selected-oof-run"
    selected_run.mkdir(parents=True, exist_ok=True)
    (selected_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "selected-oof-run",
                "train_sha256": audit["files"]["train"]["sha256"],
                "split_sha256": split["split_sha256"],
                "models": {"word": MODEL_CONFIG},
            }
        ),
        encoding="utf-8",
    )
    (selected_run / "metrics.json").write_text(
        json.dumps({"models": {"word": {"balanced_accuracy": 0.5}}}),
        encoding="utf-8",
    )
    return config_path, project_root / "kaggle" / "test.csv", test_rows


def test_fit_final_predict_and_validate_offline_test_sample(tmp_path):
    config_path, test_path, test_rows = make_inference_project(tmp_path)
    # Final fitting must not open test.csv after the audit/split are complete.
    test_path.write_bytes(b"not a readable CSV; fit-final must not open test data")

    fit = fit_final_from_config(config_path)

    model_dir = fit["model_dir"]
    assert (model_dir / "model.joblib").is_file()
    assert (model_dir / "manifest.json").is_file()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate"] == "word"
    assert manifest["train_sha256"]
    assert manifest["model_sha256"]
    assert manifest["competition_entry_readiness"] == "blocked"

    write_csv(test_path, ["id", "text"], test_rows)
    first_path = tmp_path / "reports" / "submissions" / "first.parquet"
    second_path = tmp_path / "reports" / "submissions" / "second.parquet"
    predict_from_artifact(model_dir, test_path, first_path)
    predict_from_artifact(model_dir, test_path, second_path)

    first = parquet.read_table(first_path)
    second = parquet.read_table(second_path)
    assert first.schema.names == ["id", "model"]
    assert first.to_pylist() == second.to_pylist()
    assert [row["id"] for row in first.to_pylist()] == [row["id"] for row in test_rows]
    result = validate_submission_file(first_path, test_path, LABELS)
    assert result["status"] == "passed"


def test_final_fit_stops_when_training_readiness_is_blocked(tmp_path):
    config_path, _, _ = make_inference_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["competition"]["rules_verified"] = False
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_from_config(config_path)

    with pytest.raises(InferenceError, match="training readiness is blocked"):
        fit_final_from_config(config_path)

    assert not (tmp_path / "artifacts" / "final_models").exists()


def test_predict_rejects_invalid_test_without_writing_submission(tmp_path):
    config_path, test_path, _ = make_inference_project(tmp_path)
    fit = fit_final_from_config(config_path)
    write_csv(test_path, ["id", "text"], [{"id": "bad-row", "text": "   "}])
    output = tmp_path / "reports" / "submissions" / "bad.parquet"

    with pytest.raises(InferenceError, match="blank text"):
        predict_from_artifact(fit["model_dir"], test_path, output)

    assert not output.exists()


def test_fit_predict_and_validate_cli_commands(tmp_path, capsys):
    config_path, test_path, test_rows = make_inference_project(tmp_path)

    assert main(["fit-final", "--config", str(config_path)]) == 0
    fit_output = capsys.readouterr().out
    model_path = fit_output.strip().splitlines()[-1].split(": ", 1)[-1]
    submission = tmp_path / "reports" / "submissions" / "sample-preview.parquet"

    assert main(
        [
            "predict",
            "--model",
            model_path,
            "--input",
            str(test_path),
            "--output",
            str(submission),
        ]
    ) == 0
    assert main(
        [
            "validate-submission",
            "--input",
            str(submission),
            "--test",
            str(test_path),
            "--config",
            str(config_path),
        ]
    ) == 0
    assert main(
        [
            "predict",
            "--model",
            model_path,
            "--input",
            str(test_path),
            "--config",
            str(config_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "validate-submission: passed" in output
    assert parquet.read_table(submission).num_rows == len(test_rows)
    assert parquet.read_table(tmp_path / "reports" / "submissions" / "submission.parquet").num_rows == len(test_rows)


@pytest.mark.parametrize(
    ("columns", "rows", "message"),
    [
        (["id", "model", "extra"], [{"id": "a", "model": "model-a", "extra": 1}], "exactly columns"),
        (["id", "model"], [{"id": "a", "model": "not-allowed"}], "outside the official allowlist"),
        (["id", "model"], [{"id": "a", "model": None}], "null/blank"),
    ],
)
def test_submission_validator_rejects_schema_nulls_and_invalid_labels(tmp_path, columns, rows, message):
    _, test_path, _ = make_inference_project(tmp_path)
    submission = tmp_path / "bad.parquet"
    parquet.write_table(pa.Table.from_pylist(rows), submission)

    result = validate_submission_file(submission, test_path, LABELS)

    assert result["status"] == "failed"
    assert any(message in issue for issue in result["issues"])


def test_submission_validator_reports_missing_duplicate_and_extra_ids(tmp_path):
    _, test_path, _ = make_inference_project(tmp_path)
    submission = tmp_path / "bad-ids.parquet"
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {"id": "test-b", "model": "model-a"},
                {"id": "test-b", "model": "model-b"},
                {"id": "extra", "model": "model-a"},
            ]
        ),
        submission,
    )

    result = validate_submission_file(submission, test_path, LABELS)

    assert result["status"] == "failed"
    assert result["duplicate_ids"] == 1
    assert result["missing_ids"] == 1
    assert result["extra_ids"] == 1
