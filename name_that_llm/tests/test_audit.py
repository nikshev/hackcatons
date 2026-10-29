# verifies: FR-001-03
# verifies: FR-001-04
# verifies: FR-001-02
# verifies: FR-001-05
# verifies: FR-001-07
# verifies: FR-001-08

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ntllm.audit import audit_from_config, build_manifest_entry, profile_csv
from ntllm.__main__ import main
from ntllm.data import AuditDataError, inspect_csv


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_baseline_config_matches_current_official_kaggle_allowlist():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["version"] == 13
    assert "prompt_id" not in config["data"]["files"]["train"]["required_roles"]
    assert config["competition"]["rules_verified"] is True
    assert config["competition"]["eligibility_verified"] is False
    assert config["competition"]["allowed_labels"] == [
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "qwen3.8-max",
        "glm-5.3",
        "glm-5.3-flash",
        "kimi-k3",
    ]


def test_inspect_csv_uses_configured_roles_and_reports_observed_schema(tmp_path):
    source = tmp_path / "train.csv"
    write_csv(
        source,
        ["record_id", "prompt", "passage", "author"],
        [["r1", "p1", "A sample passage", "model-a"]],
    )

    result = inspect_csv(
        source,
        column_roles={
            "id": "record_id",
            "prompt_id": "prompt",
            "text": "passage",
            "label": "author",
        },
        required_roles=("id", "prompt_id", "text", "label"),
    )

    assert result.columns == ("record_id", "prompt", "passage", "author")
    assert result.row_count == 1
    assert result.role_columns["label"] == "author"


def test_inspect_csv_rejects_missing_configured_column_with_actionable_error(tmp_path):
    source = tmp_path / "train.csv"
    write_csv(source, ["id", "prompt_id", "text"], [["r1", "p1", "text"]])

    with pytest.raises(AuditDataError, match="model.*label"):
        inspect_csv(
            source,
            column_roles={"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"},
            required_roles=("id", "prompt_id", "text", "label"),
        )


def test_inspect_csv_rejects_duplicate_header_names(tmp_path):
    source = tmp_path / "train.csv"
    write_csv(source, ["id", "text", "text"], [["r1", "first", "second"]])

    with pytest.raises(AuditDataError, match="duplicate column"):
        inspect_csv(
            source,
            column_roles={"id": "id", "text": "text"},
            required_roles=("id", "text"),
        )


def test_inspect_csv_rejects_rows_with_wrong_number_of_fields(tmp_path):
    source = tmp_path / "train.csv"
    source.write_text("id,text\nr1,one,extra\n", encoding="utf-8")

    with pytest.raises(AuditDataError, match="row 2.*2 columns"):
        inspect_csv(
            source,
            column_roles={"id": "id", "text": "text"},
            required_roles=("id", "text"),
        )


def test_inspect_csv_rejects_non_utf8_input_with_clear_error(tmp_path):
    source = tmp_path / "train.csv"
    source.write_bytes(b"id,text\nr1,\xff\n")

    with pytest.raises(AuditDataError, match="not valid UTF-8"):
        inspect_csv(
            source,
            column_roles={"id": "id", "text": "text"},
            required_roles=("id", "text"),
        )


def test_manifest_records_provenance_checksum_rows_and_columns_without_mutation(tmp_path):
    source = tmp_path / "train.csv"
    write_csv(source, ["id", "prompt_id", "text", "model"], [["r1", "p1", "hello", "model-a"]])
    original_bytes = source.read_bytes()
    expected_hash = hashlib.sha256(original_bytes).hexdigest()

    entry = build_manifest_entry(
        source,
        source_name="Kaggle Name That LLM",
        retrieved_at="2026-10-26",
        license_or_terms="competition terms",
        column_roles={"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"},
        required_roles=("id", "prompt_id", "text", "label"),
    )

    assert entry.source == "Kaggle Name That LLM"
    assert entry.retrieved_at == "2026-10-26"
    assert entry.license_or_terms == "competition terms"
    assert entry.sha256 == expected_hash
    assert entry.row_count == 1
    assert entry.columns == ("id", "prompt_id", "text", "model")
    assert source.read_bytes() == original_bytes


def test_profile_csv_reports_nulls_text_lengths_duplicates_and_label_counts(tmp_path):
    source = tmp_path / "train.csv"
    write_csv(
        source,
        ["id", "prompt_id", "text", "model"],
        [
            ["a", "p1", "alpha", "model-a"],
            ["b", "p1", "alpha", "model-b"],
            ["c", "p2", "longer", "model-a"],
            ["c", "p2", "longer", "model-a"],
            ["d", "p2", "", ""],
        ],
    )

    profile = profile_csv(
        source,
        column_roles={"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"},
    )

    assert profile["null_counts"] == {"id": 0, "prompt_id": 0, "text": 1, "model": 1}
    assert profile["text_length"] == {"count": 4, "min": 5, "median": 5.5, "max": 6}
    assert profile["duplicates"] == {"exact_row_count": 1, "exact_text_count": 2}
    assert profile["label_counts"] == {"model-a": 3, "model-b": 1}


def test_profile_csv_omits_label_counts_for_unlabeled_test_file(tmp_path):
    source = tmp_path / "test.csv"
    write_csv(source, ["id", "text"], [["a", "sample"]])

    profile = profile_csv(source, column_roles={"id": "id", "text": "text", "label": "model"})

    assert profile["label_counts"] is None


def make_config(project_root, *, train_exists=True):
    train_path = project_root / "data" / "raw" / "train.csv"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    if train_exists:
        write_csv(
            train_path,
            ["id", "prompt_id", "text", "model"],
            [["r1", "p1", "private fixture text", "model-a"]],
        )

    config_path = project_root / "configs" / "baseline.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "data": {
            "column_roles": {
                "id": "id",
                "prompt_id": "prompt_id",
                "text": "text",
                "label": "model",
            },
            "files": {
                "train": {
                    "path": "data/raw/train.csv",
                    "required_roles": ["id", "prompt_id", "text", "label"],
                    "source": "fixture",
                    "retrieved_at": "unverified",
                    "license_or_terms": "unverified",
                }
            },
        },
        "competition": {
            "rules_verified": False,
            "eligibility_verified": False,
            "allowed_labels": [],
        },
        "report": {"path": "reports/audit/audit.json"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_audit_from_config_writes_manifest_without_exposing_raw_text(tmp_path):
    config_path = make_config(tmp_path)

    report = audit_from_config(config_path)

    assert report["audit_status"] == "passed"
    assert report["files"]["train"]["row_count"] == 1
    assert report["files"]["train"]["quality"]["label_counts"] == {"model-a": 1}
    assert report["files"]["train"]["integrity"]["prompt_groups"]["available"] is True
    assert report["configuration"]["version"] == 1
    assert len(report["configuration"]["sha256"]) == 64
    assert report["training_readiness"]["status"] == "blocked"
    assert "competition rules" in " ".join(report["training_readiness"]["blockers"])
    report_path = tmp_path / "reports" / "audit" / "audit.json"
    serialized = report_path.read_text(encoding="utf-8")
    assert json.loads(serialized) == report
    assert "private fixture text" not in serialized


def test_audit_from_config_reports_missing_input_as_failed_and_blocked(tmp_path):
    config_path = make_config(tmp_path, train_exists=False)

    report = audit_from_config(config_path)

    assert report["audit_status"] == "failed"
    assert "does not exist" in report["files"]["train"]["error"]
    assert report["training_readiness"]["status"] == "blocked"
    assert (tmp_path / "reports" / "audit" / "audit.json").is_file()


def test_audit_cli_returns_failure_for_missing_data(tmp_path):
    config_path = make_config(tmp_path, train_exists=False)

    assert main(["audit", "--config", str(config_path)]) == 1


def test_audit_cli_succeeds_when_structure_passes_but_training_is_blocked(tmp_path, capsys):
    config_path = make_config(tmp_path)

    assert main(["audit", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "audit: passed" in output
    assert "training readiness: blocked" in output


def test_audit_rejects_label_outside_configured_allowlist(tmp_path):
    config_path = make_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["competition"]["allowed_labels"] = ["permitted-model"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = audit_from_config(config_path)

    assert report["audit_status"] == "failed"
    assert "unrecognized label" in report["files"]["train"]["error"]
    assert "model-a" in report["files"]["train"]["error"]


def test_audit_fails_and_reports_duplicate_ids_and_blank_text(tmp_path):
    config_path = make_config(tmp_path)
    write_csv(
        tmp_path / "data" / "raw" / "train.csv",
        ["id", "prompt_id", "text", "model"],
        [["same", "p1", "first", "model-a"], ["same", "p2", "  ", "model-a"]],
    )

    report = audit_from_config(config_path)

    assert report["audit_status"] == "failed"
    assert report["files"]["train"]["integrity"]["row_integrity"] == {
        "duplicate_id_count": 1,
        "missing_id_count": 0,
        "invalid_text_count": 1,
    }
    assert "duplicate id" in report["files"]["train"]["error"]
    assert "empty text" in report["files"]["train"]["error"]


def test_audit_reports_unavailable_prompt_groups_without_silently_failing_schema(tmp_path):
    config_path = make_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["files"]["train"]["required_roles"] = ["id", "text", "label"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_csv(
        tmp_path / "data" / "raw" / "train.csv",
        ["id", "text", "model"],
        [["a", "sample", "model-a"]],
    )

    report = audit_from_config(config_path)

    assert report["audit_status"] == "passed"
    assert report["files"]["train"]["integrity"]["prompt_groups"]["available"] is False
    assert report["files"]["train"]["issues"] == [
        "prompt group column 'prompt_id' is unavailable"
    ]


def test_training_readiness_opens_only_when_rules_data_and_allowlist_are_verified(tmp_path):
    config_path = make_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["competition"].update(
        rules_verified=True,
        eligibility_verified=False,
        payout_verified=False,
        allowed_labels=["model-a"],
    )
    train = config["data"]["files"]["train"]
    train["retrieved_at"] = "2026-10-26"
    train["license_or_terms"] = "verified competition terms"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = audit_from_config(config_path)

    assert report["audit_status"] == "passed"
    assert report["training_readiness"] == {"status": "ready", "blockers": []}
    assert report["competition_entry_readiness"] == {
        "status": "blocked",
        "blockers": ["participant eligibility is not verified", "payout conditions are not verified"],
    }
