# verifies: FR-002-01
# verifies: FR-002-02
# verifies: FR-002-03
# verifies: FR-002-04
# verifies: FR-002-05
# verifies: FR-002-06
# verifies: FR-002-07
# verifies: FR-002-08

import csv
import hashlib
from pathlib import Path

import pytest
import yaml

from ntllm.splits import SplitError, assign_prompt_folds, split_from_config
from ntllm.__main__ import main


LABELS = ("model-a", "model-b", "model-c")


def balanced_rows(group_count=20):
    return [
        {
            "id": f"row-{group}-{label}",
            "prompt_id": f"prompt-{group:03}",
            "model": label,
        }
        for group in range(group_count)
        for label in LABELS
    ]


def split(rows, *, n_splits=5, seed=42):
    return assign_prompt_folds(
        rows,
        id_column="id",
        group_column="prompt_id",
        label_column="model",
        allowed_labels=LABELS,
        n_splits=n_splits,
        seed=seed,
    )


def test_baseline_config_has_explicit_split_seed_and_artifact_paths():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["version"] == 9
    assert config["validation"] == {
        "prompt_group_role": "prompt_id",
        "n_splits": 5,
        "seed": 42,
        "assignments_path": "reports/splits/prompt_folds.parquet",
        "report_path": "reports/splits/prompt_folds.json",
    }


def test_assign_prompt_folds_keeps_groups_together_and_assigns_each_id_once():
    rows = balanced_rows(10)

    result = split(rows)

    assignments = result.assignments
    assert len(assignments) == len(rows)
    assert len({row["id"] for row in assignments}) == len(rows)
    folds_by_prompt = {}
    for row in assignments:
        folds_by_prompt.setdefault(row["prompt_id"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in folds_by_prompt.values())


def test_assign_prompt_folds_balances_uniform_prompt_groups():
    result = split(balanced_rows())

    assert result.row_counts == (12, 12, 12, 12, 12)
    assert result.group_counts == (4, 4, 4, 4, 4)
    assert result.label_counts == tuple({label: 4 for label in LABELS} for _ in range(5))
    assert len(result.split_hash) == 64


def test_assign_prompt_folds_is_stable_when_input_rows_are_reordered():
    rows = balanced_rows()

    original = split(rows)
    reordered = split(list(reversed(rows)))

    assert original.split_hash == reordered.split_hash
    assert original.assignments == reordered.assignments


@pytest.mark.parametrize(
    ("row_update", "message"),
    [
        ({"id": ""}, "blank id"),
        ({"prompt_id": "  "}, "blank prompt group"),
        ({"model": ""}, "blank label"),
        ({"model": "outside-allowlist"}, "unrecognized label"),
    ],
)
def test_assign_prompt_folds_rejects_invalid_fields(row_update, message):
    rows = balanced_rows(5)
    rows[0] = {**rows[0], **row_update}

    with pytest.raises(SplitError, match=message):
        split(rows, n_splits=5)


def test_assign_prompt_folds_rejects_duplicate_ids():
    rows = balanced_rows(5)
    rows[1]["id"] = rows[0]["id"]

    with pytest.raises(SplitError, match="duplicate id"):
        split(rows, n_splits=5)


@pytest.mark.parametrize("n_splits", [0, 1, -2, True])
def test_assign_prompt_folds_rejects_invalid_fold_count(n_splits):
    with pytest.raises(SplitError, match="at least two folds"):
        split(balanced_rows(5), n_splits=n_splits)


def test_assign_prompt_folds_rejects_fewer_prompt_groups_than_folds():
    with pytest.raises(SplitError, match="fewer prompt groups than folds"):
        split(balanced_rows(4), n_splits=5)


def write_split_project(project_root, *, duplicate_id=False):
    config_path = project_root / "configs" / "baseline.yaml"
    train_path = project_root / "data" / "raw" / "train.csv"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    rows = balanced_rows(10)
    if duplicate_id:
        rows[-1]["id"] = rows[0]["id"]
    with train_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "prompt_id", "text", "model"])
        writer.writeheader()
        writer.writerows(
            {**row, "text": f"text for {row['id']}"}
            for row in rows
        )

    config = {
        "version": 4,
        "data": {
            "column_roles": {
                "id": "id",
                "prompt_id": "prompt_id",
                "text": "text",
                "label": "model",
            },
            "files": {
                "train": {"path": "data/raw/train.csv"},
                # This file is intentionally absent: split must never read test.
                "test": {"path": "data/raw/test.csv"},
            },
        },
        "competition": {"allowed_labels": list(LABELS)},
        "validation": {
            "prompt_group_role": "prompt_id",
            "n_splits": 5,
            "seed": 42,
            "assignments_path": "reports/splits/folds.parquet",
            "report_path": "reports/splits/report.json",
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_split_from_config_writes_parquet_and_reproducibility_report(tmp_path):
    config_path = write_split_project(tmp_path)
    train_path = tmp_path / "data" / "raw" / "train.csv"
    raw_before = train_path.read_bytes()

    report = split_from_config(config_path)

    assert report["configuration"]["version"] == 4
    assert len(report["configuration"]["sha256"]) == 64
    assert report["seed"] == 42
    assert report["n_splits"] == 5
    assert report["row_count"] == 30
    assert report["dataset_sha256"] == hashlib.sha256(raw_before).hexdigest()
    assert len(report["split_sha256"]) == 64
    assert [fold["row_count"] for fold in report["folds"]] == [6, 6, 6, 6, 6]
    assert report["folds"][0]["label_counts"] == {label: 2 for label in LABELS}
    assert train_path.read_bytes() == raw_before

    import pyarrow.parquet as parquet

    assignments_path = tmp_path / "reports" / "splits" / "folds.parquet"
    report_path = tmp_path / "reports" / "splits" / "report.json"
    assignments = parquet.read_table(assignments_path).to_pylist()
    assert len(assignments) == 30
    assert set(assignments[0]) == {"id", "prompt_id", "fold"}
    assert {row["fold"] for row in assignments} == {0, 1, 2, 3, 4}
    assert report_path.is_file()
    report_contents = report_path.read_text(encoding="utf-8")
    assert "text for" not in report_contents
    assert yaml.safe_load(report_contents) == report


def test_split_from_config_writes_no_artifacts_for_invalid_training_rows(tmp_path):
    config_path = write_split_project(tmp_path, duplicate_id=True)

    with pytest.raises(SplitError, match="duplicate id"):
        split_from_config(config_path)

    output_dir = tmp_path / "reports" / "splits"
    assert not output_dir.exists()


def test_split_cli_runs_configured_grouped_split(tmp_path, capsys):
    config_path = write_split_project(tmp_path)

    assert main(["split", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "split: wrote 30 rows across 5 folds" in output
    assert "split sha256:" in output


def test_split_cli_returns_failure_without_artifacts_for_invalid_input(tmp_path, capsys):
    config_path = write_split_project(tmp_path, duplicate_id=True)

    assert main(["split", "--config", str(config_path)]) == 1

    error = capsys.readouterr().err
    assert "duplicate id" in error
    assert not (tmp_path / "reports" / "splits").exists()
