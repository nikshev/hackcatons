# verifies: FR-005-01
# verifies: FR-005-02
# verifies: FR-005-03
# verifies: FR-005-04
# verifies: FR-005-05
# verifies: FR-005-06
# verifies: FR-005-07
# verifies: FR-005-08

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet
import pytest
import yaml

from scripts.build_submission_notebook import build_notebook


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_notebook_is_generated_from_current_baseline_config():
    project_root = Path(__file__).resolve().parents[1]
    notebook = build_notebook(project_root / "configs" / "baseline.yaml")
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert "CONFIG_VERSION = 12" in code
    assert "MODEL_CANDIDATE = 'char_tfidf_svc'" in code
    assert "claude-fable-5" in code
    assert "submission.parquet" in code
    assert "pip install" not in code
    assert "requests" not in code
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_committed_notebook_matches_the_current_generator():
    project_root = Path(__file__).resolve().parents[1]
    generated = build_notebook(project_root / "configs" / "baseline.yaml")
    committed = json.loads((project_root / "notebooks" / "submission.ipynb").read_text(encoding="utf-8"))

    assert committed == generated


def test_generated_notebook_runs_offline_on_fixture_and_writes_valid_submission(tmp_path, monkeypatch):
    labels = ["model-a", "model-b"]
    config = {
        "version": 8,
        "data": {
            "column_roles": {"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"},
            "files": {},
        },
        "competition": {"allowed_labels": labels},
        "training": {
            "models": {
                "char_tfidf_svc": {
                    "type": "char_tfidf",
                    "ngram_range": [2, 4],
                    "min_df": 1,
                    "max_features": 2000,
                    "sublinear_tf": True,
                    "c": 1.0,
                }
            }
        },
        "validation": {"seed": 42},
        "inference": {"model_candidate": "char_tfidf_svc"},
    }
    config_path = tmp_path / "project" / "configs" / "baseline.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    data_dir = tmp_path / "competition-data"
    write_csv(
        data_dir / "train.csv",
        ["id", "prompt_id", "text", "model"],
        [
            {"id": "a1", "prompt_id": "p1", "text": "alpha blue flower", "model": "model-a"},
            {"id": "a2", "prompt_id": "p2", "text": "alpha green garden", "model": "model-a"},
            {"id": "b1", "prompt_id": "p3", "text": "beta orange engine", "model": "model-b"},
            {"id": "b2", "prompt_id": "p4", "text": "beta yellow machine", "model": "model-b"},
        ],
    )
    write_csv(
        data_dir / "test.csv",
        ["id", "text"],
        [
            {"id": "test-1", "text": "alpha flower"},
            {"id": "test-2", "text": "beta engine"},
        ],
    )
    output_dir = tmp_path / "notebook-output"
    monkeypatch.setenv("NTLLM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("NTLLM_OUTPUT_DIR", str(output_dir))
    notebook = build_notebook(config_path)
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    exec(compile(code, "submission-notebook-cell.py", "exec"), {"__name__": "__main__"})

    submission = parquet.read_table(output_dir / "submission.parquet")
    assert submission.schema.names == ["id", "model"]
    rows = submission.to_pylist()
    assert [row["id"] for row in rows] == ["test-1", "test-2"]
    assert all(row["model"] in labels for row in rows)
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["train_rows"] == 4
    assert summary["test_rows"] == 2
    assert summary["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert len(summary["train_sha256"]) == 64
    assert len(summary["test_sha256"]) == 64
    assert summary["fit_seconds"] >= 0
    assert summary["predict_seconds"] >= 0
    assert "alpha blue flower" not in json.dumps(summary)


def test_notebook_generator_rejects_non_char_selected_model(tmp_path):
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "data": {"column_roles": {"id": "id", "text": "text", "label": "model"}},
                "competition": {"allowed_labels": ["a"]},
                "training": {"models": {"word": {"type": "word_tfidf"}}},
                "inference": {"model_candidate": "word"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="char_tfidf"):
        build_notebook(config_path)


def test_generated_notebook_rejects_unlisted_train_label_before_output(tmp_path, monkeypatch):
    labels = ["model-a", "model-b"]
    config = {
        "version": 8,
        "data": {
            "column_roles": {"id": "id", "prompt_id": "prompt_id", "text": "text", "label": "model"}
        },
        "competition": {"allowed_labels": labels},
        "training": {
            "models": {
                "char_tfidf_svc": {
                    "type": "char_tfidf",
                    "ngram_range": [2, 3],
                    "min_df": 1,
                    "max_features": 1000,
                    "sublinear_tf": True,
                    "c": 1.0,
                }
            }
        },
        "validation": {"seed": 42},
        "inference": {"model_candidate": "char_tfidf_svc"},
    }
    config_path = tmp_path / "project" / "configs" / "baseline.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    data_dir = tmp_path / "bad-data"
    write_csv(
        data_dir / "train.csv",
        ["id", "prompt_id", "text", "model"],
        [{"id": "a", "prompt_id": "p1", "text": "text", "model": "outside"}],
    )
    write_csv(data_dir / "test.csv", ["id", "text"], [{"id": "t", "text": "test"}])
    output_dir = tmp_path / "bad-output"
    monkeypatch.setenv("NTLLM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("NTLLM_OUTPUT_DIR", str(output_dir))
    code = "".join(
        "".join(cell["source"])
        for cell in build_notebook(config_path)["cells"]
        if cell["cell_type"] == "code"
    )

    with pytest.raises(ValueError, match="official allowlist"):
        exec(compile(code, "submission-notebook.py", "exec"), {"__name__": "__main__"})

    assert not (output_dir / "submission.parquet").exists()
