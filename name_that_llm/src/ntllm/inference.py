"""Final fitting, offline prediction, and Kaggle submission validation.

impl: FR-004-01, FR-004-02
impl: FR-004-03, FR-004-04
impl: FR-004-05, FR-004-06
impl: FR-004-07, FR-004-08
impl: FR-004-09
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import pyarrow as pa
import pyarrow.parquet as parquet
import yaml

from ntllm.data import AuditDataError, inspect_csv
from ntllm.training import (
    TrainingError,
    _git_provenance,
    _package_version,
    _sha256,
    _source_sha256,
    build_estimator,
)


class InferenceError(ValueError):
    """Final fit, prediction, or submission validation failed."""


def _read_yaml(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        content = path.read_bytes()
        value = yaml.safe_load(content)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InferenceError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InferenceError(f"{path}: configuration must be a YAML mapping")
    return value, content


def _project_path(project_root: Path, value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InferenceError(f"configuration field '{key}' must be a nonblank path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def default_submission_output(config_path: str | Path) -> Path:
    """Resolve the versioned default submission destination from config."""
    source_config = Path(config_path).resolve()
    config, _ = _read_yaml(source_config)
    inference = config.get("inference")
    if not isinstance(inference, dict):
        raise InferenceError(f"{source_config}: 'inference' mapping is required")
    return _project_path(
        source_config.parent.parent,
        inference.get("submission_output"),
        key="inference.submission_output",
    )


def _json_file(path: Path, *, purpose: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError(f"cannot read {purpose} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InferenceError(f"{purpose} {path} must contain a JSON object")
    return value


def fit_final_from_config(config_path: str | Path) -> dict[str, object]:
    """Fit the OOF-selected model on all audited train rows and save a bundle."""
    source_config = Path(config_path).resolve()
    if not source_config.is_file():
        raise InferenceError(f"configuration file does not exist: {source_config}")
    config, config_bytes = _read_yaml(source_config)
    config_version = config.get("version")
    project_root = source_config.parent.parent
    config_hash = hashlib.sha256(config_bytes).hexdigest()

    report_config = config.get("report")
    if not isinstance(report_config, dict):
        raise InferenceError(f"{source_config}: 'report' mapping is required")
    audit_path = _project_path(project_root, report_config.get("path"), key="report.path")
    audit = _json_file(audit_path, purpose="audit report")
    audit_config = audit.get("configuration", {})
    if audit_config.get("sha256") != config_hash or audit_config.get("version") != config_version:
        raise InferenceError("audit report is stale for this config; rerun `ntllm audit`")
    if audit.get("audit_status") != "passed":
        raise InferenceError("data audit did not pass")
    readiness = audit.get("training_readiness", {})
    if readiness.get("status") != "ready":
        blockers = "; ".join(readiness.get("blockers", []))
        raise InferenceError(f"training readiness is blocked: {blockers}")

    data_config = config.get("data")
    if not isinstance(data_config, dict) or not isinstance(data_config.get("column_roles"), dict):
        raise InferenceError(f"{source_config}: 'data.column_roles' mapping is required")
    roles = data_config["column_roles"]
    files = data_config.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("train"), dict):
        raise InferenceError(f"{source_config}: 'data.files.train' mapping is required")
    train_path = _project_path(project_root, files["train"].get("path"), key="data.files.train.path")
    audited_train = audit.get("files", {}).get("train", {})
    train_sha = audited_train.get("sha256")
    if not isinstance(train_sha, str) or not train_sha or _sha256(train_path) != train_sha:
        raise InferenceError("train checksum differs from the passing audit")

    inference = config.get("inference")
    training = config.get("training")
    competition = config.get("competition")
    if not isinstance(inference, dict) or not isinstance(training, dict) or not isinstance(competition, dict):
        raise InferenceError(f"{source_config}: inference/training/competition mappings are required")
    candidate = inference.get("model_candidate")
    selected_run_id = inference.get("selected_from_run")
    models = training.get("models")
    allowed_labels = competition.get("allowed_labels")
    if (
        not isinstance(candidate, str)
        or not isinstance(selected_run_id, str)
        or Path(selected_run_id).name != selected_run_id
        or not isinstance(models, dict)
        or candidate not in models
        or not isinstance(allowed_labels, list)
        or not allowed_labels
    ):
        raise InferenceError("selected model, OOF run, candidate config, and official labels are required")

    training_root = _project_path(project_root, training.get("artifact_root"), key="training.artifact_root")
    selected_run_dir = training_root / selected_run_id
    selected_manifest = _json_file(selected_run_dir / "manifest.json", purpose="selected OOF manifest")
    selected_metrics = _json_file(selected_run_dir / "metrics.json", purpose="selected OOF metrics")
    if selected_manifest.get("train_sha256") != train_sha:
        raise InferenceError("selected OOF run uses a different train dataset")
    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise InferenceError(f"{source_config}: 'validation' mapping is required")
    split_report_path = _project_path(
        project_root,
        validation.get("report_path"),
        key="validation.report_path",
    )
    current_split = _json_file(split_report_path, purpose="current split report")
    if (
        selected_manifest.get("split_sha256") != current_split.get("split_sha256")
        or current_split.get("dataset_sha256") != train_sha
    ):
        raise InferenceError("selected OOF model uses a different prompt-group split")
    selected_models = selected_manifest.get("models")
    if not isinstance(selected_models, dict) or selected_models.get(candidate) != models[candidate]:
        raise InferenceError("selected OOF model parameters differ from current training config")
    if candidate not in selected_metrics.get("models", {}):
        raise InferenceError(f"selected OOF run has no metrics for '{candidate}'")

    roles_required = ("id", "text", "label")
    try:
        inspection = inspect_csv(train_path, column_roles=roles, required_roles=roles_required)
        with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
            train_rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error, AuditDataError) as exc:
        raise InferenceError(f"cannot load audited train data: {exc}") from exc
    if _sha256(train_path) != train_sha:
        raise InferenceError("train data changed while final fitting was being prepared")

    label_column = roles["label"]
    observed_labels = {row.get(label_column) for row in train_rows}
    if observed_labels != set(allowed_labels):
        raise InferenceError("train labels do not exactly match the official allowlist")
    id_column = roles["id"]
    text_column = roles["text"]
    seed = config.get("validation", {}).get("seed")
    estimator = build_estimator(candidate, models[candidate], seed=seed)
    try:
        estimator.fit(
            [row[text_column] for row in train_rows],
            [row[label_column] for row in train_rows],
        )
    except Exception as exc:
        raise InferenceError(f"final fit failed for '{candidate}': {exc}") from exc
    if _sha256(train_path) != train_sha:
        raise InferenceError("raw train file changed during final fitting")

    source_hash = _source_sha256(project_root)
    model_payload = {
        "config_sha256": config_hash,
        "train_sha256": train_sha,
        "selected_run_id": selected_run_id,
        "candidate": candidate,
        "source_sha256": source_hash,
    }
    model_id = hashlib.sha256(
        json.dumps(model_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    artifact_root = _project_path(
        project_root,
        inference.get("artifact_root"),
        key="inference.artifact_root",
    )
    model_dir = artifact_root / model_id
    if model_dir.exists():
        existing = _json_file(model_dir / "manifest.json", purpose="final model manifest")
        existing_model = model_dir / "model.joblib"
        if (
            existing.get("model_id") != model_id
            or not existing_model.is_file()
            or _sha256(existing_model) != existing.get("model_sha256")
        ):
            raise InferenceError(f"existing model artifact is stale or incomplete: {model_dir}")
        return {"model_id": model_id, "model_dir": model_dir, "manifest": existing, "reused": True}

    commit, dirty = _git_provenance(project_root)
    manifest: dict[str, object] = {
        **model_payload,
        "model_id": model_id,
        "config_version": config_version,
        "candidate_parameters": models[candidate],
        "row_count": inspection.row_count,
        "id_column": id_column,
        "text_column": text_column,
        "label_column": label_column,
        "allowed_labels": allowed_labels,
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python_version": platform.python_version(),
        "library_versions": {
            package: _package_version(package)
            for package in ("numpy", "scikit-learn", "scipy", "pyarrow", "joblib")
        },
        "fit_at": datetime.now(timezone.utc).isoformat(),
        "competition_entry_readiness": (
            "ready"
            if competition.get("eligibility_verified") is True
            and competition.get("payout_verified") is True
            else "blocked"
        ),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{model_id}.", dir=artifact_root))
    try:
        model_path = temporary_dir / "model.joblib"
        joblib.dump(estimator, model_path, compress=3)
        manifest["model_sha256"] = _sha256(model_path)
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, model_dir)
    except Exception as exc:
        import shutil

        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise InferenceError(f"cannot save final model artifact: {exc}") from exc

    return {"model_id": model_id, "model_dir": model_dir, "manifest": manifest, "reused": False}


def predict_from_artifact(
    model_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Predict offline from a serialized bundle and preserve test-row order."""
    bundle = Path(model_dir).resolve()
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    manifest = _json_file(bundle / "manifest.json", purpose="final model manifest")
    model_path = bundle / "model.joblib"
    if not model_path.is_file() or _sha256(model_path) != manifest.get("model_sha256"):
        raise InferenceError(f"model artifact checksum mismatch in {bundle}")
    if output == source or output.is_relative_to(bundle) or output == model_path.resolve():
        raise InferenceError("prediction output must not overwrite an input/model artifact")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise InferenceError(f"cannot load model artifact {model_path}: {exc}") from exc

    id_column = manifest.get("id_column")
    text_column = manifest.get("text_column")
    allowed_labels = manifest.get("allowed_labels")
    if not isinstance(id_column, str) or not isinstance(text_column, str) or not isinstance(allowed_labels, list):
        raise InferenceError(f"model manifest is missing inference fields: {bundle}")
    before = _sha256(source) if source.is_file() else None
    if before is None:
        raise InferenceError(f"test input does not exist: {source}")
    try:
        inspect_csv(source, column_roles={"id": id_column, "text": text_column}, required_roles=("id", "text"))
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error, AuditDataError) as exc:
        raise InferenceError(f"cannot read test CSV {source}: {exc}") from exc
    if not rows:
        raise InferenceError(f"test input contains no rows: {source}")
    ids = [row.get(id_column) for row in rows]
    texts = [row.get(text_column) for row in rows]
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids):
        raise InferenceError(f"test input has a blank ID in '{id_column}'")
    if len(ids) != len(set(ids)):
        raise InferenceError(f"test input has duplicate IDs in '{id_column}'")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise InferenceError(f"test input has blank text in '{text_column}'")
    if before != _sha256(source):
        raise InferenceError(f"test input changed while it was being read: {source}")
    try:
        predictions = model.predict(texts)
    except Exception as exc:
        raise InferenceError(f"model prediction failed: {exc}") from exc
    if len(predictions) != len(ids):
        raise InferenceError("model returned a different number of predictions than test rows")
    invalid = sorted({str(label) for label in predictions if label not in allowed_labels})
    if invalid:
        raise InferenceError(f"model predicted labels outside the official allowlist: {', '.join(invalid)}")

    output_rows = [
        {"id": identifier, "model": str(prediction)}
        for identifier, prediction in zip(ids, predictions, strict=True)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        parquet.write_table(pa.Table.from_pylist(output_rows), temporary)
        os.replace(temporary, output)
        temporary = None
    except (OSError, pa.ArrowException) as exc:
        raise InferenceError(f"cannot write prediction file {output}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {"model_id": manifest["model_id"], "rows": len(output_rows), "output": output}


def validate_submission_file(
    submission_path: str | Path,
    test_path: str | Path,
    allowed_labels: Sequence[str],
    *,
    id_column: str = "id",
    label_column: str = "model",
) -> dict[str, object]:
    """Validate exact Kaggle submission schema, row IDs, nulls, and labels."""
    submission_file = Path(submission_path)
    test_file = Path(test_path)
    issues: list[str] = []
    try:
        table = parquet.read_table(submission_file)
    except (OSError, pa.ArrowException) as exc:
        raise InferenceError(f"cannot read submission Parquet {submission_file}: {exc}") from exc
    if table.column_names != [id_column, label_column]:
        issues.append(f"submission must have exactly columns '{id_column}' and '{label_column}' in order")

    try:
        inspect_csv(test_file, column_roles={"id": id_column}, required_roles=("id",))
        with test_file.open("r", encoding="utf-8-sig", newline="") as stream:
            test_rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error, AuditDataError) as exc:
        raise InferenceError(f"cannot read test IDs from {test_file}: {exc}") from exc

    submission_rows = table.to_pylist()
    test_ids = [row.get(id_column) for row in test_rows]
    submission_ids = [row.get(id_column) for row in submission_rows] if id_column in table.column_names else []
    labels = [row.get(label_column) for row in submission_rows] if label_column in table.column_names else []
    test_id_values = [value for value in test_ids if isinstance(value, str) and value.strip()]
    submission_id_values = [value for value in submission_ids if isinstance(value, str) and value.strip()]
    allowed = set(allowed_labels)

    null_values = 0
    duplicate_ids = len(submission_id_values) - len(set(submission_id_values))
    test_duplicate_ids = len(test_id_values) - len(set(test_id_values))
    invalid_labels = 0
    invalid_test_ids = sum(value is None or not isinstance(value, str) or not value.strip() for value in test_ids)
    if id_column in table.column_names:
        null_values += sum(value is None or not str(value).strip() for value in submission_ids)
    else:
        null_values += len(submission_rows)
    if label_column in table.column_names:
        null_values += sum(value is None or not str(value).strip() for value in labels)
        invalid_labels = sum(value not in allowed for value in labels if value is not None)
    else:
        null_values += len(submission_rows)

    missing_ids = len(set(test_id_values) - set(submission_id_values))
    extra_ids = len(set(submission_id_values) - set(test_id_values))
    if duplicate_ids:
        issues.append(f"submission contains {duplicate_ids} duplicate ID(s)")
    if test_duplicate_ids:
        issues.append(f"test file contains {test_duplicate_ids} duplicate ID(s)")
    if invalid_test_ids:
        issues.append(f"test file contains {invalid_test_ids} null/blank ID(s)")
    if null_values:
        issues.append(f"submission contains {null_values} null/blank value(s)")
    if missing_ids:
        issues.append(f"submission is missing {missing_ids} test ID(s)")
    if extra_ids:
        issues.append(f"submission contains {extra_ids} unknown ID(s)")
    if invalid_labels:
        issues.append(f"submission contains {invalid_labels} label(s) outside the official allowlist")

    return {
        "status": "passed" if not issues else "failed",
        "submission_rows": len(submission_rows),
        "test_rows": len(test_rows),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "null_values": null_values,
        "invalid_labels": invalid_labels,
        "issues": issues,
    }


def validate_submission_from_config(
    submission_path: str | Path,
    test_path: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    """Validate using the current versioned official roles and label allowlist."""
    source_config = Path(config_path).resolve()
    config, _ = _read_yaml(source_config)
    data_config = config.get("data")
    competition = config.get("competition")
    if not isinstance(data_config, dict) or not isinstance(data_config.get("column_roles"), dict):
        raise InferenceError(f"{source_config}: 'data.column_roles' mapping is required")
    roles = data_config["column_roles"]
    if not isinstance(competition, dict):
        raise InferenceError(f"{source_config}: 'competition' mapping is required")
    allowed_labels = competition.get("allowed_labels")
    id_column = roles.get("id")
    label_column = roles.get("label")
    if (
        not isinstance(allowed_labels, list)
        or not allowed_labels
        or not isinstance(id_column, str)
        or not isinstance(label_column, str)
    ):
        raise InferenceError(f"{source_config}: ID, label, and official allowed labels are required")
    return validate_submission_file(
        submission_path,
        test_path,
        allowed_labels,
        id_column=id_column,
        label_column=label_column,
    )
