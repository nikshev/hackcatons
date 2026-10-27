"""Aggregate OOF error analysis and bounded TF-IDF experiment runs.

impl: FR-006-01, FR-006-02, FR-006-03
impl: FR-006-04, FR-006-05, FR-006-06
impl: FR-006-07, FR-006-08, FR-006-09
impl: FR-007-01, FR-007-03, FR-007-04
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import pyarrow.parquet as parquet
import yaml
from sklearn.metrics import f1_score, recall_score

from ntllm.data import AuditDataError, inspect_csv
from ntllm.training import TrainingError, _git_provenance, _sha256, _source_sha256, evaluate_oof


class AnalysisError(ValueError):
    """OOF analysis or tuning provenance failed validation."""


def _load_yaml(path: Path, purpose: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise AnalysisError(f"{purpose} file does not exist: {path}")
    try:
        content = path.read_bytes()
        config = yaml.safe_load(content)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AnalysisError(f"cannot read {purpose} file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise AnalysisError(f"{purpose} file {path} must contain a YAML mapping")
    return config, content


def _project_path(project_root: Path, value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"configuration field '{key}' must be a nonblank path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_train(config: dict[str, object], project_root: Path, expected_sha: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    data = config.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("column_roles"), dict):
        raise AnalysisError("baseline config requires data.column_roles")
    roles = data["column_roles"]
    files = data.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("train"), dict):
        raise AnalysisError("baseline config requires data.files.train")
    train_path = _project_path(project_root, files["train"].get("path"), key="data.files.train.path")
    if _sha256(train_path) != expected_sha:
        raise AnalysisError("current train checksum does not match the selected OOF run")
    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise AnalysisError("baseline config requires a validation mapping")
    group_role = validation.get("prompt_group_role")
    try:
        inspect_csv(
            train_path,
            column_roles=roles,
            required_roles=("id", group_role, "text", "label"),
        )
        with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error, AuditDataError) as exc:
        raise AnalysisError(f"cannot read OOF training data: {exc}") from exc
    id_column = roles["id"]
    if len({row[id_column] for row in rows}) != len(rows):
        raise AnalysisError("train data contains duplicate IDs")
    by_id = {row[id_column]: row for row in rows}
    return rows, by_id


def _verify_baseline_readiness(config_path: Path, config: dict[str, object], content: bytes) -> tuple[Path, dict[str, object]]:
    project_root = config_path.parent.parent
    report_config = config.get("report")
    if not isinstance(report_config, dict):
        raise AnalysisError("baseline config requires a report mapping")
    audit_path = _project_path(project_root, report_config.get("path"), key="report.path")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read audit report {audit_path}: {exc}") from exc
    audit_configuration = audit.get("configuration", {})
    if (
        audit_configuration.get("sha256") != hashlib.sha256(content).hexdigest()
        or audit_configuration.get("version") != config.get("version")
    ):
        raise AnalysisError("audit report is stale for the current baseline config")
    if audit.get("audit_status") != "passed":
        raise AnalysisError("data audit did not pass")
    readiness = audit.get("training_readiness", {})
    if readiness.get("status") != "ready":
        raise AnalysisError(
            "training readiness is blocked: " + "; ".join(readiness.get("blockers", []))
        )
    return project_root, audit


def _fold_score(rows: list[dict[str, object]], labels: list[str]) -> dict[str, object]:
    actual = [str(row["actual"]) for row in rows]
    predicted = [str(row["prediction"]) for row in rows]
    recall = recall_score(actual, predicted, labels=labels, average=None, zero_division=0)
    return {
        "row_count": len(rows),
        "balanced_accuracy": float(statistics.mean(recall)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
    }


def analyze_oof_run(
    run_id: str,
    config_path: str | Path = "configs/baseline.yaml",
    *,
    candidate_name: str | None = None,
) -> dict[str, object]:
    """Write an aggregate-only error report for one completed grouped OOF run."""
    if not run_id or Path(run_id).name != run_id:
        raise AnalysisError("run_id must be a single path component")
    source_config = Path(config_path).resolve()
    config, content = _load_yaml(source_config, "baseline config")
    project_root, _ = _verify_baseline_readiness(source_config, config, content)
    training = config.get("training")
    validation = config.get("validation")
    competition = config.get("competition")
    if not isinstance(training, dict) or not isinstance(validation, dict) or not isinstance(competition, dict):
        raise AnalysisError("baseline config requires training, validation, and competition mappings")

    run_dir = _project_path(project_root, training.get("artifact_root"), key="training.artifact_root") / run_id
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics_report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        prediction_rows = parquet.read_table(run_dir / "predictions_oof.parquet").to_pylist()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, parquet.ArrowException) as exc:
        raise AnalysisError(f"cannot read OOF run {run_dir}: {exc}") from exc
    if manifest.get("run_id") != run_id:
        raise AnalysisError("OOF manifest run_id does not match the requested run")

    labels = competition.get("allowed_labels")
    if not isinstance(labels, list) or not labels:
        raise AnalysisError("official competition.allowed_labels are required")
    train_sha = manifest.get("train_sha256")
    split_sha = manifest.get("split_sha256")
    if not isinstance(train_sha, str) or not isinstance(split_sha, str):
        raise AnalysisError("OOF manifest is missing train/split checksums")
    train_rows, train_by_id = _read_train(config, project_root, train_sha)
    if _sha256(_project_path(project_root, config["data"]["files"]["train"]["path"], key="data.files.train.path")) != train_sha:
        raise AnalysisError("train data changed while error analysis was being prepared")
    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise AnalysisError("baseline config requires a validation mapping")
    current_split_path = _project_path(
        project_root,
        validation.get("report_path"),
        key="validation.report_path",
    )
    try:
        current_split = json.loads(current_split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read current split report {current_split_path}: {exc}") from exc
    if current_split.get("split_sha256") != split_sha or current_split.get("dataset_sha256") != train_sha:
        raise AnalysisError("current split report does not match the selected OOF run")

    candidates = sorted({str(row.get("candidate")) for row in prediction_rows})
    selected = candidate_name or config.get("inference", {}).get("model_candidate")
    if not isinstance(selected, str) or selected not in candidates:
        raise AnalysisError(f"OOF run does not contain candidate '{selected}'")
    if selected not in metrics_report.get("models", {}):
        raise AnalysisError(f"OOF metrics do not contain candidate '{selected}'")
    candidate_rows = [row for row in prediction_rows if row.get("candidate") == selected]
    ids = [row.get("id") for row in candidate_rows]
    if len(ids) != len(train_rows) or len(set(ids)) != len(ids) or set(ids) != set(train_by_id):
        raise AnalysisError("OOF candidate has incomplete or duplicate train ID coverage")

    group_role = validation.get("prompt_group_role")
    roles = config.get("data", {}).get("column_roles", {})
    group_column = roles.get(group_role) if isinstance(group_role, str) else None
    if not isinstance(group_column, str):
        raise AnalysisError("prompt-group column role is not configured")
    folds_by_group: dict[str, set[int]] = defaultdict(set)
    class_support: Counter[str] = Counter()
    class_correct: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    errors_by_group: Counter[str] = Counter()
    errors_by_length: Counter[str] = Counter()
    rows_by_length: Counter[str] = Counter()
    fold_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    length_bins = ((0, 500, "0-499"), (500, 1000, "500-999"), (1000, 1500, "1000-1499"), (1500, 10**12, "1500+"))

    for prediction in candidate_rows:
        identifier = prediction["id"]
        train_row = train_by_id[identifier]
        group = train_row[group_column]
        actual = str(prediction["actual"])
        predicted = str(prediction["prediction"])
        fold = int(prediction["fold"])
        if prediction.get("prompt_id") != group:
            raise AnalysisError("OOF prompt group differs from the audited train row")
        if actual != train_row[roles["label"]] or actual not in labels or predicted not in labels:
            raise AnalysisError("OOF prediction contains inconsistent or unrecognized labels")
        folds_by_group[group].add(fold)
        class_support[actual] += 1
        if actual == predicted:
            class_correct[actual] += 1
        else:
            confusion[(actual, predicted)] += 1
            errors_by_group[group] += 1
        text_length = len(train_row[roles["text"]])
        length_label = next(label for lower, upper, label in length_bins if lower <= text_length < upper)
        rows_by_length[length_label] += 1
        if actual != predicted:
            errors_by_length[length_label] += 1
        fold_rows[fold].append(prediction)

    if any(len(folds) != 1 for folds in folds_by_group.values()):
        raise AnalysisError("OOF predictions split a prompt group across folds")
    if len(folds_by_group) != len({row[group_column] for row in train_rows}):
        raise AnalysisError("OOF prompt groups do not cover the audited training data")

    class_metrics = {
        label: {
            "support": class_support[label],
            "recall": class_correct[label] / class_support[label] if class_support[label] else 0.0,
            "errors": class_support[label] - class_correct[label],
        }
        for label in labels
    }
    top_confusions = [
        {"actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in sorted(
            confusion.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:10]
    ]
    per_fold = [
        {"fold": fold, **_fold_score(fold_rows[fold], labels)}
        for fold in sorted(fold_rows)
    ]
    length_summary = {
        name: {
            "rows": rows_by_length[name],
            "errors": errors_by_length[name],
            "error_rate": errors_by_length[name] / rows_by_length[name] if rows_by_length[name] else None,
        }
        for _, _, name in length_bins
    }
    overall_metrics = _fold_score(candidate_rows, labels)
    summary = {
        "run_id": run_id,
        "candidate": selected,
        "train_sha256": train_sha,
        "split_sha256": split_sha,
        "row_count": len(candidate_rows),
        "balanced_accuracy": overall_metrics["balanced_accuracy"],
        "macro_f1": overall_metrics["macro_f1"],
        "prompt_group_count": len(folds_by_group),
        "error_count": sum(confusion.values()),
        "class_metrics": class_metrics,
        "top_confusions": top_confusions,
        "fold_metrics": per_fold,
        "prompt_group_errors": {
            "groups_with_errors": sum(count > 0 for count in errors_by_group.values()),
            "total_groups": len(folds_by_group),
            "error_rows_by_group_count": dict(sorted(Counter(errors_by_group.values()).items())),
        },
        "text_length_errors": length_summary,
    }
    report_lines = [
        f"# OOF Error Analysis — {run_id}",
        "",
        f"Candidate: `{selected}`  ",
        f"Train SHA-256: `{train_sha}`  ",
        f"Split SHA-256: `{split_sha}`  ",
        f"Rows: **{len(candidate_rows)}** · prompt groups: **{len(folds_by_group)}** · errors: **{summary['error_count']}**",
        "",
        "This report contains aggregates only; it intentionally omits row IDs, prompt IDs, and passage text.",
        "",
        "## Per-class recall",
        "",
        "| Label | Support | Recall | Errors |",
        "|---|---:|---:|---:|",
    ]
    for label in labels:
        item = class_metrics[label]
        report_lines.append(f"| `{label}` | {item['support']} | {item['recall']:.4f} | {item['errors']} |")
    report_lines += ["", "## Most frequent confusions", "", "| Actual | Predicted | Count |", "|---|---|---:|"]
    report_lines += [
        f"| `{item['actual']}` | `{item['predicted']}` | {item['count']} |"
        for item in top_confusions
    ] or ["| — | — | 0 |"]
    report_lines += ["", "## Fold metrics", "", "| Fold | Rows | Balanced accuracy | Macro F1 |", "|---:|---:|---:|---:|"]
    report_lines += [
        f"| {item['fold']} | {item['row_count']} | {item['balanced_accuracy']:.4f} | {item['macro_f1']:.4f} |"
        for item in per_fold
    ]
    group_error_counts = {
        group: errors_by_group.get(group, 0)
        for group in folds_by_group
    }
    summary["prompt_group_errors"]["error_rows_by_group_count"] = dict(
        sorted(Counter(group_error_counts.values()).items())
    )
    report_lines += ["", "## Prompt-group error incidence", ""]
    report_lines.append(
        f"Groups with errors: **{summary['prompt_group_errors']['groups_with_errors']}** / **{summary['prompt_group_errors']['total_groups']}**."
    )
    report_lines += ["", "## Text-length bins", "", "| Characters | Rows | Errors | Error rate |", "|---|---:|---:|---:|"]
    for _, _, name in length_bins:
        item = length_summary[name]
        rate = "—" if item["error_rate"] is None else f"{item['error_rate']:.4f}"
        report_lines.append(f"| {name} | {item['rows']} | {item['errors']} | {rate} |")
    report_path = run_dir / "error_analysis.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"summary": summary, "report_path": str(report_path)}


def tune_from_config(
    config_path: str | Path,
    experiment_path: str | Path,
) -> dict[str, object]:
    """Run a bounded candidate grid on one current, saved prompt-group split."""
    source_config = Path(config_path).resolve()
    config, config_bytes = _load_yaml(source_config, "baseline config")
    project_root, _ = _verify_baseline_readiness(source_config, config, config_bytes)
    source_experiment = Path(experiment_path).resolve()
    experiment, experiment_bytes = _load_yaml(source_experiment, "experiment config")
    experiment_name = experiment.get("name")
    experiment_version = experiment.get("version")
    candidates = experiment.get("candidates")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise AnalysisError("experiment config requires a nonblank 'name'")
    if not isinstance(experiment_version, (int, str)) or not str(experiment_version).strip():
        raise AnalysisError("experiment config requires a nonblank 'version'")
    if not isinstance(candidates, dict) or not candidates:
        raise AnalysisError("experiment config requires a nonempty 'candidates' mapping")

    data = config.get("data", {})
    roles = data.get("column_roles", {})
    validation = config.get("validation", {})
    competition = config.get("competition", {})
    if not all(isinstance(value, dict) for value in (data, roles, validation, competition)):
        raise AnalysisError("baseline config data/roles/validation/competition sections are invalid")
    group_role = validation.get("prompt_group_role")
    group_column = roles.get(group_role) if isinstance(group_role, str) else None
    id_column = roles.get("id")
    text_column = roles.get("text")
    label_column = roles.get("label")
    if not all(isinstance(column, str) and column.strip() for column in (id_column, group_column, text_column, label_column)):
        raise AnalysisError("baseline config must define id/group/text/label column roles")
    allowed_labels = competition.get("allowed_labels")
    if not isinstance(allowed_labels, list) or not allowed_labels:
        raise AnalysisError("baseline config must define the official allowed-label list")

    train_file = data.get("files", {}).get("train", {})
    train_path = _project_path(project_root, train_file.get("path"), key="data.files.train.path")
    audit_path = _project_path(project_root, config.get("report", {}).get("path"), key="report.path")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train_sha = audit.get("files", {}).get("train", {}).get("sha256")
    if not isinstance(train_sha, str) or _sha256(train_path) != train_sha:
        raise AnalysisError("train checksum differs from the current passing audit")

    assignments_path = _project_path(
        project_root,
        validation.get("assignments_path"),
        key="validation.assignments_path",
    )
    split_report_path = _project_path(
        project_root,
        validation.get("report_path"),
        key="validation.report_path",
    )
    try:
        split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
        assignments = parquet.read_table(assignments_path).to_pylist()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, parquet.ArrowException) as exc:
        raise AnalysisError(f"cannot read current split artifacts: {exc}") from exc
    if (
        split_report.get("configuration", {}).get("sha256") != hashlib.sha256(config_bytes).hexdigest()
        or split_report.get("configuration", {}).get("version") != config.get("version")
        or split_report.get("dataset_sha256") != train_sha
        or split_report.get("n_splits") != validation.get("n_splits")
        or split_report.get("seed") != validation.get("seed")
    ):
        raise AnalysisError("split artifacts are stale for the current baseline config/data")

    train_rows, _ = _read_train(config, project_root, train_sha)
    valid_candidates: list[dict[str, object]] = []
    failed_candidates: list[dict[str, str]] = []
    for name, model_config in candidates.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(model_config, Mapping):
            failed_candidates.append({"name": str(name), "error": "candidate must be a named mapping"})
            continue
        try:
            result = evaluate_oof(
                train_rows,
                assignments,
                id_column=id_column,
                group_column=group_column,
                text_column=text_column,
                label_column=label_column,
                allowed_labels=allowed_labels,
                n_splits=validation["n_splits"],
                seed=validation["seed"],
                models={name: model_config},
            )
        except TrainingError as exc:
            failed_candidates.append({"name": name, "error": str(exc)})
            continue

        metrics = result.metrics[name]
        fold_metrics = metrics["fold_metrics"]
        fold_scores = [float(item["balanced_accuracy"]) for item in fold_metrics]
        fold_f1 = [float(item["macro_f1"]) for item in fold_metrics]
        valid_candidates.append(
            {
                "name": name,
                "parameters": dict(model_config),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "balanced_accuracy_std": float(statistics.pstdev(fold_scores)),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_f1_std": float(statistics.pstdev(fold_f1)),
                "fold_balanced_accuracy": fold_scores,
                "fold_macro_f1": fold_f1,
                "recall_by_label": metrics["recall_by_label"],
                "runtime_seconds": metrics["runtime_seconds"],
            }
        )

    valid_candidates.sort(
        key=lambda item: (-item["balanced_accuracy"], -item["macro_f1"], item["name"])
    )
    for rank, candidate in enumerate(valid_candidates, start=1):
        candidate["rank"] = rank

    source_hash = _source_sha256(project_root)
    experiment_hash = hashlib.sha256(experiment_bytes).hexdigest()
    run_key = json.dumps(
        {
            "baseline_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "experiment_sha256": experiment_hash,
            "train_sha256": train_sha,
            "split_sha256": split_report["split_sha256"],
            "source_sha256": source_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    experiment_run_id = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:16]
    output_path = _project_path(
        project_root,
        experiment.get("output_path", f"reports/tuning/{experiment_name}/summary.json"),
        key="experiment.output_path",
    )
    if output_path.resolve() in {source_config, source_experiment, train_path.resolve()}:
        raise AnalysisError("tuning summary must not overwrite config or raw train data")

    commit, dirty = _git_provenance(project_root)
    summary: dict[str, object] = {
        "status": "failed" if failed_candidates else "passed",
        "experiment_name": experiment_name,
        "experiment_version": experiment_version,
        "experiment_run_id": experiment_run_id,
        "hypothesis": experiment.get("hypothesis"),
        "experiment_sha256": experiment_hash,
        "baseline_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "baseline_config_version": config.get("version"),
        "train_sha256": train_sha,
        "split_sha256": split_report["split_sha256"],
        "source_sha256": source_hash,
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "n_splits": validation["n_splits"],
        "seed": validation["seed"],
        "candidate_count": len(candidates),
        "successful_count": len(valid_candidates),
        "failures": failed_candidates,
        "candidates": valid_candidates,
        "summary_path": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    try:
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary
