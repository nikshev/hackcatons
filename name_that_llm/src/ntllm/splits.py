"""Deterministic prompt-group cross-validation fold assignments.

impl: FR-002-01, FR-002-02
impl: FR-002-03, FR-002-04
impl: FR-002-05, FR-002-06
impl: FR-002-07, FR-002-08
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as parquet
import yaml

from ntllm.data import inspect_csv


class SplitError(ValueError):
    """Split inputs or settings violate a validation invariant."""


@dataclass(frozen=True)
class SplitResult:
    """Canonical assignments and the fold-level summaries they produce."""

    assignments: tuple[dict[str, str | int], ...]
    row_counts: tuple[int, ...]
    group_counts: tuple[int, ...]
    label_counts: tuple[dict[str, int], ...]
    split_hash: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible summary without source text."""
        return {
            "assignments": [dict(row) for row in self.assignments],
            "folds": [
                {
                    "fold": fold,
                    "row_count": self.row_counts[fold],
                    "group_count": self.group_counts[fold],
                    "label_counts": self.label_counts[fold],
                }
                for fold in range(len(self.row_counts))
            ],
            "split_sha256": self.split_hash,
        }


def _stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()


def assign_prompt_folds(
    records: Iterable[Mapping[str, str]],
    *,
    id_column: str,
    group_column: str,
    label_column: str,
    allowed_labels: Sequence[str] | None,
    n_splits: int,
    seed: int,
) -> SplitResult:
    """Assign complete prompt groups to deterministic, approximately balanced folds.

    Groups are indivisible. Invalid row identifiers, groups, labels, or fold
    settings are rejected before any assignments are returned.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise SplitError("n_splits must request at least two folds")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    for field_name, column in (
        ("id", id_column),
        ("prompt-group", group_column),
        ("label", label_column),
    ):
        if not isinstance(column, str) or not column.strip():
            raise SplitError(f"{field_name} column must be configured")

    if allowed_labels is None:
        allowed = None
    else:
        if not allowed_labels or any(
            not isinstance(label, str) or not label.strip() for label in allowed_labels
        ):
            raise SplitError("allowed_labels must contain nonblank class names")
        allowed = set(allowed_labels)

    group_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_ids: set[str] = set()
    total_labels: Counter[str] = Counter()
    row_count = 0

    for row_number, record in enumerate(records, start=1):
        values: dict[str, str] = {}
        for field_name, column in (
            ("id", id_column),
            ("prompt group", group_column),
            ("label", label_column),
        ):
            value = record.get(column)
            if not isinstance(value, str) or not value.strip():
                raise SplitError(f"row {row_number} has a blank {field_name} in '{column}'")
            values[field_name] = value

        identifier = values["id"]
        group = values["prompt group"]
        label = values["label"]
        if identifier in seen_ids:
            raise SplitError(f"duplicate id '{identifier}' in '{id_column}'")
        if allowed is not None and label not in allowed:
            raise SplitError(f"unrecognized label '{label}' in '{label_column}'")

        seen_ids.add(identifier)
        group_rows[group].append((identifier, label))
        total_labels[label] += 1
        row_count += 1

    if not row_count:
        raise SplitError("training data has no rows to split")
    if len(group_rows) < n_splits:
        raise SplitError(
            f"fewer prompt groups than folds: {len(group_rows)} groups for {n_splits} folds"
        )

    labels = tuple(sorted(total_labels))
    target_rows = row_count / n_splits
    target_labels = {label: total_labels[label] / n_splits for label in labels}
    fold_rows = [0] * n_splits
    fold_groups = [0] * n_splits
    fold_labels = [Counter() for _ in range(n_splits)]
    group_assignment: dict[str, int] = {}

    ordered_groups = sorted(
        group_rows,
        key=lambda group: (-len(group_rows[group]), _stable_key(seed, group), group),
    )

    def place(group: str, fold: int) -> None:
        group_assignment[group] = fold
        fold_rows[fold] += len(group_rows[group])
        fold_groups[fold] += 1
        fold_labels[fold].update(label for _, label in group_rows[group])

    # Seed every fold so that the greedy balancing step cannot leave one empty.
    for fold, group in enumerate(ordered_groups[:n_splits]):
        place(group, fold)

    for group in ordered_groups[n_splits:]:
        group_label_counts = Counter(label for _, label in group_rows[group])

        def cost(fold: int) -> tuple[float, int, int, int]:
            projected_rows = fold_rows[fold] + len(group_rows[group])
            row_error = (
                (projected_rows - target_rows) ** 2 - (fold_rows[fold] - target_rows) ** 2
            ) / max(target_rows, 1)
            label_error = sum(
                ((fold_labels[fold][label] + group_label_counts[label] - target_labels[label]) ** 2
                 - (fold_labels[fold][label] - target_labels[label]) ** 2)
                / max(target_labels[label], 1)
                for label in labels
            )
            return (
                row_error + label_error,
                fold_rows[fold],
                fold_groups[fold],
                fold,
            )

        place(group, min(range(n_splits), key=cost))

    assignments = tuple(
        {
            "id": identifier,
            "prompt_id": group,
            "fold": group_assignment[group],
        }
        for group, group_records in group_rows.items()
        for identifier, _ in group_records
    )
    assignments = tuple(sorted(assignments, key=lambda assignment: str(assignment["id"])))
    canonical = json.dumps(
        assignments,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    split_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return SplitResult(
        assignments=assignments,
        row_counts=tuple(fold_rows),
        group_counts=tuple(fold_groups),
        label_counts=tuple(
            {label: fold_labels[fold][label] for label in labels}
            for fold in range(n_splits)
        ),
        split_hash=split_hash,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_project_path(project_root: Path, value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"configuration field '{key}' must be a nonblank path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _write_split_artifacts(
    assignments_path: Path,
    report_path: Path,
    assignments: Sequence[Mapping[str, str | int]],
    report: Mapping[str, object],
) -> None:
    if assignments_path.resolve() == report_path.resolve():
        raise SplitError("assignment and report outputs must use different paths")

    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_temp: Path | None = None
    report_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{assignments_path.name}.",
            suffix=".tmp",
            dir=assignments_path.parent,
            delete=False,
        ) as stream:
            assignment_temp = Path(stream.name)
        with tempfile.NamedTemporaryFile(
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=report_path.parent,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as stream:
            report_temp = Path(stream.name)
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

        table = pa.Table.from_pylist([dict(row) for row in assignments])
        parquet.write_table(table, assignment_temp)
        os.replace(assignment_temp, assignments_path)
        assignment_temp = None
        os.replace(report_temp, report_path)
        report_temp = None
    except (OSError, pa.ArrowException) as exc:
        raise SplitError(f"cannot write split artifacts: {exc}") from exc
    finally:
        for temporary in (assignment_temp, report_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def split_from_config(config_path: str | Path) -> dict[str, object]:
    """Create grouped folds and persist assignments plus reproducibility report."""
    source_config = Path(config_path).resolve()
    if not source_config.is_file():
        raise SplitError(f"configuration file does not exist: {source_config}")
    try:
        config_bytes = source_config.read_bytes()
        config = yaml.safe_load(config_bytes)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SplitError(f"cannot read YAML configuration {source_config}: {exc}") from exc
    if not isinstance(config, dict):
        raise SplitError(f"{source_config}: configuration must be a YAML mapping")
    config_version = config.get("version")
    if not isinstance(config_version, (int, str)) or not str(config_version).strip():
        raise SplitError(f"{source_config}: a nonblank configuration version is required")

    data_config = config.get("data")
    if not isinstance(data_config, dict) or not isinstance(data_config.get("column_roles"), dict):
        raise SplitError(f"{source_config}: 'data.column_roles' mapping is required")
    roles = data_config["column_roles"]
    files = data_config.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("train"), dict):
        raise SplitError(f"{source_config}: 'data.files.train' mapping is required")
    train_config = files["train"]

    competition = config.get("competition")
    if not isinstance(competition, dict):
        raise SplitError(f"{source_config}: 'competition' mapping is required")
    allowed_labels = competition.get("allowed_labels")
    if not isinstance(allowed_labels, list) or not allowed_labels:
        raise SplitError(f"{source_config}: official 'competition.allowed_labels' is required")

    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise SplitError(f"{source_config}: 'validation' mapping is required")
    group_role = validation.get("prompt_group_role")
    group_column = roles.get(group_role) if isinstance(group_role, str) else None
    id_column = roles.get("id")
    label_column = roles.get("label")
    if not all(isinstance(column, str) and column.strip() for column in (id_column, group_column, label_column)):
        raise SplitError(
            f"{source_config}: validation group role and data id/label column roles must be configured"
        )

    project_root = source_config.parent.parent
    train_path = _absolute_project_path(
        project_root,
        train_config.get("path"),
        key="data.files.train.path",
    )
    assignments_path = _absolute_project_path(
        project_root,
        validation.get("assignments_path"),
        key="validation.assignments_path",
    )
    report_path = _absolute_project_path(
        project_root,
        validation.get("report_path"),
        key="validation.report_path",
    )
    if assignments_path.resolve() == train_path.resolve() or report_path.resolve() == train_path.resolve():
        raise SplitError("split outputs must not overwrite the raw training file")
    if assignments_path.resolve() == source_config or report_path.resolve() == source_config:
        raise SplitError("split outputs must not overwrite the configuration file")

    n_splits = validation.get("n_splits")
    seed = validation.get("seed")
    checksum_before = _file_sha256(train_path) if train_path.is_file() else None
    if checksum_before is None:
        raise SplitError(f"training file does not exist: {train_path}")
    inspection = inspect_csv(
        train_path,
        column_roles=roles,
        required_roles=("id", group_role, "label"),
    )
    if id_column not in inspection.columns or group_column not in inspection.columns or label_column not in inspection.columns:
        raise SplitError(f"{train_path}: required split columns are missing")

    try:
        with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SplitError(f"cannot read training rows from {train_path}: {exc}") from exc

    result = assign_prompt_folds(
        records,
        id_column=id_column,
        group_column=group_column,
        label_column=label_column,
        allowed_labels=allowed_labels,
        n_splits=n_splits,
        seed=seed,
    )
    checksum_after = _file_sha256(train_path)
    if checksum_before != checksum_after:
        raise SplitError(f"{train_path}: training data changed while the split was being created")
    config_hash = hashlib.sha256(config_bytes).hexdigest()

    report: dict[str, object] = {
        "configuration": {
            "path": str(source_config.relative_to(project_root)),
            "version": config_version,
            "sha256": config_hash,
        },
        "dataset_sha256": checksum_after,
        "seed": seed,
        "n_splits": n_splits,
        "row_count": len(result.assignments),
        "split_sha256": result.split_hash,
        "folds": [
            {
                "fold": fold,
                "row_count": result.row_counts[fold],
                "group_count": result.group_counts[fold],
                "label_counts": result.label_counts[fold],
            }
            for fold in range(n_splits)
        ],
        "artifacts": {
            "assignments": str(assignments_path.relative_to(project_root))
            if assignments_path.is_relative_to(project_root)
            else str(assignments_path),
            "report": str(report_path.relative_to(project_root))
            if report_path.is_relative_to(project_root)
            else str(report_path),
        },
    }
    _write_split_artifacts(assignments_path, report_path, result.assignments, report)
    return report
