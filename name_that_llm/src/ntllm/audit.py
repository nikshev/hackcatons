"""Dataset provenance records for the raw-data audit.

impl: FR-001-04
impl: FR-001-05
impl: FR-001-06
impl: FR-001-07
impl: FR-001-08
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from ntllm.data import AuditDataError, inspect_csv
from ntllm.validation import validate_rows


@dataclass(frozen=True)
class DatasetManifestEntry:
    """Auditable metadata for one unchanged input file."""

    path: str
    source: str
    retrieved_at: str | None
    license_or_terms: str | None
    sha256: str
    row_count: int
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["columns"] = list(self.columns)
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_entry(
    path: str | Path,
    *,
    source_name: str,
    retrieved_at: str | None,
    license_or_terms: str | None,
    column_roles: Mapping[str, str],
    required_roles: Iterable[str],
) -> DatasetManifestEntry:
    """Build provenance metadata after validating the configured CSV schema."""
    source_path = Path(path)
    if not source_path.is_file():
        raise AuditDataError(f"input file does not exist or is not a file: {source_path}")

    checksum_before = _sha256(source_path)
    inspection = inspect_csv(
        source_path,
        column_roles=column_roles,
        required_roles=required_roles,
    )
    checksum_after = _sha256(source_path)
    if checksum_before != checksum_after:
        raise AuditDataError(f"{source_path}: input changed while it was being audited")

    return DatasetManifestEntry(
        path=source_path.name,
        source=source_name,
        retrieved_at=retrieved_at,
        license_or_terms=license_or_terms,
        sha256=checksum_after,
        row_count=inspection.row_count,
        columns=inspection.columns,
    )


def profile_csv(
    path: str | Path,
    *,
    column_roles: Mapping[str, str],
) -> dict[str, object]:
    """Summarize missing values, text lengths, exact duplicates, and labels.

    Blank or whitespace-only cells count as null. Duplicate counts count rows
    beyond the first occurrence. Raw cell values are never included in output.
    """
    source = Path(path)
    inspection = inspect_csv(source, column_roles={}, required_roles=())
    text_column = column_roles.get("text")
    if not text_column or text_column not in inspection.columns:
        raise AuditDataError(
            f"{source}: configured text column is missing; "
            f"observed columns: {', '.join(inspection.columns)}"
        )
    label_column = column_roles.get("label")
    if label_column not in inspection.columns:
        label_column = None

    null_counts = {column: 0 for column in inspection.columns}
    text_lengths: list[int] = []
    row_counts: dict[tuple[str | None, ...], int] = {}
    text_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                for column in inspection.columns:
                    value = row.get(column)
                    if value is None or not value.strip():
                        null_counts[column] += 1

                row_key = tuple(row.get(column) for column in inspection.columns)
                row_counts[row_key] = row_counts.get(row_key, 0) + 1

                text = row.get(text_column)
                if text is not None and text.strip():
                    text_lengths.append(len(text))
                    text_counts[text] = text_counts.get(text, 0) + 1

                if label_column is not None:
                    label = row.get(label_column)
                    if label is not None and label.strip():
                        label_counts[label] = label_counts.get(label, 0) + 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditDataError(f"{source}: cannot profile CSV data: {exc}") from exc

    duplicate_rows = sum(count - 1 for count in row_counts.values() if count > 1)
    duplicate_texts = sum(count - 1 for count in text_counts.values() if count > 1)
    text_summary: dict[str, int | float | None] = {
        "count": len(text_lengths),
        "min": min(text_lengths) if text_lengths else None,
        "median": statistics.median(text_lengths) if text_lengths else None,
        "max": max(text_lengths) if text_lengths else None,
    }

    return {
        "null_counts": null_counts,
        "text_length": text_summary,
        "duplicates": {
            "exact_row_count": duplicate_rows,
            "exact_text_count": duplicate_texts,
        },
        "label_counts": dict(sorted(label_counts.items())) if label_column else None,
    }


def _validate_label_allowlist(
    path: Path,
    *,
    label_column: str,
    allowed_labels: list[str],
) -> None:
    """Reject blank or out-of-allowlist target values without normalization."""
    allowed = set(allowed_labels)
    observed: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row_number, row in enumerate(reader, start=2):
                label = row.get(label_column)
                if label is None or not label.strip():
                    raise AuditDataError(
                        f"{path}: row {row_number} has an empty label in '{label_column}'"
                    )
                observed.add(label)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditDataError(f"{path}: cannot read labels from '{label_column}': {exc}") from exc

    unexpected = sorted(observed - allowed)
    if unexpected:
        rendered = ", ".join(repr(value) for value in unexpected)
        raise AuditDataError(
            f"{path}: unrecognized label(s) in '{label_column}': {rendered}"
        )


def audit_from_config(config_path: str | Path) -> dict[str, object]:
    """Audit configured local files and write a JSON report.

    Paths in the configuration are relative to the project root, defined as
    the parent of the configuration directory (normally ``configs/``).
    Structural audit success is independent from training readiness.
    """
    source_config = Path(config_path).resolve()
    if not source_config.is_file():
        raise AuditDataError(f"configuration file does not exist: {source_config}")
    try:
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AuditDataError(f"cannot read YAML configuration {source_config}: {exc}") from exc
    if not isinstance(config, dict):
        raise AuditDataError(f"{source_config}: configuration must be a YAML mapping")
    config_version = config.get("version")
    if not isinstance(config_version, (int, str)) or not str(config_version).strip():
        raise AuditDataError(f"{source_config}: a non-empty configuration 'version' is required")

    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise AuditDataError(f"{source_config}: missing 'data' mapping")
    roles = data_config.get("column_roles")
    files = data_config.get("files")
    if not isinstance(roles, dict) or not isinstance(files, dict) or not files:
        raise AuditDataError(
            f"{source_config}: 'data.column_roles' and non-empty 'data.files' mappings are required"
        )
    competition = config.get("competition", {})
    if not isinstance(competition, dict):
        raise AuditDataError(f"{source_config}: 'competition' must be a YAML mapping")
    allowed_labels = competition.get("allowed_labels", [])
    if not isinstance(allowed_labels, list) or any(
        not isinstance(label, str) or not label.strip() for label in allowed_labels
    ):
        raise AuditDataError(
            f"{source_config}: competition.allowed_labels must be a list of non-empty strings"
        )

    project_root = source_config.parent.parent
    audited_files: dict[str, object] = {}
    audit_failed = False
    for name, file_config in files.items():
        if not isinstance(file_config, dict) or not file_config.get("path"):
            raise AuditDataError(
                f"{source_config}: data.files.{name} must define a path"
            )
        file_path = Path(file_config["path"])
        if not file_path.is_absolute():
            file_path = project_root / file_path
        required_roles = file_config.get("required_roles", [])
        if not isinstance(required_roles, list):
            raise AuditDataError(
                f"{source_config}: data.files.{name}.required_roles must be a list"
            )
        try:
            entry = build_manifest_entry(
                file_path,
                source_name=str(file_config.get("source", "unverified")),
                retrieved_at=file_config.get("retrieved_at"),
                license_or_terms=file_config.get("license_or_terms"),
                column_roles=roles,
                required_roles=required_roles,
            )
            file_result = entry.to_dict()
            audited_files[str(name)] = file_result
            if str(name) == "train" and allowed_labels:
                label_column = roles.get("label")
                if not isinstance(label_column, str) or not label_column:
                    raise AuditDataError(
                        f"{file_path}: a configured label column is required to check the allowlist"
                    )
                _validate_label_allowlist(
                    file_path,
                    label_column=label_column,
                    allowed_labels=allowed_labels,
                )
            file_result["quality"] = profile_csv(
                file_path,
                column_roles=roles,
            )
            integrity = validate_rows(file_path, column_roles=roles)
            file_result["integrity"] = integrity
            if integrity["issues"]:
                file_result["issues"] = integrity["issues"]
            if integrity["blocking_issues"]:
                file_result["error"] = "; ".join(integrity["blocking_issues"])
                audit_failed = True
            if _sha256(file_path) != entry.sha256:
                raise AuditDataError(f"{file_path}: input changed while it was being audited")
        except AuditDataError as exc:
            audit_failed = True
            file_result = audited_files.get(str(name), {"path": str(file_config["path"])})
            file_result["error"] = str(exc)
            audited_files[str(name)] = file_result

    blockers: list[str] = []
    if competition.get("rules_verified") is not True:
        blockers.append("competition rules are not verified")
    if not allowed_labels:
        blockers.append("official allowed-label list is not confirmed")
    train_config = files.get("train")
    if not isinstance(train_config, dict) or "train" not in audited_files:
        blockers.append("an audited labeled train file is not available")
    elif "error" in audited_files["train"]:
        blockers.append("the configured train file did not pass the audit")
    else:
        if audited_files["train"].get("row_count", 0) == 0:
            blockers.append("the configured train file contains no labeled rows")
        retrieved_at = train_config.get("retrieved_at")
        if (
            not isinstance(retrieved_at, str)
            or not retrieved_at.strip()
            or retrieved_at.strip().lower().startswith("unverified")
        ):
            blockers.append("train data retrieval date is not verified")
        if str(train_config.get("license_or_terms", "unverified")).strip().lower().startswith(
            "unverified"
        ):
            blockers.append("train data license or terms are not verified")

    entry_blockers: list[str] = []
    if competition.get("eligibility_verified") is not True:
        entry_blockers.append("participant eligibility is not verified")
    if competition.get("payout_verified") is not True:
        entry_blockers.append("payout conditions are not verified")

    report_config = config.get("report", {})
    if not isinstance(report_config, dict) or not report_config.get("path"):
        raise AuditDataError(f"{source_config}: 'report.path' is required")
    report_path = Path(report_config["path"])
    if not report_path.is_absolute():
        report_path = project_root / report_path

    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "path": str(source_config.relative_to(project_root)),
            "version": config_version,
            "sha256": _sha256(source_config),
        },
        "files": audited_files,
        "audit_status": "failed" if audit_failed else "passed",
        "training_readiness": {
            "status": "blocked" if blockers else "ready",
            "blockers": blockers,
        },
        "competition_entry_readiness": {
            "status": "blocked" if entry_blockers else "ready",
            "blockers": entry_blockers,
        },
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AuditDataError(f"cannot write audit report {report_path}: {exc}") from exc

    return report
