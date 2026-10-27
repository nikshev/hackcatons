"""Row and prompt-group integrity checks for competition data.

impl: FR-001-06
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from ntllm.data import AuditDataError, inspect_csv


def validate_rows(
    path: str | Path,
    *,
    column_roles: Mapping[str, str],
) -> dict[str, object]:
    """Report prompt grouping and blocking row-integrity issues.

    Prompt-group metadata is optional: its absence is reported as a warning so
    callers can choose a documented fallback. Duplicate/missing IDs and blank
    text are blocking issues. No cell values are included in the result.
    """
    source = Path(path)
    inspection = inspect_csv(
        source,
        column_roles=column_roles,
        required_roles=("id", "text"),
    )
    id_column = inspection.role_columns["id"]
    text_column = inspection.role_columns["text"]
    configured_group_column = column_roles.get("prompt_id")
    group_column = (
        configured_group_column
        if configured_group_column in inspection.columns
        else None
    )

    seen_ids: set[str] = set()
    prompt_groups: set[str] = set()
    duplicate_id_count = 0
    missing_id_count = 0
    invalid_text_count = 0
    missing_group_count = 0

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                identifier = row.get(id_column)
                if identifier is None or not identifier.strip():
                    missing_id_count += 1
                elif identifier in seen_ids:
                    duplicate_id_count += 1
                else:
                    seen_ids.add(identifier)

                text = row.get(text_column)
                if text is None or not text.strip():
                    invalid_text_count += 1

                if group_column is not None:
                    group = row.get(group_column)
                    if group is None or not group.strip():
                        missing_group_count += 1
                    else:
                        prompt_groups.add(group)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditDataError(f"{source}: cannot validate CSV rows: {exc}") from exc

    blocking_issues: list[str] = []
    if duplicate_id_count:
        blocking_issues.append(
            f"{duplicate_id_count} duplicate id value(s) in '{id_column}'"
        )
    if missing_id_count:
        blocking_issues.append(f"{missing_id_count} missing id value(s) in '{id_column}'")
    if invalid_text_count:
        blocking_issues.append(
            f"{invalid_text_count} empty text value(s) in '{text_column}'"
        )

    issues: list[str] = []
    if group_column is None:
        issues.append(
            f"prompt group column '{configured_group_column or '<unconfigured>'}' is unavailable"
        )
        missing_group_count = inspection.row_count
    elif missing_group_count:
        issues.append(
            f"{missing_group_count} row(s) have no prompt group in '{group_column}'"
        )

    return {
        "prompt_groups": {
            "available": group_column is not None,
            "column": group_column,
            "unique_count": len(prompt_groups) if group_column is not None else None,
            "missing_count": missing_group_count,
        },
        "row_integrity": {
            "duplicate_id_count": duplicate_id_count,
            "missing_id_count": missing_id_count,
            "invalid_text_count": invalid_text_count,
        },
        "issues": issues,
        "blocking_issues": blocking_issues,
    }
