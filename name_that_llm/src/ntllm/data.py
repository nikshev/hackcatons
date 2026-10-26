"""Read-only inspection of configured competition CSV schemas.

impl: FR-001-03
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class AuditDataError(ValueError):
    """An input file cannot be interpreted using its configured schema."""


@dataclass(frozen=True)
class CSVInspection:
    """Observed CSV structure and the configured logical field mapping."""

    path: Path
    columns: tuple[str, ...]
    row_count: int
    role_columns: dict[str, str]


def inspect_csv(
    path: str | Path,
    *,
    column_roles: Mapping[str, str],
    required_roles: Iterable[str],
) -> CSVInspection:
    """Inspect a CSV header and row widths without changing the source file.

    The actual column names are discovered from the file. Logical roles such as
    ``id`` and ``label`` must be mapped explicitly by configuration.
    """
    source = Path(path)
    if not source.is_file():
        raise AuditDataError(f"input file does not exist or is not a file: {source}")

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None:
                raise AuditDataError(f"{source}: CSV is empty and has no header")
            if not header or any(not name.strip() for name in header):
                raise AuditDataError(f"{source}: CSV contains an empty column name")

            columns = tuple(header)
            seen: set[str] = set()
            duplicate_columns: set[str] = set()
            for name in columns:
                if name in seen:
                    duplicate_columns.add(name)
                seen.add(name)
            if duplicate_columns:
                names = ", ".join(sorted(duplicate_columns))
                raise AuditDataError(f"{source}: duplicate column name(s): {names}")

            role_columns: dict[str, str] = {}
            for role in required_roles:
                configured_name = column_roles.get(role)
                if not configured_name:
                    raise AuditDataError(
                        f"{source}: required role '{role}' has no configured column mapping"
                    )
                if configured_name not in columns:
                    raise AuditDataError(
                        f"{source}: configured column '{configured_name}' for required "
                        f"role '{role}' is missing; observed columns: {', '.join(columns)}"
                    )
                role_columns[role] = configured_name

            if len(set(role_columns.values())) != len(role_columns):
                raise AuditDataError(
                    f"{source}: multiple required roles map to the same column"
                )

            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise AuditDataError(
                        f"{source}: row {row_number} has {len(row)} fields; "
                        f"expected {len(columns)} columns"
                    )
                row_count += 1
    except UnicodeDecodeError as exc:
        raise AuditDataError(f"{source}: CSV is not valid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise AuditDataError(f"{source}: invalid CSV: {exc}") from exc

    return CSVInspection(
        path=source,
        columns=columns,
        row_count=row_count,
        role_columns=role_columns,
    )
