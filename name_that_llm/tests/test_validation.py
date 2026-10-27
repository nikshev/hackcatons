# verifies: FR-001-06

import csv

from ntllm.validation import validate_rows


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_validate_rows_reports_missing_prompt_group_and_invalid_rows(tmp_path):
    source = tmp_path / "data.csv"
    write_csv(
        source,
        ["id", "text"],
        [["dup", "first"], ["dup", "  "], ["", "third"]],
    )

    result = validate_rows(
        source,
        column_roles={"id": "id", "prompt_id": "prompt_id", "text": "text"},
    )

    assert result["prompt_groups"] == {
        "available": False,
        "column": None,
        "unique_count": None,
        "missing_count": 3,
    }
    assert result["row_integrity"] == {
        "duplicate_id_count": 1,
        "missing_id_count": 1,
        "invalid_text_count": 1,
    }
    assert result["blocking_issues"] == [
        "1 duplicate id value(s) in 'id'",
        "1 missing id value(s) in 'id'",
        "1 empty text value(s) in 'text'",
    ]
    assert result["issues"][0] == "prompt group column 'prompt_id' is unavailable"


def test_validate_rows_counts_available_prompt_groups_and_missing_values(tmp_path):
    source = tmp_path / "data.csv"
    write_csv(
        source,
        ["id", "prompt_id", "text"],
        [["a", "p1", "one"], ["b", "p1", "two"], ["c", "", "three"]],
    )

    result = validate_rows(
        source,
        column_roles={"id": "id", "prompt_id": "prompt_id", "text": "text"},
    )

    assert result["prompt_groups"] == {
        "available": True,
        "column": "prompt_id",
        "unique_count": 1,
        "missing_count": 1,
    }
    assert result["row_integrity"] == {
        "duplicate_id_count": 0,
        "missing_id_count": 0,
        "invalid_text_count": 0,
    }
    assert result["blocking_issues"] == []
    assert result["issues"] == ["1 row(s) have no prompt group in 'prompt_id'"]
