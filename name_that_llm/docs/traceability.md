# Матриця трасування

<!-- ГЕНЕРУЄТЬСЯ scripts/trace.py — не редагувати вручну -->

Вимог: **8** · задач: **16** (виконано 10) · вимог у роботі: **4** · порушень: **0**

## 001-competition-readiness-audit

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-001-01` | The competition record MUST distinguish confirmed, unverified, | `T-001` | — | — |
| `FR-001-02` | The record MUST cover eligibility, current deadlines, prizes, | `T-002` | — | `tests/test_audit.py` |
| `FR-001-03` | The data audit MUST inspect the actual input files and report | `T-003`, `T-004`, `T-015` | `src/ntllm/data.py`, `src/ntllm/__main__.py` | `tests/test_audit.py` |
| `FR-001-04` | For every audited raw file, the manifest MUST record source, | `T-005`, `T-006`, `T-016` | `src/ntllm/audit.py` | `tests/test_audit.py` |
| `FR-001-05` | The audit MUST report missing values, text-length summaries, | `T-007`, `T-008` | — | — |
| `FR-001-06` | The audit MUST report prompt-group availability and identify | `T-009`, `T-010` | — | — |
| `FR-001-07` | The project MUST mark supervised baseline training as blocked | `T-011`, `T-012` | `src/ntllm/audit.py` | `tests/test_audit.py` |
| `FR-001-08` | An unknown schema, unreadable file, or unrecognized class MUST | `T-013`, `T-014` | `src/ntllm/__main__.py`, `src/ntllm/audit.py` | `tests/test_audit.py` |

## Критичні задачі

- `T-011` (001-competition-readiness-audit) — виконано
- `T-012` (001-competition-readiness-audit) — виконано
