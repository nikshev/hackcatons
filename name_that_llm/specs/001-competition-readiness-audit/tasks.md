# Tasks: Competition Readiness and Data Audit

**Input**: Design documents from `specs/001-competition-readiness-audit/`
**Prerequisites**: `plan.md`, `spec.md`, `data-model.md`, and `contracts/audit-cli.md`

## Phase 1: Competition readiness record (US1, Priority: P1)

- [ ] T-001 [FR-001-01] Update `docs/competition.md` with authoritative links, check dates, and verification status for each material competition claim.
- [ ] T-002 [FR-001-02] Verify participant eligibility and payout conditions, and resolve host clarification for paid API-generated data; leave unknowns blocked in `docs/competition.md`.

## Phase 2: Local file inspection (US2, Priority: P2)

- [x] T-003 [FR-001-03] Write failing tests for role-based schema discovery and unknown/missing columns in `tests/test_audit.py`.
- [x] T-004 [FR-001-03] Implement CSV schema inspection and configured-role validation in `src/ntllm/data.py`.
- [x] T-005 [FR-001-04] Write failing tests for SHA-256, provenance metadata, row counts, ordered columns, and unchanged raw inputs in `tests/test_audit.py`.
- [x] T-006 [FR-001-04] Implement a per-file dataset manifest in `src/ntllm/audit.py`.
- [x] T-007 [FR-001-05] Write failing tests for null counts, text-length summaries, exact duplicates, and label counts in `tests/test_audit.py`.
- [x] T-008 [FR-001-05] Implement data-quality summaries in `src/ntllm/audit.py`.
- [x] T-009 [FR-001-06] Write failing tests for prompt-group availability, duplicate IDs, and invalid text in `tests/test_validation.py`.
- [x] T-010 [FR-001-06] Implement group and row-integrity checks in `src/ntllm/validation.py`.
- [x] T-011 [FR-001-07] critical: Write failing tests proving unresolved rules, missing train labels, or an unknown allowlist block supervised readiness.
- [x] T-012 [FR-001-07] critical: Implement explicit readiness blockers in `src/ntllm/audit.py`.
- [x] T-013 [FR-001-08] Write failing tests for missing files, unreadable CSVs, unknown schema, and unrecognized classes.
- [x] T-014 [FR-001-08] Implement actionable CLI failures and audit status in `src/ntllm/__main__.py`.
- [x] T-015 [FR-001-03] Add `python -m ntllm audit --config configs/baseline.yaml` argument parsing and JSON report output.
- [x] T-016 [FR-001-04] Add fixture-based CLI/report coverage and document how to run the audit.

## Dependencies and Execution Order

- The documentation tasks T-001 and T-002 are independent; unresolved facts
  stay blocked and do not prevent fixture-based audit development.
- Write each test task before its matching implementation task; run it red for
  the expected reason before implementing.
- The critical readiness pair T-011/T-012 must be covered by tests before the
  readiness logic is treated as complete.
- The end-to-end CLI task depends on the schema, manifest, profile, validation,
  and readiness components.
