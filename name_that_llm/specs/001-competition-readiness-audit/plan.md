# Implementation Plan: Competition Readiness and Data Audit

**Branch**: `001-competition-readiness-audit` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-competition-readiness-audit/spec.md`

## Summary

Deliver a source-backed competition readiness record and a repeatable local
audit of the competition CSVs. The audit reports observed schema and data
quality, produces a provenance manifest without modifying raw files, and makes
supervised-training blockers explicit. It does not train a model. The initial
Kaggle facts and unresolved questions are recorded in `docs/competition.md`.

## Technical Context

**Language/Version**: Python 3.12 or newer
**Primary Dependencies**: pandas, pyarrow, PyYAML; versions fixed after a Kaggle smoke test
**Storage**: Read-only raw CSV files; YAML configuration; JSON audit report and manifest
**Testing**: pytest with offline synthetic fixtures
**Target Platform**: Local Python environment and later Kaggle Notebook
**Project Type**: Python CLI and reusable package
**Performance Goals**: Audit the small official training set without approaching the Notebook runtime budget
**Constraints**: No network access from tests; no raw input mutation; no guessed class allowlist; do not train while readiness blockers remain
**Scale/Scope**: Official competition data (currently described as 2,699 training rows and about 4,000 hidden test rows); one developer

## Constitution Check

- **Traceability**: Every FR maps to one or more tasks; completed tasks require
  code and tests carrying the matching FR marker.
- **Test-first**: Write and observe tests failing before implementation.
- **Reproducibility**: Record config, data checksum, seed/split metadata in
  later training features; audit does not fit learned transformations.
- **Provenance/leakage**: Raw files stay unchanged; data lineage is explicit;
  external/synthetic data remain disabled pending rule clarification.
- **Stop gates**: This feature can audit without resolving eligibility, but
  supervised training remains blocked until permitted labeled data, exact
  schema/classes, and applicable rules are confirmed.

No constitution violations identified.

## Design

### Audit flow

1. Load a version-controlled YAML configuration with input paths and intended
   field roles.
2. Read each CSV without writing to its source; report observed columns before
   applying configured roles.
3. Validate required columns and IDs/text/labels; do not infer a missing role.
4. Compute file checksum, row count, nulls, text lengths, duplicates, prompt
   group availability, and observed label counts.
5. Write a JSON report/manifest and return a clear failure status for missing
   files, incompatible schema, or invalid labels.
6. Mark training readiness separately from audit success. Readiness remains
   blocked if rule, license, schema, or allowlist evidence is unresolved.

### Data and interfaces

- The local schema mapping is configuration, not a code constant.
- `docs/competition.md` records human-verified competition constraints and
  their evidence status.
- `python -m ntllm audit --config configs/baseline.yaml` is the first CLI
  operation, matching the PRD.
- `reports/audit/` contains generated JSON; raw `data/` is not committed.
- The default audit must not claim that local three-row test examples represent
  the hidden evaluation distribution.

### Project Structure

```text
configs/baseline.yaml
src/ntllm/
├── __init__.py
├── __main__.py
├── audit.py
├── data.py
└── validation.py
tests/
├── fixtures/
└── test_audit.py
docs/competition.md
reports/audit/                 # generated, not committed
```

**Structure Decision**: Keep a single Python package and CLI inside the
`name_that_llm` solution folder. Its `.specify/` and `specs/` are local to this
folder even though the workspace Git repository is shared.

## Risks and Open Questions

- The current allowed labels and submission/runtime requirements are captured
  from Kaggle's official pages; verify them against downloaded files and recheck
  the pages before submission.
- Participant-specific eligibility/payout conditions and clarification of paid
  API-generated training data remain unresolved.
- Kaggle's current official pages render dynamically in this environment; the
  dated record clearly distinguishes indexed facts from unresolved checks.
- Near-duplicate detection and language identification may add dependencies or
  runtime; begin with exact duplicates and length/null profiles, then measure
  before expanding the audit.

## Complexity Tracking

No constitution violations to justify.
