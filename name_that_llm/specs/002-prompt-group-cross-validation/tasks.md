# Tasks: Prompt-Group Cross-Validation

**Input**: Design documents from `specs/002-prompt-group-cross-validation/`
**Prerequisites**: `plan.md`, `spec.md`, `data-model.md`, and `contracts/split-cli.md`

## Phase 1: Grouped assignment engine (US1, Priority: P1)

- [x] T-001 [FR-002-01] Write tests proving every prompt group stays within one fold and each ID is assigned once in `tests/test_splits.py`.
- [x] T-002 [FR-002-01] Implement group-to-fold assignment in `src/ntllm/splits.py`.
- [x] T-003 [FR-002-04] Write determinism tests across repeated runs and input row order in `tests/test_splits.py`.
- [x] T-004 [FR-002-04] Implement canonical split hashing and stable seed-based ordering in `src/ntllm/splits.py`.
- [x] T-005 [FR-002-05] Write tests for row/class balance and per-fold summaries in `tests/test_splits.py`.
- [x] T-006 [FR-002-05] Implement deterministic greedy balancing and fold summaries in `src/ntllm/splits.py`.

## Phase 2: Validation and artifacts (US2, Priority: P1)

- [x] T-007 [FR-002-02] Write failing tests for duplicate/missing IDs, groups, labels, and unknown classes in `tests/test_splits.py`.
- [x] T-008 [FR-002-02] Implement strict input validation in `src/ntllm/splits.py`.
- [x] T-009 [FR-002-03] Write tests for invalid fold counts and too few prompt groups.
- [x] T-010 [FR-002-03] Implement fold-setting validation; never fall back to row-wise splitting.
- [x] T-011 [FR-002-06] Write tests for Parquet assignments and report metadata/hash.
- [x] T-012 [FR-002-06] Implement configuration-driven split reporting and Parquet output.
- [x] T-013 [FR-002-07] Write tests proving only train is read and raw bytes are unchanged.
- [x] T-014 [FR-002-07] Keep split generation read-only and exclude test input from the split flow.
- [x] T-015 [FR-002-08] Write tests that invalid inputs leave no output artifacts.
- [x] T-016 [FR-002-08] Stage output artifacts only after all validation and fold assignment succeeds.
- [x] T-017 [FR-002-06] Add `python -m ntllm split --config configs/baseline.yaml` and document it in `docs/quickstart.md`.

## Dependencies and Execution Order

- Write and observe unit tests red before implementing the assignment engine.
- Complete T-001 through T-006 before CLI artifact work.
- Complete all validation tests before writing any output files.
- The real-data split remains blocked until the training CSV is locally present
  and has passed feature 001's audit.
