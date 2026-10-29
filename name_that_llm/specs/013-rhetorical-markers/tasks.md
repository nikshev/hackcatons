# Tasks: Rhetorical and Formatting Markers

**Input**: Design documents from `specs/013-rhetorical-markers/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: Marker inventory (US1, Priority: P1)

- [x] T-001 [FR-013-02] Write tests for marker rates on crafted passages, determinism, and finiteness.
- [x] T-002 [FR-013-02] Implement marker lists and rates in `src/ntllm/stylometry.py`.
- [x] T-003 [FR-013-03] Verify scaling stays per-fold via existing pipeline tests.

## Phase 2: Fair comparison (US1, Priority: P1)

- [x] T-004 [FR-013-01] Rerun the identical flat/style/fusion grid on the saved folds.
- [x] T-005 [FR-013-04] Test mean/per-fold metrics and deterministic ranking.
- [x] T-006 [FR-013-05] Verify tune never reads `test.csv`.

## Dependencies and Execution Order

- Red tests before implementation.
- Grid config unchanged; only the extractor changes.
- No baseline promotion inside this feature.
