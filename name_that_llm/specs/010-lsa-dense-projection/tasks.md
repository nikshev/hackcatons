# Tasks: LSA Dense Projection over Char TF-IDF

**Input**: Design documents from `specs/010-lsa-dense-projection/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: SVD pipeline option (US1, Priority: P1)

- [x] T-001 [FR-010-02] Write failing tests for the `svd_components` pipeline layer, determinism, and dummy rejection.
- [x] T-002 [FR-010-02] Implement the SVD layer in `build_estimator` with fixed random state.
- [x] T-003 [FR-010-03] Write tests for invalid component counts rejected before fitting.
- [x] T-004 [FR-010-06] Reject `svd_components` on dummy candidates.

## Phase 2: Fair comparison (US1, Priority: P1)

- [x] T-005 [FR-010-01] Add experiment YAML with flat vs SVD candidates on identical features.
- [x] T-006 [FR-010-04] Test mean/per-fold metrics and deterministic ranking for the SVD grid.
- [x] T-007 [FR-010-05] Verify tune never reads `test.csv` and adds no network dependency.

## Dependencies and Execution Order

- Red tests before implementation.
- Grid is flat-vs-SVD-512 on char 2–5 only.
- No baseline promotion inside this feature.
