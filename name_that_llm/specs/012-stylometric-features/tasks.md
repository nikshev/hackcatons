# Tasks: Stylometric Features Alone and Fused

**Input**: Design documents from `specs/012-stylometric-features/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: Style extractor and estimators (US1, Priority: P1)

- [x] T-001 [FR-012-02] Write tests for fixed feature order, finiteness, determinism, and blank-input rejection.
- [x] T-002 [FR-012-02] Implement `extract_style_matrix` in `src/ntllm/stylometry.py`.
- [x] T-003 [FR-012-03] Write tests for fold-train-only scaling and new estimator types.
- [x] T-004 [FR-012-03] Implement `style_svc` and `style_char_tfidf` in `build_estimator`.

## Phase 2: Fair comparison (US1, Priority: P1)

- [x] T-005 [FR-012-01] Add experiment YAML with flat/style/fusion on identical features and folds.
- [x] T-006 [FR-012-04] Test mean/per-fold metrics and deterministic ranking for the style grid.
- [x] T-007 [FR-012-05] Verify tune never reads `test.csv`.

## Dependencies and Execution Order

- Red tests before implementation in each phase.
- Grid is flat vs style vs fusion on char 2–5.
- No baseline promotion inside this feature.
