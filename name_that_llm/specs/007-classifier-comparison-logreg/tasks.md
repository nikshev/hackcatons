# Tasks: Classifier Comparison on Fixed Char Features

**Input**: Design documents from `specs/007-classifier-comparison-logreg/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: Classifier option (US1, Priority: P1)

- [x] T-001 [FR-007-02] Write failing tests for `classifier: logreg` pipeline construction and invalid classifier rejection in `tests/test_training.py`.
- [x] T-002 [FR-007-02] Implement `classifier` option in `src/ntllm/training.py` (default LinearSVC, opt-in multinomial logreg).
- [x] T-003 [FR-007-01] Write fixture test proving SVC and logreg candidates share folds/checksum/allowlist via `tune_from_config`.
- [x] T-004 [FR-007-01] Add versioned experiment YAML `configs/experiments/char_logreg_vs_svc.yaml` with identical features across candidates.

## Phase 2: Fair ranking (US1, Priority: P1)

- [x] T-005 [FR-007-03] Test mean/per-fold metrics and deterministic ranking for the classifier grid.
- [x] T-006 [FR-007-04] Test tune never reads `test.csv` and failures are explicit, not ranked.

## Dependencies and Execution Order

- Test tasks go red before their matching implementation tasks.
- Run the real grid only after unit tests pass.
- Do not promote the winner to baseline/final-model in this feature.
