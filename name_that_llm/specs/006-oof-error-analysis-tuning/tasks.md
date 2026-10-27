# Tasks: OOF Error Analysis and Controlled Tuning

**Input**: Design documents from `specs/006-oof-error-analysis-tuning/`
**Prerequisites**: `plan.md`, `spec.md`, `data-model.md`, and `contracts/analysis-cli.md`

## Phase 1: OOF error report (US1, Priority: P1)

- [x] T-001 [FR-006-01] Write fixtures/tests for checksum, ID coverage, and group-leak validation in `tests/test_analysis.py`.
- [x] T-002 [FR-006-01] Validate the named OOF run against train and split provenance in `src/ntllm/analysis.py`.
- [x] T-003 [FR-006-02] Write known-error fixture tests for class recall/confusion, fold, group, and length aggregates.
- [x] T-004 [FR-006-02] Implement aggregate OOF error analysis without raw IDs/text in `src/ntllm/analysis.py`.
- [x] T-005 [FR-006-03] Test report serialization excludes IDs, prompt IDs, and passage text.
- [x] T-006 [FR-006-03] Write `error_analysis.md` under the run directory.
- [x] T-007 [FR-006-01] Add `ntllm evaluate --run-id` and document its output.

## Phase 2: Bounded grouped-fold tuning (US2, Priority: P1)

- [x] T-008 [FR-006-04] Add a versioned char/word TF-IDF candidate grid at `configs/experiments/char_tfidf_grid.yaml`.
- [x] T-009 [FR-006-04] Test all candidates use the same saved folds and same train checksum.
- [x] T-010 [FR-006-05] Implement candidate evaluation with mean/per-fold metrics and per-class recall.
- [x] T-011 [FR-006-06] Test tuning never reads `test.csv` or test labels.
- [x] T-012 [FR-006-06] Keep tuning inputs train-only and grouped by the persisted split.
- [x] T-013 [FR-006-07] Test candidate failure/stale provenance is not ranked as a valid result.
- [x] T-014 [FR-006-07] Implement explicit error states for failed candidates and stale data/split.
- [x] T-015 [FR-006-08] Test deterministic ranking and retention of per-fold spread.
- [x] T-016 [FR-006-08] Rank candidates by mean balanced accuracy while reporting variability; do not claim significance.
- [x] T-017 [FR-006-09] Test experiment/data/split/code hashes in the summary.
- [x] T-018 [FR-006-09] Write `ntllm tune` summary under `reports/tuning/`.

## Dependencies and Execution Order

- Complete OOF artifact/provenance checks before computing any aggregate.
- Complete error analysis before interpreting candidate-level confusion
  changes.
- Keep the first grid small and run on the fixed five-fold split.
- Do not use test or leaderboard labels to select a candidate.
