# Tasks: Gemini-Flash Trio Rerank Specialist

**Input**: Design documents from `specs/009-gemini-rerank-specialist/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: Rerank evaluation (US1, Priority: P1)

- [x] T-001 [FR-009-02] Write tests for trio-only rerank scope and trio-label output in `tests/test_training.py`.
- [x] T-002 [FR-009-03] Test rerank scope vs flat, per-ID coverage, and determinism across row order in `tests/test_training.py`.
- [x] T-003 [FR-009-02] Implement trio constant and `evaluate_gemini_rerank_oof` in `src/ntllm/training.py`.
- [x] T-004 [FR-009-01] Extend `tune_from_config` with the `gemini_rerank` architecture on identical folds.
- [x] T-005 [FR-009-04] Test mean/per-fold metrics and deterministic ranking for the rerank grid.
- [x] T-006 [FR-009-05] Add experiment YAML and verify tune never reads `test.csv`.

## Dependencies and Execution Order

- Red tests before implementation.
- Grid is flat-vs-rerank on char 2–5 only.
- No baseline promotion inside this feature.
