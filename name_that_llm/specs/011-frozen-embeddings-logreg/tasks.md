# Tasks: Frozen Embeddings with Logistic Regression

**Input**: Design documents from `specs/011-frozen-embeddings-logreg/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: Embedding evaluation (US1, Priority: P1)

- [x] T-001 [FR-011-02] Write tests for encoder loading, normalized output shape, determinism, and missing-dir failure (stubbed encoder).
- [x] T-002 [FR-011-02] Implement `src/ntllm/embeddings.py` with frozen local MiniLM loading and offline encode.
- [x] T-003 [FR-011-03] Write test proving the classifier fits only fold-train vectors (stubbed encoder + spy).
- [x] T-004 [FR-011-03] Implement `evaluate_embedding_oof` reusing prompt-group OOF invariants.
- [x] T-005 [FR-011-01] Route `type: sbert_logreg` candidates through tune on identical folds/checksum/allowlist.
- [x] T-006 [FR-011-04] Test mean/per-fold metrics and deterministic ranking for the embedding grid.
- [x] T-007 [FR-011-05] Add experiment YAML; verify tune never reads `test.csv`.

## Phase 2: Packaging note (US1, Priority: P1)

- [ ] T-008 [FR-011-06] Verify `docs/embeddings_packaging.md` with a real Kaggle upload (draft written; unverified).

## Dependencies and Execution Order

- Red tests before implementation in each phase.
- Grid is flat-vs-embedding on identical folds.
- No baseline promotion inside this feature.
