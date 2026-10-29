# Tasks: Hierarchical Provider-Then-Model Classification

**Input**: Design documents from `specs/008-hierarchical-provider-model/`
**Prerequisites**: `plan.md`, `spec.md`

## Phase 1: Two-stage OOF evaluation (US1, Priority: P1)

- [x] T-001 [FR-008-02] Write tests for deterministic provider derivation and unknown-prefix rejection.
- [x] T-002 [FR-008-02] Implement provider mapping in `src/ntllm/training.py`.
- [x] T-003 [FR-008-03] Write spy tests proving neither stage fits on held-out fold rows/labels.
- [x] T-004 [FR-008-03] Implement `evaluate_hierarchical_oof` with per-fold provider fit and per-provider specialists.
- [x] T-005 [FR-008-01] Write test proving flat and hierarchical runs share folds/checksum/allowlist via `tune_from_config`.
- [x] T-006 [FR-008-04] Report hierarchical metrics in the same schema as flat candidates.

## Phase 2: Fair comparison (US1, Priority: P1)

- [x] T-007 [FR-008-04] Add experiment YAML with flat vs hierarchical candidates on identical features.
- [x] T-008 [FR-008-05] Test tune never reads `test.csv`; failures are explicit, not ranked.

## Dependencies and Execution Order

- Red tests before implementation in each phase.
- Keep the grid to flat-vs-hierarchical on char 2–5 features.
- No baseline promotion inside this feature.
