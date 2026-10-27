# Tasks: Grouped TF-IDF Baseline Benchmark

**Input**: Design documents from `specs/003-tfidf-grouped-baseline/`
**Prerequisites**: `plan.md`, `spec.md`, `data-model.md`, and `contracts/train-cli.md`

## Phase 1: OOF evaluation engine (US1, Priority: P1)

- [x] T-001 [FR-003-01] Write tests for consuming saved prompt folds and validating exact ID coverage in `tests/test_training.py`.
- [x] T-002 [FR-003-01] Implement fold/report consistency validation in `src/ntllm/training.py`.
- [x] T-003 [FR-003-02] Write a spy-estimator test proving held-out fold texts/labels are never passed to `fit`.
- [x] T-004 [FR-003-02] Implement fresh per-fold pipelines and OOF prediction assembly in `src/ntllm/training.py`.
- [x] T-005 [FR-003-03] Write tests selecting all configured baseline candidate types from YAML.
- [x] T-006 [FR-003-03] Implement dummy, word, char, and combined TF-IDF linear candidates.
- [x] T-007 [FR-003-04] Write tests for exactly one OOF prediction per ID/model and fold provenance.
- [x] T-008 [FR-003-04] Enforce complete OOF coverage and valid allowlist predictions.
- [x] T-009 [FR-003-05] Write metric tests against known labels/predictions, including balanced accuracy and confusion matrix.
- [x] T-010 [FR-003-05] Implement aggregate OOF metrics and per-class recall in `src/ntllm/training.py`.

## Phase 2: Reproducibility and CLI artifacts (US2, Priority: P2)

- [x] T-011 [FR-003-06] Write manifest/artifact tests for hashes, config, split, seed, model params, versions, and code state.
- [x] T-012 [FR-003-06] Implement deterministic run IDs and JSON/Parquet artifacts under `reports/runs/`.
- [x] T-013 [FR-003-07] Write a test proving train-only execution does not open `test.csv`.
- [x] T-014 [FR-003-07] Keep OOF training isolated to train rows and saved folds.
- [x] T-015 [FR-003-08] Write readiness-blocker and stale-split tests; assert no completed run output is published.
- [x] T-016 [FR-003-08] Gate training on current audit readiness and split/config/data hashes.
- [x] T-017 [FR-003-09] Write a test for predictions outside the official allowlist.
- [x] T-018 [FR-003-09] Validate all OOF predictions against the configured official labels.
- [x] T-019 [FR-003-06] Add `python -m ntllm train --config configs/baseline.yaml` and document local usage.

## Dependencies and Execution Order

- Write and observe tests red before implementing the OOF engine.
- Complete tasks T-001 through T-010 before artifact/CLI integration.
- Do not read or load the configured test path anywhere in this feature.
- The actual run remains gated on a current audit and matching split report.
