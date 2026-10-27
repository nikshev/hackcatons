# Tasks: Offline Final Fit and Submission Pipeline

**Input**: Design documents from `specs/004-offline-submission-pipeline/`
**Prerequisites**: `plan.md`, `spec.md`, `data-model.md`, and `contracts/inference-cli.md`

## Phase 1: Final model bundle (US1, Priority: P1)

- [x] T-001 [FR-004-01] Write tests for training-readiness gating and selected model choice in `tests/test_inference.py`.
- [x] T-002 [FR-004-01] Implement final-fit config/readiness/data validation in `src/ntllm/inference.py`.
- [x] T-003 [FR-004-02] Write tests for complete pipeline serialization/reload and train-only fit.
- [x] T-004 [FR-004-02] Fit the selected pipeline on all train rows and serialize the artifact bundle.
- [x] T-005 [FR-004-03] Write manifest tests for data/config/model/code/library provenance.
- [x] T-006 [FR-004-03] Persist deterministic model ID and bundle manifest.

## Phase 2: Offline prediction and submission validation (US2, Priority: P1)

- [x] T-007 [FR-004-04] Write tests for input-order-preserving `id`/`text` predictions and one row per ID.
- [x] T-008 [FR-004-04] Implement offline model loading and Parquet prediction output.
- [x] T-009 [FR-004-05] Write tests for null and out-of-allowlist prediction rejection.
- [x] T-010 [FR-004-05] Enforce official labels and non-null output values.
- [x] T-011 [FR-004-06] Write validator tests for schema, missing/extra/duplicate IDs, and order.
- [x] T-012 [FR-004-06] Implement the full submission contract validator.
- [x] T-013 [FR-004-07] Write repeated-inference determinism tests.
- [x] T-014 [FR-004-07] Preserve deterministic input/model-to-prediction behavior.
- [x] T-015 [FR-004-08] Write tests proving final fit uses train only and predict is offline/no-fit.
- [x] T-016 [FR-004-08] Keep train, predict, and validator IO boundaries explicit.
- [x] T-017 [FR-004-09] Write tests proving failed fit/predict/validation leave no partial output.
- [x] T-018 [FR-004-09] Publish artifacts atomically after successful checks.
- [x] T-019 [FR-004-04] Add `fit-final`, `predict`, and `validate-submission` CLI commands and update quickstart.

## Dependencies and Execution Order

- Write and observe model/artifact tests red before implementing final-fit code.
- Complete the bundle before prediction; complete prediction before validator
  end-to-end verification.
- Use the actual three-row test only as a smoke test; hidden-set quality remains
  unknown until Kaggle evaluates a notebook submission.
