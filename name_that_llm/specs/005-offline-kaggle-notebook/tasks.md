# Tasks: Offline Kaggle Submission Notebook

**Input**: Design documents from `specs/005-offline-kaggle-notebook/`
**Prerequisites**: `plan.md`, `spec.md`, `data-model.md`, and `contracts/notebook.md`

## Phase 1: Config-driven notebook generation (US1, Priority: P1)

- [x] T-001 [FR-005-02] Write tests that generated model/label parameters and config hash match YAML in `tests/test_submission_notebook.py`.
- [x] T-002 [FR-005-02] Implement `scripts/build_submission_notebook.py` to render a self-contained notebook from the baseline config.
- [x] T-003 [FR-005-07] Write a test proving the committed notebook is reproducible from the generator.
- [x] T-004 [FR-005-07] Generate and maintain `notebooks/submission.ipynb` from YAML.

## Phase 2: Offline prediction and contract validation (US1, Priority: P1)

- [x] T-005 [FR-005-01] Write tests that the notebook code runs with only local train/test inputs and no network/install calls.
- [x] T-006 [FR-005-03] Implement generated char-TF-IDF full-train fit and test prediction.
- [x] T-007 [FR-005-04] Write tests for exact `id`/`model` output, row order, and one prediction per test ID.
- [x] T-008 [FR-005-04] Write `submission.parquet` to Kaggle working directory with local override for smoke tests.
- [x] T-009 [FR-005-05] Write tests for nulls, duplicates, ID mismatch, and invalid labels.
- [x] T-010 [FR-005-05] Enforce the submission schema/ID/label contract before output.
- [x] T-011 [FR-005-06] Write tests for checksums, runtime summary, and absence of raw text in logs.
- [x] T-012 [FR-005-06] Emit compact run provenance and execution timing.
- [x] T-013 [FR-005-08] Execute the generated notebook code locally against fixtures and Kaggle samples.

## Dependencies and Execution Order

- Test generator/config synchronization before generating the committed
  Notebook.
- Keep the Kaggle paths as defaults; use only explicit local environment
  overrides in smoke tests.
- The local three-row test validates schema only; the Kaggle platform smoke
  remains a separate final gate.
