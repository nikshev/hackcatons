# Feature Specification: Offline Kaggle Submission Notebook

**Feature Branch**: `005-offline-kaggle-notebook`
**Created**: 2026-10-26
**Status**: Draft
**Input**: User description: "Build a self-contained Kaggle Notebook generated from baseline.yaml that runs offline, trains the selected char-TF-IDF model on competition train.csv, predicts test.csv, validates the official id/model submission contract, and writes submission.parquet to the working directory."

## User Scenarios & Testing

### User Story 1 - Run the selected baseline in Kaggle (Priority: P1)

As the project developer, I need a self-contained offline notebook that loads
the competition files, fits the OOF-selected baseline, and produces Kaggle's
submission file without depending on a network service.

**Why this priority**: Kaggle submissions are evaluated through a Notebook with
internet disabled and a fixed runtime limit.

**Independent Test**: Generate the notebook from the current versioned config,
execute its code locally with the same train/test schema, and verify the output
against the submission contract.

**Acceptance Scenarios**:

1. **Given** the competition files are mounted and required packages are
   available, **When** the notebook runs, **Then** it fits the configured final
   model on train and writes `submission.parquet` in the Notebook working
   directory.
2. **Given** the notebook runs with internet disabled, **When** all cells
   complete, **Then** it makes no network request and attempts no dependency
   installation or model download.
3. **Given** test rows in a particular order, **When** predictions are written,
   **Then** there is exactly one row per test ID in that order, with `id` and
   `model` columns and labels in the official allowlist.
4. **Given** the baseline configuration changes, **When** the notebook is
   regenerated, **Then** its embedded config version/hash and model parameters
   match the YAML source.

### User Story 2 - Inspect notebook readiness locally (Priority: P2)

As the project developer, I need to smoke-test the exact generated notebook code
on local data fixtures before attaching inputs and submitting it through Kaggle.

**Why this priority**: An offline Notebook has no package/network recovery path;
schema and output failures should be caught before Kaggle execution.

**Independent Test**: Execute the generated code against fixtures and verify
runtime summary, output schema, labels, and row-to-ID coverage without exposing
passage text in logs or artifacts.

**Acceptance Scenarios**:

1. **Given** valid fixture CSVs, **When** the generated code executes, **Then**
   it writes a valid Parquet output and a compact metadata summary.
2. **Given** malformed schema, duplicate IDs, or invalid labels, **When** the
   notebook runs, **Then** it stops with a clear error before claiming a valid
   submission.

### Edge Cases

- Kaggle competition input path is unavailable or the test file is empty.
- The local test sample contains only three rows and is not representative of
  the hidden rerun's runtime or distribution.
- Train/test schemas differ from the local audit or labels are outside the
  allowlist.
- A config changes after notebook generation, leaving stale embedded settings.
- Runtime exceeds the nine-hour CPU/GPU limit or package versions differ in
  Kaggle's offline image.

## Requirements

### Functional Requirements

- **FR-005-01**: The notebook MUST read only the competition's mounted train and
  test files and MUST NOT depend on an internet connection, API, or downloaded
  package/model during execution.
- **FR-005-02**: The notebook MUST embed the baseline config version/hash, chosen
  model, all model parameters, and official allowlist from the generated
  versioned configuration.
- **FR-005-03**: The notebook MUST fit the selected model on all permitted
  training rows; learned text transformations MUST be fit only on train.
- **FR-005-04**: The notebook MUST predict each test row once and write
  `submission.parquet` with exactly the columns `id` and `model` in test-row
  order.
- **FR-005-05**: Before writing a valid-output status, the notebook MUST reject
  null/blank IDs/text/predictions, duplicate IDs, missing/extra test IDs, or
  predictions outside the official label allowlist.
- **FR-005-06**: The notebook MUST print a compact run summary with config/data
  checksums, package versions, fit/predict runtime, and row count; it MUST NOT
  print or persist raw passage text.
- **FR-005-07**: The committed notebook MUST be reproducible from
  `configs/baseline.yaml` using the local generator; drift in model parameters
  or allowlist MUST be detected by tests.
- **FR-005-08**: Local smoke execution MUST allow explicit input/output path
  overrides while preserving Kaggle's default mounted paths.

### Key Entities

- **Generated Notebook**: Self-contained Kaggle code with a versioned config
  snapshot and no external calls.
- **Submission file**: Ordered mapping of every test ID to one official model
  label.
- **Notebook run summary**: Compact provenance, environment, and runtime record.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Generated Notebook parameters and labels exactly match the current
  baseline YAML and its recorded checksum.
- **SC-002**: Local smoke execution writes a file with exact columns `id` and
  `model`, one row per test ID, and no invalid labels/nulls/duplicates.
- **SC-003**: Static inspection and local smoke execution confirm zero network
  requests, package installs, or remote model downloads.
- **SC-004**: Fit and prediction runtime are measured and the complete Notebook
  can be benchmarked against the official nine-hour limit.
- **SC-005**: No raw passage text appears in generated logs, summary, or
  submission artifacts.

## Assumptions

- Kaggle mounts competition input at `/kaggle/input/competitions/name-that-llm/`
  and expects output at `/kaggle/working/submission.parquet`.
- The OOF-selected model is `char_tfidf_svc` in baseline config version 9.
- Kaggle's image contains the configured scientific Python dependencies; the
  precise Kaggle image versions must be checked in a Notebook smoke test.
- The local three-row test file supports contract smoke tests only, not hidden
  runtime or accuracy estimates.

## Scope Boundaries

This feature packages the already selected local baseline for offline Kaggle
Notebook execution. It does not tune on test/leaderboard labels, submit to
Kaggle, or assert participant eligibility or prize qualification.
