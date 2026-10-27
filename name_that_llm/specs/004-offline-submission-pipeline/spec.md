# Feature Specification: Offline Final Fit and Submission Pipeline

**Feature Branch**: `004-offline-submission-pipeline`
**Created**: 2026-10-26
**Status**: Draft
**Input**: User description: "Fit the selected grouped-CV baseline on all audited official training rows; save a portable model bundle, predict the offline Kaggle test CSV into submission.parquet, and validate the exact id/model schema, IDs, nulls, and official labels. Keep inference network-free and preserve source/config provenance."

## User Scenarios & Testing

### User Story 1 - Fit a final model from the chosen baseline (Priority: P1)

As the project developer, I need to fit the selected baseline on all permitted
training data and keep its complete preprocessing/model artifact so inference
does not fit or download anything.

**Why this priority**: The OOF benchmark establishes a local model choice; a
submission needs one final artifact trained on all available labeled rows.

**Independent Test**: Fit against a small fixture after a successful audit,
then reload the saved artifact and predict the fixture's test texts without
network access or additional fitting.

**Acceptance Scenarios**:

1. **Given** current audit readiness and a selected configured model, **When**
   final fitting runs, **Then** every permitted train row is used and the
   preprocessing plus classifier are saved together with provenance metadata.
2. **Given** training readiness is blocked or the config/data checksum is stale,
   **When** final fitting is requested, **Then** it fails before publishing a
   model bundle.

### User Story 2 - Predict and validate Kaggle's submission contract (Priority: P1)

As the project developer, I need offline predictions and a strict submission
validator so a malformed or out-of-allowlist file cannot be treated as ready.

**Why this priority**: Kaggle evaluates a specific one-row-per-ID Parquet
contract with exact official model labels, and the notebook has no internet.

**Independent Test**: Load a fitted fixture artifact, predict its test rows,
and validate the generated file; then alter IDs, labels, nulls, and columns and
verify each invalid fixture is rejected.

**Acceptance Scenarios**:

1. **Given** an artifact and a test CSV, **When** predictions are generated,
   **Then** exactly one prediction is written per input ID in input order, with
   no network request or inference-time fit.
2. **Given** a valid Kaggle submission file, **When** validation runs, **Then**
   its exact `id`/`model` schema, IDs, null constraints, and official labels pass.
3. **Given** missing/extra/duplicate IDs, null values, extra columns, or an
   unlisted label, **When** validation runs, **Then** it fails with an
   actionable diagnosis.

### Edge Cases

- A saved artifact's config, data, or model checksum does not match the current
  inputs.
- Test IDs are duplicated, missing, blank, or contain a different order than
  the sample submission.
- A model produces an unlisted label or invalid-length prediction array.
- The test file contains only the three interactive sample rows; this is a
  smoke test, not evidence of hidden-set performance.
- Dependencies required by the artifact are unavailable in the offline
  Notebook environment.
- An existing output path would overwrite raw competition data or another run.

## Requirements

### Functional Requirements

- **FR-004-01**: Final fitting MUST require `training_readiness: ready`, use the
  model explicitly selected in versioned configuration, and fit on all audited
  labeled training rows.
- **FR-004-02**: The saved artifact MUST contain learned text preprocessing and
  the fitted classifier; inference MUST load them without fitting or network
  access.
- **FR-004-03**: The model manifest MUST record train/config/model hashes,
  config version, allowed labels, code revision, library versions, and fit date.
- **FR-004-04**: Prediction MUST read the actual test schema and produce exactly
  one `model` prediction for each nonblank test `id`, in test-row order.
- **FR-004-05**: Prediction and submission validation MUST use only the current
  official allowlist and MUST reject null/blank or unlisted labels.
- **FR-004-06**: Submission validation MUST require exactly the columns `id`
  and `model`, one nonblank ID per test row, no duplicate IDs, and exact equality
  between submission and test ID sets.
- **FR-004-07**: Repeated prediction with the same artifact and test input MUST
  produce the same ordered ID/label rows.
- **FR-004-08**: Test data MUST NOT be used to fit the final model or tune its
  parameters; no internet request is allowed during prediction.
- **FR-004-09**: Failed readiness, artifact, prediction, or validation checks
  MUST not leave a partial submission file.

### Key Entities

- **Final model bundle**: Fitted preprocessing and classifier plus a manifest of
  the data/config/code/environment used to fit them.
- **Prediction file**: One official model label per test ID in input order.
- **Submission validation result**: Contract check covering schema, IDs, nulls,
  and allowlisted labels.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A successful final fit records all train IDs and the current
  config/train checksums in its manifest.
- **SC-002**: For a test file of N valid rows, prediction produces exactly N
  rows with the same ordered ID sequence and no null/out-of-allowlist labels.
- **SC-003**: Repeated inference from the same artifact/input yields identical
  ordered predictions.
- **SC-004**: The validator rejects each fixture with wrong schema, nulls,
  missing/extra/duplicate IDs, or invalid labels.
- **SC-005**: A readiness or validation failure leaves no new completed artifact
  or submission file.

## Assumptions

- Kaggle's sample submission confirms the output columns `id` and `model`, and
  one row per test ID.
- The current local interactive test sample has only three rows; production
  scoring uses a hidden rerun.
- The OOF benchmark supports selecting the character TF-IDF model as the first
  final-fit candidate; later selection changes require a config/changelog update.
- The local model bundle will be attached to a Kaggle Notebook as an offline
  artifact; Kaggle package compatibility still requires a notebook smoke test.

## Scope Boundaries

This feature fits the selected model, predicts a provided test CSV, and validates
the submission file. It does not submit to Kaggle, optimize against leaderboard
scores, or guarantee eligibility/payout. The final Kaggle Notebook packaging and
runtime smoke test are a subsequent feature.
