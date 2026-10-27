# Feature Specification: Grouped TF-IDF Baseline Benchmark

**Feature Branch**: `003-tfidf-grouped-baseline`
**Created**: 2026-10-26
**Status**: Draft
**Input**: User description: "Run a reproducible baseline benchmark on the deterministic prompt-group folds: compare a dummy classifier, word TF-IDF, character TF-IDF, and combined word/character TF-IDF with a linear classifier; generate out-of-fold predictions, balanced accuracy, macro F1, per-class recall, confusion matrices, and run metadata. Fit each vectorizer only on that fold's training rows and preserve the official 18-label allowlist."

## User Scenarios & Testing

### User Story 1 - Measure a prompt-held-out baseline (Priority: P1)

As the project developer, I need repeatable out-of-fold predictions and
comparable metrics from simple text baselines so I can decide whether the
competition data supports a useful modeling direction before spending more time
or resources.

**Why this priority**: The split is prompt-grouped to reflect the hidden test's
unseen prompts; baseline metrics must use that split and remain comparable.

**Independent Test**: Run a small labeled fixture through every configured
baseline and verify one OOF prediction per row/model, all fold rows predicted
only by models that did not train on their prompt group, and metrics calculated
from those predictions.

**Acceptance Scenarios**:

1. **Given** audited train data and valid saved prompt-group folds, **When** the
   developer runs the baseline benchmark, **Then** it compares the configured
   no-skill and text-feature models using the same folds.
2. **Given** a validation fold, **When** a model is fit for that fold, **Then**
   all learned text transforms and the classifier use only the other folds.
3. **Given** a complete run, **When** the developer reviews its artifacts,
   **Then** OOF predictions and balanced accuracy, macro F1, per-class recall,
   and confusion matrices are available for each model.
4. **Given** identical data, config, saved splits, and seed, **When** the run is
   repeated, **Then** its OOF predictions and metrics are identical.

### User Story 2 - Preserve comparable experiment evidence (Priority: P2)

As the project developer, I need each run tied to its data, config, split, and
code revision so later experiments can be compared without guessing which
inputs changed.

**Why this priority**: Model scores are not useful evidence if the dataset,
folds, or parameters cannot be reconstructed.

**Independent Test**: Inspect a generated run manifest and match its checksums,
versions, split seed/hash, model settings, and code revision to the run inputs.

**Acceptance Scenarios**:

1. **Given** a completed run, **When** its manifest is read, **Then** it
   identifies the raw train checksum, config version/hash, split hash/seed,
   model parameters, and code revision.
2. **Given** missing or stale split assignments, **When** training is
   requested, **Then** the command reports the mismatch and creates no partial
   run artifacts.

### Edge Cases

- Any class is absent from a fold's training rows.
- A model fails to fit or emits a prediction outside the official allowlist.
- A fold assignment omits or repeats an ID.
- The split report references a different train checksum, seed, or fold count.
- Training-readiness is blocked or the official label allowlist is unavailable.
- The working tree has no commit corresponding to the current code state.

## Requirements

### Functional Requirements

- **FR-003-01**: The benchmark MUST consume the saved prompt-group assignments
  from feature 002 and use the same fold boundaries for every candidate model.
- **FR-003-02**: Every learned text transform and classifier MUST be fit only
  on the training folds for the current OOF fold; validation rows and labels
  MUST NOT participate in fitting.
- **FR-003-03**: The benchmark MUST compare a no-skill classifier, word-based
  TF-IDF, character-based TF-IDF, and combined word/character TF-IDF with a
  linear classifier, using parameters from versioned YAML configuration.
- **FR-003-04**: For each candidate model, the run MUST produce exactly one
  out-of-fold prediction for every training ID; each prediction's fold MUST be
  the fold that held its prompt group out from fitting.
- **FR-003-05**: For each model, the run MUST report balanced accuracy, macro
  F1, recall for each official class, and a confusion matrix.
- **FR-003-06**: Each run MUST record train-data checksum, config version/hash,
  split hash/seed/fold count, source-code checksum, model parameters, software
  versions, code revision, metrics, and runtime.
- **FR-003-07**: The benchmark MUST read only training data; test rows and labels
  MUST NOT be loaded or used in fitting, validation, or model selection.
- **FR-003-08**: Model fitting MUST be blocked when feature 001 reports
  `training_readiness: blocked`, the saved split is stale, or the configured
  allowed-label list is missing or inconsistent with train labels.
- **FR-003-09**: OOF prediction artifacts MUST contain no duplicate/missing
  training IDs and MUST use only labels from the official allowlist.

### Key Entities

- **Baseline model**: One configured no-skill or text-feature classifier.
- **OOF prediction**: A prediction for a training ID from a model that did not
  fit on that ID's prompt group.
- **Fold metric set**: Per-model aggregate and per-class scores computed from
  OOF predictions.
- **Run manifest**: Provenance and configuration needed to reconstruct and
  compare the benchmark run.

## Success Criteria

### Measurable Outcomes

- **SC-001**: For each model, every audited training ID has exactly one OOF
  prediction and no validation prompt group appears in that fold's training
  set.
- **SC-002**: The report contains balanced accuracy, macro F1, all 18 class
  recalls, and an 18-by-18 confusion matrix for every configured model.
- **SC-003**: Repeating a run with identical inputs, config, split, and code
  revision yields identical OOF predictions and metrics.
- **SC-004**: Each completed run has a manifest with all required provenance
  fields; a stale split or blocked readiness creates no completed run.
- **SC-005**: The whole baseline benchmark completes locally without paid
  services or external training data.

## Assumptions

- The training CSV, official label list, and prompt-group folds have passed the
  preceding audit/split features.
- Scikit-learn and its dependencies are available in the local development
  environment; Kaggle availability will be checked separately before notebook
  packaging.
- Balanced accuracy is the primary metric; macro F1, class recall, and
  confusion matrix are diagnostics.
- The baseline compares candidate models on exactly the same saved folds.

## Scope Boundaries

This feature evaluates fixed baseline candidates using OOF predictions. It does
not tune parameters on the hidden test, create the final Kaggle Notebook, submit
predictions, or use external/synthetic training data.
