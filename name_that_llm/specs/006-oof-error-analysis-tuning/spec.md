# Feature Specification: OOF Error Analysis and Controlled Tuning

**Feature Branch**: `006-oof-error-analysis-tuning`
**Created**: 2026-10-27
**Status**: Draft
**Input**: User description: "Analyze prompt-group OOF errors by class, confusion pair, fold, prompt group, and text length without saving raw text or identifiers; run a bounded, versioned TF-IDF hyperparameter experiment on the same saved folds and compare stability before selecting a candidate."

## User Scenarios & Testing

### User Story 1 - Understand baseline errors (Priority: P1)

As the project developer, I need an aggregate report of the OOF mistakes so I
can form a specific model-improvement hypothesis without inspecting test labels
or exposing individual passages.

**Why this priority**: A single mean score hides which classes, prompt groups,
folds, or text lengths limit the baseline.

**Independent Test**: Build a small OOF fixture with known wrong predictions,
then verify class recalls, top confusion pairs, fold results, group error rates,
and length-bin counts without IDs or text in the report.

**Acceptance Scenarios**:

1. **Given** a complete OOF run, **When** error analysis is generated,
   **Then** it reports per-class support/recall and the most frequent
   misclassification pairs.
2. **Given** the same OOF run, **When** fold, prompt-group, and text-length
   analyses are computed, **Then** aggregates are traceable to the run and
   contain no example IDs or passage text.
3. **Given** OOF predictions or train checksums do not match the run manifest,
   **When** analysis is requested, **Then** it stops with a clear integrity
   error rather than mixing runs.

### User Story 2 - Compare bounded baseline hypotheses (Priority: P1)

As the project developer, I need a small, explicit TF-IDF parameter sweep on the
same grouped folds so I can compare alternatives without leakage or accidental
test-set tuning.

**Why this priority**: The baseline is above the dummy score but still has weak
recall for several labels; modest, interpretable changes should be tested before
adding external data or large models.

**Independent Test**: Run a small fixture grid, then verify every candidate uses
the same saved folds, per-fold balanced accuracy is reported, rankings are
deterministic, and no test file is opened.

**Acceptance Scenarios**:

1. **Given** a versioned experiment YAML, **When** tuning runs, **Then** each
   candidate uses the same train checksum, prompt folds, and official label
   allowlist.
2. **Given** a candidate, **When** results are summarized, **Then** mean and
   per-fold balanced accuracy, macro F1, and runtime are reported.
3. **Given** a candidate improves only one fold or class while degrading the
   rest, **When** candidates are compared, **Then** the per-fold results expose
   the instability instead of hiding it behind one score.

### Edge Cases

- OOF predictions omit, duplicate, or add training IDs.
- A run's train checksum differs from the current local file.
- A prompt group is split across folds in the run artifact.
- A candidate config has unsupported vectorizer/classifier settings.
- A candidate fails on one fold or predicts a label outside the allowlist.
- Text length bins are empty or a class is absent from a fold.
- A run has OOF predictions but no source train file available for length/group
  aggregates.

## Requirements

### Functional Requirements

- **FR-006-01**: Error analysis MUST consume one named OOF run and verify its
  train checksum, split hash, ID coverage, and prompt-group fold exclusivity.
- **FR-006-02**: The report MUST include per-class support/recall, top
  off-diagonal confusion pairs, per-fold metrics, aggregate prompt-group error
  rates, and error counts by configured text-length bins.
- **FR-006-03**: Error-analysis output MUST NOT include raw passage text, row
  IDs, or individual prompt IDs.
- **FR-006-04**: Tuning candidates MUST come from a versioned experiment YAML and
  use the same saved prompt-group folds, train checksum, seed, and allowlist.
- **FR-006-05**: Each candidate MUST report mean and per-fold balanced accuracy,
  macro F1, runtime, and class-level recall.
- **FR-006-06**: Tuning MUST NOT load or inspect `test.csv`, hidden labels, or
  leaderboard scores.
- **FR-006-07**: Unsupported model settings, data/split mismatch, incomplete
  OOF coverage, or a candidate failure MUST be reported and MUST NOT be silently
  ranked as a valid result.
- **FR-006-08**: Candidate ranking MUST use the primary balanced-accuracy metric
  and retain fold-level variability; the report MUST not claim statistical
  significance from a single grouped split.
- **FR-006-09**: Generated reports MUST record the experiment config hash,
  dataset checksum, split hash, and baseline code revision.

### Key Entities

- **OOF error**: A training-row prediction made by a fold model that excluded
  its prompt group.
- **Confusion pair**: An actual label and a different predicted label with an
  aggregate count.
- **Experiment candidate**: One versioned model/vectorizer parameter set
  evaluated on the fixed folds.
- **Experiment report**: Aggregate and per-fold scores plus provenance for all
  candidates in one sweep.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All 2,699 current training IDs and 150 prompt groups are covered
  exactly once by the named OOF run before analysis.
- **SC-002**: The generated error report includes counts for all 18 official
  labels and per-fold metrics, with zero raw text/ID values.
- **SC-003**: Every configured tuning candidate has five fold results on the
  same split; no candidate uses test data.
- **SC-004**: Re-running the same experiment config/data/split yields identical
  OOF predictions and metrics (runtime may differ).
- **SC-005**: The summary ranks candidates by mean balanced accuracy while
  displaying per-fold spread and macro F1 for interpretation.

## Assumptions

- The saved OOF artifacts in `reports/runs/` and split assignments are the
  source of evaluation truth; the current Kaggle test sample has no labels.
- The selected baseline is char TF-IDF, with word/combined candidates retained
  as comparisons.
- The local prompt-group split remains fixed during a tuning sweep; changing
  fold assignments requires rerunning the original baselines.
- Paid API data and external corpora are not part of this experiment.

## Scope Boundaries

This feature analyzes existing OOF artifacts and evaluates a bounded config
grid. It does not train on test, use leaderboard feedback, add external data,
change the fixed split, or claim the best local OOF candidate is a winning
competition submission.
