# Feature Specification: Classifier Comparison on Fixed Char Features

**Feature Branch**: `007-classifier-comparison-logreg`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Compare LinearSVC versus multinomial logistic regression on identical char 2–5 TF-IDF features and the same saved prompt-group folds; one main change (classifier only), no test data.

## User Scenarios & Testing

### User Story 1 - Compare classifiers fairly (Priority: P1)

As the project developer, I need both classifiers evaluated on exactly the same
features, folds, labels, and seed so the difference measures the classifier,
not data or split noise.

**Independent Test**: Run a small fixture grid with one SVC and one logreg
candidate sharing features/folds; verify identical fold coverage, deterministic
ranking, and per-fold spread for both.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** tuning runs,
   **Then** every candidate uses identical assignments, seed, and allowlist.
2. **Given** completed candidates, **When** ranked, **Then** the report shows
   mean and per-fold balanced accuracy for each, with no significance claim.

### Edge Cases

- Logreg fails to converge on a fold; report it as a failed candidate, not a rank.
- A candidate predicts a label outside the allowlist; fail explicitly.
- Experiment config names an unknown classifier; reject before any fit.

## Requirements

### Functional Requirements

- **FR-007-01**: Candidates MUST share the same saved prompt-group folds, train
  checksum, seed, and official allowlist; only the classifier may differ.
- **FR-007-02**: The pipeline MUST support a `classifier: logreg` option for
  TF-IDF candidates, defaulting to the existing LinearSVC behavior when absent.
- **FR-007-03**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall for every candidate.
- **FR-007-04**: Tuning MUST NOT open `test.csv`, test labels, or leaderboard
  data; failures leave no partial ranking.

### Key Entities

- **Classifier candidate**: One `classifier` setting on fixed char 2–5 features.
- **Experiment report**: Ranked aggregate plus per-fold metrics and provenance.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All candidates have five fold results on the identical split.
- **SC-002**: Ranking is deterministic and preserves per-fold spread.
- **SC-003**: No test data is touched; failed candidates are explicit, not ranked.

## Assumptions

- Char 2–5 TF-IDF settings are frozen from the promoted baseline.
- Logistic regression uses multinomial loss with a fixed iteration cap.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature compares classifiers only. It does not change the production
baseline config, refit the final model, or regenerate the Notebook unless the
winner is later promoted in a separate explicit step.
