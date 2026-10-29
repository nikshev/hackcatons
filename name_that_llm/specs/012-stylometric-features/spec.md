# Feature Specification: Stylometric Features Alone and Fused

**Feature Branch**: `012-stylometric-features`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Evaluate handcrafted stylometric features (punctuation, case, length, repetition, function words) alone and fused with char TF-IDF on identical saved prompt-group folds; compare against flat char baseline with per-fold spread. No test data.

## User Scenarios & Testing

### User Story 1 - Test explicit style signals (Priority: P1)

As the project developer, I need a style-only and a style+char candidate on the
same folds, because n-gram distributions mix content with style while explicit
markers (em-dashes, casing, sentence rhythm, function words) may survive the
prompt change better.

**Why this priority**: All sparse/dense token-distribution tweaks plateau near
0.30; error analysis points at authorial habits (punctuation, casing) that
n-grams capture only indirectly.

**Independent Test**: Run flat, style-only, and fusion candidates on fixtures
and the real split; verify identical fold coverage, deterministic ranking, and
per-fold spread.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** all three run,
   **Then** assignments, seed, and allowlist are identical; only features differ.
2. **Given** a validation fold, **When** the pipeline fits, **Then** scaler and
   classifier see only fold-train rows.
3. **Given** completed candidates, **When** ranked, **Then** mean and per-fold
   balanced accuracy are shown with no hidden-test claim.

### Edge Cases

- A passage has no sentence terminators; length features fall back to whole-text
  values without NaN.
- A text is very short; ratios use guarded denominators, never NaN/inf.
- The extractor receives a blank or non-string input; reject explicitly.

## Requirements

### Functional Requirements

- **FR-012-01**: All candidates MUST use the same saved prompt-group folds,
  train checksum, seed, and official allowlist; only the feature set may differ.
- **FR-012-02**: The extractor MUST emit a fixed ordered vector of prompt-
  independent style ratios/counts with no NaN/inf, deterministically.
- **FR-012-03**: Style features MUST be scaled inside the per-fold pipeline fit
  on fold-train rows only.
- **FR-012-04**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall in the same schema as flat candidates.
- **FR-012-05**: Tuning MUST NOT open `test.csv` or test labels; failures are
  explicit and never silently ranked.

### Key Entities

- **Style vector**: Fixed-order dense authorship statistics per passage.
- **Fusion candidate**: Char TF-IDF unioned with scaled style features.
- **Experiment report**: Ranked flat/style/fusion comparison with per-fold
  spread and provenance.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every candidate has five fold results on the identical split.
- **SC-002**: Extractor output is finite and deterministic for identical input.
- **SC-003**: Ranking is deterministic and preserves per-fold spread.

## Assumptions

- English-majority texts; function-word lists are fixed linguistic resources,
  not tuned constants.
- Same char 2–5 settings and LinearSVC in the fusion branch.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature tests stylometric features only. It does not change the
production baseline, refit the final model, or touch test data. External
training data is a separate, explicitly gated track.
