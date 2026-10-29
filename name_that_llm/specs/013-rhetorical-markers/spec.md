# Feature Specification: Rhetorical and Formatting Markers

**Feature Branch**: `013-rhetorical-markers`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Add rhetorical/formatting marker rates (discourse connectives, hedging, self-reference, markdown structure, closings) to the style vector; compare flat, style-only, and fusion on identical saved prompt-group folds. No test data.

## User Scenarios & Testing

### User Story 1 - Test authorial voice markers (Priority: P1)

As the project developer, I need explicit voice markers because n-grams mix
them with topic content while per-author habits (discourse scaffolding, hedging
style, markdown formatting) may separate models that share vocabulary.

**Why this priority**: Family-level confusion persists; style+char helped, so a
richer voice inventory is the next single-variable step on the same protocol.

**Independent Test**: Run flat, style-only, and fusion on fixtures and the real
split; verify identical fold coverage, deterministic ranking, and per-fold
spread.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** candidates run,
   **Then** assignments, seed, and allowlist are identical; only features differ.
2. **Given** any passage, **When** markers are extracted, **Then** the vector is
   finite, deterministic, and blank inputs are rejected.
3. **Given** completed candidates, **When** ranked, **Then** mean and per-fold
   balanced accuracy are shown with no hidden-test claim.

### Edge Cases

- A passage has no markers at all; all marker rates are zero, never NaN.
- Marker phrases overlap (e.g. "as well as"); counting is deterministic by
  fixed list order.
- Very short texts; all ratios guard against zero denominators.

## Requirements

### Functional Requirements

- **FR-013-01**: All candidates MUST use the same saved prompt-group folds,
  train checksum, seed, and official allowlist; only the feature inventory may
  differ.
- **FR-013-02**: Marker rates MUST cover discourse connectives, hedging,
  self-reference, markdown structure, and closings as fixed documented lists;
  extraction MUST be deterministic and finite.
- **FR-013-03**: Style features MUST be scaled inside the per-fold pipeline on
  fold-train rows only.
- **FR-013-04**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall in the same schema as flat candidates.
- **FR-013-05**: Tuning MUST NOT open `test.csv` or test labels; failures are
  explicit and never silently ranked.

### Key Entities

- **Marker rate**: Occurrences of a fixed phrase/class per 1,000 words (or per
  line for structural marks).
- **Experiment report**: Ranked flat/style/fusion comparison with per-fold
  spread and provenance.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every candidate has five fold results on the identical split.
- **SC-002**: Extraction is deterministic and finite on fixtures and real data.
- **SC-003**: Ranking is deterministic and preserves per-fold spread.

## Assumptions

- English-majority texts; marker lists are fixed linguistic resources.
- Same char 2–5 settings and LinearSVC as the fusion baseline.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature extends the style inventory only. It does not change the
production baseline, refit the final model, or touch test data.
