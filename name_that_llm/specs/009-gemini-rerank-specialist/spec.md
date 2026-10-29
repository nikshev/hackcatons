# Feature Specification: Gemini-Flash Trio Rerank Specialist

**Feature Branch**: `009-gemini-rerank-specialist`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Rerank flat char 2–5 predictions inside the confused Gemini flash trio (3.6, 3.7, 3.8) with a per-fold specialist trained only on trio rows; compare end-to-end balanced accuracy against flat on identical saved folds. No test data.

## User Scenarios & Testing

### User Story 1 - Disambiguate the worst confusion cluster (Priority: P1)

As the project developer, I need a second-stage classifier that re-scores only
rows the flat model already assigns to the Gemini flash trio, because error
analysis shows that trio holds the largest mutual-confusion counts.

**Why this priority**: It is the single biggest error cluster; a targeted fix
is cheaper and safer than changing the whole architecture.

**Independent Test**: On a fixture with Gemini-trio rows plus other labels,
verify rerank touches only trio-predicted rows, every ID gets exactly one
prediction, and no fold leaks into either fit.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** flat and rerank
   run, **Then** both use identical assignments, seed, and allowlist.
2. **Given** a validation fold, **When** either stage fits, **Then** it sees
   only non-held-out rows.
3. **Given** completed candidates, **When** ranked, **Then** mean and per-fold
   balanced accuracy are shown with no hidden-test claim.

### Edge Cases

- The flat model predicts no trio rows in a fold; the specialist is still fit
  but unused for that fold.
- A trio class is absent from a fold-train trio subset; the specialist trains
  on the remaining trio classes.
- The specialist predicts outside the trio; reject explicitly.

## Requirements

### Functional Requirements

- **FR-009-01**: Both candidates MUST use the same saved prompt-group folds,
  train checksum, seed, and official allowlist; only the trio rerank may
  differ.
- **FR-009-02**: The trio is exactly `gemini-3.6-flash`, `gemini-3.7-flash`,
  `gemini-3.8-flash`; rerank MUST apply only to rows the flat stage assigns to
  the trio and MUST output a trio label.
- **FR-009-03**: Both stages MUST fit only on non-held-out rows; trio
  specialists fit only on trio rows of the fold-train set.
- **FR-009-04**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall in the same schema as flat candidates.
- **FR-009-05**: Tuning MUST NOT open `test.csv` or test labels; failures are
  explicit and never silently ranked.

### Key Entities

- **Trio prediction**: A flat-stage prediction inside the three confused labels.
- **Trio specialist**: Per-fold classifier trained only on trio fold-train rows.
- **Experiment report**: Ranked flat-vs-rerank comparison with per-fold spread.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every candidate has five fold results on the identical split.
- **SC-002**: Every training ID has exactly one final prediction; non-trio
  flat predictions pass through unchanged.
- **SC-003**: Ranking is deterministic and preserves per-fold spread.

## Assumptions

- Same char 2–5 TF-IDF features and LinearSVC settings in both stages.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature tests the trio rerank only. It does not change the production
baseline, refit the final model, or touch test data. Promotion, if any, is a
separate explicit step.
