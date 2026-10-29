# Feature Specification: Hierarchical Provider-Then-Model Classification

**Feature Branch**: `008-hierarchical-provider-model`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Evaluate a two-stage provider-then-model classifier on the same saved prompt-group folds; compare end-to-end balanced accuracy against the flat char baseline with per-fold spread. No test data.

## User Scenarios & Testing

### User Story 1 - Test provider routing (Priority: P1)

As the project developer, I need a two-stage OOF evaluation where a provider
classifier routes each row to a within-provider specialist, using identical
features and the same saved folds as the flat baseline, so the comparison
measures architecture only.

**Why this priority**: Error analysis shows the heaviest confusion inside
provider families (Gemini flash variants, GPT variants). Routing first may
isolate those decisions.

**Independent Test**: Build a small two-provider fixture and verify one
prediction per ID, no fold leakage in either stage, deterministic ranking, and
per-fold spread for both architectures.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** flat and
   hierarchical candidates run, **Then** both use identical assignments, seed,
   and allowlist.
2. **Given** a validation fold, **When** either stage fits, **Then** it sees
   only non-held-out rows.
3. **Given** completed candidates, **When** ranked, **Then** mean and per-fold
   balanced accuracy are shown with no hidden-test claim.

### Edge Cases

- A predicted provider has no specialist for the fold; fall back
  deterministically to that provider's most frequent fold-train label.
- A provider label prefix is unrecognized; fail explicitly before any fit.
- A single-class provider needs no classifier; its specialist is constant.

## Requirements

### Functional Requirements

- **FR-008-01**: Both architectures MUST use the same saved prompt-group folds,
  train checksum, seed, and official allowlist; only the decision structure
  may differ.
- **FR-008-02**: Provider labels MUST derive deterministically from official
  model names; unknown prefixes MUST fail before fitting.
- **FR-008-03**: Both stages MUST fit only on non-held-out rows; validation
  rows and labels MUST NOT participate in either fit.
- **FR-008-04**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall for the hierarchical candidate in the same
  format as flat candidates.
- **FR-008-05**: Tuning MUST NOT open `test.csv` or test labels; failures are
  explicit and never silently ranked.

### Key Entities

- **Provider**: One of six labs derived from the model name prefix.
- **Specialist**: Within-provider classifier fit on that provider's fold-train
  rows only.
- **Experiment report**: Ranked flat-vs-hierarchical comparison with per-fold
  spread and provenance.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every candidate has five fold results on the identical split.
- **SC-002**: Every training ID has exactly one hierarchical OOF prediction;
  no prompt group crosses folds in either stage.
- **SC-003**: Ranking is deterministic and preserves per-fold spread.

## Assumptions

- Six providers: OpenAI (gpt), Google (gemini), Anthropic (claude), Qwen
  (qwen), ZAI (glm), Moonshot (kimi).
- Same char 2–5 TF-IDF features and LinearSVC settings in both stages.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature evaluates the hierarchical architecture only. It does not change
the production baseline, refit the final model, or touch test data. Promotion,
if any, is a separate explicit step.
