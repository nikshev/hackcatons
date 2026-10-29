# Feature Specification: Frozen Embeddings with Logistic Regression

**Feature Branch**: `011-frozen-embeddings-logreg`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Evaluate frozen MiniLM sentence embeddings + logistic regression on the same saved prompt-group folds vs the flat char TF-IDF baseline; vendor model weights locally (Git-ignored) and design for offline Kaggle packaging without a sentence-transformers dependency. No test data.

## User Scenarios & Testing

### User Story 1 - Test dense neural features (Priority: P1)

As the project developer, I need an OOF comparison of frozen MiniLM embeddings
+ logreg against flat char TF-IDF on identical folds, because sparse n-grams
plateau near 0.30 and the confusion analysis points at semantic similarity
within provider families.

**Independent Test**: Run flat vs embedding candidates on fixtures with a
stubbed encoder, then on real data; verify identical fold coverage,
deterministic ranking, and per-fold spread.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** both run,
   **Then** assignments, seed, and allowlist are identical; only features differ.
2. **Given** a validation fold, **When** encoding or fitting happens, **Then**
   no validation text leaks into any fit (encoder is frozen, classifier fits
   fold-train only).
3. **Given** completed candidates, **When** ranked, **Then** mean and per-fold
   balanced accuracy are shown with no hidden-test claim.

### Edge Cases

- The vendored model directory is missing; fail explicitly before encoding.
- An embedding candidate fails on a fold; report it, never silently rank it.
- Encoder output dimension mismatches the classifier input; fail explicitly.

## Requirements

### Functional Requirements

- **FR-011-01**: Both candidates MUST use the same saved prompt-group folds,
  train checksum, seed, and official allowlist; only the feature space may
  differ.
- **FR-011-02**: Embeddings MUST come from the vendored local MiniLM directory
  (Git-ignored); encoding MUST NOT touch the network.
- **FR-011-03**: The embedding encoder MUST be frozen; the classifier MUST fit
  only on non-held-out fold rows.
- **FR-011-04**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall in the same schema as flat candidates.
- **FR-011-05**: Tuning MUST NOT open `test.csv` or test labels; failures are
  explicit and never silently ranked.
- **FR-011-06**: A Kaggle packaging note MUST describe offline inference with
  torch+transformers mean pooling (no sentence-transformers install) and a
  weights dataset attached to the Notebook.

### Key Entities

- **Frozen embedding**: L2-normalized MiniLM vector per passage, encoder never
  fine-tuned.
- **Experiment report**: Ranked flat-vs-embedding comparison with per-fold
  spread and provenance, including the vendored model identity.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every candidate has five fold results on the identical split.
- **SC-002**: Encoder weights are fixed for the whole run; classifier sees no
  validation rows.
- **SC-003**: Ranking is deterministic and preserves per-fold spread.

## Assumptions

- Vendored model: `all-MiniLM-L6-v2` under Git-ignored `models/`.
- Classifier: multinomial logistic regression on normalized embeddings.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature runs the local embedding experiment and writes the Kaggle
packaging note. It does not upload datasets to Kaggle, change the production
baseline, or touch test labels. Promotion is a separate explicit step.
