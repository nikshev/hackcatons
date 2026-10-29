# Feature Specification: LSA Dense Projection over Char TF-IDF

**Feature Branch**: `010-lsa-dense-projection`
**Created**: 2026-10-27
**Status**: Draft
**Input**: Insert a TruncatedSVD dense projection between char 2–5 TF-IDF and LinearSVC on identical saved folds; compare against the sparse flat baseline with per-fold spread. No new dependencies, no test data.

## User Scenarios & Testing

### User Story 1 - Test whether dense features generalize better (Priority: P1)

As the project developer, I need a sparse-vs-dense comparison on the same
features, folds, seed, and allowlist, because sparse char n-grams may memorize
prompt-specific tokens while dense components may transfer across unseen
prompts.

**Why this priority**: Three targeted tweaks failed to move the needle; the
next cheapest representation change is LSA, which needs no downloads and stays
offline-safe.

**Independent Test**: Run flat vs SVD candidates on a fixture and on the real
saved split; verify identical fold coverage, deterministic ranking, and
per-fold spread for both.

**Acceptance Scenarios**:

1. **Given** the same saved folds and train checksum, **When** flat and SVD
   run, **Then** both use identical assignments, seed, and allowlist.
2. **Given** a validation fold, **When** the pipeline fits, **Then** TF-IDF and
   SVD fit only on non-held-out rows.
3. **Given** completed candidates, **When** ranked, **Then** mean and per-fold
   balanced accuracy are shown with no hidden-test claim.

### Edge Cases

- Requested components exceed what the fold-train matrix supports; fail
  explicitly before fitting.
- SVD candidate on a dummy model type; reject as unsupported.
- A candidate fails on a fold; report it as failed, never silently rank it.

## Requirements

### Functional Requirements

- **FR-010-01**: Both candidates MUST use the same saved prompt-group folds,
  train checksum, seed, and official allowlist; only the SVD layer may differ.
- **FR-010-02**: TF-IDF configs MAY specify `svd_components`; the projection
  MUST be fit inside the per-fold pipeline on fold-train rows only, with a
  fixed random state.
- **FR-010-03**: Invalid component counts (non-positive, or exceeding
  fold-train limits) MUST fail explicitly before fitting.
- **FR-010-04**: Tuning MUST report mean and per-fold balanced accuracy, macro
  F1, runtime, and per-class recall in the same schema as flat candidates.
- **FR-010-05**: Tuning MUST NOT open `test.csv` or test labels; SVD adds no
  network, download, or external-data dependency.
- **FR-010-06**: `svd_components` on a dummy candidate MUST be rejected; dummy
  stays a pure prior baseline.

### Key Entities

- **SVD layer**: TruncatedSVD projection between vectorizer and classifier,
  refit per fold.
- **Experiment report**: Ranked flat-vs-SVD comparison with per-fold spread.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every candidate has five fold results on the identical split.
- **SC-002**: SVD never sees validation rows; same-row determinism holds.
- **SC-003**: Ranking is deterministic and preserves per-fold spread.

## Assumptions

- Same char 2–5 settings as the production baseline.
- SVD dimensions tested: 512 first; other widths only as a follow-up if 512
  shows a consistent signal.
- Local OOF only; hidden-test conclusions are out of scope.

## Scope Boundaries

This feature tests the SVD layer only. It does not change the production
baseline, refit the final model, add neural dependencies, or touch test data.
Neural embeddings are a separate decision gated on this result.
