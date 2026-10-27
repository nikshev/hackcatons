# Feature Specification: Prompt-Group Cross-Validation

**Feature Branch**: `002-prompt-group-cross-validation`
**Created**: 2026-10-26
**Status**: Draft
**Input**: User description: "Create deterministic leakage-aware cross-validation folds for Name That LLM by assigning all examples with the same prompt_id to one fold, balancing fold sizes and class counts, and saving assignments plus reproducibility metadata. Reject missing IDs, labels, prompt IDs, duplicate IDs, or invalid fold settings; never fall back to a row-wise split."

## User Scenarios & Testing

### User Story 1 - Validate on unseen prompts (Priority: P1)

As the project developer, I need a repeatable validation split that keeps all
answers to the same prompt together so that validation measures generalization
to unseen prompts rather than memorization of prompt-specific content.

**Why this priority**: The competition's hidden test prompts do not occur in
training. A row-wise split can put sibling answers to the same prompt in both
training and validation and misstate model quality.

**Independent Test**: Create folds from a fixture with multiple prompt groups
and labels, then verify each row is assigned once, no prompt crosses folds, and
the same input/configuration/seed produces the same split hash.

**Acceptance Scenarios**:

1. **Given** a valid labeled file with enough prompt groups, **When** the
   developer creates folds, **Then** every row in one prompt group receives the
   same fold assignment.
2. **Given** the same records, fold count, and seed, **When** assignments are
   created repeatedly or input rows are reordered, **Then** the id-to-fold map
   and split hash remain identical.
3. **Given** a valid split, **When** the developer reviews its report, **Then**
   each fold's row count and class distribution are available for comparison.

### User Story 2 - Refuse unsafe split inputs (Priority: P1)

As the project developer, I need invalid group or row identifiers to stop split
creation with a clear error so that a misleading validation result is never
produced silently.

**Why this priority**: A split with missing groups, repeated IDs, or too few
groups cannot support a defensible leakage check.

**Independent Test**: Supply one fixture for each invalid condition and verify
that split creation fails without writing a partial assignment artifact.

**Acceptance Scenarios**:

1. **Given** a missing or blank prompt ID, **When** a split is requested,
   **Then** the command reports the affected field and produces no split.
2. **Given** repeated or missing row IDs, **When** a split is requested,
   **Then** the command reports the problem and produces no split.
3. **Given** invalid fold settings or fewer groups than folds, **When** a split
   is requested, **Then** it fails with a specific actionable message rather
   than reverting to a row-wise split.

### Edge Cases

- One prompt group has fewer labeled responses than other groups.
- A class is absent from a fold because that class is absent from its prompt
  groups.
- The number of prompt groups is exactly the requested fold count.
- The same prompt ID is represented by non-adjacent rows.
- Prompt IDs, row IDs, or labels contain blank/whitespace values.
- Two distinct input orders describe the same set of rows.
- The configured group field is absent; split creation must stop unless another
  explicit, valid group field is configured.

## Requirements

### Functional Requirements

- **FR-002-01**: The split MUST assign every row to exactly one fold and MUST
  keep all rows with the same configured prompt-group value in that same fold.
- **FR-002-02**: The split MUST reject missing/blank row IDs, duplicate row IDs,
  missing/blank labels, and missing/blank group IDs with an actionable error.
- **FR-002-03**: The split MUST reject fewer than two folds, fewer prompt groups
  than folds, and other invalid fold settings; it MUST NOT fall back to a
  row-wise split.
- **FR-002-04**: Given the same dataset, configuration, and seed, the id-to-fold
  assignments and split hash MUST be stable, including when input rows are
  reordered.
- **FR-002-05**: Fold assignment MUST balance total row counts and per-class
  counts as evenly as the indivisible prompt groups allow.
- **FR-002-06**: The split report MUST include the seed, fold count, source-data
  checksum, configuration version, split hash, row count per fold, and class
  counts per fold.
- **FR-002-07**: Split creation MUST use labeled training rows only and MUST NOT
  fit preprocessing, inspect test labels, or modify raw input files.
- **FR-002-08**: Assignment artifacts MUST be written only after all input
  validation succeeds; a failed split MUST leave no partial output.

### Key Entities

- **Prompt group**: The set of labeled responses sharing one prompt identifier.
- **Fold assignment**: A mapping from one unique row ID to exactly one fold.
- **Split report**: Reproducibility metadata and per-fold row/class counts.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Zero prompt groups occur in more than one fold.
- **SC-002**: Every valid training row appears exactly once in the assignment
  artifact; missing and duplicate row assignments are zero.
- **SC-003**: Repeated runs with the same dataset, configuration, and seed
  produce the same id-to-fold mapping and split hash.
- **SC-004**: Every fold has at least one prompt group, and the report includes
  row and per-class counts for all folds.
- **SC-005**: Every invalid-input fixture fails with a non-zero command status
  and leaves no assignment artifact.

## Assumptions

- Kaggle's current official data page provides `prompt_id` on the training data;
  the local file must still be audited before split generation.
- The number of folds and randomization seed are explicit versioned
  configuration values, not hidden defaults.
- Prompt groups are indivisible, so exact equal fold sizes may be impossible.
- When no valid group field is available, the safe outcome is blocked split
  generation; no row-wise fallback is acceptable.

## Scope Boundaries

This feature creates and validates reusable fold assignments and their report.
It does not fit models, tune hyperparameters, or compute OOF predictions; those
are later features that consume the saved assignments.
