# Feature Specification: Competition Readiness and Data Audit

**Feature Branch**: `001-competition-readiness-audit`
**Created**: 2026-10-26
**Status**: Draft
**Input**: User description: "Establish a verified competition and data-readiness gate for Name That LLM before any supervised modeling: document current official rules, eligibility, data schema, labels, local data availability, and unresolved external-data constraints."

## User Scenarios & Testing

### User Story 1 - Verify competition readiness (Priority: P1)

As the project developer, I need a dated record of the current competition
requirements and unresolved questions so I can decide which work and data are
permitted before spending time on modeling.

**Why this priority**: Stale rules, unknown eligibility, or unapproved external
data could invalidate the solution or prize eligibility.

**Independent Test**: Review the competition record and confirm every material
claim has a source and check date, while every unchecked condition is clearly
marked unverified or blocked.

**Acceptance Scenarios**:

1. **Given** an official requirement can be checked, **When** the competition
   record is updated, **Then** it includes the exact source and the date checked.
2. **Given** a requirement cannot be confirmed, **When** the record is reviewed,
   **Then** it remains explicitly unresolved and is not presented as a fact.
3. **Given** external or synthetic data rules are unclear, **When** a readiness
   decision is made, **Then** use of that data and associated spending remain
   blocked.

---

### User Story 2 - Audit available competition data (Priority: P2)

As the project developer, I need to inspect downloaded competition files without
changing them so I can detect missing, malformed, or mismatched data before
training.

**Why this priority**: A repeatable data audit is the entry point for the ML
pipeline and prevents silent assumptions about columns, labels, or data quality.

**Independent Test**: Run the audit against small representative train/test
fixtures, then verify the reported schema and counts, an unchanged source-file
checksum, and a clear failure for an unknown schema.

**Acceptance Scenarios**:

1. **Given** valid train and test files, **When** the developer audits them,
   **Then** the report identifies their schemas, row counts, missing values,
   text-length summaries, duplicate summaries, and available label counts.
2. **Given** a candidate data file has an unknown schema or unexpected labels,
   **When** the audit runs, **Then** it names the file and issue and does not
   silently guess the target column or accepted classes.
3. **Given** a file is missing or unreadable, **When** the audit runs, **Then**
   it reports the blocker without altering any raw input.

### Edge Cases

- Kaggle's current pages or rules are unavailable, changed, or disagree with the
  previously recorded PRD.
- The local `test.csv` is only the interactive sample and differs in size from
  the hidden evaluation rerun.
- The train file has missing labels, unexpected class names, or no prompt-group
  identifier.
- A file is empty, malformed, has duplicate IDs, or contains null/blank text.
- The same source file is audited more than once or is changed after its
  manifest was recorded.
- Rules for external/synthetic data or participant eligibility remain
  unresolved.

## Requirements

### Functional Requirements

- **FR-001-01**: The competition record MUST distinguish confirmed, unverified,
  changed, and blocked requirements; each confirmed external fact MUST include a
  source URL and date checked.
- **FR-001-02**: The record MUST cover eligibility, current deadlines, prizes,
  submission requirements, notebook runtime/network constraints, data usage
  terms, and the exact allowed model labels. Unknown items MUST remain
  explicitly unverified.
- **FR-001-03**: The data audit MUST inspect the actual input files and report
  their observed columns; it MUST NOT assume a schema or infer an unknown target
  column silently.
- **FR-001-04**: For every audited raw file, the manifest MUST record source,
  retrieval date, applicable license/terms, checksum, row count, and observed
  columns. Raw input files MUST remain unchanged.
- **FR-001-05**: The audit MUST report missing values, text-length summaries,
  exact-duplicate counts, and label counts when a labeled file is available.
- **FR-001-06**: The audit MUST report prompt-group availability and identify
  duplicate IDs or invalid/missing text; detected issues MUST be visible rather
  than silently dropped.
- **FR-001-07**: The project MUST mark supervised baseline training as blocked
  until permitted labeled training data, its schema, and the exact allowed class
  list are confirmed.
- **FR-001-08**: An unknown schema, unreadable file, or unrecognized class MUST
  produce a clear, actionable failure and a non-success audit status.

### Key Entities

- **Competition requirement**: A rule, date, runtime, eligibility condition, or
  submission constraint, with source, check date, and verification status.
- **Dataset file**: A raw competition input, identified by source, checksum,
  schema, and row count.
- **Audit result**: A repeatable profile of available data, including quality
  issues and whether supervised training is unblocked.
- **Model label**: One exact allowed competition class associated with a
  labeled passage.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every competition fact used to make a project decision is either
  linked to a dated source or explicitly marked unverified.
- **SC-002**: For every available input file, an audit result records its
  checksum, row count, observed columns, and applicable quality summaries.
- **SC-003**: Auditing a valid fixture leaves its raw-file checksum unchanged.
- **SC-004**: Unknown schemas, invalid files, and unrecognized labels are
  reported as failures with the file and offending field/value identified.
- **SC-005**: No supervised baseline is reported as ready while any data,
  allowlist, or competition-rule blocker in FR-001-07 remains unresolved.

## Assumptions

- The competition publishes CSV files for training and interactive testing;
  actual downloaded files are authoritative over this assumption.
- `model` is the likely target column and `prompt_id` is the likely grouping
  field based on current official data-page excerpts; both must be checked
  against the local files.
- The user's competition eligibility and any payout eligibility require direct
  confirmation; the project cannot infer them from public dataset metadata.
- The current local solution folder contains no competition data; fixture-based
  checks can proceed until the user obtains permitted data.

## Scope Boundaries

This feature covers competition readiness documentation and raw-data audit. It
does not train or evaluate a classifier, create a leakage-aware validation
split, build the Kaggle submission notebook, or submit predictions. Those are
subsequent features and depend on the readiness gates above.
