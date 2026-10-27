# Implementation Plan: Prompt-Group Cross-Validation

**Branch**: `002-prompt-group-cross-validation` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-prompt-group-cross-validation/spec.md`

## Summary

Create deterministic grouped folds from the audited training CSV. Prompt groups
are indivisible, assignments balance row and label counts as far as group
boundaries permit, and the command writes a Parquet assignment file plus JSON
reproducibility report. Invalid or missing grouping data blocks generation;
there is no row-wise fallback.

## Technical Context

**Language/Version**: Python 3.12 or newer
**Primary Dependencies**: Python standard library, PyYAML, PyArrow
**Storage**: Read-only CSV input; Parquet assignments; JSON split report
**Testing**: pytest with offline synthetic fixtures
**Target Platform**: Local Python environment and later Kaggle Notebook
**Project Type**: Python package and CLI
**Performance Goals**: Create folds for the official training set in seconds
**Constraints**: No network access; no input mutation; deterministic seed; grouping is mandatory; use training rows only
**Scale/Scope**: Current official train has 2,699 rows, about 150 prompts, and 18 labels

## Constitution Check

- **Traceability**: Each split invariant maps to FR-002 requirements, tasks,
  implementation markers, and tests.
- **Test-first**: Split invariants and invalid-input behavior are tested before
  implementation.
- **Reproducibility**: Dataset/config checksums, seed, fold counts, and a
  canonical split hash are stored with assignments.
- **Leakage**: No prompt group may cross folds; test rows/labels are not read.
- **Stop gates**: Missing prompt IDs, duplicate IDs, invalid labels, and invalid
  fold settings fail before any output artifact is created.

No constitution violations identified.

## Design

1. Load the training-file path, configured `id`/`prompt_id`/`label` roles,
   allowlist, fold count, and seed from `configs/baseline.yaml`.
2. Validate unique nonblank IDs, prompt groups, labels, and fold settings.
3. Aggregate rows and class counts by prompt group.
4. Order groups deterministically from the configured seed; seed one group per
   fold, then greedily assign remaining groups to reduce fold row/class-count
   imbalance, with stable tie-breaking.
5. Canonicalize assignments by ID and calculate a stable SHA-256 split hash.
6. Write Parquet assignments and a JSON report only after all data validation
   and fold assignment has succeeded.

The split algorithm is deterministic, uses no model or learned transform, and
does not inspect the configured test file.

## Project Structure

```text
configs/baseline.yaml                   # versioned seed/fold/output settings
src/ntllm/splits.py                     # grouped fold assignment and report
src/ntllm/__main__.py                   # `ntllm split` command
tests/test_splits.py                    # group, stability, balance, errors
specs/002-prompt-group-cross-validation/
├── data-model.md
└── contracts/split-cli.md
reports/splits/                         # generated Parquet and JSON
```

**Structure Decision**: Add a split-focused module and a CLI command under the
existing `ntllm` package. Keep generated assignments out of source code and
preserve the existing audit command.

## Risks and Open Questions

- Prompt-level groups are indivisible; exact row/class balance may be
  impossible for groups with substantially different sizes or labels.
- The official training file is not present locally. Fixture tests establish
  invariants; a real split report requires the downloaded, audited file.
- The current dataset exposes `prompt_id`; a future schema without a reliable
  group field must remain blocked until a separately validated fallback exists.

## Complexity Tracking

No constitution violations to justify.
