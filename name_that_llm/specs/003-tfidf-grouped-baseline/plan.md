# Implementation Plan: Grouped TF-IDF Baseline Benchmark

**Branch**: `003-tfidf-grouped-baseline` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-tfidf-grouped-baseline/spec.md`

## Summary

Run a no-skill baseline and three sparse text representations on the same saved
prompt-group folds. Fit every text transform within each fold's training side,
produce one out-of-fold prediction per training row/model, calculate balanced
accuracy and supporting class metrics, and save run provenance under an
ignored `reports/runs/<run-id>/` directory.

## Technical Context

**Language/Version**: Python 3.12 or newer
**Primary Dependencies**: pandas, PyArrow, scikit-learn, SciPy, PyYAML
**Storage**: Read-only CSV train input; Parquet OOF predictions; JSON metrics and manifest
**Testing**: pytest with offline synthetic fixtures
**Target Platform**: Local Python environment; Kaggle compatibility smoke test later
**Project Type**: Python package and CLI
**Performance Goals**: Complete five folds and four candidates well within Kaggle's nine-hour notebook limit
**Constraints**: No network/external data; no test-file reads; no transform fit on held-out folds; all result-changing parameters in versioned YAML
**Scale/Scope**: 2,699 training rows, 150 prompt groups, 18 official labels

## Constitution Check

- **Leakage**: OOF fold membership comes from the saved prompt-group split;
  vectorizer and classifier fit only on the other folds.
- **Test-first**: Unit/integration fixtures verify fold isolation, label/schema
  checks, OOF completeness, and metric calculations before implementation.
- **Reproducibility**: Store train/config/split hashes, code revision and dirty
  state, software versions, seed, model parameters, metrics, and runtimes.
- **Data boundary**: Read only Kaggle Competition Data; raw CSVs stay unchanged;
  no external/synthetic data or network calls.
- **Readiness**: Local training requires `training_readiness: ready`. Personal
  eligibility/payout is tracked separately and gates competition entry only.

No constitution violations identified.

## Design

1. Re-run the data audit and require `training_readiness: ready`.
2. Verify the saved split report refers to the current train checksum, config
   version, fold count, and seed; verify assignment IDs exactly match train IDs.
3. For each candidate and each fold, fit a fresh model pipeline on rows whose
   saved fold differs from the held-out fold, then predict only held-out rows.
4. Require one prediction per training ID/model and only official labels.
5. Calculate aggregate OOF balanced accuracy, macro F1, per-class recall, and
   confusion matrix.
6. Write an OOF Parquet file, metrics JSON, and manifest with hashes and run
   metadata under a deterministic run ID derived from data, config, split,
   source code, and library versions. Do not store source passage text.

Candidates are config-defined: most-frequent dummy; word TF-IDF + LinearSVC;
character TF-IDF + LinearSVC; combined word/character TF-IDF + LinearSVC.

## Project Structure

```text
configs/baseline.yaml                   # model/search parameters and run root
src/ntllm/training.py                   # estimators, grouped OOF evaluation
src/ntllm/__main__.py                   # `ntllm train` command
tests/test_training.py                  # leakage, metrics, artifacts, failures
reports/runs/<run-id>/
├── manifest.json
├── metrics.json
└── predictions_oof.parquet
```

**Structure Decision**: Keep training and OOF evaluation in one module under
`ntllm`; the CLI consumes the existing audit and split contracts, without
re-reading or modifying test data.

## Risks and Open Questions

- Character n-grams may consume substantial memory; max feature counts are
  versioned config values and will be measured on the local 2,699-row set.
- scikit-learn is declared by the project but is not installed in the system
  Python; install the project dependencies in the ignored local virtualenv for
  full-model verification.
- Notebook package versions and runtime still require a Kaggle smoke test.

## Complexity Tracking

No constitution violations to justify.
