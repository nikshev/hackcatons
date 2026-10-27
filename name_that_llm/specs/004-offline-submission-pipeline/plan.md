# Implementation Plan: Offline Final Fit and Submission Pipeline

**Branch**: `004-offline-submission-pipeline` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/004-offline-submission-pipeline/spec.md`

## Summary

Fit the chosen `char_tfidf_svc` candidate on all audited training rows, store
the fitted vectorizer/classifier as one offline-loadable artifact, predict the
actual test CSV, and validate the result against Kaggle's exact submission
contract. The model choice is recorded in versioned configuration; competition
eligibility and payout remain separate from local inference readiness.

## Technical Context

**Language/Version**: Python 3.12 or newer
**Primary Dependencies**: scikit-learn, joblib, PyArrow, PyYAML
**Storage**: Joblib model bundle; Parquet predictions/submission; JSON manifest
**Testing**: pytest with synthetic fixtures and the local three-row Kaggle sample
**Target Platform**: Local Python environment; later offline Kaggle Notebook
**Project Type**: Python package and CLI
**Performance Goals**: Local fit/predict and 9-hour Kaggle Notebook compatibility
**Constraints**: No network calls; fit only on train; inference never fits/downloads; exact 18-label allowlist; raw CSV remains unchanged
**Scale/Scope**: 2,699 training rows and an interactive three-row test sample

## Constitution Check

- **Readiness**: Final fit requires a current passing audit with
  `training_readiness: ready`; entry eligibility is not a local model gate.
- **Offline**: Inference loads the complete vectorizer/classifier bundle and
  uses only provided test text; no network access or fit is performed.
- **Submission contract**: Exactly one row per test ID, columns `id` and
  `model`, values in the official allowlist.
- **Traceability**: Tests cover bundle provenance, deterministic prediction,
  exact ID coverage, and every validator rejection case.

No constitution violations identified.

## Design

1. Read and validate the current baseline config, audit report, train checksum,
   and readiness state.
2. Select the model named by `inference.model_candidate`; build it from the
   corresponding training configuration and fit on all train rows.
3. Save model and manifest into a temporary directory; publish the complete
   artifact directory only after serialization succeeds.
4. At predict time, load only the model bundle and input `id`/`text` columns.
   Preserve input order and reject invalid/unknown predictions.
5. At validation time, compare the generated file's exact schema and ID set to
   the test CSV and verify nulls/labels.

## Project Structure

```text
configs/baseline.yaml
src/ntllm/inference.py                   # final fit, load, predict, validate
src/ntllm/__main__.py                    # fit-final/predict/validate-submission
tests/test_inference.py                  # artifacts, offline predict, contract
artifacts/final_model/                   # generated, Git-ignored
reports/submissions/                     # generated, Git-ignored
```

**Structure Decision**: Keep final-fit, prediction, and validation functions in
one module so the artifact interface is shared by local CLI and a later offline
Notebook wrapper.

## Risks and Open Questions

- The local test CSV is only a three-row interactive sample; successful
  validation proves schema/IDs on that sample, not hidden-set performance.
- Kaggle's exact library versions and offline artifact attachment format still
  need a Notebook smoke test.
- The final model is selected from local OOF results; no leaderboard claims are
  made.

## Complexity Tracking

No constitution violations to justify.
