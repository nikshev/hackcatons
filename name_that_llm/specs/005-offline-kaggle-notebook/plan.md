# Implementation Plan: Offline Kaggle Submission Notebook

**Branch**: `005-offline-kaggle-notebook` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/005-offline-kaggle-notebook/spec.md`

## Summary

Generate a self-contained Kaggle Notebook from the versioned baseline config.
Inside Kaggle it reads mounted train/test CSVs, fits the selected character
TF-IDF model on train, predicts test, validates the official submission
contract, and writes `submission.parquet`. Local path overrides let tests execute
the same generated code against fixtures and the supplied three-row sample.

## Technical Context

**Language/Version**: Python 3.12 or newer
**Primary Dependencies**: Python standard library, pandas, scikit-learn, PyArrow, PyYAML
**Storage**: Generated `.ipynb`; `submission.parquet`; compact JSON run summary
**Testing**: pytest; execute the generated notebook code cell locally, offline
**Target Platform**: Kaggle Notebook with CPU/GPU <= 9 hours and internet disabled
**Project Type**: Notebook generator and self-contained Kaggle Notebook
**Performance Goals**: Measure fit + predict time and keep it below the official runtime
**Constraints**: No package installs, external data, network calls, or test labels; exact 18-label output
**Scale/Scope**: 2,699 train rows, 150 prompt groups, three local interactive test rows, about 4,000 hidden rows

## Constitution Check

- **Source of truth**: Kaggle official rules/data and local audit/schema govern;
  notebook constants are rendered from the versioned YAML config.
- **Offline**: Notebook has no network, package install, API, or external model
  download path.
- **Leakage**: Final text model fits only on train; test is only read for
  prediction and contract checks.
- **Reproducibility**: Config hash/version, dataset checksums, package versions,
  and runtime are printed in a compact summary.
- **Privacy**: Raw text is never printed or copied to submission/metadata.

No constitution violations identified.

## Design

1. A local generator reads `configs/baseline.yaml` and emits a clean notebook
   with embedded config version/hash, official labels, and the selected
   `char_tfidf_svc` parameters.
2. The notebook selects Kaggle-mounted data paths by default; environment/path
   overrides support local smoke execution.
3. Validate train/test column contracts, row integrity, and label allowlist.
4. Fit char TF-IDF + LinearSVC on all train rows and predict test rows.
5. Validate output IDs, order, nulls, labels, and exact Parquet columns before
   writing to the notebook working directory.
6. Print checksums, class counts, package versions, and elapsed times without
   exposing text.

The notebook trains from official Competition Data rather than requiring an
attached serialized model, avoiding scikit-learn pickle-version coupling.

## Project Structure

```text
scripts/build_submission_notebook.py       # render from YAML config
notebooks/submission.ipynb                 # generated, offline Kaggle entry
tests/test_submission_notebook.py          # config sync + local execution
configs/baseline.yaml                      # canonical model/label parameters
```

**Structure Decision**: Keep model hyperparameters in YAML; generate a
self-contained notebook cell from them so Kaggle does not need the repository
package or access to the internet.

## Risks and Open Questions

- Kaggle's actual notebook package versions/runtime may differ from local
  versions and still need an online-platform smoke check.
- The local test file is only a three-row sample; it validates the contract,
  not hidden inference throughput or score.
- The local output must remain a preview until a Kaggle Notebook run succeeds.

## Complexity Tracking

No constitution violations to justify.
