# Grouped Baseline CLI Contract

## Command

```text
python -m ntllm train --config configs/baseline.yaml
```

## Inputs

- Current config and `training_readiness: ready` from the data audit.
- The audited train CSV and its saved prompt-group assignment/report.
- No test file, network, external corpus, or paid service.

## Outputs

- `reports/runs/<run-id>/predictions_oof.parquet` with one row per train ID and
  candidate model.
- `metrics.json` with OOF metrics and per-label diagnostics.
- `manifest.json` with data/config/split hashes, code revision, model parameters,
  source-code checksum, library versions, seed, and runtime.
- No raw passage text.

## Exit status

- `0`: all candidates generated complete OOF predictions and artifacts.
- non-zero: readiness blocked, missing/stale split, invalid labels/IDs, fit or
  prediction failure, or incomplete OOF coverage. No completed run directory is
  published on failure.

## Invariants

- A fold's OOF model is fit only on rows assigned to other folds.
- Every train ID receives exactly one OOF prediction for each candidate.
- All predicted classes belong to the current official allowlist.
- Same data/config/split/model settings produce the same OOF predictions and
  metrics; runtime/timestamp metadata may differ.
