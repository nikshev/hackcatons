# OOF Analysis and Tuning CLI Contract

## Commands

```text
python -m ntllm evaluate --run-id RUN_ID --config configs/baseline.yaml
python -m ntllm tune --config configs/baseline.yaml --experiment configs/experiments/char_tfidf_grid.yaml
```

## Inputs

- An existing OOF run directory with manifest, metrics, and prediction Parquet.
- The audited train CSV for group/text-length aggregation and checksum match.
- A versioned tuning experiment YAML for `tune`.
- No `test.csv`, test labels, network, or external data.

## Outputs

- `error_analysis.md` inside the analyzed run directory.
- A ranked tuning summary with per-fold metrics under `reports/tuning/`.
- Aggregate class/group/length statistics only; no passage text, IDs, or prompt
  IDs.

## Failure conditions

- Train checksum, OOF ID coverage, or prompt-group exclusivity mismatch.
- Candidate settings invalid or model fails on a fold.
- OOF predictions contain labels outside the configured allowlist.
- Input audit/readiness is blocked.

Failures return a non-zero status and do not emit a success report.
