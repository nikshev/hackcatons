# Offline Inference and Submission Contract

## Commands

```text
python -m ntllm fit-final --config configs/baseline.yaml
python -m ntllm predict --model artifacts/final_model --input kaggle/test.csv --output reports/submissions/submission.parquet
python -m ntllm validate-submission --input reports/submissions/submission.parquet --test kaggle/test.csv --config configs/baseline.yaml
```

## Invariants

- `fit-final` requires current `training_readiness: ready` and writes a complete
  model bundle only after fit/serialization succeeds.
- `predict` reads test IDs/text only; it does not fit, access labels, or use the
  network.
- Prediction output has exactly columns `id` and `model`, one row per input ID
  in the same order, no nulls, and labels from the config allowlist.
- `validate-submission` checks exact schema, nulls, duplicate IDs, missing/extra
  IDs, and official label membership.
- A failed command does not leave a partial new model/submission artifact.
