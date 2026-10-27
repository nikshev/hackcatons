# Kaggle Notebook Contract

## Inputs

- `/kaggle/input/competitions/name-that-llm/train.csv`
- `/kaggle/input/competitions/name-that-llm/test.csv`
- Preinstalled Kaggle Python/scientific stack; no network or package install.
- Local smoke overrides: `NTLLM_DATA_DIR` and `NTLLM_OUTPUT_DIR`.

## Output

- Kaggle: `/kaggle/working/submission.parquet`
- Exactly two columns in order: `id`, `model`.
- One non-null prediction per test ID, in test input order.
- Each `model` value is in the official allowlist from the embedded config.

## Execution guarantees

- The selected final model is fit on train only; test labels are never read.
- The Notebook validates duplicates, nulls, exact ID coverage, and allowed
  classes before writing the final output.
- No network/API/package-download call is allowed.
- The Notebook reports input checksums and runtime without printing passage text.
