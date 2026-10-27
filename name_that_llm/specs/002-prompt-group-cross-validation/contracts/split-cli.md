# Prompt-Group Split CLI Contract

## Command

```text
python -m ntllm split --config configs/baseline.yaml
```

## Inputs

- Versioned YAML configuration with the training CSV, field roles, official
  label allowlist, fold count, seed, and output paths.
- The audited training CSV only. The test CSV is not opened.
- No network access.

## Outputs

- Parquet assignments containing `id`, `prompt_id`, and `fold`.
- JSON report with dataset/config hashes, seed, fold count/hash, row counts,
  prompt-group counts, and per-fold label counts.
- No raw text or complete source rows.

## Exit status

- `0`: all input and split invariants passed and both output artifacts were
  written.
- non-zero: missing/invalid input, invalid fold settings, duplicate/missing IDs,
  missing groups/labels, or artifact-write failure. Invalid input creates no
  assignment artifacts.

## Invariants

- Each training ID occurs exactly once in the assignments.
- A prompt group occurs in exactly one fold.
- The same data/config/seed produces the same canonical split hash.
- Raw CSV bytes remain unchanged.
