# Data Model: Prompt-Group Cross-Validation

## FoldAssignment

- `id`: one unique training-row identifier.
- `prompt_id`: the prompt group for the row.
- `fold`: integer in `[0, n_splits)`.

Every training `id` appears once, and all rows with the same `prompt_id` have
the same `fold`.

## SplitReport

- `configuration`: configuration path, version, and SHA-256.
- `dataset_sha256`: checksum of the raw training CSV bytes.
- `seed`: explicit deterministic assignment seed.
- `n_splits`: requested fold count.
- `split_sha256`: hash of canonical `(id, prompt_id, fold)` assignments.
- `row_counts`: number of rows in each fold.
- `label_counts`: number of rows per label in each fold.
- `group_counts`: number of prompt groups in each fold.

The report contains no raw passage text.
