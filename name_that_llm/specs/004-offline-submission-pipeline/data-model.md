# Data Model: Offline Final Fit and Submission Pipeline

## FinalModelManifest

- `model_id`: deterministic ID derived from config, train checksum, and chosen
  candidate.
- `candidate`: configured OOF-validated model name.
- `train_sha256`, `config_sha256`, `config_version`: fit provenance.
- `allowed_labels`: exact current Kaggle prediction classes.
- `id_column`, `text_column`, `label_column`: observed field roles.
- `source_sha256`, `git_commit`, `working_tree_dirty`: code provenance.
- `python_version`, `library_versions`, `fit_at`: execution environment.

The artifact directory contains the manifest and one fitted preprocessing/model
pipeline. It contains no raw training rows.

## PredictionRow

- `id`: unique test passage identifier.
- `model`: predicted official model label.

Rows retain the input test order and contain no text.

## ValidationResult

- `status`: `passed` or `failed`.
- `test_rows`, `submission_rows`: compared row totals.
- `missing_ids`, `extra_ids`, `duplicate_ids`, `null_values`,
  `invalid_labels`: counts and bounded diagnostics.

The validator does not modify the submission file.
