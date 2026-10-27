# Data Model: Grouped TF-IDF Baseline Benchmark

## OOFRecord

- `id`: unique training row ID.
- `prompt_id`: held-out prompt group.
- `fold`: fold used as validation for this prediction.
- `candidate`: configured baseline name.
- `actual`: official train label.
- `prediction`: model prediction from the fold model that did not fit this group.

One record exists for each training ID and each candidate model. No source text
is copied into the OOF artifact.

## ModelMetrics

- `balanced_accuracy`: mean recall over the official label allowlist.
- `macro_f1`: unweighted mean of per-label F1 scores.
- `recall_by_label`: recall for every allowed class, including zero-valued
  classes if one is not predicted.
- `confusion_matrix`: square matrix ordered by the official allowlist.
- `runtime_seconds`: measured OOF evaluation time for the candidate.

## RunManifest

- `run_id`: deterministic identifier derived from input/config/split hashes.
- `train_sha256`, `config_sha256`, `split_sha256`: exact input provenance.
- `source_sha256`: checksum of the Python package source used for the run.
- `config_version`, `seed`, `fold_count`: validation settings.
- `git_commit`, `working_tree_dirty`: code revision at run time.
- `python_version`, `library_versions`: execution environment.
- `models`: candidate names and effective parameters.
- `completed_at`: run timestamp.

The run directory contains `manifest.json`, `metrics.json`, and
`predictions_oof.parquet`.
