# Data Model: OOF Error Analysis and Controlled Tuning

## OOFErrorSummary

- `run_id`, `train_sha256`, `split_sha256`: source run provenance.
- `candidate`: OOF candidate being analyzed.
- `class_metrics`: support and recall per official class.
- `confusion_pairs`: aggregate actual/predicted class pair counts, excluding
  correct predictions.
- `fold_metrics`: balanced accuracy and macro F1 per fold.
- `group_errors`: number/distribution of prompt groups with incorrect rows;
  no prompt IDs are retained.
- `length_errors`: aggregate counts/accuracy by configured text-length bins;
  no text is retained.

## TuningCandidate

- `name`: stable experiment-local candidate name.
- `parameters`: complete model/vectorizer configuration.
- `mean_balanced_accuracy`, `fold_balanced_accuracy`: primary overall score and
  per-fold values.
- `mean_macro_f1`, `fold_macro_f1`: secondary overall score and per-fold values.
- `runtime_seconds`: measured evaluation time.
- `class_recall`: per-label diagnostics.

## TuningSummary

- `experiment_name`, `experiment_version`, `experiment_sha256`.
- `train_sha256`, `split_sha256`, `baseline_config_sha256`, `code_sha256`.
- `seed`, `fold_count`, and ranked candidate results.

Reports contain no row IDs, prompt IDs, or raw text.
