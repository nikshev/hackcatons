# Data Model: Offline Kaggle Submission Notebook

## NotebookConfigSnapshot

- `config_version`: baseline config version used to render the notebook.
- `config_sha256`: checksum of the full YAML configuration.
- `candidate`: selected model name.
- `model_parameters`: exact candidate settings rendered from config.
- `allowed_labels`: exact current official label list.

## NotebookRunSummary

- `train_sha256`, `test_sha256`: checksums of input CSV files.
- `train_rows`, `test_rows`: input row counts.
- `model`, `fit_seconds`, `predict_seconds`: run and runtime information.
- `prediction_rows`, `submission_path`: output summary.
- `python_version`, `pandas_version`, `scikit_learn_version`: environment
  provenance.

No text values, IDs, or labels are printed as diagnostic summaries beyond the
submission itself.
