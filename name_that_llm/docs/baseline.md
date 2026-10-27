# First grouped OOF baseline

**Run**: `80d257594c97c589`
**Config**: version 8
**Train checksum**: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
**Split checksum**: `29ae008bff2004518bfafb93e81c2f281d82b61c57f997ec4acdefed35b2111a`
**Source checksum**: `aec7dc256d0402a95079e3f76c20c69dab4f320d1e8b8b7853738361f10f48ae`
**Validation**: five folds, grouped by `prompt_id`, 150 groups / 2,699 rows

## OOF results

| Candidate | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Most-frequent dummy | 0.0553 | 0.0105 |
| Word TF-IDF + LinearSVC | 0.2634 | 0.2480 |
| Character TF-IDF + LinearSVC | 0.2960 | 0.2849 |
| Word + character TF-IDF + LinearSVC | 0.2967 | 0.2840 |

Balanced accuracy is the Kaggle primary metric. The dummy result is near the
uniform baseline of `1/18`; the text baselines improve substantially under the
prompt-group split. Character and combined features are nearly tied, so this is
not evidence that combining them is a meaningful improvement. These are local
OOF measurements, not leaderboard estimates or submission results.

The version 8 inference config selects the simpler character TF-IDF model for
final fit; its 0.0007 balanced-accuracy gap from the combined model is too small
to justify the extra feature branch at this stage.

## Data and reproducibility notes

- Each fold holds out 30 complete prompt groups; four folds contain 540 rows
  and one contains 539 because one official prompt group lacks
  `claude-fable-5`.
- All 2,699 train IDs received exactly one OOF prediction per candidate. No
  prompt group crossed folds, and test data was not used.
- Full metrics, class recalls, confusion matrices, OOF predictions, library
  versions, and the manifest are in the ignored local directory
  `reports/runs/80d257594c97c589/`.
- The audit is `reports/audit/audit.json`; fold assignments and the split report
  are `reports/splits/prompt_folds.parquet` and
  `reports/splits/prompt_folds.json`.
- Personal eligibility and payout conditions remain unverified, so competition
  entry readiness is still blocked independently of these local measurements.
- The run manifest records a dirty working tree; the source checksum captures
  the implementation used for this local run.

## Final fit and sample inference

- The selected `char_tfidf_svc` model was refit on all 2,699 train rows and
  saved as model bundle `96c1db12207bd5c7` under the Git-ignored
  `artifacts/final_models/` directory.
- `kaggle/test.csv` produced a three-row preview at
  `reports/submissions/sample-preview.parquet`; the submission validator passed
  exact `id`/`model` schema, ID coverage, null, and allowlist checks.
- Final model bundle: `artifacts/final_models/d4cad7959bb854e2/` (config
  version 9, candidate `char_tfidf_svc`, selected OOF run
  `80d257594c97c589`). Its manifest includes the model checksum and source hash.
- The generated `notebooks/submission.ipynb` was executed locally against the
  same CSVs; char-TF-IDF fit in about 8 seconds and the three-row Parquet output
  passed the submission validator. Kaggle's own offline runtime remains
  untested.
- This validates the local offline path against Kaggle's interactive sample;
  it does not represent a hidden-set submission or leaderboard score.
