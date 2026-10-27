# Grouped OOF Baseline, Tuning, and Final Sample Smoke Test

## Current OOF run

**Run**: `595089c1e1fe0029`
**Config**: version 11
**Train checksum**: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
**Split checksum**: `29ae008bff2004518bfafb93e81c2f281d82b61c57f997ec4acdefed35b2111a`
**Source checksum**: `8786ba121dde6530ccc6425d9709ae24a6bd0ad381e5686693b8599971606c91`
**Validation**: five folds, grouped by `prompt_id`, 150 groups / 2,699 rows

| Candidate | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Most-frequent dummy | 0.0553 | 0.0105 |
| Word TF-IDF + LinearSVC | 0.2634 | 0.2480 |
| Character TF-IDF, 2–5 grams + LinearSVC | 0.3008 | 0.2902 |
| Word + character TF-IDF + LinearSVC | 0.2967 | 0.2840 |

The tuning sweep evaluated eight candidates on the same grouped folds. The
char 2–5 gram candidate improved over the original char 3–5 baseline on four
of five folds, with lower fold spread (0.0166 vs 0.0199). A separate seed-123
confirmation split also favored char 2–5: **0.3174 ± 0.0015** versus
**0.3119 ± 0.0089** for char 3–5. The confirmation split is a robustness check,
not a hidden-test estimate. Experiment summaries are in
`reports/tuning/char-tfidf-grid-v1/summary.json` and
`reports/tuning/char-tfidf-seed123/summary.json`.

## Error analysis

The current OOF report is `reports/runs/595089c1e1fe0029/error_analysis.md`.
The largest confusion family is between adjacent Gemini variants. Lowest recall
remains for `glm-5.3-flash` (0.093) and `claude-fable-5` (0.134). Shorter
passages are harder: the error rate is 0.771 for passages under 500 characters
and declines across longer bins. Reports contain aggregate counts, not row IDs,
prompt IDs, or passage text.

## Final fit and submission smoke test

- Config version 12 selects `char_tfidf_svc` with character n-grams `[2, 5]`,
  based on OOF run `595089c1e1fe0029`.
- Final model bundle, fit on all 2,699 train rows:
  `artifacts/final_models/8ae9a2dae39207a2/`.
- The local three-row `kaggle/test.csv` produced
  `reports/submissions/sample-preview.parquet`; the exact `id`/`model` schema,
  ID coverage, nulls, and allowlist passed validation.
- Generated `notebooks/submission.ipynb` ran locally offline; fit took about
  12.5 seconds and its preview also passed validation.

These are local prompt-group OOF and interactive-sample measurements, not a
Kaggle hidden-test score or accepted competition submission. The actual Kaggle
Notebook runtime remains to be smoke-tested. Personal eligibility and payout
conditions remain unverified, so competition-entry readiness is blocked while
local model training is ready.

## Classifier experiment (SVC vs logreg, fixed char 2–5)

`configs/experiments/char_logreg_vs_svc.yaml` compared LinearSVC and
multinomial logreg on identical features and the same five folds
(`reports/tuning/char-logreg-vs-svc-v1/summary.json`):

- char SVC: balanced accuracy **0.30079** (±0.0166), macro F1 **0.2902**.
- char logreg: balanced accuracy **0.30077** (±0.0216), macro F1 **0.2842**.

The means are effectively tied; SVC has lower fold spread and better macro F1,
so the production baseline keeps LinearSVC. No config promotion from this
experiment.
