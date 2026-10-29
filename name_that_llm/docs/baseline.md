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

## Final fit and submission smoke test (promoted fusion model)

- Config version 13 selects `style_char_fusion` (char 2–5 TF-IDF + 67
  stylometric features, LinearSVC), based on OOF run `ebd7f3471379aa3e`
  (balanced accuracy 0.3526, per-fold spread ±0.0079 — wins every fold).
- Final model bundle, fit on all 2,699 train rows:
  `artifacts/final_models/a62b372f62d35767/`. The liblinear convergence
  warning is known and identical to the validated OOF configuration
  (deterministic given the same code, config, and data).
- The local three-row `kaggle/test.csv` produced
  `reports/submissions/sample-preview.parquet`; the exact `id`/`model` schema,
  ID coverage, nulls, and allowlist passed validation.
- Generated `notebooks/submission.ipynb` (now embedding the style extractor,
  no new pip packages) ran locally offline; fit took about 3 minutes and its
  preview also passed validation.

These are local prompt-group OOF and interactive-sample measurements, not a
Kaggle hidden-test score or accepted competition submission. The actual Kaggle
Notebook runtime remains to be smoke-tested. Personal eligibility and payout
conditions remain unverified, so competition-entry readiness is blocked while
local model training is ready.

## Rhetorical markers (neutral)

Extending the style vector with 11 rhetorical/formatting marker rates
(`docs/baseline.md` inventory: discourse, hedging, self-reference, closings,
headers, numbered/block/quoted lines, code fences, bold, inline code, emoji):

- seed-42 fusion with markers: **0.3526** vs **0.3530** without (noise).
- seed-123 fusion with markers: **0.3482** vs **0.3548** without (noise).
- Spread tightened on seed-42 (±0.0054 vs ±0.0079); macro F1 comparable.

Markers neither help nor hurt materially; the fusion advantage is independent
of them. They stay in the extractor as cheap robustness features. Promotion of
the fusion model to the production baseline is a separate explicit step.

## Stylometric features (confirmed on two splits)

`configs/experiments/style_grid.yaml` compared flat char 2–5, style-only, and
style+char fusion on identical folds (`reports/tuning/style-grid-v1/summary.json`):

- style+char fusion: balanced accuracy **0.3530** (±0.0079), macro F1 **0.3356**.
- flat char 2–5: balanced accuracy **0.3008** (±0.0166), macro F1 **0.2902**.
- style-only: balanced accuracy **0.2519** (±0.0035), macro F1 **0.2233**.

The fusion wins all five folds with lower spread and better macro F1. The
seed-123 confirmation split agrees: fusion **0.3548** vs flat **0.3174**.
Style-only alone is weak but the fusion is additive, so this family is a
promotion candidate. Promotion (config version bump, final refit, notebook
regen) is a separate explicit step, not done here.

## Frozen MiniLM embeddings (rejected)

`configs/experiments/sbert_logreg.yaml` compared frozen all-MiniLM-L6-v2
embeddings + logistic regression against flat char 2–5 on identical folds
(`reports/tuning/sbert-logreg-v1/summary.json`):

- MiniLM + logreg: balanced accuracy **0.1345** (±0.0138), macro F1 **0.1238**.
- flat char 2–5: balanced accuracy **0.3008** (±0.0166), macro F1 **0.2902**.

The gap is consistent across all five folds, so no confirmation split was
needed. Interpretation: semantic embeddings cluster passages by prompt (which
is held out in validation) rather than by author; surface char n-grams carry
the authorial signal on identical prompts. The vendored weights live in the
Git-ignored `models/` directory. A Kaggle packaging runbook is in
`docs/embeddings_packaging.md`; no baseline change.

## LSA dense projection (rejected)

`configs/experiments/char_svd.yaml` added a per-fold TruncatedSVD(512) layer
between char 2–5 TF-IDF and LinearSVC (`reports/tuning/char-svd-v1/summary.json`):

- SVD-512: balanced accuracy **0.3067** (±0.0210), macro F1 **0.2948**.
- flat char 2–5: balanced accuracy **0.3008** (±0.0166), macro F1 **0.2902**.

SVD won 2 folds, tied 2, lost 1 — but the seed-123 confirmation split
rejected it: flat **0.3174** vs SVD **0.3019**. The dense projection does not
generalize across splits the way the sparse baseline does, so the flat model
stays. No new dependencies were added; SVD remains an opt-in config key.

## Gemini-trio rerank (rejected)

`configs/experiments/gemini_rerank.yaml` rescored only flat-predicted trio
rows with a per-fold specialist (`reports/tuning/gemini-rerank-v1/summary.json`):

- rerank: balanced accuracy **0.30079** (±0.0156), macro F1 **0.29027**.
- flat char 2–5: balanced accuracy **0.30079** (±0.0166), macro F1 **0.29023**.

Identical means; per-fold wins split 2–2 with one tie. The seed-123
confirmation split rejected the rerank: flat **0.3174** vs rerank **0.3126**.
Routing errors outside the trio are unrecoverable, so the flat model stays.

## Hierarchical experiment (provider→model, fixed char 2–5)

`configs/experiments/hierarchical_provider.yaml` routed rows through a
provider classifier into within-provider specialists on the same five folds
(`reports/tuning/hierarchical-provider-v1/summary.json`):

- hierarchical: balanced accuracy **0.3056** (±0.0196), macro F1 **0.2747**.
- flat char 2–5: balanced accuracy **0.3008** (±0.0166), macro F1 **0.2902**.

Hierarchical led on 3 folds, tied 1, lost 1 — but a seed-123 confirmation
split reversed the ranking: flat **0.3174** (±0.0015) vs hierarchical
**0.3075** (±0.0079). The architecture advantage is split-dependent, and
hierarchical macro F1 is worse on both splits, so the production baseline
keeps the flat model. No config promotion from this experiment.

## Classifier experiment (SVC vs logreg, fixed char 2–5)

`configs/experiments/char_logreg_vs_svc.yaml` compared LinearSVC and
multinomial logreg on identical features and the same five folds
(`reports/tuning/char-logreg-vs-svc-v1/summary.json`):

- char SVC: balanced accuracy **0.30079** (±0.0166), macro F1 **0.2902**.
- char logreg: balanced accuracy **0.30077** (±0.0216), macro F1 **0.2842**.

The means are effectively tied; SVC has lower fold spread and better macro F1,
so the production baseline keeps LinearSVC. No config promotion from this
experiment.
