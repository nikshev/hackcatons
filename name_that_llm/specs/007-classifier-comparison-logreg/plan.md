# Implementation Plan: Classifier Comparison on Fixed Char Features

**Branch**: `007-classifier-comparison-logreg` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

One-variable experiment: LinearSVC vs multinomial logistic regression on frozen
char 2–5 TF-IDF features, evaluated on the saved five prompt-group folds.

## Design

1. Extend `build_estimator` with opt-in `classifier: logreg` (multinomial,
   lbfgs, fixed `max_iter`, seeded); default stays LinearSVC.
2. Add `configs/experiments/char_logreg_vs_svc.yaml` with 2 candidates sharing
   all feature settings; only `classifier` differs (plus one C variant only if
   runtime allows — default 2 candidates).
3. Run `ntllm tune` on the current saved split; rank by mean balanced accuracy
   with per-fold spread; no test data.
4. Record results in `docs/baseline.md` as an experiment note; no baseline
   promotion in this feature.

## Risks

- Logreg on 100k sparse features x 18 classes may be slower than SVC; cap
  iterations and keep the grid to 2 candidates.
- A single grouped split can overstate a small gap; report spread, claim nothing
  about the hidden test.
