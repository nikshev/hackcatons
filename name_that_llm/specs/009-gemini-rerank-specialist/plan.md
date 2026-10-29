# Implementation Plan: Gemini-Flash Trio Rerank Specialist

**Branch**: `009-gemini-rerank-specialist` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

One-variable experiment: flat char 2–5 SVC vs the same flat pass plus a
per-fold trio specialist rerank. Identical features and the same five saved
folds.

## Design

1. Add `evaluate_gemini_rerank_oof` to `src/ntllm/training.py`: per fold, fit
   flat estimator on fold-train, fit trio specialist on trio fold-train rows,
   rerank only trio-predicted validation rows.
2. Extend `tune_from_config` architectures with `gemini_rerank` (default stays
   `flat`).
3. Add `configs/experiments/gemini_rerank.yaml` with two candidates on
   identical settings; run on the saved split, then confirm on the seed-123
   split only if seed-42 favors rerank.
4. Rank by mean balanced accuracy with spread; claim nothing about hidden data.

## Risks

- Routing errors are unrecoverable: rows the flat stage misroutes outside the
  trio never reach the specialist.
- Small trio subsets per fold (~90 rows); specialist variance may be high.

## Complexity Tracking

No constitution violations to justify.
