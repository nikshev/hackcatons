# Implementation Plan: Stylometric Features Alone and Fused

**Branch**: `012-stylometric-features` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

One family, two candidates (style-only, style+char fusion) vs flat char 2–5,
on the identical five saved folds.

## Design

1. `src/ntllm/stylometry.py`: pure-stdlib `extract_style_matrix` returning a
   fixed-order dense matrix (lengths, case, punctuation rates, structure,
   function-word and contraction rates, repetition), NaN-free by construction.
2. `build_estimator`: `style_svc` (style scaler + SVC) and
   `style_char_tfidf` (FeatureUnion of char TF-IDF and scaled style) types;
   reject `svd_components` with style types as untested.
3. `configs/experiments/style_grid.yaml`: flat ref, style-only, fusion.
4. Run on the saved split; confirm on seed-123 only on a signal. Rank by mean
   balanced accuracy with spread; no promotion inside this feature.

## Risks

- Style features may duplicate what char n-grams already capture; the
  style-only candidate measures the standalone signal.
- English function-word lists on potentially mixed-language texts degrade to
  zeros; still deterministic and finite.

## Complexity Tracking

No constitution violations to justify.
