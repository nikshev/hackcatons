# Implementation Plan: LSA Dense Projection over Char TF-IDF

**Branch**: `010-lsa-dense-projection` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

One-variable experiment: flat char 2–5 SVC vs the same features plus a
per-fold TruncatedSVD(512) projection, on the identical five saved folds.

## Design

1. Add optional `svd_components` to TF-IDF model configs; insert
   `TruncatedSVD(n_components, random_state=seed)` between vectorizer and
   classifier; reject for dummy models and invalid counts.
2. Add `configs/experiments/char_svd.yaml` with two candidates on identical
   settings; run on the saved split, then confirm on seed-123 only if seed-42
   favors SVD.
3. Rank by mean balanced accuracy with spread; claim nothing about hidden data.

## Risks

- SVD fit per fold on ~2,160×100k sparse rows costs minutes; keep the grid to
  two candidates (flat reference + SVD-512).
- Dense projection may discard the rare token signals that separate close
  providers; per-class recall will show it.

## Complexity Tracking

No constitution violations to justify.
