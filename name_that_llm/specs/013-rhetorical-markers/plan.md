# Implementation Plan: Rhetorical and Formatting Markers

**Branch**: `013-rhetorical-markers` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

Extend the style vector with rhetorical/formatting marker rates and rerun the
identical flat/style/fusion grid on the same five saved folds.

## Design

1. Add fixed marker lists and per-1k-word (or per-line) rates to
   `src/ntllm/stylometry.py`; keep names ordered and values finite.
2. Rerun `style_grid.yaml` unchanged (same candidates read the extended
   extractor); confirm on seed-123 only on a seed-42 signal.
3. Rank by mean balanced accuracy with spread; no promotion inside this feature.

## Risks

- Marker lists encode hypotheses about authorial voice; if wrong, fusion may
  drop — the per-fold report will show it.
- Longer vectors slightly increase LinearSVC fit time; still minutes.

## Complexity Tracking

No constitution violations to justify.
