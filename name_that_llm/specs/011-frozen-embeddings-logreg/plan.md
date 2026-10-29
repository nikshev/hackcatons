# Implementation Plan: Frozen Embeddings with Logistic Regression

**Branch**: `011-frozen-embeddings-logreg` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

One-variable experiment: flat char 2–5 SVC vs frozen MiniLM embeddings +
logreg, same five saved folds. Weights vendored locally; tune reads the model
directory from the experiment YAML (no baseline config change).

## Design

1. `src/ntllm/embeddings.py`: load-once encoder from the vendored directory,
   batch encode with normalization, fail loudly when the directory is absent.
2. `evaluate_embedding_oof` in `training.py`: reuse the standard provenance
   checks, encode train texts once, fit logreg per fold on fold-train vectors.
3. `tune_from_config`: route `type: sbert_logreg` candidates to the embedding
   evaluator; record the embedding model identity in the summary.
4. `configs/experiments/sbert_logreg.yaml`: flat reference vs embedding
   candidate; run on the saved split, confirm on seed-123 only on a signal.
5. `docs/embeddings_packaging.md`: offline Kaggle runbook (weights dataset +
   torch/transformers mean-pooling snippet, no sentence-transformers install).

## Risks

- torch + sentence-transformers add ~1GB to the local venv only (ignored by
  Git); Kaggle uses its preinstalled torch/transformers instead.
- MiniLM on 2,699 passages: minutes on CPU; encode once and reuse per fold.
- Dense vectors may still not beat sparse n-grams on this task; keep the grid
  to two candidates and report spread.

## Complexity Tracking

No constitution violations to justify.
