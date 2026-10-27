# Implementation Plan: OOF Error Analysis and Controlled Tuning

**Branch**: `006-oof-error-analysis-tuning` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/006-oof-error-analysis-tuning/spec.md`

## Summary

Turn existing grouped OOF predictions into an aggregate error report and run a
small, versioned TF-IDF sweep on the same 5-fold prompt split. Report per-fold
stability and error patterns by class, confusion pair, prompt-group incidence,
and text-length bin; do not save passage text, IDs, or prompt IDs in the report.

## Technical Context

**Language/Version**: Python 3.12 or newer
**Primary Dependencies**: pandas, PyArrow, scikit-learn, PyYAML
**Storage**: Existing OOF Parquet; Markdown error report; JSON tuning summary
**Testing**: pytest with offline synthetic OOF artifacts
**Target Platform**: Local Python environment
**Project Type**: Python package and CLI
**Performance Goals**: Error report in seconds; bounded sweep in minutes on the 2,699-row dataset
**Constraints**: Fixed prompt-group split, no test access, no external data, no raw text/ID in reports
**Scale/Scope**: 18 classes, 150 prompt groups, 5 folds, a small predeclared candidate grid

## Constitution Check

- **Leakage**: Analysis uses only existing OOF predictions; tuning uses the
  exact same prompt-group fold assignments for every candidate.
- **Test-first**: Verify aggregation and ranking with known fixtures before
  reading the real OOF run.
- **Reproducibility**: Record train/config/split/experiment/code checksums,
  seed, candidate parameters, and per-fold metrics.
- **Privacy**: Aggregate by class/fold/group/length bin; never export an example
  ID, prompt ID, or raw text in Markdown/JSON summaries.
- **No test optimization**: The configured test path is not opened by either
  command.

No constitution violations identified.

## Design

### OOF error analysis

1. Resolve the OOF run by run ID and verify its train checksum and split hash.
2. Reconcile OOF IDs and prompt groups with the audited train file.
3. Produce per-class support/recall, top off-diagonal confusion pairs,
   per-fold scores, group error-incidence aggregates, and error counts by text
   length bin.
4. Write `error_analysis.md` under the run directory; omit IDs and passage text.

### Controlled tuning

1. Load candidate definitions from a separate versioned experiment YAML.
2. Verify the current audit and saved split, then evaluate each candidate with
   the same training rows/folds/allowlist.
3. Record mean and per-fold balanced accuracy, macro F1, runtime, and per-class
   recall.
4. Rank by mean balanced accuracy but preserve fold spread; no test/leaderboard
   data participates.
5. Write aggregate results under `reports/tuning/<experiment>/`.

### Project Structure

```text
configs/experiments/char_tfidf_grid.yaml
src/ntllm/analysis.py
src/ntllm/__main__.py                # evaluate and tune commands
tests/test_analysis.py
reports/runs/<run-id>/error_analysis.md
reports/tuning/<experiment>/summary.json
```

**Structure Decision**: Reuse the OOF run and fixed split contracts from
features 002/003. Keep tuning candidates separate from production baseline
settings until a candidate is deliberately promoted.

## Risks and Open Questions

- Selecting the best from many candidates can overfit the fixed OOF folds; keep
  the grid bounded and report fold-level spread before promotion.
- The hidden test distribution is not represented by the three-row local test
  sample; no hidden-set conclusions are possible.
- Length/group aggregates may identify systematic domain patterns but must not
  expose individual examples.

## Complexity Tracking

No constitution violations to justify.
