# Матриця трасування

<!-- ГЕНЕРУЄТЬСЯ scripts/trace.py — не редагувати вручну -->

Вимог: **55** · задач: **108** (виконано 106) · вимог у роботі: **2** · порушень: **0**

## 001-competition-readiness-audit

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-001-01` | The competition record MUST distinguish confirmed, unverified, | `T-001` | — | — |
| `FR-001-02` | The record MUST cover eligibility, current deadlines, prizes, | `T-002` | — | `tests/test_audit.py` |
| `FR-001-03` | The data audit MUST inspect the actual input files and report | `T-003`, `T-004`, `T-015` | `src/ntllm/data.py`, `src/ntllm/__main__.py` | `tests/test_audit.py` |
| `FR-001-04` | For every audited raw file, the manifest MUST record source, | `T-005`, `T-006`, `T-016` | `src/ntllm/audit.py` | `tests/test_audit.py` |
| `FR-001-05` | The audit MUST report missing values, text-length summaries, | `T-007`, `T-008` | `src/ntllm/audit.py` | `tests/test_audit.py` |
| `FR-001-06` | The audit MUST report prompt-group availability and identify | `T-009`, `T-010` | `src/ntllm/validation.py`, `src/ntllm/audit.py` | `tests/test_validation.py` |
| `FR-001-07` | The project MUST mark supervised baseline training as blocked | `T-011`, `T-012` | `src/ntllm/audit.py` | `tests/test_audit.py` |
| `FR-001-08` | An unknown schema, unreadable file, or unrecognized class MUST | `T-013`, `T-014` | `src/ntllm/__main__.py`, `src/ntllm/audit.py` | `tests/test_audit.py` |

## 002-prompt-group-cross-validation

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-002-01` | The split MUST assign every row to exactly one fold and MUST | `T-001`, `T-002` | `src/ntllm/splits.py` | `tests/test_splits.py` |
| `FR-002-02` | The split MUST reject missing/blank row IDs, duplicate row IDs, | `T-007`, `T-008` | `src/ntllm/splits.py` | `tests/test_splits.py` |
| `FR-002-03` | The split MUST reject fewer than two folds, fewer prompt groups | `T-009`, `T-010` | `src/ntllm/splits.py` | `tests/test_splits.py` |
| `FR-002-04` | Given the same dataset, configuration, and seed, the id-to-fold | `T-003`, `T-004` | `src/ntllm/splits.py` | `tests/test_splits.py` |
| `FR-002-05` | Fold assignment MUST balance total row counts and per-class | `T-005`, `T-006` | `src/ntllm/splits.py` | `tests/test_splits.py` |
| `FR-002-06` | The split report MUST include the seed, fold count, source-data | `T-011`, `T-012`, `T-017` | `src/ntllm/splits.py`, `src/ntllm/__main__.py` | `tests/test_splits.py` |
| `FR-002-07` | Split creation MUST use labeled training rows only and MUST NOT | `T-013`, `T-014` | `src/ntllm/splits.py` | `tests/test_splits.py` |
| `FR-002-08` | Assignment artifacts MUST be written only after all input | `T-015`, `T-016` | `src/ntllm/splits.py`, `src/ntllm/__main__.py` | `tests/test_splits.py` |

## 003-tfidf-grouped-baseline

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-003-01` | The benchmark MUST consume the saved prompt-group assignments | `T-001`, `T-002` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-003-02` | Every learned text transform and classifier MUST be fit only | `T-003`, `T-004` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-003-03` | The benchmark MUST compare a no-skill classifier, word-based | `T-005`, `T-006` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-003-04` | For each candidate model, the run MUST produce exactly one | `T-007`, `T-008` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-003-05` | For each model, the run MUST report balanced accuracy, macro | `T-009`, `T-010` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-003-06` | Each run MUST record train-data checksum, config version/hash, | `T-011`, `T-012`, `T-019` | `src/ntllm/training.py`, `src/ntllm/__main__.py` | `tests/test_training.py` |
| `FR-003-07` | The benchmark MUST read only training data; test rows and labels | `T-013`, `T-014` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-003-08` | Model fitting MUST be blocked when feature 001 reports | `T-015`, `T-016` | `src/ntllm/training.py`, `src/ntllm/__main__.py` | `tests/test_training.py` |
| `FR-003-09` | OOF prediction artifacts MUST contain no duplicate/missing | `T-017`, `T-018` | `src/ntllm/training.py` | `tests/test_training.py` |

## 004-offline-submission-pipeline

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-004-01` | Final fitting MUST require `training_readiness: ready`, use the | `T-001`, `T-002` | `src/ntllm/__main__.py`, `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-02` | The saved artifact MUST contain learned text preprocessing and | `T-003`, `T-004` | `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-03` | The model manifest MUST record train/config/model hashes, | `T-005`, `T-006` | `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-04` | Prediction MUST read the actual test schema and produce exactly | `T-007`, `T-008`, `T-019` | `src/ntllm/__main__.py`, `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-05` | Prediction and submission validation MUST use only the current | `T-009`, `T-010` | `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-06` | Submission validation MUST require exactly the columns `id` | `T-011`, `T-012` | `src/ntllm/__main__.py`, `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-07` | Repeated prediction with the same artifact and test input MUST | `T-013`, `T-014` | `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-08` | Test data MUST NOT be used to fit the final model or tune its | `T-015`, `T-016` | `src/ntllm/inference.py` | `tests/test_inference.py` |
| `FR-004-09` | Failed readiness, artifact, prediction, or validation checks | `T-017`, `T-018` | `src/ntllm/__main__.py`, `src/ntllm/inference.py` | `tests/test_inference.py` |

## 005-offline-kaggle-notebook

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-005-01` | The notebook MUST read only the competition's mounted train and | `T-005` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-02` | The notebook MUST embed the baseline config version/hash, chosen | `T-001`, `T-002` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-03` | The notebook MUST fit the selected model on all permitted | `T-006` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-04` | The notebook MUST predict each test row once and write | `T-007`, `T-008` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-05` | Before writing a valid-output status, the notebook MUST reject | `T-009`, `T-010` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-06` | The notebook MUST print a compact run summary with config/data | `T-011`, `T-012` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-07` | The committed notebook MUST be reproducible from | `T-003`, `T-004` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |
| `FR-005-08` | Local smoke execution MUST allow explicit input/output path | `T-013` | `scripts/build_submission_notebook.py` | `tests/test_submission_notebook.py` |

## 006-oof-error-analysis-tuning

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-006-01` | Error analysis MUST consume one named OOF run and verify its | `T-001`, `T-002`, `T-007` | `src/ntllm/analysis.py`, `src/ntllm/__main__.py` | `tests/test_analysis.py` |
| `FR-006-02` | The report MUST include per-class support/recall, top | `T-003`, `T-004` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-03` | Error-analysis output MUST NOT include raw passage text, row | `T-005`, `T-006` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-04` | Tuning candidates MUST come from a versioned experiment YAML and | `T-008`, `T-009` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-05` | Each candidate MUST report mean and per-fold balanced accuracy, | `T-010` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-06` | Tuning MUST NOT load or inspect `test.csv`, hidden labels, or | `T-011`, `T-012` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-07` | Unsupported model settings, data/split mismatch, incomplete | `T-013`, `T-014` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-08` | Candidate ranking MUST use the primary balanced-accuracy metric | `T-015`, `T-016` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-006-09` | Generated reports MUST record the experiment config hash, | `T-017`, `T-018` | `src/ntllm/analysis.py`, `src/ntllm/__main__.py` | `tests/test_analysis.py` |

## 007-classifier-comparison-logreg

| Вимога | Опис | Задачі | Імплементація | Тести |
|---|---|---|---|---|
| `FR-007-01` | Candidates MUST share the same saved prompt-group folds, train | `T-003`, `T-004` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-007-02` | The pipeline MUST support a `classifier: logreg` option for | `T-001`, `T-002` | `src/ntllm/training.py` | `tests/test_training.py` |
| `FR-007-03` | Tuning MUST report mean and per-fold balanced accuracy, macro | `T-005` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |
| `FR-007-04` | Tuning MUST NOT open `test.csv`, test labels, or leaderboard | `T-006` | `src/ntllm/analysis.py` | `tests/test_analysis.py` |

## Критичні задачі

- `T-011` (001-competition-readiness-audit) — виконано
- `T-012` (001-competition-readiness-audit) — виконано
