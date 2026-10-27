# Local quickstart

1. Review the dated facts and open checks in `docs/competition.md`.
2. Place permitted competition files unchanged under `kaggle/` (the directory
   is ignored by Git); the baseline expects `train.csv` and `test.csv` there.
3. Update provenance and schema roles in `configs/baseline.yaml` from the
   actual files and official current sources. Do not fill `allowed_labels` by
   inference.
4. Install the package and its dependencies in the local environment:

   ```bash
   python -m venv .venv
   .venv/bin/python -m pip install -e '.[dev]'
   ```

5. Run the data audit:

   ```bash
   .venv/bin/python -m ntllm audit --config configs/baseline.yaml
   ```

6. After the train CSV passes the audit, generate prompt-group validation folds:

   ```bash
   .venv/bin/python -m ntllm split --config configs/baseline.yaml
   ```

   The command reads only `train.csv` and writes
   `reports/splits/prompt_folds.parquet` plus
   `reports/splits/prompt_folds.json`. All rows for a given `prompt_id` stay in
   one fold; missing or duplicate IDs/groups stop generation.

7. Run the grouped OOF baseline:

   ```bash
   .venv/bin/python -m ntllm train --config configs/baseline.yaml
   ```

   The deterministic run directory contains OOF predictions, metrics, and a
   manifest. The first real-data result is summarized in `docs/baseline.md`.

8. Fit the selected final model on all audited train rows:

   ```bash
   .venv/bin/python -m ntllm fit-final --config configs/baseline.yaml
   ```

9. Predict the local Kaggle test sample and validate the file:

   Replace `MODEL_DIR` with the bundle path printed by `fit-final`.

   ```bash
   .venv/bin/python -m ntllm predict \
     --model MODEL_DIR \
     --input kaggle/test.csv \
     --output reports/submissions/sample-preview.parquet
   .venv/bin/python -m ntllm validate-submission \
     --input reports/submissions/sample-preview.parquet \
     --test kaggle/test.csv \
     --config configs/baseline.yaml
   ```

10. Regenerate the self-contained offline Kaggle Notebook after any config
    change:

    ```bash
    .venv/bin/python scripts/build_submission_notebook.py --config configs/baseline.yaml
    ```

    Upload `notebooks/submission.ipynb` to Kaggle with the competition data
    attached and internet disabled. `docs/baseline.md` records the local smoke
    result; this is not a Kaggle-runtime smoke test.

The local `test.csv` is only the three-row interactive sample. Its accepted
validation is a pipeline smoke test, not a hidden-test or leaderboard result.

The audit command writes `reports/audit/audit.json`. A structurally valid audit
may still report `training_readiness: blocked` until permitted labeled data,
schema, data-use terms, and the official allowlist are verified. Personal
eligibility and payout are tracked separately as
`competition_entry_readiness`; they block competition entry, not local
experiments. Missing files, incompatible schemas, duplicate/missing IDs, or
blank text produce a failed audit. A missing prompt-group column is a
non-blocking audit issue, but split generation remains blocked without a valid
configured group field.

Run the offline tests with `.venv/bin/python -m pytest`.
