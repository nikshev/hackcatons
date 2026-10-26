# Local quickstart

1. Review the dated facts and open checks in `docs/competition.md`.
2. Obtain permitted competition files and place them unchanged under
   `data/raw/` (the directory is ignored by Git).
3. Update provenance and schema roles in `configs/baseline.yaml` from the
   actual files and official current sources. Do not fill `allowed_labels` by
   inference.
4. Install the package and its dependencies in the local environment:

   ```bash
   python -m pip install -e '.[dev]'
   ```

5. Run the data audit:

   ```bash
   python -m ntllm audit --config configs/baseline.yaml
   ```

The command writes `reports/audit/audit.json`. A structurally valid audit may
still report `training_readiness: blocked`; that is expected until competition
rules, eligibility, training data, and the exact official label allowlist are
verified. Missing files or an incompatible schema produce a failed audit.

Run the offline tests with `python -m pytest`.
