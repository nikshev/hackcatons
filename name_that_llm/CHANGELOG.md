# Changelog

## Unreleased

- Add CSV null/length/duplicate/label profiles and prompt-group/row-integrity
  validation to the local audit report.
- Report a missing prompt-group column as a non-blocking audit issue, while
  duplicate/missing IDs and blank text fail the audit.
- Bump baseline configuration to version 3 because prompt-group metadata is now
  optional for inspection and explicitly reported when unavailable.
- Add version 4 validation settings for five deterministic prompt-group folds
  with seed 42 and configured assignment/report destinations.
- Add `ntllm split` with prompt-group exclusivity, deterministic balancing,
  canonical split hashes, Parquet assignments, and per-fold class-count reports.
- Bump baseline configuration to version 5 and point train/test inputs at the
  locally supplied, Git-ignored `kaggle/` directory.
- Bump baseline configuration to version 6 with the official Kaggle Rule 4(a)
  data-use status and workspace receipt date. Separate model-training readiness
  from participant eligibility/payout readiness.
- Bump baseline configuration to version 7 with four grouped-OOF baseline
  candidates and explicit TF-IDF/LinearSVC parameters.
- Add the first real-data grouped OOF benchmark and its concise result summary in
  `docs/baseline.md`.
- Bump baseline configuration to version 8, select char-TF-IDF from grouped OOF
  run `4f88aa2f4630e9b8`, and configure final-model/submission artifact paths.
- Include the source-code checksum and library versions in the deterministic run
  ID so code changes cannot reuse stale OOF artifacts.
- Declare `joblib` directly for serializing the final offline model bundle and
  record its version in training/inference manifests.
- Generate a self-contained offline Kaggle Notebook from the baseline YAML and
  smoke-test its code against the supplied local competition files.
- Bump baseline configuration to version 9 and bind final fitting explicitly
  to the repeated grouped OOF benchmark run `80d257594c97c589`.
- Bump baseline configuration to version 10 and bind final fitting to the
  current-source OOF run `174e931a846ecb42`.
- Confirm char 2–5 n-grams on an independent seed-123 grouped split and promote
  that configuration for the next baseline run.
- Bump baseline configuration to version 12 and bind the final-fit selection to
  OOF run `595089c1e1fe0029` using the promoted character range.
- Compare flat vs provider-routed hierarchical classification on identical char
  2–5 features: hierarchical leads on seed-42 (+0.0048) but loses on the seed-123
  confirmation split; no promotion, flat baseline kept.
- Add `classifier: logreg` pipeline option; multinomial logreg ties LinearSVC on
  mean balanced accuracy with worse spread and macro F1, so SVC is kept.
- Compare LinearSVC vs multinomial logreg on frozen char 2–5 features: means
  tied (0.30079 vs 0.30077), SVC kept for lower spread and better macro F1;
  no baseline promotion.
- Add handcrafted stylometric features (punctuation, case, length, repetition,
  function words) with `style_svc` and `style_char_tfidf` estimators; the
  style+char fusion beats flat char 2–5 on both seed-42 (+0.052) and seed-123
  (+0.037) splits. Promotion is a separate explicit step.
- Rerank flat-predicted Gemini flash trio rows with a per-fold trio specialist:
  means tie exactly on seed-42 (0.30079) and flat wins seed-123 (0.3174 vs
  0.3126); no promotion, flat baseline kept.
- Promote style+char fusion to the production baseline (config v13, OOF run
  `ebd7f3471379aa3e`, balanced accuracy 0.3526): final refit, offline
  prediction, submission validation, and a regenerated self-contained Notebook
  with the embedded style extractor all pass locally.
- Extend style vectors with rhetorical/formatting markers (discourse, hedging,
  self-reference, markdown structure): neutral on the mean on both splits,
  tighter spread; fusion advantage confirmed independently of markers.
- Compare frozen MiniLM embeddings + logreg against flat char 2–5 on identical
  folds: embeddings lose decisively and consistently (0.1345 vs 0.3008, every
  fold); no promotion. Weights stay Git-ignored under `models/`; Kaggle
  offline packaging is documented in `docs/embeddings_packaging.md`.
- Add opt-in `svd_components` LSA projection (TruncatedSVD, sklearn-only, fit
  per fold): SVD-512 leads on seed-42 (+0.0059) but loses seed-123 (0.3019 vs
  0.3174); no promotion, flat baseline kept.
- Add final-fit, offline prediction, and strict submission validation; the
  local three-row Kaggle sample passes the full pipeline smoke test.

## 0.1.0 — 2026-10-26

- Initialize the independent `name_that_llm` solution context and SDD feature
  `001-competition-readiness-audit`.
- Record partial current Kaggle facts, corrections to the supplied PRD, and
  unresolved rule/eligibility checks in `docs/competition.md`.
- Add versioned baseline audit configuration and a read-only CSV schema and
  provenance-manifest implementation; supervised readiness defaults to blocked.

## 0.1.1 — 2026-10-26

- Make current official Kaggle Overview/Data/Rules pages the competition source
  of truth and reconcile the PRD's outdated test-row description.
- Record Kaggle's current 18-label allowlist, submission contract, Notebook
  constraints, prize schedule, and freely available external-data policy.
- Bump the baseline config version and populate the official allowlist; personal
  eligibility, payout terms, dataset licenses, and paid API-generation rules
  remain verification gates.
