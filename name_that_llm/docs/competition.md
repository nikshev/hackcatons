# Name That LLM — competition facts and open checks

**Checked:** 2026-10-26
**Status:** partial verification; do not treat unresolved items as confirmed.

**Source of truth:** Current official Kaggle Overview, Data, and Rules pages
override this PRD or any older notes when details conflict. Record changes here
with their check date; the PRD describes project intent and implementation
scope, not the live competition contract.

## Confirmed from Kaggle's current official pages

The official-page search excerpts retrieved on 2026-10-26 state:

- The task is to identify the author among 18 language models.
- The primary metric is balanced accuracy (mean per-class recall).
- `train.csv` contains 2,699 passages across 150 prompts. The documented fields
  are `id`, `prompt_id`, `text`, and `model` (the label).
- Interactive `test.csv` contains three example rows; the hidden rerun is about
  4,000 passages. Hidden prompts do not appear in train.
- The listed entry/team-merger deadline is 2027-01-06 and the final submission
  deadline is 2027-01-13, both 11:59 PM UTC. Kaggle says the timeline may change.
- The competition offers ten prizes of $5,000 each ($50,000 total).
- Kaggle Rules §4(a) permits access to and use of Competition Data for any
  purpose, including participating in the competition; checked 2026-10-26.
- Submissions are made through a Kaggle Notebook. The notebook must write
  `submission.parquet` with one row per test `id` and columns `id` and `model`.
  CPU and GPU notebooks have a nine-hour runtime limit; internet is disabled.
- Kaggle's overview explicitly allows freely and publicly available external
  data, including pretrained models. Applicable licenses and the competition
  rules still apply.

### Exact allowed labels

Kaggle's current official Overview > Evaluation page lists these 18 labels. The
same values are set in `configs/baseline.yaml`; compare them against the actual
downloaded train file before training.

| Provider | Model label |
|---|---|
| OpenAI | `gpt-5.5` |
| OpenAI | `gpt-5.6-luna` |
| OpenAI | `gpt-5.6-sol` |
| OpenAI | `gpt-5.6-terra` |
| OpenAI | `gpt-6-astra` |
| Google | `gemini-3.1-pro-preview` |
| Google | `gemini-3.6-flash` |
| Google | `gemini-3.7-flash` |
| Google | `gemini-3.8-flash` |
| Anthropic | `claude-fable-5` |
| Anthropic | `claude-opus-4-6` |
| Anthropic | `claude-opus-4-8` |
| Anthropic | `claude-opus-5` |
| Anthropic | `claude-sonnet-5` |
| Qwen | `qwen3.8-max` |
| ZAI | `glm-5.3` |
| ZAI | `glm-5.3-flash` |
| Moonshot AI | `kimi-k3` |

This corrects the PRD's one-row test description and records the current label
allowlist. Kaggle rejects null predictions, repeated IDs, and labels outside
this set.

## Still needs direct verification

The following items still need participant-specific or external-data verification:

- participant eligibility, territorial availability, and prize payout
  conditions;
- the license and terms for any external dataset beyond Kaggle Competition Data;
- any host clarification about paid API generation or publicly shared
  synthetic/API-generated training data.

The current public discussion includes a request for clarification on API-
generated passages and the reasonableness of paid generation. Freely and
publicly available external datasets are permitted by the overview; paid API
generation stays disabled until its treatment is clarified and explicitly
budgeted. The PRD keeps paid resources off by default. Personal eligibility and
payout checks gate entering the competition, not local experiments on the
official competition train data.

## Local data state

No competition files were present in this solution folder at initialization.
The documented train schema is therefore a discovery aid, not a substitute for
auditing the downloaded files. Once data is obtained, record source, retrieval
date, license/terms, checksum, and row count in its manifest; keep raw files
unchanged and out of Git where appropriate.

As of 2026-10-26, local `kaggle/train.csv`, `kaggle/test.csv`, and
`kaggle/sample_submission.parquet` are present. The entire `kaggle/` directory is
Git-ignored; the audit and split reports record checksums and derived metadata
without copying passage text.

The local audit passed on 2026-10-26:

- `train.csv`: 2,699 rows; columns `id`, `prompt_id`, `text`, `model`; 150
  prompt groups; no null/blank values, duplicate IDs, exact duplicate rows, or
  exact duplicate passages. All 18 official classes are present; 17 have 150
  rows and `claude-fable-5` has 149.
- `test.csv`: three interactive rows; observed columns are `id` and `text`, with
  no `prompt_id` column in this local sample. The split uses train only, so this
  does not affect prompt-group validation.
- `sample_submission.parquet`: observed columns are `id` and `model`.

Kaggle Rule 4(a) is verified for use of the official Competition Data, so local
training readiness is `ready`. Participant eligibility and payout conditions
remain unverified; `competition_entry_readiness` is still `blocked`.

The local five-fold prompt-group split contains 30 groups per fold, with 540
rows in four folds and 539 in the fold containing the incomplete prompt group.
The saved report confirms no prompt group crosses folds. Checksums, exact
per-fold class counts, and the split hash are in `reports/audit/audit.json` and
`reports/splits/prompt_folds.json`.

## Sources

- [Competition overview](https://www.kaggle.com/competitions/name-that-llm/overview)
- [Competition data](https://www.kaggle.com/competitions/name-that-llm/data)
- [Competition rules](https://www.kaggle.com/competitions/name-that-llm/rules)
- [Public question about Rule 2.6 and API-generated data](https://www.kaggle.com/competitions/name-that-llm/discussion/751954)

Direct page retrieval rendered only the page title in this environment. The
confirmed bullets above came from indexed excerpts of the official Kaggle pages
on the checked date. Re-open the links and update this document before the first
modeling run and again before submission.
