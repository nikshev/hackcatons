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

The following items still need participant-specific or direct terms verification:

- participant eligibility, territorial availability, and prize payout
  conditions;
- the applicable data license/terms for any downloaded or external dataset;
- any host clarification about paid API generation or publicly shared
  synthetic/API-generated training data.

The current public discussion includes a request for clarification on API-
generated passages and the reasonableness of paid generation. Freely and
publicly available external datasets are permitted by the overview; paid API
generation stays disabled until its treatment is clarified and explicitly
budgeted. The PRD keeps paid resources off by default.

## Local data state

No competition files were present in this solution folder at initialization.
The documented train schema is therefore a discovery aid, not a substitute for
auditing the downloaded files. Once data is obtained, record source, retrieval
date, license/terms, checksum, and row count in its manifest; keep raw files
unchanged and out of Git where appropriate.

## Sources

- [Competition overview](https://www.kaggle.com/competitions/name-that-llm/overview)
- [Competition data](https://www.kaggle.com/competitions/name-that-llm/data)
- [Competition rules](https://www.kaggle.com/competitions/name-that-llm/rules)
- [Public question about Rule 2.6 and API-generated data](https://www.kaggle.com/competitions/name-that-llm/discussion/751954)

Direct page retrieval rendered only the page title in this environment. The
confirmed bullets above came from indexed excerpts of the official Kaggle pages
on the checked date. Re-open the links and update this document before the first
modeling run and again before submission.
