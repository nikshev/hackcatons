# Audit CLI Contract

## Command

```text
python -m ntllm audit --config configs/baseline.yaml
```

## Inputs

- A YAML configuration containing configured input file paths, field roles,
  provenance metadata, and report destination.
- One or more local competition CSV files. Input bytes are read-only.
- No network access is required.

## Outputs

- A JSON audit report/manifest in the configured output location.
- A concise terminal summary with per-file status and any readiness blockers.
- Per-file quality summaries: blank-cell counts, nonblank text character
  lengths, repeated exact rows/text, and observed label counts.
- Per-file integrity summaries: prompt-group availability, unique/missing group
  counts, duplicate/missing IDs, and blank-text rows.
- No raw text or complete source rows in generated reports.

## Exit status

- `0`: the audit completed and all configured files passed schema and blocking
  row-integrity checks. Training readiness may still be `blocked`; an unavailable
  prompt-group column is reported as a non-blocking issue.
- non-zero: configuration, file access, schema, or validation failure. The
  error identifies the affected file and field and does not silently guess.

## Invariants

- Input checksums before and after audit are identical.
- Observed columns are reported; configured roles are validated against them.
- Label counts describe observed labels and do not define the official
  allowlist.
- An audit pass is not evidence that competition eligibility or data-use rules
  are satisfied.
