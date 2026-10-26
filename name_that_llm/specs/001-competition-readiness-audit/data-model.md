# Data Model: Competition Readiness and Data Audit

## CompetitionRequirement

- `key`: stable identifier for one rule, deadline, eligibility condition, or
  submission constraint.
- `value`: current value when confirmed; otherwise absent.
- `status`: `confirmed`, `unverified`, `changed`, or `blocked`.
- `source_url`: official or authoritative source where available.
- `checked_at`: date the source was checked.
- `notes`: explanation, unresolved question, or consequence for this project.

## DatasetManifestEntry

- `path`: path relative to the solution root or configured source name.
- `source`: competition/source attribution.
- `retrieved_at`: date/time obtained, or explicitly unverified.
- `license_or_terms`: applicable license or terms, or explicitly unverified.
- `sha256`: checksum of the unmodified input bytes.
- `row_count`: number of parsed data rows, excluding the header.
- `columns`: ordered list of observed column names.

## FileAudit

- `manifest`: one `DatasetManifestEntry`.
- `null_counts`: number of missing values for each observed column.
- `text_length`: count, minimum, median, and maximum for the configured text
  column when present.
- `duplicate_counts`: exact duplicate row/text counts and duplicate ID count.
- `prompt_groups`: prompt-group column and number of unique groups when present.
- `label_counts`: observed target counts when the configured label column is
  present; this is not itself an official allowlist.
- `issues`: actionable validation findings, including file and field.

## AuditReport

- `created_at`: audit timestamp.
- `configuration`: path, checksum, and version for the configuration used.
- `files`: one `FileAudit` per configured input.
- `audit_status`: `passed` or `failed` based on file/schema validity.
- `training_readiness`: `ready` or `blocked`, independent of audit status.
- `blockers`: reasons training is blocked, including missing data, unresolved
  rules/terms, or unconfirmed allowed classes.

Raw rows and text are not copied into the manifest or report.
