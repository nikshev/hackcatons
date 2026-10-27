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
- `quality`: blank/whitespace null counts for each observed column, character
  length count/min/median/max for nonblank text, exact duplicate-row/text counts
  beyond the first occurrence, and observed nonblank label counts when a label
  column is present. Label counts are not an official allowlist.
- `integrity`: prompt-group availability/counts and duplicate/missing ID and
  invalid-text counts.
- `issues`: non-blocking findings such as an unavailable prompt-group column.
- `blocking_issues`: duplicate/missing IDs and invalid text that fail the audit.

## AuditReport

- `created_at`: audit timestamp.
- `configuration`: path, checksum, and version for the configuration used.
- `files`: one `FileAudit` per configured input.
- `audit_status`: `passed` or `failed` based on file/schema validity.
- `training_readiness`: `ready` or `blocked`, independent of audit status.
- `competition_entry_readiness`: `ready` or `blocked` for participant-specific
  eligibility and payout conditions; this does not block local model experiments.
- `blockers`: reasons training is blocked, including missing data, unresolved
  rules/terms, or unconfirmed allowed classes.

Raw rows and text are not copied into the manifest or report.
