# Specification Quality Checklist: Rhetorical and Formatting Markers

**Purpose**: Validate specification completeness before planning
**Created**: 2026-10-27
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Single inventory extension, identical grid and folds
- [x] No test/leaderboard tuning
- [x] Mandatory sections completed

## Requirement Completeness

- [x] Requirements are testable and measurable
- [x] Per-fold spread and failure behavior specified
- [x] Confirmation split performed before any promotion

## Notes

Markers are neutral on the mean (seed-42: -0.0004, seed-123: -0.0066) but
tighten spread and lift macro F1 on seed-42. The fusion advantage itself is
confirmed on both splits independently of markers.
