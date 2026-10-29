# Implementation Plan: Hierarchical Provider-Then-Model Classification

**Branch**: `008-hierarchical-provider-model` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary

Single-variable architecture experiment: flat char 2–5 LinearSVC vs a
provider router plus within-provider specialists, on the identical saved
five-fold split.

## Design

1. Derive provider labels deterministically from official model-name prefixes;
   reject unknown prefixes before any fit.
2. Per fold: fit provider pipeline on fold-train rows; fit one specialist per
   provider present on that provider's fold-train rows (constant prediction for
   single-class providers); route validation rows through provider prediction
   then the matching specialist, with a deterministic most-frequent fallback if
   a predicted provider lacks a specialist.
3. Extend `tune_from_config` candidates with `architecture: hierarchical`
   (default `flat`); validate the value explicitly; keep metric schema
   identical so ranking stays fair.
4. Add `configs/experiments/hierarchical_provider.yaml` with two candidates on
   identical features; run once on the saved split; record spread, do not
   promote.

## Risks

- Two-stage errors compound: routing mistakes are unrecoverable downstream.
- More fits per fold than flat (provider + up to six specialists); subsets are
  small so runtime should stay within minutes.
- One grouped split only; report spread, claim nothing about hidden data.
