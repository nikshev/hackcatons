# Hackathons workspace

This is a workspace for independent solutions, not one product. Each top-level
solution folder has its own PRD, `.specify/` constitution, `specs/`, code,
configuration, and documentation. Run SDD initialization from the selected
solution folder and keep its requirements and implementation inside that
folder. Do not reuse another solution's specs, plans, or artifacts.

The folders share this workspace's Git repository; do not create nested `.git`
directories. The root `sdd-init/` is a local copy of initialization tooling and
is ignored by Git.

Workspace-level commits validate each initialized solution through its own
`scripts/trace.py`; each validator scans only the solution folder that contains
it.
