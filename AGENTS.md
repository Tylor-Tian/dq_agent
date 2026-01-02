# AGENTS.md

## Goal
Build a runnable, testable CLI demo: `python -m dq_agent demo` + `pytest -q`.

## Commands
- Run demo: `python -m dq_agent demo`
- Run tests: `pytest -q`

## Constraints
- Keep deps minimal (pandas, pyarrow, pydantic, pyyaml, typer, pytest).
- Default demo must run offline (no external downloads).
- Prefer small, readable modules; no heavy frameworks.

## Definition of Done
- Demo command works and generates `artifacts/<run_id>/report.json`
- `pytest -q` passes
