---
name: dq-agent-m7-schema-first
description: Use this skill when implementing M7 in dq_agent: enforce schema-first typed boundaries for report.json and run_record.json, add schema/validate CLI, and tests + README updates.
metadata:
  short-description: dq_agent M7 schema-first workflow
---

# dq_agent M7: Schema-first / Typed boundaries

## When to use
Use this skill for any work that:
- touches `report.json` or `run_record.json`
- adds new output fields
- adds downstream consumption requirements (dashboard/alerts)
- needs strict validation + stable contracts

## Non-negotiable constraints
- Keep existing commands working:
  - `python -m dq_agent demo`
  - `python -m dq_agent run ...`
  - `python -m dq_agent replay --run-record ...`
- Validate at write-time: if model validation fails, CLI must fail fast with a clear error.
- Prefer strictness at boundaries:
  - Pydantic v2, with stronger settings (e.g., forbid extra fields where appropriate).
- Keep changes minimal and testable.

## Required deliverables
1) Add `schema_version: 1` to:
   - report.json (top-level)
   - run_record.json (top-level)
   Write it in code (not configurable).

2) Make JSON outputs typed + validated:
   - report JSON is produced from a Pydantic model
   - run record JSON is produced from a Pydantic model

3) Add CLI commands (Typer):
   - `python -m dq_agent schema --kind report|run_record [--out PATH]`
     - prints JSON Schema from Pydantic model_json_schema()
   - `python -m dq_agent validate --kind report|run_record --path FILE`
     - exit 0 ok
     - exit 2 on validation failure
     - exit 1 on I/O or unexpected error

4) Tests (fast + hermetic):
   - Ensure demo generates report.json + run_record.json that pass validate
   - Ensure `schema` outputs valid JSON (json.load ok)

5) README update:
   - Document schema_version
   - Document schema/validate commands

## Implementation checklist
- Identify/extend existing Pydantic models:
  - report: prefer `dq_agent/report/schema.py`
  - run_record: new module ok, but keep it clearly named and imported
- Writers should:
  - build model -> validate -> model_dump -> write json
- Keep replay comparisons stable:
  - avoid introducing volatile fields into report.json unless necessary

## Acceptance commands
- `python -m pip install -e ".[test]"`
- `python -m dq_agent demo`
- `python -m dq_agent validate --kind report --path artifacts/<run_id>/report.json`
- `python -m dq_agent validate --kind run_record --path artifacts/<run_id>/run_record.json`
- `python -m dq_agent schema --kind report | python -c "import sys,json; json.load(sys.stdin); print('ok')"`
- `python -m dq_agent schema --kind run_record | python -c "import sys,json; json.load(sys.stdin); print('ok')"`
- `python -m pytest -q`

