# Contributing

Thanks for contributing to `dq_agent`.

## Environment

- Python 3.11+
- Linux/macOS shell (examples use `bash`)

## Local development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[test]"
```

## Run tests

```bash
python -m pytest -q
```

## Run demo

```bash
python -m dq_agent demo
```

## Add a new rule

1. Implement the check function in `dq_agent/rules/checks.py`.
2. Register it with `@register_check("your_check_name")` from `dq_agent/rules/base.py`.
3. Add/extend tests under `tests/` for behavior, edge cases, and failure modes.
4. Run `python -m pytest -q`.

## Add a new anomaly detector

1. Implement the detector in `dq_agent/anomalies/detectors.py`.
2. Register it with `@register_anomaly("your_detector_name")` from `dq_agent/anomalies/base.py`.
3. Add/extend tests under `tests/` for correctness and threshold behavior.
4. Run `python -m pytest -q`.

## Update benchmarks

Run the benchmark scripts and refresh benchmark summaries:

```bash
bash scripts/run_raha_and_save.sh
bash scripts/run_ped_and_save.sh
python scripts/update_readme_benchmarks.py
```

If you only need quick checks, use the smoke scripts:

```bash
OUTPUT_DIR=/tmp/dq_artifacts bash scripts/smoke_replay.sh
OUTPUT_DIR=/tmp/dq_artifacts bash scripts/smoke_failure.sh
```

## Pull request checklist

- Keep core behavior stable unless the PR explicitly changes behavior.
- Include tests for any functional change.
- Keep docs and examples in sync with output schema changes.
- Ensure CI passes.
