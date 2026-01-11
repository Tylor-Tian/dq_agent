# dq_agent

[![CI](https://github.com/Tylor-Tian/dq_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Tylor-Tian/dq_agent/actions/workflows/ci.yml)

A minimal, reproducible **Data Quality + Anomaly Detection CLI** (offline demo with synthetic data).

- **Input**: a table (CSV / Parquet) + rules config (YAML / JSON)
- **Output**: a machine-readable `report.json` + a human-readable `report.md`
- **Demo**: `python -m dq_agent demo`

## Quickstart (Demo)

Requirements: **Python 3.11+**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[test]"
```

Run the demo:

```bash
python -m dq_agent demo
```

It prints something like:

```json
{"report_json_path": "artifacts/<run_id>/report.json", "report_md_path": "artifacts/<run_id>/report.md", "run_record_path": "artifacts/<run_id>/run_record.json"}
```

## Outputs

A demo/run creates a new run directory:

- `artifacts/<run_id>/report.json`
- `artifacts/<run_id>/report.md`
- `artifacts/<run_id>/run_record.json` (replayable run record)

`report.json` and `run_record.json` include `schema_version: 1` at the top level. `report.json` includes an
`observability` section with timing and rule/anomaly counts.

Sample outputs (committed for quick preview):

- `examples/report.md`
- `examples/report.json`

> Note: `artifacts/` is ignored by git. Use `examples/` if you want committed sample reports.

## Run on your own data

```bash
python -m dq_agent run --data path/to/table.parquet --config path/to/rules.yml
```

Supported:
- Data: CSV / Parquet
- Config: YAML / JSON

Fail the run when issues reach a severity threshold:

```bash
python -m dq_agent run --data path/to/table.parquet --config path/to/rules.yml --fail-on ERROR
```

Exit code behavior:
- `0`: run completed without triggering `--fail-on`
- `1`: config parsing or missing files
- `2`: run completed but issues/anomalies met the `--fail-on` severity

See all CLI options:

```bash
python -m dq_agent --help
python -m dq_agent run --help
```

## Replay a run

After a run completes, use the `run_record.json` to replay deterministically:

```bash
python -m dq_agent replay --run-record artifacts/<run_id>/run_record.json --strict
```

## Schema + validation

Print JSON Schema for each output:

```bash
python -m dq_agent schema --kind report
python -m dq_agent schema --kind run_record
```

Validate a generated output:

```bash
python -m dq_agent validate --kind report --path artifacts/<run_id>/report.json
python -m dq_agent validate --kind run_record --path artifacts/<run_id>/run_record.json
```

## Config format (YAML)

Minimal example:

```yaml
version: 1
dataset:
  name: demo_orders
  primary_key: [order_id]
  time_column: created_at

columns:
  order_id:
    type: string
    required: true
    checks:
      - unique: true

  user_id:
    type: string
    required: true
    checks:
      - not_null: { max_null_rate: 0.01 }
    anomalies:
      - missing_rate: { max_null_rate: 0.02 }

  amount:
    type: float
    required: true
    checks:
      - range: { min: 0, max: 10000 }
    anomalies:
      - outlier_mad: { z: 6.0 }

  status:
    type: string
    required: true
    checks:
      - allowed_values: { values: ["PAID","REFUND","CANCEL","PENDING"] }
```

Demo config lives at: `dq_agent/resources/demo_rules.yml`.

## What it checks

Deterministic rules:
- `not_null`
- `unique`
- `range`
- `allowed_values`

Statistical anomalies:
- `outlier_mad` (robust outliers via MAD z-score)
- `missing_rate` (null-rate anomaly)

## Dev / Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

If you see:
- `ModuleNotFoundError: typer` / `No module named pytest`

You almost certainly forgot to activate venv:

```bash
source .venv/bin/activate
```

## Project layout (high-level)

- `dq_agent/cli.py` – Typer CLI (`run`, `demo`)
- `dq_agent/loader.py` – CSV/Parquet loader
- `dq_agent/config.py` – config loading (YAML/JSON)
- `dq_agent/contract.py` – contract validation
- `dq_agent/rules/` – deterministic checks (registry-based)
- `dq_agent/anomalies/` – anomaly detectors (registry-based)
- `dq_agent/report/` – JSON + Markdown report writers
- `tests/` – unit + integration tests

## Spec / Roadmap

Full design doc: `A0_SPEC.md`

## License

Apache-2.0
