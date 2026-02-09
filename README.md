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

Use idempotency keys to keep outputs stable across reruns:

```bash
python -m dq_agent demo --idempotency-key demo-001 --idempotency-mode reuse
```

It prints something like:

```json
{"report_json_path": "artifacts/<run_id>/report.json", "report_md_path": "artifacts/<run_id>/report.md", "run_record_path": "artifacts/<run_id>/run_record.json", "trace_path": "artifacts/<run_id>/trace.jsonl"}
```

## Outputs

A demo/run creates a new run directory:

- `artifacts/<run_id>/report.json`
- `artifacts/<run_id>/report.md`
- `artifacts/<run_id>/run_record.json` (replayable run record)
- `artifacts/<run_id>/trace.jsonl` (run trace events, NDJSON)

`report.json` and `run_record.json` include `schema_version: 1` at the top level. `report.json` includes an
`observability` section with timing and rule/anomaly counts.

Sample outputs (committed for quick preview):

- `examples/report.md`
- `examples/report.json`

> Note: `artifacts/` is ignored by git. Use `examples/` if you want committed sample reports.

### Trace file

Each run writes a minimal trace log to `trace.jsonl` (newline-delimited JSON). The trace includes `run_start`,
`stage_start`, `stage_end`, and `run_end` events with elapsed milliseconds since start.

Inspect it with standard shell tools:

```bash
tail -n +1 artifacts/<run_id>/trace.jsonl | head
```

## Run on your own data

```bash
python -m dq_agent run --data path/to/table.parquet --config path/to/rules.yml
```

Supported:
- Data: CSV / Parquet
- Config: YAML / JSON

Guardrails (optional safety limits):

```bash
python -m dq_agent run \
  --data path/to/table.parquet \
  --config path/to/rules.yml \
  --max-input-mb 50 \
  --max-rows 500000 \
  --max-cols 200 \
  --max-rules 2000 \
  --max-anomalies 200 \
  --max-wall-time-s 30
```

Fail the run when issues reach a severity threshold:

```bash
python -m dq_agent run --data path/to/table.parquet --config path/to/rules.yml --fail-on ERROR
```

Exit code behavior:
- `0`: run completed without triggering `--fail-on`
- `1`: I/O or config parsing errors (missing/unreadable files, invalid config)
- `2`: guardrail violation or schema validation failures, plus `--fail-on` severity

Idempotency controls:

```bash
python -m dq_agent run \
  --data path/to/table.parquet \
  --config path/to/rules.yml \
  --idempotency-key run-001 \
  --idempotency-mode reuse
```

Modes:
- `reuse` (default): if `report.json` + `run_record.json` already exist for the key, return their paths and skip the pipeline.
- `overwrite`: re-run and overwrite artifacts in the deterministic run directory.
- `fail`: return a structured error (`idempotency_conflict`) with exit code 2 when artifacts exist.

### Failure contract (typed errors)

Failures are first-class artifacts. When a run fails, both `report.json` and `run_record.json` include an `error`
object, and the CLI prints a JSON payload with the error and any written output paths.

Error schema fields:
- `type`: `guardrail_violation` | `io_error` | `config_error` | `schema_validation_error` | `internal_error` | `idempotency_conflict` | `regression`
- `code`: short, stable machine code (e.g., `max_rows`, `data_not_found`, `invalid_config`)
- `message`: stable human-readable message
- `is_retryable`: boolean retry hint
- `suggested_next_step`: short actionable hint
- `details`: optional, small JSON object

Failure output example:

```json
{
  "error": {
    "type": "guardrail_violation",
    "code": "max_rows",
    "message": "Row count 2500 exceeds limit 10.",
    "is_retryable": false,
    "suggested_next_step": "Adjust guardrail limits or reduce input size before retrying.",
    "details": {"limit": 10, "observed": 2500}
  },
  "report_json_path": "artifacts/<run_id>/report.json",
  "run_record_path": "artifacts/<run_id>/run_record.json"
}
```

See all CLI options:

```bash
python -m dq_agent --help
python -m dq_agent run --help
```

## Shadow (baseline vs candidate)

Compare two configs against the same data before promoting changes:

```bash
python -m dq_agent shadow \
  --data path/to/table.parquet \
  --baseline-config path/to/baseline.yml \
  --candidate-config path/to/candidate.yml \
  --fail-on-regression
```

Outputs are grouped under one shadow run directory:

```
artifacts/<shadow_run_id>/
  baseline/<baseline_run_id>/...
  candidate/<candidate_run_id>/...
  shadow_diff.json
```

The `--fail-on-regression` flag exits with code `2` when the candidate is worse than the baseline, while still
writing `shadow_diff.json` and both run artifacts. The CLI prints a single JSON payload with the baseline and
candidate report paths plus any typed error details.

## Replay a run

After a run completes, use the `run_record.json` to replay deterministically:

```bash
python -m dq_agent replay --run-record artifacts/<run_id>/run_record.json --strict
```

## Resume a run with checkpoints

Each run writes a `checkpoint.json` alongside the other artifacts. If artifacts go missing, resume can repair them:

```bash
python -m dq_agent resume --run-dir artifacts/<run_id>
```

Example flow:

```bash
python -m dq_agent demo --output-dir artifacts
rm artifacts/<run_id>/report.md
python -m dq_agent resume --run-dir artifacts/<run_id>
python -m dq_agent validate --kind report --path artifacts/<run_id>/report.json
python -m dq_agent validate --kind run_record --path artifacts/<run_id>/run_record.json
python -m dq_agent validate --kind checkpoint --path artifacts/<run_id>/checkpoint.json
```

## Schema + validation

Print JSON Schema for each output:

```bash
python -m dq_agent schema --kind report
python -m dq_agent schema --kind run_record
python -m dq_agent schema --kind checkpoint
```

Validate a generated output:

```bash
python -m dq_agent validate --kind report --path artifacts/<run_id>/report.json
python -m dq_agent validate --kind run_record --path artifacts/<run_id>/run_record.json
python -m dq_agent validate --kind checkpoint --path artifacts/<run_id>/checkpoint.json
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
      - string_noise: { contains: ["*", "''"], max_rate: 0.0 }
    anomalies:
      - missing_rate: { max_rate: 0.02 }

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
- `string_noise` (substring / regex pattern based)

Statistical anomalies:
- `outlier_mad` (robust outliers via MAD z-score)
- `missing_rate` (null-rate anomaly)

## Benchmarks (real labeled datasets)

We evaluate dq_agent as an **error detector** on datasets that have real labels
via paired `dirty.csv` / `clean.csv`.

Metrics:
- **cell-level** precision / recall / F1 (detect wrong cells)
- **row-level** precision / recall / F1 (detect rows containing any wrong cell)

Reproduce (Raha):

```bash
bash scripts/run_raha_and_save.sh
```

Notes:
- Benchmark harness: `scripts/eval_dirty_clean.py` (auto-generates a config from profile data).
- The `string_noise` check is **enabled by default** for open-domain string columns in the harness
  (use `--no-string-noise` to ablate).

Reproduce (PED):

```bash
bash scripts/run_ped_and_save.sh
```

To refresh the README benchmark block from `benchmarks/` artifacts:

```bash
python scripts/update_readme_benchmarks.py
```

<!-- BENCHMARKS:START -->
### Raha (7 datasets; dirty vs clean profiles)

| profile | datasets | macro_cell_f1 | micro_cell_f1 | macro_row_f1 | micro_row_f1 | cell_tp/fp/fn | row_tp/fp/fn |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 7 | 0.684955 | 0.227177 | 0.824243 | 0.209508 | (15957, 105323, 3244) | (12296, 91535, 1253) |
| dirty | 7 | 0.358737 | 0.058786 | 0.588284 | 0.142788 | (3784, 104080, 17091) | (8081, 91389, 5638) |

Full breakdown: `benchmarks/raha_compare.md`.

### Raha string-noise ablation (patterns: `*`, `''`)

| metric | base | union | Δ |
|---|---:|---:|---:|
| macro_cell_f1 | 0.760961 | 0.817760 | 0.056798 |
| micro_cell_f1 | 0.806936 | 0.841877 | 0.034940 |
| macro_row_f1  | 0.859008 | 0.915870 | 0.056862 |
| micro_row_f1  | 0.876437 | 0.925301 | 0.048864 |

Largest per-dataset gain (from the committed compare file):

| dataset | base cell_f1 | union cell_f1 | Δ | base row_f1 | union row_f1 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| raha/tax | 0.319744 | 0.718032 | 0.398288 | 0.324004 | 0.722462 | 0.398457 |

Full breakdown: `benchmarks/raha_noise_union/compare.md`.

### PED (additional dirty/clean datasets)

Not generated yet. Run:

```bash
bash scripts/run_ped_and_save.sh
```
<!-- BENCHMARKS:END -->

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
