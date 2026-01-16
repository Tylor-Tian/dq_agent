#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"

echo "==> [1/3] Trigger guardrail failure (max_rows=10)"
set +e
OUT=$(python -m dq_agent demo --output-dir "$OUTPUT_DIR" --max-rows 10)
EC=$?
set -e

echo "$OUT"
echo "exit_code=$EC"

if [[ "$EC" -ne 2 ]]; then
  echo "❌ expected exit code 2 for guardrail violation, got $EC"
  exit 1
fi

echo "==> [2/3] Extract artifact paths + validate"
REPORT_JSON=$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["report_json_path"])' <<<"$OUT")
RUN_RECORD=$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_record_path"])' <<<"$OUT")

echo "REPORT_JSON=$REPORT_JSON"
echo "RUN_RECORD=$RUN_RECORD"

python -m dq_agent validate --kind report --path "$REPORT_JSON"
python -m dq_agent validate --kind run_record --path "$RUN_RECORD"

echo "==> [2.5/3] Assert report.json contains non-empty error on failure"
python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); \
assert str(d.get("status","")).startswith("FAILED"), d.get("status"); \
assert d.get("error"), "expected non-empty error in report.json on failure"; \
print("status:", d.get("status")); print("error:", d.get("error")); print("✅ failure contract ok")' "$REPORT_JSON"

echo "==> [3/3] Replay gate (strict) for failure run"
python -m dq_agent replay --run-record "$RUN_RECORD" --strict

echo "✅ smoke_failure passed"
