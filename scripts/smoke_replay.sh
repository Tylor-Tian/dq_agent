#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/smoke_replay.sh
#   OUTPUT_DIR=artifacts ./scripts/smoke_replay.sh
#   OUTPUT_DIR=/tmp/dq_artifacts ./scripts/smoke_replay.sh
#
# Note:
# - This script runs demo -> validate -> schema export check -> replay --strict
# - It is intended as a regression gate for behavior drift.

OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"

echo "==> [1/4] Run demo (OUTPUT_DIR=$OUTPUT_DIR)"
OUT=$(python -m dq_agent demo --output-dir "$OUTPUT_DIR")
echo "$OUT"

echo "==> [2/4] Extract artifact paths"
REPORT_JSON=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["report_json_path"])' "$OUT")
RUN_RECORD=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_record_path"])' "$OUT")

echo "REPORT_JSON=$REPORT_JSON"
echo "RUN_RECORD=$RUN_RECORD"

echo "==> [3/4] Validate artifacts"
python -m dq_agent validate --kind report --path "$REPORT_JSON"
python -m dq_agent validate --kind run_record --path "$RUN_RECORD"

echo "==> [3.5/4] Schema commands output valid JSON"
python -m dq_agent schema --kind report | python -c 'import sys,json; json.load(sys.stdin); print("report schema ok")'
python -m dq_agent schema --kind run_record | python -c 'import sys,json; json.load(sys.stdin); print("run_record schema ok")'

echo "==> [4/4] Replay gate (strict)"
python -m dq_agent replay --run-record "$RUN_RECORD" --strict

echo "✅ smoke_replay passed"

