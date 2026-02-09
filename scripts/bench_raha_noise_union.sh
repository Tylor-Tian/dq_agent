#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PROFILE_SOURCE=clean FAIL_ON=INFO bash scripts/bench_raha_noise_union.sh
# Env:
#   PROFILE_SOURCE=clean|dirty   (default: clean)
#   FAIL_ON=INFO|WARN|ERROR      (default: INFO)
#   RAHA_SRC=/path/to/raha/datasets (optional)
#   OUT_ROOT=/path/to/output (optional)

log() { echo -e "\n==> $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# best-effort venv activate (ok if already activated)
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PROFILE_SOURCE="${PROFILE_SOURCE:-clean}"
FAIL_ON="${FAIL_ON:-INFO}"

RAHA_SRC="${RAHA_SRC:-/root/dq_benchmarks/src/raha/datasets}"
OUT_ROOT="${OUT_ROOT:-/root/dq_benchmarks/out/raha_noise_union_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

SUMMARY_JSONL="$OUT_ROOT/summary_union.jsonl"
SUMMARY_CSV="$OUT_ROOT/summary_union.csv"
: > "$SUMMARY_JSONL"

log "OUT_ROOT=$OUT_ROOT"
log "PROFILE_SOURCE=$PROFILE_SOURCE  FAIL_ON=$FAIL_ON"
log "patterns: '*' and \"''\""
log "RAHA_SRC=$RAHA_SRC"

if [[ ! -d "$RAHA_SRC" ]]; then
  echo "ERROR: RAHA dataset dir not found: $RAHA_SRC" >&2
  echo "Hint: run scripts/bench_raha.sh once to clone raha, or set RAHA_SRC." >&2
  exit 1
fi

mapfile -t DATASETS < <(find "$RAHA_SRC" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
log "found datasets: ${#DATASETS[@]}"
printf ' - %s\n' "${DATASETS[@]}"

for ds in "${DATASETS[@]}"; do
  log "[$ds] eval_dirty_clean (BASE dq_agent)"

  DS_OUT="$OUT_ROOT/$ds"
  DS_BASE="$DS_OUT/base"
  mkdir -p "$DS_BASE"

  # 关键修复点：先 mkdir，再 tee 写 summary.json（避免 > 目录不存在 直接炸）
  python scripts/eval_dirty_clean.py \
    --dirty "$RAHA_SRC/$ds/dirty.csv" \
    --clean "$RAHA_SRC/$ds/clean.csv" \
    --out  "$DS_BASE" \
    --dataset-name "raha/$ds" \
    --profile-source "$PROFILE_SOURCE" \
    --no-string-noise \
    --fail-on "$FAIL_ON" \
    --max-domain 2000 \
    --min-domain-coverage 0.98 \
    --numeric-success-threshold 0.98 \
    | tee "$DS_BASE/summary.json"

  METRICS="$DS_BASE/metrics.json"
  if [[ ! -f "$METRICS" ]]; then
    echo "[FAIL] missing metrics.json for $ds at $METRICS" >&2
    echo "Check: $DS_BASE/dq_raw.log and $DS_BASE/summary.json" >&2
    exit 1
  fi

  log "[$ds] probe_string_noise_union (BASE/NOISE/UNION)"
  python scripts/probe_string_noise_union.py \
    --metrics "$METRICS" \
    --dirty "$RAHA_SRC/$ds/dirty.csv" \
    --clean "$RAHA_SRC/$ds/clean.csv" \
    --contains "*" \
    --contains "''" \
    --numeric-success-threshold 0.98 \
    | tee "$DS_OUT/noise_union.log"

  log "[$ds] extract UNION numbers -> summary_union.jsonl"
  python - <<'PY' "$ds" "$DS_OUT/noise_union.log" >> "$SUMMARY_JSONL"
import ast, json, re, sys

ds, logp = sys.argv[1], sys.argv[2]
lines = open(logp, "r", encoding="utf-8").read().splitlines()

# find UNION block
start = None
for i, line in enumerate(lines):
    if line.strip() == "=== UNION (BASE ∪ NOISE) ===":
        start = i
        break
if start is None:
    raise SystemExit(f"no UNION section in {logp}")

blk = lines[start:start+12]

def parse_dict(prefix: str):
    for line in blk:
        if line.startswith(prefix):
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            return ast.literal_eval(m.group(0))
    return None

cell = parse_dict("cell: ")
row  = parse_dict("row : ")

if not cell or not row:
    raise SystemExit(f"cannot parse UNION cell/row in {logp}")

out = {"dataset": f"raha/{ds}", "union_cell": cell, "union_row": row}
print(json.dumps(out, ensure_ascii=False))
PY

done

log "write CSV: $SUMMARY_CSV"
python - <<'PY' "$SUMMARY_JSONL" "$SUMMARY_CSV"
import json, sys
import pandas as pd

jsonl, out_csv = sys.argv[1], sys.argv[2]
rows=[]
for line in open(jsonl, "r", encoding="utf-8"):
    d=json.loads(line)
    c=d["union_cell"]; r=d["union_row"]
    rows.append({
        "dataset": d["dataset"],
        "cell_tp": c["tp"], "cell_fp": c["fp"], "cell_fn": c["fn"],
        "cell_precision": c["precision"], "cell_recall": c["recall"], "cell_f1": c["f1"],
        "row_tp": r["tp"], "row_fp": r["fp"], "row_fn": r["fn"],
        "row_precision": r["precision"], "row_recall": r["recall"], "row_f1": r["f1"],
    })
df=pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(df.sort_values("cell_f1").head(10))
PY

log "DONE"
log "OUT_ROOT=$OUT_ROOT"
log "UNION JSONL: $SUMMARY_JSONL"
log "UNION CSV : $SUMMARY_CSV"
