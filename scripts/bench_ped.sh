#!/usr/bin/env bash
set -euo pipefail

# Benchmark dq_agent on dirty/clean datasets from twinklelittlestars/PED.
#
# PED repo provides multiple dataset folders under `data/`, each containing:
#   - dirty.csv
#   - clean.csv
#
# Outputs:
#   $OUT_ROOT/summary.jsonl   (one {"metrics_json": "..."} per dataset)
#   $OUT_ROOT/summary.csv     (flat table)
#   $OUT_ROOT/summary.md      (markdown table)
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/bench_ped.sh
#
# Optional knobs:
#   LIMIT=20 PROFILE_SOURCE=clean FAIL_ON=INFO bash scripts/bench_ped.sh
#   LIMIT=20 PROFILE_SOURCE=dirty bash scripts/bench_ped.sh

log() { echo -e "\n==> $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BENCH_ROOT="${BENCH_ROOT:-$HOME/dq_benchmarks}"
SRC_ROOT="$BENCH_ROOT/src"
OUT_ROOT="$BENCH_ROOT/out/ped_$(date +%Y%m%d_%H%M%S)"

PED_REPO="$SRC_ROOT/ped"
PED_REPO_URL="${PED_REPO_URL:-https://github.com/twinklelittlestars/PED.git}"

LIMIT="${LIMIT:-20}"
PROFILE_SOURCE="${PROFILE_SOURCE:-clean}"   # clean|dirty
MAX_DOMAIN="${MAX_DOMAIN:-2000}"
NULL_SLACK="${NULL_SLACK:-0.01}"
NUMERIC_SUCCESS_THRESHOLD="${NUMERIC_SUCCESS_THRESHOLD:-0.98}"
FAIL_ON="${FAIL_ON:-INFO}"

mkdir -p "$SRC_ROOT" "$OUT_ROOT"

log "Repo root      : $REPO_ROOT"
log "Bench src root : $SRC_ROOT"
log "Bench out root : $OUT_ROOT"
log "Dataset source : $PED_REPO_URL"
log "LIMIT=$LIMIT PROFILE_SOURCE=$PROFILE_SOURCE FAIL_ON=$FAIL_ON"

# Clone/update dataset repo
if [[ -d "$PED_REPO/.git" ]]; then
  log "Update PED repo"
  git -C "$PED_REPO" pull --ff-only
else
  log "Clone PED repo"
  git clone --depth 1 "$PED_REPO_URL" "$PED_REPO"
fi

DATASETS_DIR="$PED_REPO/data"
if [[ ! -d "$DATASETS_DIR" ]]; then
  echo "ERROR: datasets dir not found at: $DATASETS_DIR" >&2
  exit 1
fi

log "Discover dataset pairs under: $DATASETS_DIR"
mapfile -t DIRS < <(
  find "$DATASETS_DIR" -type f -name "dirty.csv" -print0 \
    | xargs -0 -n1 dirname \
    | sort -u
)

log "Found ${#DIRS[@]} candidate datasets"

SUMMARY_JSONL="$OUT_ROOT/summary.jsonl"
: > "$SUMMARY_JSONL"

FAIL_LOG="$OUT_ROOT/failures.log"
: > "$FAIL_LOG"

count=0
ok=0
for d in "${DIRS[@]}"; do
  dirty="$d/dirty.csv"
  clean="$d/clean.csv"

  [[ -f "$dirty" && -f "$clean" ]] || continue

  count=$((count+1))
  if [[ "$count" -gt "$LIMIT" ]]; then
    break
  fi

  rel="${d#$DATASETS_DIR/}"
  safe_name="${rel//\//__}"

  out="$OUT_ROOT/$safe_name"
  log "[$count/$LIMIT] Evaluate: $rel"

  set +e
  meta="$(python "$REPO_ROOT/scripts/eval_dirty_clean.py" \
    --dirty "$dirty" \
    --clean "$clean" \
    --out "$out" \
    --dataset-name "ped/$rel" \
    --profile-source "$PROFILE_SOURCE" \
    --max-domain "$MAX_DOMAIN" \
    --null-slack "$NULL_SLACK" \
    --numeric-success-threshold "$NUMERIC_SUCCESS_THRESHOLD" \
    --fail-on "$FAIL_ON" \
    2>&1
  )"
  ec=$?
  set -e

  if [[ "$ec" -ne 0 ]]; then
    echo "[FAIL] $rel ec=$ec" | tee -a "$FAIL_LOG" >&2
    echo "$meta" >> "$FAIL_LOG"
    continue
  fi

  last_line="$(echo "$meta" | tail -n 1)"
  echo "$last_line" >> "$SUMMARY_JSONL"
  ok=$((ok+1))
done

log "Finished. attempted=$count success=$ok"
log "Summary jsonl: $SUMMARY_JSONL"
log "Failures log : $FAIL_LOG"

if [[ "$ok" -eq 0 ]]; then
  echo "ERROR: No successful runs to aggregate. See $FAIL_LOG" >&2
  exit 1
fi

log "Aggregate summary -> summary.csv + summary.md"
python - "$SUMMARY_JSONL" "$OUT_ROOT/summary.csv" "$OUT_ROOT/summary.md" <<'PY'
import csv, json, math, os, sys

summary_jsonl, out_csv, out_md = sys.argv[1], sys.argv[2], sys.argv[3]

def is_json_line(s: str) -> bool:
    s = s.strip()
    return s.startswith("{") and s.endswith("}")

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not is_json_line(s):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("dataset"):
                rows.append(obj)
    return rows

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6g}"
    return str(v)

rows = load_jsonl(summary_jsonl)
if not rows:
    raise SystemExit(f"ERROR: parsed 0 JSON rows from {summary_jsonl}.")

csv_fields = ["dataset","rows","cols","profile","truth_err_cells","predicted","cell_f1","row_f1","truncated","metrics_path"]
md_fields  = ["dataset","rows","cols","profile","truth_err_cells","predicted","cell_f1","row_f1","truncated"]

for r in rows:
    for k in csv_fields:
        r.setdefault(k, None)

rows.sort(key=lambda r: str(r.get("dataset")))
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(csv_fields)
    for r in rows:
        w.writerow([fmt(r.get(k)) for k in csv_fields])

lines = []
lines.append("# dq_agent benchmark (PED dirty/clean)\n")
lines.append("| " + " | ".join(md_fields) + " |")
lines.append("|" + "|".join(["---"] + ["---:"]*(len(md_fields)-1)) + "|")
for r in rows:
    lines.append("| " + " | ".join(fmt(r.get(k)) for k in md_fields) + " |")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("wrote:", out_csv)
print("wrote:", out_md)
PY

log "DONE"
log "Out root: $OUT_ROOT"
log "Summary:  $OUT_ROOT/summary.md"
