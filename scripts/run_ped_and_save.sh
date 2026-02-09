#!/usr/bin/env bash
set -euo pipefail

# Run PED dirty/clean benchmark and save the aggregated results into repo-local
# `benchmarks/` (committable artifacts).
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/run_ped_and_save.sh

log() { echo -e "\n==> $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIMIT="${LIMIT:-20}"

log "Run PED benchmark (PROFILE_SOURCE=clean)"
OUT_CLEAN=$(LIMIT="$LIMIT" PROFILE_SOURCE=clean bash "$REPO_ROOT/scripts/bench_ped.sh" \
  | grep -E '^Out root:' | tail -n1 | awk '{print $3}')

log "Run PED benchmark (PROFILE_SOURCE=dirty)"
OUT_DIRTY=$(LIMIT="$LIMIT" PROFILE_SOURCE=dirty bash "$REPO_ROOT/scripts/bench_ped.sh" \
  | grep -E '^Out root:' | tail -n1 | awk '{print $3}')

if [[ -z "$OUT_CLEAN" || -z "$OUT_DIRTY" ]]; then
  echo "ERROR: failed to capture output roots from bench_ped.sh" >&2
  exit 1
fi

log "Copy summaries into repo benchmarks/"
mkdir -p "$REPO_ROOT/benchmarks/ped_clean" "$REPO_ROOT/benchmarks/ped_dirty"

cp -f "$OUT_CLEAN/summary.md"   "$REPO_ROOT/benchmarks/ped_clean/summary.md"
cp -f "$OUT_CLEAN/summary.csv"  "$REPO_ROOT/benchmarks/ped_clean/summary.csv"
cp -f "$OUT_CLEAN/summary.jsonl" "$REPO_ROOT/benchmarks/ped_clean/summary.jsonl"

cp -f "$OUT_DIRTY/summary.md"   "$REPO_ROOT/benchmarks/ped_dirty/summary.md"
cp -f "$OUT_DIRTY/summary.csv"  "$REPO_ROOT/benchmarks/ped_dirty/summary.csv"
cp -f "$OUT_DIRTY/summary.jsonl" "$REPO_ROOT/benchmarks/ped_dirty/summary.jsonl"

log "Aggregate compare (dirty vs clean) -> benchmarks/ped_compare.*"
python "$REPO_ROOT/scripts/agg_bench.py" \
  --clean "$OUT_CLEAN" \
  --dirty "$OUT_DIRTY" \
  --out-md "$REPO_ROOT/benchmarks/ped_compare.md" \
  --out-json "$REPO_ROOT/benchmarks/ped_compare.json" \
  --title "PED benchmark summary (dirty vs clean profiles)"

log "DONE"
log "Saved: benchmarks/ped_clean/*, benchmarks/ped_dirty/*, benchmarks/ped_compare.*"