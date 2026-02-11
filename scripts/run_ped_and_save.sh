#!/usr/bin/env bash
set -euo pipefail

# Run PED dirty/clean benchmark and save aggregated results into repo-local `benchmarks/`.

# logs -> stderr (so stdout can be used to return paths)
log() { echo -e "\n==> $*" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${LIMIT:=20}"

run_one () {
  local profile="$1"

  log "Run PED benchmark (PROFILE_SOURCE=$profile LIMIT=$LIMIT)"

  local tmp
  tmp="$(mktemp)"

  # Show bench output to user (stderr), while keeping a copy for parsing
  PROFILE_SOURCE="$profile" LIMIT="$LIMIT" \
    bash "$REPO_ROOT/scripts/bench_ped.sh" | tee "$tmp" 1>&2

  local out=""
  # bench_ped.sh prints: "==> Out root: /path"
  out="$(awk -F': ' '/^==> Out root:/ {print $2}' "$tmp" | tail -n1 | tr -d '\r')"

  # Fallback: parse the header line "==> Bench out root : /path"
  if [[ -z "${out:-}" ]]; then
    out="$(awk -F' : ' '/^==> Bench out root/ {print $2}' "$tmp" | tail -n1 | tr -d '\r')"
  fi

  rm -f "$tmp"

  if [[ -z "${out:-}" ]]; then
    echo "ERROR: failed to parse output root from bench_ped.sh output" >&2
    exit 1
  fi

  echo "$out"
}

CLEAN_OUT="$(run_one clean)"
DIRTY_OUT="$(run_one dirty)"

log "CLEAN_OUT=$CLEAN_OUT"
log "DIRTY_OUT=$DIRTY_OUT"

log "Copy summaries into repo benchmarks/"
mkdir -p "$REPO_ROOT/benchmarks/ped_clean" "$REPO_ROOT/benchmarks/ped_dirty"

cp -f "$CLEAN_OUT/summary.md"     "$REPO_ROOT/benchmarks/ped_clean/summary.md"
cp -f "$CLEAN_OUT/summary.csv"    "$REPO_ROOT/benchmarks/ped_clean/summary.csv"
cp -f "$CLEAN_OUT/summary.jsonl"  "$REPO_ROOT/benchmarks/ped_clean/summary.jsonl"

cp -f "$DIRTY_OUT/summary.md"     "$REPO_ROOT/benchmarks/ped_dirty/summary.md"
cp -f "$DIRTY_OUT/summary.csv"    "$REPO_ROOT/benchmarks/ped_dirty/summary.csv"
cp -f "$DIRTY_OUT/summary.jsonl"  "$REPO_ROOT/benchmarks/ped_dirty/summary.jsonl"

log "Aggregate compare (dirty vs clean) -> benchmarks/ped_compare.*"
python "$REPO_ROOT/scripts/agg_bench.py" \
  --clean "$CLEAN_OUT" \
  --dirty "$DIRTY_OUT" \
  --out-md "$REPO_ROOT/benchmarks/ped_compare.md" \
  --out-json "$REPO_ROOT/benchmarks/ped_compare.json" \
  --title "PED benchmark summary (dirty vs clean profiles)"

log "DONE"
log "Saved: benchmarks/ped_clean/*, benchmarks/ped_dirty/*, benchmarks/ped_compare.*"
