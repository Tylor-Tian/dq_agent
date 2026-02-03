#!/usr/bin/env bash
set -euo pipefail

# logs -> stderr (so command substitution only captures pure stdout)
log() { echo -e "\n==> $*" >&2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${LIMIT:=10}"
: "${FAIL_ON:=INFO}"

run_one () {
  local profile="$1"

  log "Run raha bench: PROFILE_SOURCE=$profile LIMIT=$LIMIT FAIL_ON=$FAIL_ON"

  local tmp
  tmp="$(mktemp)"

  # IMPORTANT:
  # - bench_raha.sh output goes to stdout
  # - but we're inside $(...), so stdout would be captured
  # => redirect tee's stdout to stderr, while still saving to tmp for parsing
  PROFILE_SOURCE="$profile" LIMIT="$LIMIT" FAIL_ON="$FAIL_ON" \
    bash scripts/bench_raha.sh | tee "$tmp" 1>&2

  local out
  out="$(awk -F': ' '/^==> Bench out root/ {print $2}' "$tmp" | tail -n1 | tr -d '\r')"
  rm -f "$tmp"

  if [[ -z "${out:-}" ]]; then
    echo "ERROR: cannot parse 'Bench out root' from bench output" >&2
    exit 1
  fi

  # ONLY this line goes to stdout (captured by CLEAN_OUT/DIRTY_OUT)
  echo "$out"
}

CLEAN_OUT="$(run_one clean)"
DIRTY_OUT="$(run_one dirty)"

log "CLEAN_OUT=$CLEAN_OUT"
log "DIRTY_OUT=$DIRTY_OUT"

mkdir -p benchmarks/raha_clean benchmarks/raha_dirty

cp -f "$CLEAN_OUT/summary.md"  benchmarks/raha_clean/summary.md
cp -f "$CLEAN_OUT/summary.csv" benchmarks/raha_clean/summary.csv

cp -f "$DIRTY_OUT/summary.md"  benchmarks/raha_dirty/summary.md
cp -f "$DIRTY_OUT/summary.csv" benchmarks/raha_dirty/summary.csv

log "Aggregate macro/micro + compare"
python scripts/agg_bench.py \
  --clean "$CLEAN_OUT" \
  --dirty "$DIRTY_OUT" \
  --out-md benchmarks/raha_compare.md \
  --out-json benchmarks/raha_compare.json

log "DONE"
log "Saved:"
log "  benchmarks/raha_clean/summary.*"
log "  benchmarks/raha_dirty/summary.*"
log "  benchmarks/raha_compare.md"
log "  benchmarks/raha_compare.json"
