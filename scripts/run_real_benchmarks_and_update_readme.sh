#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper:
#   - runs Raha benchmark (dirty vs clean profiles)
#   - runs PED benchmark (dirty vs clean profiles)
#   - refreshes README benchmark block from `benchmarks/` artifacts

log() { echo -e "\n==> $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_PED="${RUN_PED:-1}"

log "Run Raha -> benchmarks/raha_*"
bash "$REPO_ROOT/scripts/run_raha_and_save.sh"

if [[ "$RUN_PED" == "1" ]]; then
  log "Run PED -> benchmarks/ped_*"
  bash "$REPO_ROOT/scripts/run_ped_and_save.sh"
else
  log "Skip PED (RUN_PED=$RUN_PED)"
fi

log "Update README benchmark block"
python "$REPO_ROOT/scripts/update_readme_benchmarks.py"

log "DONE"