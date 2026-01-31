#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "\n==> $*"; }

# -----------------------------
# Paths
# -----------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK_DIR:-$HOME/dq_realdata/adult}"

DATA_ZIP="$WORK/adult.zip"
DATA_DIR="$WORK"
PARQUET="$WORK/adult_with_id.parquet"
INJECTED="$WORK/adult_injected.parquet"
RULES="$WORK/adult_rules.yml"

EVAL_DIR="$WORK/eval"
BASE_DIR="$EVAL_DIR/baseline"
INJ_DIR="$EVAL_DIR/injected"

BASE_OUT="$BASE_DIR/out"
INJ_OUT="$INJ_DIR/out"

BASE_RAW="$BASE_DIR/baseline_raw.log"
INJ_RAW="$INJ_DIR/injected_raw.log"

BASE_EC_FILE="$BASE_DIR/baseline_exit_code.txt"
INJ_EC_FILE="$INJ_DIR/injected_exit_code.txt"

SUMMARY_JSON="$EVAL_DIR/eval_summary.json"

# -----------------------------
# Helpers
# -----------------------------
ensure_unzip() {
  if command -v unzip >/dev/null 2>&1; then
    return 0
  fi
  log "unzip not found; installing (apt-get)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y unzip
    return 0
  fi
  echo "ERROR: unzip not installed and apt-get not available" >&2
  exit 1
}

download_and_unzip() {
  mkdir -p "$WORK"
  cd "$WORK"

  log "Download adult.zip (if missing)"
  if [[ -f "$DATA_ZIP" ]]; then
    log "adult.zip already exists: $DATA_ZIP"
  else
    curl -fL -o "$DATA_ZIP" "https://archive.ics.uci.edu/static/public/2/adult.zip"
  fi

  log "Ensure unzip exists"
  ensure_unzip

  log "Unzip dataset into $DATA_DIR"
  unzip -o "$DATA_ZIP" -d "$DATA_DIR" >/dev/null
}

build_parquet_if_missing() {
  if [[ -f "$PARQUET" ]]; then
    log "adult_with_id.parquet already exists: $PARQUET"
    return 0
  fi

  log "Build adult_with_id.parquet"
  python - <<'PY' "$WORK"
import os, sys
import pandas as pd

work = sys.argv[1]
train_path = os.path.join(work, "adult.data")
test_path  = os.path.join(work, "adult.test")

cols = [
  "age","workclass","fnlwgt","education","education_num","marital_status",
  "occupation","relationship","race","sex","capital_gain","capital_loss",
  "hours_per_week","native_country","income"
]

train = pd.read_csv(
    train_path,
    header=None,
    names=cols,
    sep=",",
    skipinitialspace=True,
    na_values=["?"],
    comment="|",
)
test = pd.read_csv(
    test_path,
    header=None,
    names=cols,
    sep=",",
    skipinitialspace=True,
    na_values=["?"],
    comment="|",
    skiprows=1,  # adult.test 第一行通常是注释
)

# adult.test 的 income 常见尾巴是 "."
test["income"] = test["income"].astype("string").str.replace(r"\.$", "", regex=True)

train["split"] = "train"
test["split"]  = "test"

df = pd.concat([train, test], ignore_index=True)

# 统一 strip（避免 " Male"）
for c in df.columns:
    if str(df[c].dtype).startswith(("object", "string")):
        df[c] = df[c].astype("string").str.strip()

df.insert(0, "row_id", range(1, len(df) + 1))

out = os.path.join(work, "adult_with_id.parquet")
df.to_parquet(out, index=False)
print("wrote:", out, "shape=", df.shape)
PY
}

write_rules_deterministic() {
  log "Write adult_rules.yml (overwrite for determinism)"
  cat > "$RULES" <<'YAML'
version: 1

dataset:
  name: adult
  primary_key: [row_id]

columns:
  age:
    type: int
    required: true
    checks:
      - range: { min: 1, max: 99 }
    anomalies:
      - outlier_mad: { z: 6.0 }

  workclass:
    type: string
    required: false
    checks:
      - not_null: { max_null_rate: 0.06 }
    anomalies:
      - missing_rate: { max_rate: 0.06 }

  occupation:
    type: string
    required: false
    checks:
      - not_null: { max_null_rate: 0.06 }
    anomalies:
      - missing_rate: { max_rate: 0.06 }

  native_country:
    type: string
    required: false
    checks:
      - not_null: { max_null_rate: 0.02 }
    anomalies:
      - missing_rate: { max_rate: 0.02 }

  sex:
    type: string
    required: true
    checks:
      - allowed_values: { values: ["Male", "Female"] }

  income:
    type: string
    required: true
    checks:
      - allowed_values: { values: ["<=50K", ">50K"] }

  split:
    type: string
    required: true
    checks:
      - allowed_values: { values: ["train", "test"] }

report:
  sample_rows: 20
YAML
}

config_sanity_check() {
  log "Config sanity check"
  python - <<'PY' "$RULES"
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
cols = cfg.get("columns") or {}
print("ok: rules parsed; columns=", len(cols))
PY
}

build_injected_if_missing() {
  if [[ -f "$INJECTED" ]]; then
    log "adult_injected.parquet already exists: $INJECTED"
    return 0
  fi

  log "Build adult_injected.parquet (deterministic injections)"
  python - <<'PY' "$PARQUET" "$INJECTED"
import sys
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)

rng = np.random.default_rng(0)

# 1) age extreme outlier: 50 rows -> 999
idx = rng.choice(len(df), size=50, replace=False)
df.loc[idx, "age"] = 999

# 2) expand missing: 1000 rows native_country -> None
idx2 = rng.choice(len(df), size=1000, replace=False)
df.loc[idx2, "native_country"] = None

# 3) illegal enum: 20 rows income -> "???"
idx3 = rng.choice(len(df), size=20, replace=False)
df.loc[idx3, "income"] = "???"

df.to_parquet(out, index=False)
print("wrote:", out, "shape=", df.shape)
PY
}

extract_meta_json_from_log() {
  local raw_log="$1"
  python - <<'PY' "$raw_log"
import json, sys

raw = open(sys.argv[1], "r", encoding="utf-8").read().splitlines()

# 从后往前找最后一个可解析 JSON 的行
for line in reversed(raw):
    s = line.strip()
    if not s:
        continue
    if not (s.startswith("{") and s.endswith("}")):
        continue
    try:
        obj = json.loads(s)
        print(json.dumps(obj))
        raise SystemExit(0)
    except Exception:
        pass

raise SystemExit(1)
PY
}

run_dq() {
  local label="$1"
  local data_path="$2"
  local outdir="$3"
  local raw_log="$4"
  local ec_file="$5"

  mkdir -p "$(dirname "$raw_log")" "$outdir"

  log "Run dq_agent ($label)"
  log "  data   = $data_path"
  log "  config = $RULES"
  log "  outdir = $outdir"
  log "  meta   = $(dirname "$raw_log")"

  set +e
  python -m dq_agent run \
    --data "$data_path" \
    --config "$RULES" \
    --output-dir "$outdir" \
    --fail-on INFO \
    >"$raw_log" 2>&1
  local ec=$?
  set -e

  echo "$ec" > "$ec_file"

  cat "$raw_log"

  local meta
  if ! meta="$(extract_meta_json_from_log "$raw_log")"; then
    echo "❌ Could not parse JSON from dq_agent output for: $label" >&2
    echo "   Saved raw log to: $raw_log" >&2
    echo "   ---- first 200 lines of raw ----" >&2
    sed -n '1,200p' "$raw_log" >&2
    exit 1
  fi

  local report_json run_record
  report_json="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["report_json_path"])' <<<"$meta")"
  run_record="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_record_path"])' <<<"$meta")"

  echo "$report_json"
  echo "$run_record"
}

validate_and_replay() {
  local report_json="$1"
  local run_record="$2"
  local label="$3"

  log "Validate + replay $label"
  python -m dq_agent validate --kind report --path "$report_json" >/dev/null
  python -m dq_agent validate --kind run_record --path "$run_record" >/dev/null
  python -m dq_agent replay --run-record "$run_record" --strict >/dev/null
  log "$label ok"
}

# -----------------------------
# Main
# -----------------------------
log "Repo root: $REPO_ROOT"
log "Work dir : $WORK"

mkdir -p "$BASE_OUT" "$INJ_OUT" "$BASE_DIR" "$INJ_DIR" "$EVAL_DIR"

download_and_unzip
build_parquet_if_missing
write_rules_deterministic
config_sanity_check
build_injected_if_missing

log "Run BASELINE"
BASE_REPORT="$(run_dq baseline "$PARQUET" "$BASE_OUT" "$BASE_RAW" "$BASE_EC_FILE" | sed -n '1p')"
BASE_RUNREC="$(run_dq baseline "$PARQUET" "$BASE_OUT" "$BASE_RAW" "$BASE_EC_FILE" | sed -n '2p')"
# ↑ 上面调用了两次会重复跑。为了避免重复跑，我们改成从 raw log 解析：
BASE_META="$(extract_meta_json_from_log "$BASE_RAW")"
BASE_REPORT="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["report_json_path"])' <<<"$BASE_META")"
BASE_RUNREC="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_record_path"])' <<<"$BASE_META")"

# baseline 必须是 0
BASE_EC="$(cat "$BASE_EC_FILE" | tr -d '[:space:]')"
if [[ "$BASE_EC" != "0" ]]; then
  echo "❌ baseline exit_code expected 0, got $BASE_EC" >&2
  exit 1
fi
validate_and_replay "$BASE_REPORT" "$BASE_RUNREC" "baseline"

log "Run INJECTED"
set +e
python -m dq_agent run \
  --data "$INJECTED" \
  --config "$RULES" \
  --output-dir "$INJ_OUT" \
  --fail-on INFO \
  >"$INJ_RAW" 2>&1
INJ_EC=$?
set -e
echo "$INJ_EC" > "$INJ_EC_FILE"
cat "$INJ_RAW"

INJ_META="$(extract_meta_json_from_log "$INJ_RAW")"
INJ_REPORT="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["report_json_path"])' <<<"$INJ_META")"
INJ_RUNREC="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_record_path"])' <<<"$INJ_META")"

# injected 期望 2（fail-on INFO）
if [[ "$INJ_EC" != "2" ]]; then
  echo "❌ injected exit_code expected 2, got $INJ_EC" >&2
  exit 1
fi
validate_and_replay "$INJ_REPORT" "$INJ_RUNREC" "injected"

log "Summarize injected failures (rule_results status FAIL + anomalies status FAIL)"
python - <<'PY' "$INJ_REPORT"
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))

rr=d.get("rule_results") or []
rf=[r for r in rr if r.get("status")=="FAIL" or r.get("passed") is False]

an=d.get("anomalies") or []
af=[a for a in an if a.get("status")=="FAIL"]

print("status:", d.get("status"))
print("rule_results:", len(rr), "FAIL:", len(rf))
for r in rf:
    print("-", r.get("rule_id"), r.get("column"), "failed_count=", r.get("failed_count"), "ratio=", r.get("failing_ratio"))

print("anomalies:", len(an), "FAIL:", len(af))
for a in af:
    print("-", a.get("column"), a.get("anomaly_id"), "|", a.get("explanation"))
PY

log "Write eval_summary.json (regression-friendly)"
python - <<'PY' "$PARQUET" "$INJECTED" "$BASE_REPORT" "$INJ_REPORT" "$BASE_EC_FILE" "$INJ_EC_FILE" "$SUMMARY_JSON"
import json, sys, os
import pandas as pd

parquet, injected, base_report, inj_report, base_ec_file, inj_ec_file, out_path = sys.argv[1:]

def load_json(p):
    return json.load(open(p, "r", encoding="utf-8"))

def rule_fails(rr):
    rr = rr or []
    fails = [r for r in rr if r.get("status")=="FAIL" or r.get("passed") is False]
    return rr, fails

def anomaly_fails(an):
    an = an or []
    fails = [a for a in an if a.get("status")=="FAIL"]
    return an, fails

def pick_rule(report, rule_id):
    for r in (report.get("rule_results") or []):
        if r.get("rule_id")==rule_id:
            return r
    return None

base = load_json(base_report)
inj  = load_json(inj_report)

base_rr, base_rf = rule_fails(base.get("rule_results"))
inj_rr,  inj_rf  = rule_fails(inj.get("rule_results"))
base_an, base_af = anomaly_fails(base.get("anomalies"))
inj_an,  inj_af  = anomaly_fails(inj.get("anomalies"))

base_ec = int(open(base_ec_file).read().strip() or "0")
inj_ec  = int(open(inj_ec_file).read().strip() or "0")

# truth from injected parquet (deterministic injections)
dfi = pd.read_parquet(injected)

truth_age_999 = int((dfi["age"]==999).sum())
truth_income_bad = int((dfi["income"]=="???").sum())
truth_native_null = int(dfi["native_country"].isna().sum())

# baseline native_country nulls (for delta)
dfb = pd.read_parquet(parquet)
base_native_null = int(dfb["native_country"].isna().sum())

# rule-based recalls (行级可验证的那部分)
r_age = pick_rule(inj, "range:age")
r_inc = pick_rule(inj, "allowed_values:income")

recall_age = None
if truth_age_999 > 0 and r_age and r_age.get("failed_count") is not None:
    recall_age = min(1.0, float(r_age["failed_count"]) / float(truth_age_999))

recall_income = None
if truth_income_bad > 0 and r_inc and r_inc.get("failed_count") is not None:
    recall_income = min(1.0, float(r_inc["failed_count"]) / float(truth_income_bad))

summary = {
  "dataset": {
    "name": "adult",
    "rows": int((inj.get("summary") or {}).get("rows") or 0),
    "cols": int((inj.get("summary") or {}).get("cols") or 0),
  },
  "paths": {
    "work_dir": os.path.dirname(os.path.dirname(out_path)),
    "baseline_report": base_report,
    "injected_report": inj_report,
    "baseline_exit_code_file": base_ec_file,
    "injected_exit_code_file": inj_ec_file,
  },
  "baseline": {
    "exit_code": base_ec,
    "report_status": base.get("status"),
    "rule_total": len(base_rr),
    "rule_fail": len(base_rf),
    "anomaly_total": len(base_an),
    "anomaly_fail": len(base_af),
  },
  "injected": {
    "exit_code": inj_ec,
    "report_status": inj.get("status"),
    "rule_total": len(inj_rr),
    "rule_fail": len(inj_rf),
    "anomaly_total": len(inj_an),
    "anomaly_fail": len(inj_af),
    "rule_failures": [
      {
        "rule_id": r.get("rule_id"),
        "column": r.get("column"),
        "failed_count": r.get("failed_count"),
        "failing_ratio": r.get("failing_ratio"),
      }
      for r in inj_rf
    ],
    "anomaly_failures": [
      {
        "anomaly_id": a.get("anomaly_id"),
        "column": a.get("column"),
        "explanation": a.get("explanation"),
        "metric": a.get("metric"),
        "threshold": a.get("threshold"),
      }
      for a in inj_af
    ],
  },
  "truth": {
    "injected_age_eq_999": truth_age_999,
    "injected_income_eq_???": truth_income_bad,
    "injected_native_country_null": truth_native_null,
    "baseline_native_country_null": base_native_null,
    "delta_native_country_null": truth_native_null - base_native_null,
  },
  "derived_metrics": {
    "recall_rule_range_age_on_age_eq_999": recall_age,
    "recall_rule_allowed_values_income_on_income_eq_???": recall_income,
    "note": "Anomaly samples are truncated; use status/metric for anomaly evaluation, not row-level recall.",
  }
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("wrote:", out_path)
PY

log "DONE"
log "Baseline raw log : $BASE_RAW"
log "Injected raw log : $INJ_RAW"
log "Baseline report  : $BASE_REPORT"
log "Injected report  : $INJ_REPORT"
log "Eval summary     : $SUMMARY_JSON"

