#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate dq_agent on a dirty/clean paired dataset (e.g., Raha):
- Truth labels: cell-level mismatches between dirty vs clean after normalization.
- Prediction: cells reported by dq_agent as FAIL (rule_results/anomalies) using full samples.
- Outputs:
  - metrics.json (detailed)
  - prints one JSON line summary for bench scripts

Key design:
- Generate a dq_agent config from a "profile source" (clean or dirty).
- For high-cardinality string columns, only emit allowed_values if top-K domain coverage >= threshold.
  This prevents exploding false positives (e.g., tax.city).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml


# -----------------------------
# Utils
# -----------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _strip_bom(s: str) -> str:
    # handle UTF-8 BOM in some CSV headers
    return s.lstrip("\ufeff")


def _norm_col_name(x: Any) -> str:
    s = _strip_bom(str(x))
    return s.strip()


def normalize_string_series(s: pd.Series) -> pd.Series:
    """
    Normalize strings for both:
      - truth comparison
      - dq_agent input for string columns
    Rules:
      - strip whitespace
      - empty -> NA
      - common textual NA markers -> NA (conservative)
    """
    ss = s.astype("string")
    ss = ss.str.strip()

    # empty -> NA
    ss = ss.replace("", pd.NA)

    # conservative NA markers (optional; helps datasets with explicit NA tokens)
    ss = ss.replace(
        {
            "NA": pd.NA,
            "N/A": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "NaN": pd.NA,
            "nan": pd.NA,
        }
    )
    return ss


def numeric_success_rate(s: pd.Series) -> float:
    """
    Fraction of non-null values that can be coerced to numeric.
    """
    ss = normalize_string_series(s)
    non_null = int(ss.notna().sum())
    if non_null == 0:
        return 0.0
    num = pd.to_numeric(ss, errors="coerce")
    ok = int(num.notna().sum())
    return ok / non_null


def coerce_numeric(s: pd.Series) -> pd.Series:
    """
    Normalize then coerce to float (NaN if cannot parse).
    """
    ss = normalize_string_series(s)
    return pd.to_numeric(ss, errors="coerce")


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Define PR/F1 robustly:
      - If tp=fp=fn=0 (empty truth and empty preds), treat as perfect => (1,1,1).
      - If no preds but truth exists => precision=0, recall=0, f1=0.
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def find_last_json_line(text: str) -> Dict[str, Any]:
    """
    dq_agent prints a single JSON object line; but to be robust we scan from bottom
    for a line that parses as JSON dict.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if not (ln.startswith("{") and ln.endswith("}")):
            continue
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise RuntimeError("No JSON object found in dq_agent output")


def drop_index_like_first_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Drop an index-like first column if it looks like:
      - name is Unnamed: 0 / index / idx / blank-ish
      - values are monotonic increasing integers and mostly unique
    """
    if df.shape[1] == 0:
        return df, False

    first = df.columns[0]
    name = str(first).strip().lower()

    if not (name.startswith("unnamed") or name in ("index", "idx", "")):
        return df, False

    col = df.iloc[:, 0]
    # try numeric
    num = pd.to_numeric(normalize_string_series(col), errors="coerce")
    if num.notna().mean() < 0.95:
        return df, False

    # unique-ish and monotonic-ish
    nn = num.dropna()
    if len(nn) < 10:
        return df, False

    # check near-unique
    uniq_rate = nn.nunique(dropna=True) / len(nn)
    if uniq_rate < 0.95:
        return df, False

    # monotonic increasing
    if not nn.is_monotonic_increasing:
        return df, False

    # looks like index -> drop
    return df.iloc[:, 1:].copy(), True


def align_dirty_clean(
    dirty: pd.DataFrame, clean: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Align dirty and clean frames on a shared set of columns.
    Returns:
      dirty_aligned, clean_aligned, columns, notes
    Strategy:
      1) normalize column names (strip/BOM)
      2) intersect on names
      3) if empty, try dropping index-like first col then intersect again
      4) if still empty, fall back to positional alignment on min(ncols)
    """
    notes: List[str] = []

    dirty = dirty.copy()
    clean = clean.copy()
    dirty.columns = [_norm_col_name(c) for c in dirty.columns]
    clean.columns = [_norm_col_name(c) for c in clean.columns]

    common = sorted(set(dirty.columns) & set(clean.columns))
    if common:
        return dirty[common].copy(), clean[common].copy(), common, notes

    # try dropping index-like
    dirty2, dropped_d = drop_index_like_first_col(dirty)
    clean2, dropped_c = drop_index_like_first_col(clean)
    if dropped_d or dropped_c:
        notes.append(f"dropped index-like first column: dirty={dropped_d} clean={dropped_c}")
        common2 = sorted(set(dirty2.columns) & set(clean2.columns))
        if common2:
            return dirty2[common2].copy(), clean2[common2].copy(), common2, notes
        dirty, clean = dirty2, clean2  # continue with cleaned versions

    # fallback: positional alignment
    n = min(dirty.shape[1], clean.shape[1])
    if n <= 0:
        raise RuntimeError("No columns to align (both have 0 columns?)")

    notes.append("no common columns by name; falling back to positional alignment")
    new_cols = [f"col_{i}" for i in range(n)]
    dirty_aligned = dirty.iloc[:, :n].copy()
    clean_aligned = clean.iloc[:, :n].copy()
    dirty_aligned.columns = new_cols
    clean_aligned.columns = new_cols
    return dirty_aligned, clean_aligned, new_cols, notes


@dataclass
class AllowedValuesDecision:
    emitted: bool
    mode: str                   # "all" | "topk" | "skip" | "empty"
    nunique: int
    non_null: int
    top_k: int
    coverage: float
    values_count: int


def decide_allowed_values(
    s: pd.Series,
    *,
    max_allowed_values: int,
    max_domain: int,
    min_domain_coverage: float,
) -> Tuple[Optional[List[str]], AllowedValuesDecision]:
    """
    Decide whether to emit allowed_values for a string column.
    - If nunique <= max_allowed_values => emit all unique values.
    - Else consider top-K (K=max_domain) only if coverage >= min_domain_coverage.
    - Otherwise skip (avoid high-cardinality/open-domain false positives).
    """
    ss = normalize_string_series(s).dropna()
    non_null = int(len(ss))
    if non_null == 0:
        return None, AllowedValuesDecision(
            emitted=False, mode="empty", nunique=0, non_null=0, top_k=0, coverage=0.0, values_count=0
        )

    vc = ss.value_counts(dropna=True)
    nunique = int(len(vc))

    # small domain: full enumerate
    if nunique <= max_allowed_values:
        vals = vc.index.astype(str).tolist()
        return vals, AllowedValuesDecision(
            emitted=True,
            mode="all",
            nunique=nunique,
            non_null=non_null,
            top_k=nunique,
            coverage=1.0,
            values_count=len(vals),
        )

    # large domain: maybe top-K if coverage is high
    if max_domain <= 0:
        return None, AllowedValuesDecision(
            emitted=False, mode="skip", nunique=nunique, non_null=non_null, top_k=0, coverage=0.0, values_count=0
        )

    k = min(int(max_domain), nunique)
    top = vc.head(k)
    cov = float(top.sum()) / float(non_null) if non_null else 0.0

    if cov < float(min_domain_coverage):
        return None, AllowedValuesDecision(
            emitted=False,
            mode="skip",
            nunique=nunique,
            non_null=non_null,
            top_k=k,
            coverage=cov,
            values_count=0,
        )

    vals = top.index.astype(str).tolist()
    return vals, AllowedValuesDecision(
        emitted=True,
        mode="topk",
        nunique=nunique,
        non_null=non_null,
        top_k=k,
        coverage=cov,
        values_count=len(vals),
    )


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def try_write_parquet_else_csv(df: pd.DataFrame, out_base: Path) -> Tuple[Path, str]:
    """
    Prefer Parquet for stable dtypes; fallback to CSV if parquet engine missing.
    """
    pq = out_base.with_suffix(".parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq, "parquet"
    except Exception:
        csv = out_base.with_suffix(".csv")
        # keep empty as empty
        df.to_csv(csv, index=False)
        return csv, "csv"


def run_dq_agent(
    *,
    data_path: Path,
    config_path: Path,
    dq_out_dir: Path,
    fail_on: str,
    raw_log_path: Path,
    verbose: bool,
) -> Tuple[int, Dict[str, Any], float]:
    """
    Run `python -m dq_agent run ...` and return:
      exit_code, meta_json, wall_time_s
    """
    _ensure_dir(dq_out_dir)
    _ensure_dir(raw_log_path.parent)

    cmd = [
        sys.executable,
        "-m",
        "dq_agent",
        "run",
        "--data",
        str(data_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(dq_out_dir),
        "--fail-on",
        str(fail_on),
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall = time.time() - t0

    raw = proc.stdout or ""
    raw_log_path.write_text(raw, encoding="utf-8")

    if verbose:
        sys.stderr.write(raw)
        sys.stderr.flush()

    meta = find_last_json_line(raw)
    return proc.returncode, meta, wall


def extract_predicted_cells_from_report(report: Dict[str, Any]) -> Tuple[Set[Tuple[int, str]], Dict[str, Any]]:
    """
    Build predicted error cell set from dq_agent report:
      - rule_results with status FAIL: each sample row_index => (row_index, column)
      - anomalies with status FAIL: each sample row_index => (row_index, column)
    Also returns truncation info.
    """
    pred: Set[Tuple[int, str]] = set()
    trunc_details: List[Dict[str, Any]] = []

    # rule_results
    for r in (report.get("rule_results") or []):
        status = r.get("status")
        if status != "FAIL" and r.get("passed") is not False:
            continue

        col = r.get("column")
        if not col:
            continue

        samples = r.get("samples") or []
        failed_count = r.get("failed_count")
        # detect truncation
        if isinstance(failed_count, int) and failed_count > len(samples):
            trunc_details.append(
                {
                    "kind": "rule_result",
                    "rule_id": r.get("rule_id"),
                    "column": col,
                    "failed_count": failed_count,
                    "sample_count": len(samples),
                }
            )

        for s in samples:
            ri = s.get("row_index")
            if ri is None:
                continue
            try:
                i = int(ri)
            except Exception:
                continue
            pred.add((i, str(col)))

    # anomalies
    for a in (report.get("anomalies") or []):
        status = a.get("status")
        if status != "FAIL":
            continue

        col = a.get("column")
        if not col:
            continue

        samples = a.get("samples") or []
        # anomalies don't always have failed_count; still track if suspicious
        # (keep it simple: if FAIL but zero samples, mark as potentially truncated)
        if len(samples) == 0:
            trunc_details.append(
                {
                    "kind": "anomaly",
                    "anomaly_id": a.get("anomaly_id"),
                    "column": str(col),
                    "note": "FAIL anomaly with 0 samples (may be truncated or summary-only detector).",
                }
            )

        for s in samples:
            ri = s.get("row_index")
            if ri is None:
                continue
            try:
                i = int(ri)
            except Exception:
                continue
            pred.add((i, str(col)))

    trunc = {"truncated": len(trunc_details) > 0, "details": trunc_details}
    return pred, trunc


def compute_truth_cells(
    dirty: pd.DataFrame,
    clean: pd.DataFrame,
    cols: List[str],
    numeric_cols: Set[str],
) -> Set[Tuple[int, str]]:
    """
    Truth = dirty vs clean mismatch after normalization.
    For numeric cols: numeric coercion + isclose (tolerance).
    For string cols: strip + NA normalization + string compare.
    """
    if dirty.shape[0] != clean.shape[0]:
        raise RuntimeError(f"Row count mismatch: dirty={dirty.shape[0]} clean={clean.shape[0]}")

    truth: Set[Tuple[int, str]] = set()
    n = dirty.shape[0]

    for c in cols:
        if c in numeric_cols:
            dnum = coerce_numeric(dirty[c])
            cnum = coerce_numeric(clean[c])
            # both NaN -> equal
            both_na = dnum.isna() & cnum.isna()
            one_na = dnum.isna() ^ cnum.isna()
            # both numeric: compare isclose
            # note: if both notna, isclose works; if any NaN, handled above
            close = np.isclose(dnum.fillna(0.0).to_numpy(), cnum.fillna(0.0).to_numpy(), atol=1e-9, rtol=0.0)
            eq = both_na.to_numpy() | (~one_na.to_numpy() & close)
            mismatch_idx = np.flatnonzero(~eq)
        else:
            ds = normalize_string_series(dirty[c])
            cs = normalize_string_series(clean[c])
            both_na = ds.isna() & cs.isna()
            eq = (ds == cs) | both_na
            mismatch_idx = np.flatnonzero(~eq.fillna(False).to_numpy())

        for i in mismatch_idx:
            truth.add((int(i), c))

    return truth


def infer_numeric_cols(
    profile_df: pd.DataFrame,
    cols: List[str],
    numeric_success_threshold: float,
) -> Set[str]:
    numeric_cols: Set[str] = set()
    for c in cols:
        rate = numeric_success_rate(profile_df[c])
        if rate >= float(numeric_success_threshold):
            numeric_cols.add(c)
    return numeric_cols


# -----------------------------
# Main
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate dq_agent on dirty/clean datasets with cell-level truth labels.")

    p.add_argument("--dirty", required=True, help="Path to dirty.csv")
    p.add_argument("--clean", required=True, help="Path to clean.csv")
    p.add_argument("--out", required=True, help="Output directory for this dataset eval")
    p.add_argument("--dataset-name", required=True, help="Dataset name for reports (e.g., raha/tax)")

    # accept both hyphen and underscore variants
    p.add_argument(
        "--profile-source", "--profile_source",
        dest="profile_source",
        choices=["clean", "dirty"],
        default="clean",
        help="Build profiling rules from clean or dirty dataset."
    )
    p.add_argument(
        "--fail-on", "--fail_on",
        dest="fail_on",
        default="INFO",
        help="Pass through to dq_agent --fail-on (info|warn|error).",
    )

    # rule generation knobs
    p.add_argument(
        "--max-allowed-values", "--max_allowed_values",
        dest="max_allowed_values",
        type=int,
        default=200,
        help="Emit allowed_values for string columns only if nunique <= this."
    )
    p.add_argument(
        "--max-domain", "--max_domain",
        dest="max_domain",
        type=int,
        default=2000,
        help="If nunique > max_allowed_values, consider top-K domain of this size for allowed_values."
    )
    p.add_argument(
        "--min-domain-coverage", "--min_domain_coverage",
        dest="min_domain_coverage",
        type=float,
        default=0.98,
        help="Only emit allowed_values (top-K) if top-K covers >= this fraction of non-null rows."
    )
    p.add_argument(
        "--numeric-success-threshold", "--numeric_success_threshold",
        dest="numeric_success_threshold",
        type=float,
        default=0.98,
        help="Infer a column as numeric if >= this fraction of non-null values parse as numeric."
    )
    p.add_argument(
        "--z-outlier-mad", "--z_outlier_mad",
        dest="z_outlier_mad",
        type=float,
        default=6.0,
        help="MAD z-score threshold for outlier_mad anomaly detector."
    )
    p.add_argument(
        "--null-slack", "--null_slack",
        dest="null_slack",
        type=float,
        default=0.01,
        help="Slack added to profile null rate to set max_null_rate / max_rate thresholds."
    )

    # string-noise detector knobs (for open-domain string columns)
    p.add_argument(
        "--string-noise", "--string_noise",
        dest="string_noise",
        action="store_true",
        default=True,
        help="Enable string_noise check for string columns (default: enabled).",
    )
    p.add_argument(
        "--no-string-noise", "--no_string_noise",
        dest="string_noise",
        action="store_false",
        help="Disable string_noise check.",
    )
    p.add_argument(
        "--string-noise-scope", "--string_noise_scope",
        dest="string_noise_scope",
        choices=["open_domain", "all"],
        default="open_domain",
        help="Where to apply string_noise: open_domain (no allowed_values) or all string columns.",
    )
    p.add_argument(
        "--string-noise-contains", "--string_noise_contains",
        dest="string_noise_contains",
        action="append",
        default=None,
        help="Literal substring to flag as noise. Repeatable. Default: '*' and \"''\".",
    )
    p.add_argument(
        "--string-noise-regex", "--string_noise_regex",
        dest="string_noise_regex",
        action="append",
        default=None,
        help="Regex pattern to flag as noise. Repeatable. Default: none.",
    )
    p.add_argument(
        "--string-noise-ignore-case", "--string_noise_ignore_case",
        dest="string_noise_ignore_case",
        action="store_true",
        default=False,
        help="Case-insensitive string noise matching.",
    )
    p.add_argument(
        "--string-noise-max-rate", "--string_noise_max_rate",
        dest="string_noise_max_rate",
        type=float,
        default=0.0,
        help="Tolerated noisy fraction (failed_count/rows). Default 0.0 => any match fails.",
    )

    p.add_argument("--verbose", action="store_true", help="Print dq_agent raw output to stderr.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    dirty_path = Path(args.dirty).expanduser().resolve()
    clean_path = Path(args.clean).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    dataset_name = str(args.dataset_name)

    _ensure_dir(out_dir)

    # Read as strings (we normalize ourselves)
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False, na_filter=False)
    clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False, na_filter=False)

    dirty_aligned, clean_aligned, cols, align_notes = align_dirty_clean(dirty_df, clean_df)

    # profile source for type inference + rule generation
    profile_df = clean_aligned if args.profile_source == "clean" else dirty_aligned

    numeric_cols = infer_numeric_cols(profile_df, cols, args.numeric_success_threshold)

    # string_noise default patterns (aligned with scripts/bench_raha_noise_union.sh)
    default_noise_contains = ["*", "''"]
    string_noise_contains = (
        args.string_noise_contains if args.string_noise_contains is not None else default_noise_contains
    )
    string_noise_regex = args.string_noise_regex if args.string_noise_regex is not None else []
    string_noise_params: Dict[str, Any] = {
        "contains": string_noise_contains,
        "regex": string_noise_regex,
        "ignore_case": bool(args.string_noise_ignore_case),
        "max_rate": float(args.string_noise_max_rate),
    }

    # compute truth labels
    truth_cells = compute_truth_cells(dirty_aligned, clean_aligned, cols, numeric_cols)
    truth_rows = {r for (r, _) in truth_cells}

    # prepare dq_agent input DF (dirty)
    run_df = pd.DataFrame(index=dirty_aligned.index)

    # keep same columns order as cols
    for c in cols:
        if c in numeric_cols:
            run_df[c] = coerce_numeric(dirty_aligned[c]).astype("float64")
        else:
            run_df[c] = normalize_string_series(dirty_aligned[c])

    # row_id for dq_agent primary key
    run_df.insert(0, "row_id", np.arange(1, len(run_df) + 1, dtype=np.int64))

    # write dq_agent input
    data_base = out_dir / "dq_data"
    data_path, data_fmt = try_write_parquet_else_csv(run_df, data_base)

    # generate dq_agent config
    # thresholds derived from profile dataset (clean/dirty)
    gen_cfg_summary: Dict[str, Any] = {}

    columns_cfg: Dict[str, Any] = {
        "row_id": {
            "type": "int",
            "required": True,
            "checks": [{"unique": True}],
        }
    }

    # report.sample_rows = nrows to avoid truncation (needed for cell-level evaluation)
    nrows = int(run_df.shape[0])

    for c in cols:
        col_cfg: Dict[str, Any] = {
            "required": False,
            "checks": [],
        }

        if c in numeric_cols:
            col_cfg["type"] = "float"

            prof_num = coerce_numeric(profile_df[c])
            prof_null_rate = float(prof_num.isna().mean())
            max_null_rate = min(1.0, max(0.0, prof_null_rate + float(args.null_slack)))

            # not_null threshold
            col_cfg["checks"].append({"not_null": {"max_null_rate": max_null_rate}})

            # range from profile min/max (if any)
            nn = prof_num.dropna()
            if len(nn) > 0:
                mn = float(nn.min())
                mx = float(nn.max())
                col_cfg["checks"].append({"range": {"min": mn, "max": mx}})
                col_cfg["anomalies"] = [{"outlier_mad": {"z": float(args.z_outlier_mad)}}]
            else:
                col_cfg["anomalies"] = []

            gen_cfg_summary[c] = {
                "kind": "numeric",
                "numeric_success_rate": numeric_success_rate(profile_df[c]),
                "profile_null_rate": prof_null_rate,
                "max_null_rate": max_null_rate,
            }

        else:
            col_cfg["type"] = "string"

            prof_str = normalize_string_series(profile_df[c])
            prof_null_rate = float(prof_str.isna().mean())
            max_null_rate = min(1.0, max(0.0, prof_null_rate + float(args.null_slack)))

            col_cfg["checks"].append({"not_null": {"max_null_rate": max_null_rate}})

            allowed, decision = decide_allowed_values(
                profile_df[c],
                max_allowed_values=int(args.max_allowed_values),
                max_domain=int(args.max_domain),
                min_domain_coverage=float(args.min_domain_coverage),
            )

            # Heuristic string noise detector: primarily for open-domain columns where allowed_values is skipped.
            if bool(args.string_noise) and (string_noise_contains or string_noise_regex):
                should_apply = (
                    str(args.string_noise_scope) == "all" or (not decision.emitted and decision.mode == "skip")
                )
                if should_apply:
                    col_cfg["checks"].append({"string_noise": dict(string_noise_params)})

            if allowed:
                col_cfg["checks"].append({"allowed_values": {"values": allowed}})

            # missing_rate anomaly mirrors null threshold (kept for detector coverage)
            col_cfg["anomalies"] = [{"missing_rate": {"max_rate": max_null_rate}}]

            gen_cfg_summary[c] = {
                "kind": "string",
                "profile_null_rate": prof_null_rate,
                "max_null_rate": max_null_rate,
                "allowed_values": decision.__dict__,
                "string_noise": {
                    "enabled": bool(args.string_noise)
                    and (string_noise_contains or string_noise_regex)
                    and (
                        str(args.string_noise_scope) == "all"
                        or (not decision.emitted and decision.mode == "skip")
                    ),
                    "scope": str(args.string_noise_scope),
                    "params": dict(string_noise_params),
                },
            }

        columns_cfg[c] = col_cfg

    config_obj: Dict[str, Any] = {
        "version": 1,
        "dataset": {
            "name": dataset_name,
            "primary_key": ["row_id"],
        },
        "columns": columns_cfg,
        "report": {
            "sample_rows": nrows,
        },
    }

    config_path = out_dir / "dq_config.yml"
    write_yaml(config_path, config_obj)

    # run dq_agent
    dq_out_dir = out_dir / "dq_out"
    raw_log_path = out_dir / "dq_raw.log"

    exit_code, meta, wall = run_dq_agent(
        data_path=data_path,
        config_path=config_path,
        dq_out_dir=dq_out_dir,
        fail_on=str(args.fail_on),
        raw_log_path=raw_log_path,
        verbose=bool(args.verbose),
    )

    report_json_path = Path(meta["report_json_path"]).expanduser().resolve()
    run_record_path = Path(meta["run_record_path"]).expanduser().resolve()

    report = json.loads(report_json_path.read_text(encoding="utf-8"))

    # predicted cells
    pred_cells, trunc = extract_predicted_cells_from_report(report)
    pred_rows = {r for (r, _) in pred_cells}

    # confusion
    tp = len(pred_cells & truth_cells)
    fp = len(pred_cells - truth_cells)
    fn = len(truth_cells - pred_cells)
    cprec, crec, cf1 = prf1(tp, fp, fn)

    rtp = len(pred_rows & truth_rows)
    rfp = len(pred_rows - truth_rows)
    rfn = len(truth_rows - pred_rows)
    rprec, rrec, rf1 = prf1(rtp, rfp, rfn)

    # column coverage
    truth_cols = {c for (_, c) in truth_cells}
    pred_cols = {c for (_, c) in pred_cells}
    cov = {
        "truth_error_columns": len(truth_cols),
        "predicted_error_columns": len(pred_cols),
        "hit_columns": len(truth_cols & pred_cols),
    }

    # metrics.json
    metrics = {
        "dataset": {
            "name": dataset_name,
            "rows": int(len(run_df)),
            "cols": int(len(cols)),
            "profile_source": args.profile_source,
            "aligned_columns": cols,
            "numeric_columns": sorted(list(numeric_cols)),
            "alignment_notes": align_notes,
        },
        "generated_config": {
            "path": str(config_path),
            "data_path": str(data_path),
            "data_format": data_fmt,
            "sample_rows": nrows,
            "max_allowed_values": int(args.max_allowed_values),
            "max_domain": int(args.max_domain),
            "min_domain_coverage": float(args.min_domain_coverage),
            "numeric_success_threshold": float(args.numeric_success_threshold),
            "null_slack": float(args.null_slack),
            "z_outlier_mad": float(args.z_outlier_mad),
            "per_column_summary": gen_cfg_summary,
        },
        "dq_agent": {
            "exit_code": int(exit_code),
            "fail_on": str(args.fail_on),
            "wall_time_s": float(wall),
            "report_json_path": str(report_json_path),
            "run_record_path": str(run_record_path),
            "raw_log_path": str(raw_log_path),
        },
        "truth": {"error_cells": int(len(truth_cells))},
        "pred": {"predicted_cells": int(len(pred_cells))},
        "cell": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(cprec),
            "recall": float(crec),
            "f1": float(cf1),
        },
        "row": {
            "tp": int(rtp),
            "fp": int(rfp),
            "fn": int(rfn),
            "precision": float(rprec),
            "recall": float(rrec),
            "f1": float(rf1),
        },
        "column_coverage": cov,
        "truncation": trunc,
        "notes": [
            "Truth = dirty vs clean mismatch after normalization (strip, empty->NA, numeric coercion for numeric cols).",
            "Pred = dq_agent FAIL rule_results/anomalies samples. Config sets report.sample_rows=nrows to avoid truncation.",
        ],
        "meta": {
            "generated_at": _now_iso(),
            "script": "scripts/eval_dirty_clean.py",
            "python": sys.version.split()[0],
        },
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # bench-friendly single-line summary
    summary = {
        "dataset": dataset_name,
        "rows": int(len(run_df)),
        "cols": int(len(cols)),
        "profile": args.profile_source,
        "truth_err_cells": int(len(truth_cells)),
        "predicted": int(len(pred_cells)),
        "cell_f1": float(cf1),
        "row_f1": float(rf1),
        "truncated": bool(trunc.get("truncated", False)),
        "metrics_path": str(metrics_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

