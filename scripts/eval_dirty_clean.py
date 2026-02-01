#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate dq_agent on a (dirty.csv, clean.csv) pair with cell-level labels.

Truth labels:
  - A cell is "error" iff dirty != clean after light normalization:
      * strip whitespace
      * empty string -> NA
      * numeric coercion for numeric-like columns (so "1" == "1.0")

Pred labels:
  - Union of cell samples in dq_agent report.json where:
      * rule_results.status == "FAIL" (or passed is False)
      * anomalies.status == "FAIL"

Outputs under --out:
  - dq_rules.yml        (auto-generated config)
  - dq_raw.log          (dq_agent combined stdout/stderr)
  - dq_out/<run_id>/... (dq_agent outputs)
  - metrics.json        (detailed metrics)

Stdout:
  - prints ONE json line summary (for bench scripts to append into summary.jsonl)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml


# -----------------------------
# Logging helpers
# -----------------------------
def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# CSV reading (robust enough for RAHA variants)
# -----------------------------
_SEP_CANDIDATES: Sequence[str] = (",", "\t", ";", "|")


def _looks_like_csv_with_sep(sample: str, sep: str) -> bool:
    # Heuristic: if sep appears multiple times in first non-empty line.
    for line in sample.splitlines():
        s = line.strip()
        if not s:
            continue
        # ignore comment-y lines
        if s.startswith("#"):
            continue
        return s.count(sep) >= 1
    return False


def read_csv_flexible(path: str, *, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read CSV with a small heuristic for weird separators.
    We do an initial comma read; if it yields 1 column and sample hints another sep,
    we retry with other seps and pick the one with the most columns.

    We keep everything as string-ish so we can normalize ourselves.
    """
    # read a small sample for sep hinting
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            sample = f.read(65536)
    except Exception:
        sample = ""

    def _read(sep: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=sep,
            engine="python",          # more tolerant
            dtype="string",
            keep_default_na=False,    # keep "NA" literal; we'll normalize empties ourselves
            na_filter=False,          # keep empty as ""
        )

    df = _read(",")
    if df.shape[1] > 1:
        return df

    # If comma gives single column, try other seps if sample suggests.
    best_df = df
    best_cols = df.shape[1]

    for sep in _SEP_CANDIDATES:
        if sep == ",":
            continue
        if sample and not _looks_like_csv_with_sep(sample, sep):
            continue
        try:
            cand = _read(sep)
        except Exception:
            continue
        if cand.shape[1] > best_cols:
            best_df = cand
            best_cols = cand.shape[1]

    return best_df


# -----------------------------
# Column alignment (FIX for movies_1 and other mismatches)
# -----------------------------
_UNNAMED_RE = re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE)


def _norm_colname(x: Any) -> str:
    """
    Normalize column name for matching:
      - force str
      - strip whitespace
      - drop BOM
      - collapse whitespace
      - lowercase
    """
    s = str(x)
    s = s.replace("\ufeff", "")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    keep = []
    for c in cols:
        n = _norm_colname(c)
        if _UNNAMED_RE.match(n):
            continue
        keep.append(c)
    return df[keep].copy()


def align_dirty_clean(
    dirty: pd.DataFrame,
    clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Return (dirty_aligned, clean_aligned, notes)

    Strategy:
      1) match by normalized column names (handles int vs str, BOM, spaces)
      2) if still no common cols, drop Unnamed:* cols and retry
      3) if still no common cols but same column count, align by POSITION:
         rename clean columns to dirty columns by position
      4) else: raise with diagnostics
    """
    notes: List[str] = []

    def _align_by_norm(d: pd.DataFrame, c: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        d_norm = [_norm_colname(x) for x in d.columns]
        c_norm = [_norm_colname(x) for x in c.columns]

        # map norm -> original name (first occurrence)
        d_map: Dict[str, Any] = {}
        for orig, n in zip(d.columns, d_norm):
            d_map.setdefault(n, orig)

        c_map: Dict[str, Any] = {}
        for orig, n in zip(c.columns, c_norm):
            c_map.setdefault(n, orig)

        common = [n for n in d_norm if n in c_map]
        common_unique = []
        seen = set()
        for n in common:
            if n in seen:
                continue
            seen.add(n)
            common_unique.append(n)

        if not common_unique:
            return None

        # Keep dirty order; pick corresponding clean cols; rename clean -> dirty names
        d_cols = [d_map[n] for n in common_unique]
        c_cols = [c_map[n] for n in common_unique]

        d2 = d[d_cols].copy()
        c2 = c[c_cols].copy()
        rename = {c_col: d_col for c_col, d_col in zip(c_cols, d_cols) if c_col != d_col}
        if rename:
            notes.append(f"Aligned by normalized column names; renamed {len(rename)} clean columns to match dirty.")
            c2 = c2.rename(columns=rename)
        else:
            notes.append("Aligned by normalized column names (no renames needed).")

        return d2, c2

    # 1) norm match
    out = _align_by_norm(dirty, clean)
    if out is not None:
        d2, c2 = out
        return d2, c2, notes

    # 2) drop unnamed and retry
    dirty2 = _drop_unnamed(dirty)
    clean2 = _drop_unnamed(clean)
    if dirty2.shape[1] != dirty.shape[1] or clean2.shape[1] != clean.shape[1]:
        notes.append("Dropped Unnamed:* columns before aligning.")
    out = _align_by_norm(dirty2, clean2)
    if out is not None:
        d2, c2 = out
        return d2, c2, notes

    # 3) position fallback if same width
    if dirty2.shape[1] == clean2.shape[1] and dirty2.shape[1] > 0:
        d_cols = list(dirty2.columns)
        c_cols = list(clean2.columns)
        c2 = clean2.copy()
        c2.columns = d_cols
        notes.append(
            "No common column names after normalization; aligned by POSITION (renamed clean columns to dirty columns)."
        )
        return dirty2, c2, notes

    # 4) fail with diagnostics
    raise RuntimeError(
        "No common columns between dirty and clean after normalization.\n"
        f"dirty cols={len(dirty.columns)} -> after_drop={len(dirty2.columns)}\n"
        f"clean cols={len(clean.columns)} -> after_drop={len(clean2.columns)}\n"
        f"dirty head cols={list(dirty.columns)[:10]}\n"
        f"clean head cols={list(clean.columns)[:10]}"
    )


# -----------------------------
# Normalization + truth computation
# -----------------------------
_NUMERIC_RATE_THRESHOLD = 0.99


def _normalize_string_series(s: pd.Series) -> pd.Series:
    # s is dtype string; keep "" as "", then convert to NA
    s2 = s.astype("string")
    s2 = s2.str.strip()
    # empty -> NA
    s2 = s2.mask(s2 == "", pd.NA)
    return s2


def _is_numeric_like(col: pd.Series) -> bool:
    """
    Decide whether a column is numeric-like based on coercion success rate.
    """
    s = col
    if not pd.api.types.is_string_dtype(s.dtype):
        s = s.astype("string")
    s = _normalize_string_series(s)

    non_na = s.notna().sum()
    if non_na == 0:
        return False

    coerced = pd.to_numeric(s, errors="coerce")
    ok = coerced.notna().sum()
    rate = ok / float(non_na)
    return rate >= _NUMERIC_RATE_THRESHOLD


def normalize_pair(
    dirty: pd.DataFrame, clean: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Normalize both, and return numeric_cols set (by name) for aligned columns.
    """
    d = dirty.copy()
    c = clean.copy()

    numeric_cols: Set[str] = set()

    for col in d.columns:
        d[col] = _normalize_string_series(d[col].astype("string"))
        c[col] = _normalize_string_series(c[col].astype("string"))

    for col in d.columns:
        # numeric-like detection based on clean (or both)
        if _is_numeric_like(c[col]) and _is_numeric_like(d[col]):
            numeric_cols.add(col)
            d[col] = pd.to_numeric(d[col], errors="coerce")
            c[col] = pd.to_numeric(c[col], errors="coerce")

    return d, c, numeric_cols


def compute_truth_error_cells(
    dirty_norm: pd.DataFrame, clean_norm: pd.DataFrame, numeric_cols: Set[str]
) -> Set[Tuple[int, str]]:
    """
    Return set of (row_index, column_name) where dirty != clean (NA==NA treated equal).
    """
    if dirty_norm.shape != clean_norm.shape:
        raise RuntimeError(f"dirty/clean shape mismatch after alignment: {dirty_norm.shape} vs {clean_norm.shape}")

    truth: Set[Tuple[int, str]] = set()
    nrows = dirty_norm.shape[0]

    for col in dirty_norm.columns:
        a = dirty_norm[col]
        b = clean_norm[col]

        if col in numeric_cols:
            # float compare; NA==NA treated equal
            eq = a.eq(b)
            # eq is bool series; NaN comparisons are False; fix NA==NA
            eq = eq | (a.isna() & b.isna())
            # vectorize indices
            bad_idx = eq.index[~eq]
        else:
            # pandas StringArray eq can produce <NA> -> fill to False
            cmp = a.eq(b)
            if hasattr(cmp, "fillna"):
                cmp = cmp.fillna(False)
            eq = cmp | (a.isna() & b.isna())
            bad_idx = eq.index[~eq]

        # add tuples
        for i in bad_idx:
            # i is index label; in our case it's 0..n-1
            truth.add((int(i), col))

    # sanity check: ensure indices are in range
    for i, _ in list(truth)[:5]:
        if not (0 <= i < nrows):
            raise RuntimeError(f"Truth error row_index out of range: {i} (nrows={nrows})")

    return truth


# -----------------------------
# dq_agent config generation
# -----------------------------
@dataclass
class ConfigGenParams:
    max_allowed_values: int = 5000
    z_outlier_mad: float = 6.0
    null_slack: float = 0.0  # keep strict to catch errors


def build_dq_config(
    profile_df: pd.DataFrame,
    numeric_cols: Set[str],
    *,
    dataset_name: str,
    sample_rows: int,
    params: ConfigGenParams,
) -> Dict[str, Any]:
    """
    Build dq_agent YAML config dict from a profile dataframe (already normalized).

    Heuristics:
      - string columns with unique <= max_allowed_values -> allowed_values from profile unique set
      - all columns -> not_null check with max_null_rate = profile_null_rate + slack
      - numeric columns -> range(min,max) + outlier_mad anomaly
      - all columns -> missing_rate anomaly
    """
    cfg: Dict[str, Any] = {
        "version": 1,
        "dataset": {
            "name": dataset_name,
            "primary_key": [],  # optional; RAHA datasets don't necessarily have a stable PK
        },
        "columns": {},
        "report": {
            "sample_rows": int(sample_rows),
        },
    }

    for col in profile_df.columns:
        s = profile_df[col]
        null_rate = float(s.isna().mean()) if len(s) else 0.0
        max_null_rate = min(1.0, max(0.0, null_rate + params.null_slack))

        col_cfg: Dict[str, Any] = {
            "type": "int" if col in numeric_cols else "string",
            "required": False,
            "checks": [
                {"not_null": {"max_null_rate": max_null_rate}},
            ],
            "anomalies": [
                {"missing_rate": {"max_rate": max_null_rate}},
            ],
        }

        if col in numeric_cols:
            # range from profile numeric values
            s_num = pd.to_numeric(s, errors="coerce")
            s_num = s_num.dropna()
            if len(s_num):
                mn = float(s_num.min())
                mx = float(s_num.max())
                col_cfg["checks"].append({"range": {"min": mn, "max": mx}})
            col_cfg["anomalies"].append({"outlier_mad": {"z": float(params.z_outlier_mad)}})
        else:
            # allowed_values if feasible
            # We use normalized values (string), excluding NA.
            uniq = sorted(set(x for x in s.dropna().astype("string").tolist()))
            if len(uniq) <= params.max_allowed_values:
                col_cfg["checks"].append({"allowed_values": {"values": uniq}})
            # else: skip allowed_values to avoid huge YAML and runtime

        cfg["columns"][col] = col_cfg

    return cfg


# -----------------------------
# dq_agent runner + parsing
# -----------------------------
def extract_last_json_obj(lines: List[str]) -> Dict[str, Any]:
    """
    Scan from bottom to top; return the last line that's a JSON object.
    Prefer objects containing report_json_path/run_record_path if multiple exist.
    """
    candidates: List[Dict[str, Any]] = []
    for line in reversed(lines):
        s = line.strip()
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                candidates.append(obj)
                # best match
                if "report_json_path" in obj and "run_record_path" in obj:
                    return obj
        except Exception:
            continue

    if candidates:
        return candidates[0]
    raise RuntimeError("No JSON object found in dq_agent output.")


def run_dq_agent(
    *,
    data_path: str,
    config_path: str,
    dq_out_dir: str,
    raw_log_path: str,
    fail_on: str,
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Run `python -m dq_agent run ...` and return (exit_code, wall_time_s, meta_json).
    meta_json contains report_json_path, run_record_path, etc (as printed by dq_agent).
    """
    mkdirp(dq_out_dir)
    cmd = [
        sys.executable,
        "-m",
        "dq_agent",
        "run",
        "--data",
        data_path,
        "--config",
        config_path,
        "--output-dir",
        dq_out_dir,
        "--fail-on",
        fail_on,
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.time()

    out_text = proc.stdout or ""
    with open(raw_log_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    lines = out_text.splitlines()
    meta = extract_last_json_obj(lines)

    return proc.returncode, (t1 - t0), meta


# -----------------------------
# Prediction extraction + truncation detection
# -----------------------------
def _is_fail_status(obj: Dict[str, Any]) -> bool:
    if obj.get("status") == "FAIL":
        return True
    if obj.get("passed") is False:
        return True
    return False


def extract_predicted_cells_from_report(
    report: Dict[str, Any]
) -> Tuple[Set[Tuple[int, str]], Dict[str, Any]]:
    """
    Predicted cells: union of (row_index, column) from FAIL rule_results/anomalies samples.

    Returns:
      (pred_cells, truncation_info)
    """
    pred: Set[Tuple[int, str]] = set()
    details: List[Dict[str, Any]] = []
    truncated = False

    # rule_results
    for r in (report.get("rule_results") or []):
        if not isinstance(r, dict):
            continue
        if not _is_fail_status(r):
            continue
        col = r.get("column")
        if not col:
            continue
        samples = r.get("samples") or []
        if isinstance(samples, list):
            for s in samples:
                if not isinstance(s, dict):
                    continue
                if "row_index" not in s:
                    continue
                try:
                    i = int(s["row_index"])
                except Exception:
                    continue
                pred.add((i, str(col)))

        # truncation heuristic for rule_results
        failed_count = r.get("failed_count")
        if isinstance(failed_count, int) and isinstance(samples, list):
            if len(samples) < failed_count:
                truncated = True
                details.append(
                    {
                        "kind": "rule_result",
                        "rule_id": r.get("rule_id"),
                        "column": col,
                        "failed_count": failed_count,
                        "samples": len(samples),
                    }
                )

    # anomalies
    for a in (report.get("anomalies") or []):
        if not isinstance(a, dict):
            continue
        if a.get("status") != "FAIL":
            continue
        col = a.get("column")
        if not col:
            continue
        samples = a.get("samples") or []
        if isinstance(samples, list):
            for s in samples:
                if not isinstance(s, dict):
                    continue
                if "row_index" not in s:
                    continue
                try:
                    i = int(s["row_index"])
                except Exception:
                    continue
                pred.add((i, str(col)))

        # truncation heuristic for anomalies (if we can infer expected count)
        metric = a.get("metric") or {}
        expected = None
        if isinstance(metric, dict):
            if "null_count" in metric and isinstance(metric["null_count"], int):
                expected = metric["null_count"]
        if expected is not None and isinstance(samples, list):
            if len(samples) < expected:
                truncated = True
                details.append(
                    {
                        "kind": "anomaly",
                        "anomaly_id": a.get("anomaly_id"),
                        "column": col,
                        "expected": expected,
                        "samples": len(samples),
                    }
                )

    return pred, {"truncated": truncated, "details": details}


# -----------------------------
# Metrics
# -----------------------------
def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1(prec, rec),
    }


# -----------------------------
# Main
# ---------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)

    # 兼容 bench_raha.sh 可能传入的不同参数名（hyphen / underscore / alias）
    ap.add_argument(
        "--dirty", "--dirty-csv", "--dirty_csv", "--dirty-path", "--dirty_path",
        dest="dirty", required=True, help="Path to dirty.csv"
    )
    ap.add_argument(
        "--clean", "--clean-csv", "--clean_csv", "--clean-path", "--clean_path",
        dest="clean", required=True, help="Path to clean.csv"
    )
    ap.add_argument(
        "--out", "--out-dir", "--out_dir", "--outdir",
        dest="out", required=True, help="Output directory for this dataset evaluation"
    )
    ap.add_argument(
        "--dataset-name", "--dataset_name", "--dataset",
        dest="dataset_name", required=True, help="Dataset name (e.g., raha/beers)"
    )
    ap.add_argument(
        "--profile-source", "--profile_source", "--profile",
        dest="profile_source", choices=["clean", "dirty"], default="clean",
        help="Which table to use as profiling source for rules"
    )
    ap.add_argument(
        "--fail-on", "--fail_on",
        dest="fail_on", default="INFO",
        help="dq_agent --fail-on level (INFO/WARN/ERROR)"
    )

    ap.add_argument(
        "--max-allowed-values", "--max_allowed_values",
        dest="max_allowed_values", type=int, default=5000,
        help="Max unique values for allowed_values check"
    )
    ap.add_argument(
        "--z-outlier-mad", "--z_outlier_mad",
        dest="z_outlier_mad", type=float, default=6.0,
        help="outlier_mad z threshold"
    )
    ap.add_argument(
        "--null-slack", "--null_slack",
        dest="null_slack", type=float, default=0.0,
        help="Allowable slack added to profile null rate"
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")

    # 关键：不要因为 bench 传了额外参数就直接 SystemExit(2)
    args, unknown = ap.parse_known_args(argv)
    if unknown and getattr(args, "verbose", False):
        eprint("[WARN] ignored unknown args:", unknown)

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = os.path.abspath(args.out)
    mkdirp(out_dir)

    rules_path = os.path.join(out_dir, "dq_rules.yml")
    raw_log_path = os.path.join(out_dir, "dq_raw.log")
    dq_out_dir = os.path.join(out_dir, "dq_out")
    metrics_path = os.path.join(out_dir, "metrics.json")

    if args.verbose:
        eprint(f"[eval] dirty={args.dirty}")
        eprint(f"[eval] clean={args.clean}")
        eprint(f"[eval] out={out_dir}")

    # 1) Load dirty/clean
    dirty_df = read_csv_flexible(args.dirty)
    clean_df = read_csv_flexible(args.clean)

    # sanity row count
    if len(dirty_df) != len(clean_df):
        raise RuntimeError(f"Row count mismatch: dirty={len(dirty_df)} clean={len(clean_df)}")

    # 2) Align columns (FIX)
    dirty_aligned, clean_aligned, align_notes = align_dirty_clean(dirty_df, clean_df)

    # 3) Normalize + truth
    dirty_norm, clean_norm, numeric_cols = normalize_pair(dirty_aligned, clean_aligned)
    truth_cells = compute_truth_error_cells(dirty_norm, clean_norm, numeric_cols)

    # 4) Build config from profile source
    profile_df = clean_norm if args.profile_source == "clean" else dirty_norm

    params = ConfigGenParams(
        max_allowed_values=int(args.max_allowed_values),
        z_outlier_mad=float(args.z_outlier_mad),
        null_slack=float(args.null_slack),
    )

    cfg = build_dq_config(
        profile_df=profile_df,
        numeric_cols=numeric_cols,
        dataset_name=args.dataset_name,
        sample_rows=int(len(dirty_norm)),  # avoid sample truncation
        params=params,
    )

    with open(rules_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # 5) Run dq_agent on DIRTY
    exit_code, wall_time_s, meta = run_dq_agent(
        data_path=args.dirty,
        config_path=rules_path,
        dq_out_dir=dq_out_dir,
        raw_log_path=raw_log_path,
        fail_on=str(args.fail_on),
    )

    report_json_path = meta.get("report_json_path")
    run_record_path = meta.get("run_record_path")

    if not report_json_path or not os.path.exists(report_json_path):
        raise RuntimeError(
            "dq_agent did not produce a readable report.json\n"
            f"exit_code={exit_code}\n"
            f"report_json_path={report_json_path}\n"
            f"raw_log_path={raw_log_path}"
        )

    report = json.load(open(report_json_path, "r", encoding="utf-8"))

    # 6) Extract predictions
    pred_cells, trunc_info = extract_predicted_cells_from_report(report)

    # Keep only columns we evaluated truth on (aligned columns)
    cols_set = set(dirty_norm.columns)
    pred_cells = {(i, c) for (i, c) in pred_cells if c in cols_set and 0 <= i < len(dirty_norm)}

    # 7) Metrics
    tp = len(pred_cells & truth_cells)
    fp = len(pred_cells - truth_cells)
    fn = len(truth_cells - pred_cells)

    cell_m = compute_prf(tp, fp, fn)

    truth_rows = {i for (i, _) in truth_cells}
    pred_rows = {i for (i, _) in pred_cells}

    row_tp = len(truth_rows & pred_rows)
    row_fp = len(pred_rows - truth_rows)
    row_fn = len(truth_rows - pred_rows)
    row_m = compute_prf(row_tp, row_fp, row_fn)

    truth_cols = {c for (_, c) in truth_cells}
    pred_cols = {c for (_, c) in pred_cells}

    metrics: Dict[str, Any] = {
        "dataset": {
            "name": args.dataset_name,
            "rows": int(len(dirty_norm)),
            "cols": int(len(dirty_norm.columns)),
            "profile_source": args.profile_source,
        },
        "dq_agent": {
            "exit_code": int(exit_code),
            "fail_on": str(args.fail_on),
            "wall_time_s": float(wall_time_s),
            "report_json_path": report_json_path,
            "run_record_path": run_record_path,
            "raw_log_path": raw_log_path,
            "rules_path": rules_path,
        },
        "truth": {
            "error_cells": int(len(truth_cells)),
        },
        "pred": {
            "predicted_cells": int(len(pred_cells)),
        },
        "cell": cell_m,
        "row": row_m,
        "column_coverage": {
            "truth_error_columns": int(len(truth_cols)),
            "predicted_error_columns": int(len(pred_cols)),
            "hit_columns": int(len(truth_cols & pred_cols)),
        },
        "truncation": trunc_info,
        "notes": [
            "Truth = dirty vs clean mismatch after normalization (strip, empty->NA, numeric coercion for numeric-like cols).",
            "Pred = dq_agent FAIL rule_results/anomalies samples. Config sets report.sample_rows=nrows to avoid truncation.",
            *align_notes,
        ],
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 8) Print ONE jsonl line summary to stdout (for bench scripts)
    summary_line = {
        "dataset": args.dataset_name,
        "rows": int(len(dirty_norm)),
        "cols": int(len(dirty_norm.columns)),
        "profile": args.profile_source,
        "truth_err_cells": int(len(truth_cells)),
        "predicted": int(len(pred_cells)),
        "cell_f1": metrics["cell"].get("f1"),
        "row_f1": metrics["row"].get("f1"),
        "truncated": bool(trunc_info.get("truncated")),
        "metrics_path": metrics_path,
    }
    print(json.dumps(summary_line, ensure_ascii=False))

    if args.verbose:
        eprint(f"[eval] wrote metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

