#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probe improvement by adding simple string-noise cell predictors and taking UNION with dq_agent predictions.

- Truth: dirty vs clean mismatch after normalization (strip, empty->NA, numeric coercion for numeric-ish cols).
- Base pred: dq_agent report FAIL samples (rule_results + anomalies).
- Noise pred: values matching simple patterns in DIRTY only (e.g., contains '*', contains "''").

Example:
  python scripts/probe_string_noise_union.py \
    --metrics /tmp/tax_gate_eval_clean_profile/metrics.json \
    --dirty /root/dq_benchmarks/src/raha/datasets/tax/dirty.csv \
    --clean /root/dq_benchmarks/src/raha/datasets/tax/clean.csv \
    --contains "*" \
    --contains "''" \
    --numeric-success-threshold 0.98
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

Cell = Tuple[int, str]  # (row_index, column)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_as_str(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def norm_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace("", pd.NA)
    s = s.replace({"NA": pd.NA, "N/A": pd.NA, "NULL": pd.NA, "null": pd.NA, "NaN": pd.NA, "nan": pd.NA})
    return s


def norm_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(norm_str(s), errors="coerce")


def safe_isclose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    close = np.isclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9, rtol=0.0)
    return both_nan | (~one_nan & close)


def prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def align_columns(dirty: pd.DataFrame, clean: pd.DataFrame, metrics: dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    ds = metrics.get("dataset") or {}
    cols = ds.get("aligned_columns") or []
    if cols:
        if set(cols).issubset(set(dirty.columns)) and set(cols).issubset(set(clean.columns)):
            return dirty[cols].copy(), clean[cols].copy(), list(cols)
        # fallback positional
        n = min(len(cols), dirty.shape[1], clean.shape[1])
        d2 = dirty.iloc[:, :n].copy()
        c2 = clean.iloc[:, :n].copy()
        cols = list(cols)[:n]
        d2.columns = cols
        c2.columns = cols
        return d2, c2, cols

    common = [c for c in dirty.columns if c in set(clean.columns)]
    if not common:
        raise RuntimeError("No common columns between dirty and clean, and metrics has no aligned_columns.")
    return dirty[common].copy(), clean[common].copy(), common


def infer_numeric_cols(dirty: pd.DataFrame, clean: pd.DataFrame, cols: Sequence[str], thresh: float) -> Set[str]:
    numeric_cols: Set[str] = set()
    for c in cols:
        dn = norm_num(dirty[c])
        cn = norm_num(clean[c])
        # success rate: non-null numeric after coercion / non-null raw
        d_raw = norm_str(dirty[c])
        c_raw = norm_str(clean[c])
        d_den = max(int(d_raw.notna().sum()), 1)
        c_den = max(int(c_raw.notna().sum()), 1)
        d_rate = float(dn.notna().sum()) / d_den
        c_rate = float(cn.notna().sum()) / c_den
        if d_rate >= thresh and c_rate >= thresh:
            numeric_cols.add(c)
    return numeric_cols


def truth_by_col(dirty: pd.DataFrame, clean: pd.DataFrame, cols: Sequence[str], numeric_cols: Set[str]) -> Dict[str, Set[int]]:
    n = min(len(dirty), len(clean))
    if len(dirty) != len(clean):
        dirty = dirty.iloc[:n].copy()
        clean = clean.iloc[:n].copy()

    out: Dict[str, Set[int]] = {}
    for c in cols:
        if c in numeric_cols:
            dn = norm_num(dirty[c]).to_numpy(dtype=float)
            cn = norm_num(clean[c]).to_numpy(dtype=float)
            eq = safe_isclose(dn, cn)
        else:
            ds = norm_str(dirty[c])
            cs = norm_str(clean[c])
            both_na = ds.isna() & cs.isna()
            eq = ((ds == cs) | both_na).fillna(False).to_numpy()

        bad = set(map(int, np.flatnonzero(~eq)))
        out[c] = bad
    return out


def base_pred_by_col_from_report(report_path: str) -> Dict[str, Set[int]]:
    rep = read_json(report_path)
    pred: Dict[str, Set[int]] = {}

    def add(col: str, idx: int) -> None:
        pred.setdefault(col, set()).add(int(idx))

    for r in (rep.get("rule_results") or []):
        if not (r.get("status") == "FAIL" or r.get("passed") is False):
            continue
        col = r.get("column")
        if not col:
            continue
        for s in (r.get("samples") or []):
            if "row_index" in s:
                add(col, int(s["row_index"]))

    for a in (rep.get("anomalies") or []):
        if a.get("status") != "FAIL":
            continue
        col = a.get("column")
        if not col:
            continue
        for s in (a.get("samples") or []):
            if "row_index" in s:
                add(col, int(s["row_index"]))

    return pred


def noise_pred_by_col(
    dirty: pd.DataFrame,
    cols: Sequence[str],
    contains: Sequence[str],
    regexes: Sequence[str],
) -> Dict[str, Set[int]]:
    pred: Dict[str, Set[int]] = {}
    # precompile regex
    cre = [re.compile(p) for p in regexes]

    for c in cols:
        s = norm_str(dirty[c])
        if s.isna().all():
            continue
        mask = pd.Series(False, index=s.index)

        # contains (literal substring)
        for sub in contains:
            if not sub:
                continue
            mask = mask | s.fillna("").str.contains(re.escape(sub), regex=True)

        # regex (python)
        for rx in cre:
            mask = mask | s.fillna("").str.contains(rx, regex=True)

        idx = set(map(int, np.flatnonzero(mask.to_numpy())))
        if idx:
            pred[c] = idx
    return pred


def eval_cell(truth: Dict[str, Set[int]], pred: Dict[str, Set[int]], cols: Sequence[str]) -> dict:
    t: Set[Cell] = set()
    p: Set[Cell] = set()
    for c in cols:
        for i in truth.get(c, set()):
            t.add((i, c))
        for i in pred.get(c, set()):
            p.add((i, c))
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    out = prf(tp, fp, fn)
    out["truth_cells"] = len(t)
    out["pred_cells"] = len(p)
    return out


def eval_row(truth: Dict[str, Set[int]], pred: Dict[str, Set[int]], cols: Sequence[str]) -> dict:
    t: Set[int] = set()
    p: Set[int] = set()
    for c in cols:
        t |= truth.get(c, set())
        p |= pred.get(c, set())
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    out = prf(tp, fp, fn)
    out["truth_rows"] = len(t)
    out["pred_rows"] = len(p)
    return out


def union_pred(a: Dict[str, Set[int]], b: Dict[str, Set[int]]) -> Dict[str, Set[int]]:
    out = {k: set(v) for k, v in a.items()}
    for k, v in b.items():
        out.setdefault(k, set()).update(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--dirty", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--contains", action="append", default=[], help='literal substring, repeatable (e.g. "*" , "''")')
    ap.add_argument("--regex", action="append", default=[], help="python regex, repeatable")
    ap.add_argument("--numeric-success-threshold", type=float, default=0.98)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    m = read_json(args.metrics)
    report = (m.get("dq_agent") or {}).get("report_json_path")
    if not report or not os.path.exists(report):
        raise SystemExit(f"report_json_path missing or not found: {report}")

    dirty = read_csv_as_str(args.dirty)
    clean = read_csv_as_str(args.clean)
    dirty, clean, cols = align_columns(dirty, clean, m)

    numeric_cols = set((m.get("dataset") or {}).get("numeric_columns") or [])
    if not numeric_cols:
        numeric_cols = infer_numeric_cols(dirty, clean, cols, args.numeric_success_threshold)

    truth = truth_by_col(dirty, clean, cols, numeric_cols)
    base = base_pred_by_col_from_report(report)
    noise = noise_pred_by_col(dirty, cols, args.contains, args.regex)
    uni = union_pred(base, noise)

    base_cell = eval_cell(truth, base, cols)
    noise_cell = eval_cell(truth, noise, cols)
    uni_cell = eval_cell(truth, uni, cols)

    base_row = eval_row(truth, base, cols)
    noise_row = eval_row(truth, noise, cols)
    uni_row = eval_row(truth, uni, cols)

    print("\n=== BASE (dq_agent) ===")
    print("report:", report)
    print("cell:", {k: base_cell[k] for k in ["tp","fp","fn","precision","recall","f1"]})
    print("row :", {k: base_row[k] for k in ["tp","fp","fn","precision","recall","f1"]})

    print("\n=== NOISE (patterns on dirty) ===")
    print("contains:", args.contains, "regex:", args.regex)
    print("cell:", {k: noise_cell[k] for k in ["tp","fp","fn","precision","recall","f1"]})
    print("row :", {k: noise_row[k] for k in ["tp","fp","fn","precision","recall","f1"]})

    print("\n=== UNION (BASE ∪ NOISE) ===")
    print("cell:", {k: uni_cell[k] for k in ["tp","fp","fn","precision","recall","f1"]})
    print("row :", {k: uni_row[k] for k in ["tp","fp","fn","precision","recall","f1"]})

    # where did we gain TP?
    gained: List[Tuple[str,int,int,int]] = []  # (col, new_tp, new_fp, new_fn)
    for c in cols:
        t = truth.get(c, set())
        b = base.get(c, set())
        n = noise.get(c, set())
        u = uni.get(c, set())
        # incremental contribution of NOISE beyond BASE
        inc = (u - b)
        new_tp = len(inc & t)
        new_fp = len(inc - t)
        # new_fn not directly meaningful here; show TP/FP gain
        if new_tp or new_fp:
            gained.append((c, new_tp, new_fp, len(t)))

    gained.sort(key=lambda x: (x[1] - x[2], x[1]), reverse=True)

    print(f"\n== Top columns where NOISE adds TP (beyond BASE) top={args.topk} ==")
    for c, tp_gain, fp_gain, truth_cnt in gained[: args.topk]:
        print(f"- {c:20s} tp_gain={tp_gain:6d} fp_gain={fp_gain:6d} truth_cells={truth_cnt:6d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
