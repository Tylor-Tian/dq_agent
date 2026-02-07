#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probe how much recall we can recover on Raha dirty/clean labeled datasets
by adding simple Functional-Dependency (FD) / majority-lookup predictors,
and measuring union(pred_dq_agent, pred_fd).

This is a "what-if" analysis tool to guide engineering work:
- base preds come from dq_agent report FAIL samples (rule_results + anomalies)
- FD preds are computed from the DIRTY dataset itself:
    for each key value, take the most frequent dependent value (mode),
    and if confidence >= threshold & support >= threshold, flag rows that deviate.

Usage example (tax):
  python scripts/probe_fd_union.py \
    --metrics /tmp/tax_gate_eval_clean_profile/metrics.json \
    --dirty /root/dq_benchmarks/src/raha/datasets/tax/dirty.csv \
    --clean /root/dq_benchmarks/src/raha/datasets/tax/clean.csv \
    --pair zip:state \
    --pair zip:city \
    --min-support 50 \
    --min-confidence 0.99 \
    --show-mismatches 5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


Cell = Tuple[int, str]  # (row_index, column)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_as_str(path: str) -> pd.DataFrame:
    # keep_default_na=False + na_filter=False -> preserve empty strings; we normalize later.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    # strip BOM/whitespace in headers
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def norm_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace("", pd.NA)
    s = s.replace(
        {
            "NA": pd.NA,
            "N/A": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "NaN": pd.NA,
            "nan": pd.NA,
        }
    )
    return s


def norm_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(norm_str(s), errors="coerce")


def safe_isclose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a/b are float arrays, may contain nan
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    close = np.isclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9, rtol=0.0)
    return both_nan | (~one_nan & close)


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def align_by_metrics(
    dirty: pd.DataFrame, clean: pd.DataFrame, metrics: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Set[str]]:
    ds = metrics.get("dataset") or {}
    cols: List[str] = ds.get("aligned_columns") or []
    numeric_cols: Set[str] = set(ds.get("numeric_columns") or [])

    if cols:
        # name-based first; if missing, fallback to positional alignment with the same length
        if set(cols).issubset(set(dirty.columns)) and set(cols).issubset(set(clean.columns)):
            dirty2 = dirty[cols].copy()
            clean2 = clean[cols].copy()
        else:
            n = min(len(cols), dirty.shape[1], clean.shape[1])
            dirty2 = dirty.iloc[:, :n].copy()
            clean2 = clean.iloc[:, :n].copy()
            cols = cols[:n]
            dirty2.columns = cols
            clean2.columns = cols
        return dirty2, clean2, cols, numeric_cols

    # fallback: intersection
    common = [c for c in dirty.columns if c in set(clean.columns)]
    if not common:
        raise RuntimeError("No common columns and metrics.json has no aligned_columns.")
    dirty2 = dirty[common].copy()
    clean2 = clean[common].copy()
    return dirty2, clean2, common, set()


def compute_truth_cells(
    dirty: pd.DataFrame, clean: pd.DataFrame, cols: Sequence[str], numeric_cols: Set[str]
) -> Dict[str, Set[int]]:
    """
    Return truth mismatches per column: {col -> set(row_index)}
    truth definition is consistent with eval_dirty_clean.py notes:
      - strip whitespace
      - empty -> NA
      - numeric cols -> numeric coercion then isclose
      - string cols -> exact match after normalization
    """
    n = min(len(dirty), len(clean))
    if len(dirty) != len(clean):
        dirty = dirty.iloc[:n].copy()
        clean = clean.iloc[:n].copy()

    truth_by_col: Dict[str, Set[int]] = {}

    for c in cols:
        if c in numeric_cols:
            dn = norm_num(dirty[c])
            cn = norm_num(clean[c])
            eq = safe_isclose(dn.to_numpy(dtype=float), cn.to_numpy(dtype=float))
        else:
            ds = norm_str(dirty[c])
            cs = norm_str(clean[c])
            both_na = ds.isna() & cs.isna()
            eq = ((ds == cs) | both_na).fillna(False).to_numpy()

        bad_idx = set(map(int, np.flatnonzero(~eq)))
        truth_by_col[c] = bad_idx

    return truth_by_col


def load_pred_cells_from_report(report_json_path: str) -> Dict[str, Set[int]]:
    """
    Pred per column from dq_agent report FAIL samples:
      - rule_results with status FAIL (or passed is False)
      - anomalies with status FAIL
    """
    rep = read_json(report_json_path)

    pred_by_col: Dict[str, Set[int]] = {}

    def add(col: Optional[str], row_index: Optional[int]) -> None:
        if col is None or row_index is None:
            return
        pred_by_col.setdefault(col, set()).add(int(row_index))

    for r in (rep.get("rule_results") or []):
        if not (r.get("status") == "FAIL" or r.get("passed") is False):
            continue
        col = r.get("column")
        for s in (r.get("samples") or []):
            add(col, s.get("row_index"))

    for a in (rep.get("anomalies") or []):
        if a.get("status") != "FAIL":
            continue
        col = a.get("column")
        for s in (a.get("samples") or []):
            add(col, s.get("row_index"))

    return pred_by_col


@dataclass
class FDPrediction:
    pair: str
    key: str
    val: str
    predicted_rows: Set[int]
    stats: Dict[str, float]


def fd_majority_predict(
    df: pd.DataFrame,
    key_col: str,
    val_col: str,
    min_support: int,
    min_confidence: float,
) -> Tuple[Set[int], Dict[str, float]]:
    """
    Compute majority mapping from df itself:
      for each key, find modal val and its confidence
      if (support >= min_support) & (confidence >= min_confidence),
      flag rows where val != modal val.
    """
    if key_col not in df.columns or val_col not in df.columns:
        return set(), {"keys_considered": 0, "keys_qualified": 0, "pred_rows": 0}

    k = norm_str(df[key_col])
    v = norm_str(df[val_col])

    g = pd.DataFrame({"k": k, "v": v})
    g = g.dropna(subset=["k"])  # ignore NA keys
    if g.empty:
        return set(), {"keys_considered": 0, "keys_qualified": 0, "pred_rows": 0}

    # counts per (k,v)
    counts = g.groupby(["k", "v"], dropna=False).size().reset_index(name="n")

    total = counts.groupby("k")["n"].sum()
    idx = counts.groupby("k")["n"].idxmax()
    top = counts.loc[idx, ["k", "v", "n"]].set_index("k")
    top["total"] = total
    top["conf"] = top["n"] / top["total"]

    keys_considered = int(top.shape[0])

    good = top[(top["total"] >= min_support) & (top["conf"] >= min_confidence)]
    keys_qualified = int(good.shape[0])

    mapping: Dict[str, str] = good["v"].astype("string").to_dict()

    exp = k.map(mapping)  # expected val for qualified keys
    # rows where key is qualified and val differs
    mask = exp.notna() & (v != exp)
    pred_rows = set(map(int, np.flatnonzero(mask.fillna(False).to_numpy())))

    return pred_rows, {
        "keys_considered": float(keys_considered),
        "keys_qualified": float(keys_qualified),
        "pred_rows": float(len(pred_rows)),
    }


def eval_cells(
    truth_by_col: Dict[str, Set[int]],
    pred_by_col: Dict[str, Set[int]],
    cols: Sequence[str],
) -> Dict[str, float]:
    truth_cells: Set[Cell] = set()
    pred_cells: Set[Cell] = set()

    for c in cols:
        for i in truth_by_col.get(c, set()):
            truth_cells.add((i, c))
        for i in pred_by_col.get(c, set()):
            pred_cells.add((i, c))

    tp = len(truth_cells & pred_cells)
    fp = len(pred_cells - truth_cells)
    fn = len(truth_cells - pred_cells)

    out = prf(tp, fp, fn)
    out["truth_cells"] = float(len(truth_cells))
    out["pred_cells"] = float(len(pred_cells))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics.json produced by eval_dirty_clean.py")
    ap.add_argument("--dirty", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--pair", action="append", default=[], help="FD pair KEY:VAL (repeatable), e.g. zip:state")
    ap.add_argument("--min-support", type=int, default=50)
    ap.add_argument("--min-confidence", type=float, default=0.99)
    ap.add_argument("--show-mismatches", type=int, default=0, help="Print N mismatched examples for top FN cols")
    args = ap.parse_args()

    metrics = read_json(args.metrics)
    report_path = (metrics.get("dq_agent") or {}).get("report_json_path")
    if not report_path or not os.path.exists(report_path):
        raise SystemExit(f"report_json_path not found in metrics or not exists: {report_path}")

    dirty = read_csv_as_str(args.dirty)
    clean = read_csv_as_str(args.clean)
    dirty, clean, cols, numeric_cols = align_by_metrics(dirty, clean, metrics)

    truth_by_col = compute_truth_cells(dirty, clean, cols, numeric_cols)
    base_pred_by_col = load_pred_cells_from_report(report_path)

    # base metrics
    base = eval_cells(truth_by_col, base_pred_by_col, cols)

    print("\n=== BASE (dq_agent) ===")
    print("report:", report_path)
    print("cell:", {k: base[k] for k in ["tp","fp","fn","precision","recall","f1"]})
    print("truth_cells:", int(base["truth_cells"]), "pred_cells:", int(base["pred_cells"]))

    # FN ranking
    per_col = []
    for c in cols:
        t = truth_by_col.get(c, set())
        p = base_pred_by_col.get(c, set())
        tp = len(t & p)
        fp = len(p - t)
        fn = len(t - p)
        met = prf(tp, fp, fn)
        per_col.append((c, len(t), len(p), met["tp"], met["fp"], met["fn"], met["precision"], met["recall"], met["f1"]))

    per_col_sorted = sorted(per_col, key=lambda x: x[5], reverse=True)

    print("\n== Top FN columns (BASE) ==")
    for c, tn, pn, tp, fp, fn, prec, rec, f1v in per_col_sorted[:10]:
        print(f"{c:20s} FN={fn:6d} truth={tn:6d} pred={pn:6d} recall={rec:.3f} prec={prec:.3f} f1={f1v:.3f}")

    if args.show_mismatches > 0:
        print("\n== Sample mismatches for Top FN columns ==")
        for c, tn, pn, tp, fp, fn, prec, rec, f1v in per_col_sorted[:6]:
            if tn == 0:
                continue
            # print first N mismatch rows
            idx = sorted(list(truth_by_col[c]))[: args.show_mismatches]
            print(f"\n-- {c} (show {len(idx)}/{tn}) --")
            for i in idx:
                dv = dirty[c].iloc[i]
                cv = clean[c].iloc[i]
                print(f"row={i:7d} dirty={repr(dv)}  clean={repr(cv)}")

    # FD predictions
    fd_preds: List[FDPrediction] = []
    fd_pred_by_col: Dict[str, Set[int]] = {}

    for pair in args.pair:
        if ":" not in pair:
            raise SystemExit(f"bad --pair format: {pair} (expected KEY:VAL)")
        key, val = pair.split(":", 1)
        key = key.strip()
        val = val.strip()
        rows, stats = fd_majority_predict(dirty, key, val, args.min_support, args.min_confidence)
        fd_pred_by_col.setdefault(val, set()).update(rows)
        fd_preds.append(FDPrediction(pair=pair, key=key, val=val, predicted_rows=rows, stats=stats))

    if fd_preds:
        fd_only = eval_cells(truth_by_col, fd_pred_by_col, cols)

        # union preds
        union_pred_by_col = {c: set(base_pred_by_col.get(c, set())) for c in cols}
        for c, s in fd_pred_by_col.items():
            union_pred_by_col.setdefault(c, set()).update(s)
        union = eval_cells(truth_by_col, union_pred_by_col, cols)

        print("\n=== FD-ONLY ===")
        print("cell:", {k: fd_only[k] for k in ["tp","fp","fn","precision","recall","f1"]})
        print("truth_cells:", int(fd_only["truth_cells"]), "pred_cells:", int(fd_only["pred_cells"]))

        print("\n=== UNION (dq_agent ∪ FD) ===")
        print("cell:", {k: union[k] for k in ["tp","fp","fn","precision","recall","f1"]})
        print("truth_cells:", int(union["truth_cells"]), "pred_cells:", int(union["pred_cells"]))

        print("\n== FD pair details ==")
        for x in fd_preds:
            # per-column eval for that dependent col
            t = truth_by_col.get(x.val, set())
            p = x.predicted_rows
            met = prf(len(t & p), len(p - t), len(t - p))
            print(f"- {x.pair:15s}  keys_considered={int(x.stats['keys_considered'])}  "
                  f"keys_qualified={int(x.stats['keys_qualified'])}  pred_rows={len(p)}  "
                  f"-> {x.val}  prec={met['precision']:.3f} rec={met['recall']:.3f} f1={met['f1']:.3f}")

    else:
        print("\n(no --pair provided, FD probing skipped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
