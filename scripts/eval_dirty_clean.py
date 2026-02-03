#!/usr/bin/env python3
"""
Evaluate dq_agent on a (dirty.csv, clean.csv) pair (Raha-style).

Truth:
  cells where normalized(dirty) != normalized(clean)

Pred:
  cells that appear in dq_agent report FAIL rule_results/anomalies "samples"
  (we rely on report.sample_rows being large enough to avoid truncation)

Key improvement for dirty-profile:
  coverage-based categorical domains:
    allowed_values = most frequent values covering `--domain-cover` fraction,
    up to `--max-domain` values.
  If coverage is too low under max-domain (high-cardinality columns), we skip
  allowed_values for that column to avoid flooding false positives.

Outputs:
  - <out>/rules.yml
  - <out>/dirty.parquet
  - <out>/dq_out/<run_id>/report.json (via dq_agent)
  - <out>/metrics.json
  - one-line JSON summary to stdout (for bench scripts)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml


Cell = Tuple[int, str]  # (row_index, column_name)


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def read_csv(path: Path) -> pd.DataFrame:
    # Keep strings as-is; avoid pandas turning empty into NaN automatically.
    # We'll normalize ourselves.
    df = pd.read_csv(path, dtype="string", keep_default_na=False)
    # Drop common index columns accidentally saved.
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def normalize_string_series(s: pd.Series) -> pd.Series:
    # s is string dtype
    s2 = s.astype("string")
    s2 = s2.str.strip()
    # empty -> NA
    s2 = s2.replace("", pd.NA)
    return s2


def normalize_numeric_series(s: pd.Series) -> pd.Series:
    s2 = normalize_string_series(s)
    # coercion
    x = pd.to_numeric(s2, errors="coerce")
    return x


def equal_mask(a: pd.Series, b: pd.Series) -> pd.Series:
    # True where equal; NA==NA treated as equal.
    a_na = a.isna()
    b_na = b.isna()
    both_na = a_na & b_na
    eq = (a == b)
    # comparisons involving NA yield <NA>; treat as False
    eq = eq.fillna(False)
    return both_na | eq


def align_columns(dirty: pd.DataFrame, clean: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    dcols = list(dirty.columns)
    ccols = list(clean.columns)
    common = [c for c in dcols if c in clean.columns]

    if common:
        if verbose:
            _eprint(f"[align] common columns={len(common)}")
        dirty2 = dirty[common].copy()
        clean2 = clean[common].copy()
        return dirty2, clean2, common

    # No common columns by name: try align by position if same width.
    if dirty.shape[1] == clean.shape[1]:
        if verbose:
            _eprint("[align] no common names; aligning by position")
            _eprint("[align] dirty cols:", dcols[:10], ("..." if len(dcols) > 10 else ""))
            _eprint("[align] clean cols:", ccols[:10], ("..." if len(ccols) > 10 else ""))
        clean2 = clean.copy()
        clean2.columns = dcols
        return dirty.copy(), clean2, dcols

    raise RuntimeError(
        "No common columns between dirty and clean, and column counts differ. "
        f"dirty_cols={dirty.shape[1]} clean_cols={clean.shape[1]}"
    )


def infer_numeric_cols(
    profile: pd.DataFrame,
    cols: Sequence[str],
    numeric_success_threshold: float,
) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for c in cols:
        s = normalize_string_series(profile[c])
        nonnull = s.dropna()
        if nonnull.empty:
            out[c] = False
            continue
        x = pd.to_numeric(nonnull, errors="coerce")
        success = float(x.notna().mean())
        out[c] = success >= numeric_success_threshold
    return out


@dataclass
class DomainResult:
    allowed: Optional[List[str]]
    coverage: float
    unique: int
    used: bool
    truncated: bool
    reason: str


def coverage_domain(
    s: pd.Series,
    cover: float,
    max_domain: int,
    min_usable_coverage: float = 0.50,
) -> DomainResult:
    s2 = normalize_string_series(s)
    nonnull = s2.dropna()
    if nonnull.empty:
        return DomainResult(allowed=None, coverage=0.0, unique=0, used=False, truncated=False, reason="empty")

    vc = nonnull.value_counts(dropna=True)
    total = int(vc.sum())
    unique = int(vc.shape[0])

    # If column is extremely high-cardinality and top-k coverage is too low,
    # allowed_values would create massive false positives => skip.
    topk = vc.iloc[:max_domain]
    topk_cov = float(topk.sum()) / float(total) if total > 0 else 0.0
    if unique > max_domain and topk_cov < min_usable_coverage:
        return DomainResult(
            allowed=None,
            coverage=topk_cov,
            unique=unique,
            used=False,
            truncated=True,
            reason=f"high_cardinality(top{max_domain}_coverage={topk_cov:.3f})",
        )

    allowed: List[str] = []
    cum = 0
    for val, cnt in vc.items():
        allowed.append(str(val))
        cum += int(cnt)
        if total > 0 and (cum / total) >= cover:
            break
        if len(allowed) >= max_domain:
            break

    cov = float(cum) / float(total) if total > 0 else 0.0
    truncated = (len(allowed) < unique) and (cov < cover)
    used = True
    reason = "ok"
    if unique > max_domain and truncated:
        reason = f"truncated_to_max_domain({max_domain})"
    elif truncated:
        reason = "truncated_before_cover"

    return DomainResult(allowed=allowed, coverage=cov, unique=unique, used=used, truncated=truncated, reason=reason)


def build_rules_config(
    dataset_name: str,
    profile: pd.DataFrame,
    cols: Sequence[str],
    is_numeric: Dict[str, bool],
    nrows: int,
    domain_cover: float,
    max_domain: int,
    z_outlier_mad: float,
    null_slack: float,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns: (config_dict, notes)
    """
    notes: List[str] = []

    cfg: Dict[str, Any] = {
        "version": 1,
        "dataset": {
            "name": dataset_name,
            "primary_key": ["row_id"],
        },
        "columns": {},
        "report": {
            # critical: to avoid truncation in report samples for metrics computation
            "sample_rows": int(nrows),
        },
    }

    # Always include row_id
    cfg["columns"]["row_id"] = {
        "type": "int",
        "required": True,
        "checks": [{"range": {"min": 1, "max": int(nrows)}}],
    }

    for c in cols:
        # Evaluation columns exclude row_id; but config may include all.
        col_cfg: Dict[str, Any] = {"required": True}

        if is_numeric.get(c, False):
            # numeric
            x = normalize_numeric_series(profile[c])
            nonnull = x.dropna()
            # If no numeric values, fallback to string
            if nonnull.empty:
                col_cfg["type"] = "string"
                # minimal checks
                col_cfg["checks"] = [{"not_null": {"max_null_rate": float(null_slack)}}]
                col_cfg["anomalies"] = [{"missing_rate": {"max_rate": float(null_slack)}}]
                cfg["columns"][c] = col_cfg
                continue

            # Determine int vs float
            vals = nonnull.to_numpy()
            is_int = np.all(np.isfinite(vals) & (np.mod(vals, 1) == 0))
            col_cfg["type"] = "int" if bool(is_int) else "float"

            # Robust range: use extreme quantiles to reduce sensitivity to rare outliers.
            lo = float(np.nanpercentile(vals, 0.1))
            hi = float(np.nanpercentile(vals, 99.9))
            if lo > hi:
                lo, hi = hi, lo
            span = hi - lo
            if span == 0:
                # constant column
                lo2, hi2 = lo, hi
            else:
                lo2 = lo - 0.01 * span
                hi2 = hi + 0.01 * span

            col_cfg["checks"] = [
                {"not_null": {"max_null_rate": float(null_slack)}},
                {"range": {"min": lo2, "max": hi2}},
            ]
            col_cfg["anomalies"] = [
                {"missing_rate": {"max_rate": float(null_slack)}},
                {"outlier_mad": {"z": float(z_outlier_mad)}},
            ]
        else:
            # categorical/string
            col_cfg["type"] = "string"
            checks: List[Dict[str, Any]] = [{"not_null": {"max_null_rate": float(null_slack)}}]
            anomalies: List[Dict[str, Any]] = [{"missing_rate": {"max_rate": float(null_slack)}}]

            dom = coverage_domain(profile[c], cover=domain_cover, max_domain=max_domain)
            if verbose:
                _eprint(f"[domain] {c}: used={dom.used} unique={dom.unique} coverage={dom.coverage:.3f} reason={dom.reason}")

            if dom.used and dom.allowed:
                checks.append({"allowed_values": {"values": dom.allowed}})
                if dom.truncated:
                    notes.append(f"domain_truncated:{c}:{dom.reason}:coverage={dom.coverage:.3f}:unique={dom.unique}")
            else:
                notes.append(f"domain_skipped:{c}:{dom.reason}:coverage={dom.coverage:.3f}:unique={dom.unique}")

            col_cfg["checks"] = checks
            col_cfg["anomalies"] = anomalies

        cfg["columns"][c] = col_cfg

    return cfg, notes


def extract_last_meta_json(raw_text: str) -> Dict[str, Any]:
    """
    dq_agent prints a JSON line containing report_json_path/run_record_path.
    We search from bottom for the last JSON object that contains these keys.
    """
    for line in reversed(raw_text.splitlines()):
        s = line.strip()
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict) and ("report_json_path" in obj) and ("run_record_path" in obj):
            return obj
    raise RuntimeError("No dq_agent meta JSON found in output.")


def run_dq_agent(
    data_path: Path,
    config_path: Path,
    out_dir: Path,
    fail_on: str,
    raw_log_path: Path,
) -> Tuple[Dict[str, Any], int, float, str]:
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
        str(out_dir),
        "--fail-on",
        fail_on,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.time()
    raw = proc.stdout
    raw_log_path.write_text(raw, encoding="utf-8")
    meta = extract_last_meta_json(raw)
    return meta, int(proc.returncode), float(t1 - t0), raw


def extract_predictions(report: Dict[str, Any]) -> Tuple[Set[Cell], Dict[str, Any]]:
    pred: Set[Cell] = set()
    trunc_details: List[Dict[str, Any]] = []

    # rule_results
    rr = report.get("rule_results") or []
    for r in rr:
        if not isinstance(r, dict):
            continue
        status = r.get("status")
        passed = r.get("passed")
        is_fail = (status == "FAIL") or (passed is False)
        if not is_fail:
            continue

        col = pick(r, "column", "col")
        if not col:
            continue

        samples = r.get("samples") or []
        for s in samples:
            if not isinstance(s, dict):
                continue
            idx = pick(s, "row_index", "row", "index")
            if idx is None:
                continue
            pred.add((int(idx), str(col)))

        failed_count = r.get("failed_count")
        if failed_count is not None and isinstance(failed_count, (int, float)):
            if len(samples) < int(failed_count):
                trunc_details.append(
                    {
                        "kind": "rule_result",
                        "rule_id": r.get("rule_id"),
                        "column": str(col),
                        "failed_count": int(failed_count),
                        "samples": len(samples),
                    }
                )

    # anomalies
    an = report.get("anomalies") or []
    for a in an:
        if not isinstance(a, dict):
            continue
        status = a.get("status")
        if status != "FAIL":
            continue
        col = pick(a, "column", "col")
        if not col:
            continue

        samples = a.get("samples") or []
        for s in samples:
            if not isinstance(s, dict):
                continue
            idx = pick(s, "row_index", "row", "index")
            if idx is None:
                continue
            pred.add((int(idx), str(col)))

        metric = a.get("metric") or {}
        # best-effort truncation detection
        for k in ("null_count", "outlier_count", "n_outliers", "num_outliers", "count_outliers"):
            if k in metric and isinstance(metric[k], (int, float)):
                if len(samples) < int(metric[k]):
                    trunc_details.append(
                        {
                            "kind": "anomaly",
                            "anomaly_id": a.get("anomaly_id"),
                            "column": str(col),
                            "metric_count_key": k,
                            "metric_count": int(metric[k]),
                            "samples": len(samples),
                        }
                    )
                break

    trunc = {"truncated": bool(trunc_details), "details": trunc_details}
    return pred, trunc


def compute_truth_cells(
    dirty: pd.DataFrame,
    clean: pd.DataFrame,
    cols: Sequence[str],
    is_numeric: Dict[str, bool],
) -> Set[Cell]:
    if len(dirty) != len(clean):
        raise RuntimeError(f"Row count mismatch: dirty={len(dirty)} clean={len(clean)}")

    truth: Set[Cell] = set()
    for c in cols:
        if is_numeric.get(c, False):
            a = normalize_numeric_series(dirty[c])
            b = normalize_numeric_series(clean[c])
        else:
            a = normalize_string_series(dirty[c])
            b = normalize_string_series(clean[c])

        eq = equal_mask(a, b)
        mismatch = (~eq).to_numpy()
        idxs = np.flatnonzero(mismatch)
        for i in idxs:
            truth.add((int(i), str(c)))

    return truth


def f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # returns (precision, recall, f1), with None where undefined (no truth)
    tp = int(tp); fp = int(fp); fn = int(fn)
    pred_pos = tp + fp
    truth_pos = tp + fn

    if truth_pos == 0:
        # no positives in truth: recall undefined
        precision = 1.0 if pred_pos == 0 else 0.0
        recall = None
        f1 = None if pred_pos == 0 else 0.0
        return precision, recall, f1

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / truth_pos if truth_pos > 0 else 0.0
    if (precision + recall) == 0:
        return precision, recall, 0.0
    return precision, recall, (2 * precision * recall / (precision + recall))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dirty", required=True, help="Path to dirty.csv")
    p.add_argument("--clean", required=True, help="Path to clean.csv")
    p.add_argument("--out", required=True, help="Output directory for this dataset evaluation")
    p.add_argument("--dataset-name", "--dataset_name", required=True, dest="dataset_name")

    p.add_argument("--profile-source", "--profile_source", dest="profile_source", choices=["clean", "dirty"], default="clean")
    p.add_argument("--fail-on", "--fail_on", dest="fail_on", choices=["INFO", "WARN", "ERROR"], default="INFO")

    p.add_argument("--domain-cover", "--domain_cover", dest="domain_cover", type=float, default=0.99)
    p.add_argument("--max-domain", "--max_domain", dest="max_domain", type=int, default=2000)
    p.add_argument("--max-allowed-values", "--max_allowed_values", dest="max_allowed_values", type=int, default=None)

    p.add_argument("--numeric-success-threshold", "--numeric_success_threshold",
                   dest="numeric_success_threshold", type=float, default=0.98)
    p.add_argument("--z-outlier-mad", "--z_outlier_mad", dest="z_outlier_mad", type=float, default=6.0)
    p.add_argument("--null-slack", "--null_slack", dest="null_slack", type=float, default=0.02)

    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()

    dirty_path = Path(args.dirty).resolve()
    clean_path = Path(args.clean).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # effective max_domain
    max_domain = int(args.max_domain)
    if args.max_allowed_values is not None:
        max_domain = min(max_domain, int(args.max_allowed_values))

    dirty_df = read_csv(dirty_path)
    clean_df = read_csv(clean_path)
    dirty_df, clean_df, cols = align_columns(dirty_df, clean_df, verbose=args.verbose)

    nrows = int(len(dirty_df))
    ncols = int(len(cols))

    profile_df = clean_df if args.profile_source == "clean" else dirty_df
    is_num = infer_numeric_cols(profile_df, cols, numeric_success_threshold=float(args.numeric_success_threshold))

    # TRUTH
    truth = compute_truth_cells(dirty_df, clean_df, cols, is_num)

    # Build dirty parquet for dq_agent
    dirty_run = dirty_df.copy()
    dirty_run.insert(0, "row_id", range(1, nrows + 1))
    data_parquet = out_dir / "dirty.parquet"
    dirty_run.to_parquet(data_parquet, index=False)

    # Rules config
    cfg, cfg_notes = build_rules_config(
        dataset_name=args.dataset_name,
        profile=profile_df,
        cols=cols,
        is_numeric=is_num,
        nrows=nrows,
        domain_cover=float(args.domain_cover),
        max_domain=max_domain,
        z_outlier_mad=float(args.z_outlier_mad),
        null_slack=float(args.null_slack),
        verbose=args.verbose,
    )
    rules_path = out_dir / "rules.yml"
    rules_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Run dq_agent
    dq_out = out_dir / "dq_out"
    dq_out.mkdir(parents=True, exist_ok=True)
    raw_log = out_dir / "dq_raw.log"
    meta, dq_ec, wall, raw = run_dq_agent(data_parquet, rules_path, dq_out, args.fail_on, raw_log)

    report_json_path = Path(meta["report_json_path"]).resolve()
    run_record_path = Path(meta["run_record_path"]).resolve()

    # dq_agent can exit 2 when fail-on triggers; that's expected.
    # Exit 1 indicates unexpected failure.
    if dq_ec not in (0, 2):
        raise RuntimeError(f"dq_agent unexpected exit_code={dq_ec}. See {raw_log}")

    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    pred, trunc = extract_predictions(report)

    # Compute metrics
    # Filter to evaluation columns only (exclude synthetic row_id if it sneaks in)
    pred = {(i, c) for (i, c) in pred if c in cols}
    truth = {(i, c) for (i, c) in truth if c in cols}

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    c_prec, c_rec, c_f1 = f1_from_counts(tp, fp, fn)

    truth_rows = {i for (i, _) in truth}
    pred_rows = {i for (i, _) in pred}

    rtp = len(truth_rows & pred_rows)
    rfp = len(pred_rows - truth_rows)
    rfn = len(truth_rows - pred_rows)

    r_prec, r_rec, r_f1 = f1_from_counts(rtp, rfp, rfn)

    truth_err_cols = {c for (_, c) in truth}
    pred_err_cols = {c for (_, c) in pred}
    hit_cols = truth_err_cols & pred_err_cols

    metrics_path = out_dir / "metrics.json"
    metrics = {
        "dataset": {
            "name": args.dataset_name,
            "rows": nrows,
            "cols": ncols,
            "profile_source": args.profile_source,
        },
        "dq_agent": {
            "exit_code": dq_ec,
            "fail_on": args.fail_on,
            "wall_time_s": wall,
            "report_json_path": str(report_json_path),
            "run_record_path": str(run_record_path),
            "raw_log_path": str(raw_log),
            "rules_path": str(rules_path),
            "data_parquet_path": str(data_parquet),
        },
        "truth": {
            "error_cells": len(truth),
        },
        "pred": {
            "predicted_cells": len(pred),
        },
        "cell": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": c_prec,
            "recall": c_rec,
            "f1": c_f1,
        },
        "row": {
            "tp": rtp,
            "fp": rfp,
            "fn": rfn,
            "precision": r_prec,
            "recall": r_rec,
            "f1": r_f1,
        },
        "column_coverage": {
            "truth_error_columns": len(truth_err_cols),
            "predicted_error_columns": len(pred_err_cols),
            "hit_columns": len(hit_cols),
        },
        "truncation": trunc,
        "notes": [
            "Truth = dirty vs clean mismatch after normalization (strip, empty->NA, numeric coercion for numeric cols).",
            "Pred = union of FAIL rule_results/anomalies samples from dq_agent report.json.",
            f"Categorical domain uses coverage-based allowed_values (cover={float(args.domain_cover):.3f}, max_domain={max_domain}).",
        ]
        + cfg_notes,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dataset": args.dataset_name,
        "rows": nrows,
        "cols": ncols,
        "profile": args.profile_source,
        "truth_err_cells": len(truth),
        "predicted": len(pred),
        "cell_f1": c_f1,
        "row_f1": r_f1,
        "truncated": bool(trunc.get("truncated")),
        "metrics_path": str(metrics_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
