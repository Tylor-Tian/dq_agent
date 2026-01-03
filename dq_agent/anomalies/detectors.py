"""Anomaly detectors."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from dq_agent.anomalies.base import AnomalyResult, build_samples, register_anomaly


@register_anomaly("missing_rate")
def missing_rate(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> AnomalyResult:
    max_rate = float(params.get("max_rate", 0.0))
    null_mask = series.isna()
    total_count = int(series.shape[0])
    null_count = int(null_mask.sum())
    null_rate = float(null_count / total_count) if total_count else 0.0
    status = "FAIL" if null_rate > max_rate else "PASS"
    explanation = (
        f"Null rate {null_rate:.4f} exceeds max_rate {max_rate:.4f}."
        if status == "FAIL"
        else f"Null rate {null_rate:.4f} within max_rate {max_rate:.4f}."
    )
    return AnomalyResult(
        anomaly_id="missing_rate",
        column=column,
        status=status,
        metric={"null_count": null_count, "null_rate": null_rate, "total_count": total_count},
        threshold={"max_rate": max_rate},
        samples=build_samples(series, null_mask, sample_rows),
        explanation=explanation,
    )


@register_anomaly("outlier_mad")
def outlier_mad(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> AnomalyResult:
    threshold = float(params.get("z", 6.0))
    if not is_numeric_dtype(series):
        return AnomalyResult(
            anomaly_id="outlier_mad",
            column=column,
            status="PASS",
            metric={"reason": "non_numeric"},
            threshold={"z": threshold},
            samples=[],
            explanation="Skipped MAD outlier detection for non-numeric column.",
        )

    clean = series.dropna()
    if clean.empty:
        return AnomalyResult(
            anomaly_id="outlier_mad",
            column=column,
            status="PASS",
            metric={"count": 0, "median": None, "mad": None, "max_z": 0.0},
            threshold={"z": threshold},
            samples=[],
            explanation="No numeric values available for outlier detection.",
        )

    median = float(np.median(clean.to_numpy()))
    mad = float(np.median(np.abs(clean.to_numpy() - median)))
    if mad == 0.0:
        z_scores = pd.Series(0.0, index=series.index)
    else:
        z_scores = 0.6745 * (series - median) / mad
    z_abs = z_scores.abs()
    max_z = float(z_abs.max(skipna=True)) if not z_abs.empty else 0.0
    outlier_mask = z_abs > threshold
    status = "FAIL" if bool(outlier_mask.any()) else "PASS"
    explanation = (
        f"Found values with MAD z-score above {threshold:.2f}."
        if status == "FAIL"
        else f"No values exceed MAD z-score threshold {threshold:.2f}."
    )

    samples = []
    for idx, value in series[outlier_mask].head(sample_rows).items():
        if isinstance(idx, (int, np.integer)):
            row_index: Any = int(idx)
        else:
            row_index = str(idx)
        z_value = z_scores.loc[idx]
        if pd.isna(value):
            normalized_value = None
        elif isinstance(value, np.generic):
            normalized_value = value.item()
        else:
            normalized_value = value
        samples.append(
            {
                "row_index": row_index,
                "value": normalized_value,
                "z": float(z_value) if not pd.isna(z_value) else None,
            }
        )

    return AnomalyResult(
        anomaly_id="outlier_mad",
        column=column,
        status=status,
        metric={"median": median, "mad": mad, "max_z": max_z, "count": int(series.shape[0])},
        threshold={"z": threshold},
        samples=samples,
        explanation=explanation,
    )
