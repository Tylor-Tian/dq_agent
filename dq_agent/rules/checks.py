"""Rule checks."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from dq_agent.rules.base import RuleResult, build_samples, register_check


def _status(failing_ratio: float, threshold: float) -> str:
    return "PASS" if failing_ratio <= threshold else "FAIL"


@register_check("not_null")
def check_not_null(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> RuleResult:
    max_null_rate = float(params.get("max_null_rate", 0.0))
    total_count = int(series.shape[0])
    null_mask = series.isna()
    failed_count = int(null_mask.sum())
    failing_ratio = failed_count / total_count if total_count else 0.0
    return RuleResult(
        rule_id=f"not_null:{column}",
        column=column,
        status=_status(failing_ratio, max_null_rate),
        failing_ratio=failing_ratio,
        failed_count=failed_count,
        total_count=total_count,
        samples=build_samples(series, null_mask, sample_rows),
    )


@register_check("unique")
def check_unique(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> RuleResult:
    total_count = int(series.shape[0])
    non_null = series.notna()
    dup_mask = non_null & series.duplicated(keep=False)
    failed_count = int(dup_mask.sum())
    failing_ratio = failed_count / total_count if total_count else 0.0
    return RuleResult(
        rule_id=f"unique:{column}",
        column=column,
        status=_status(failing_ratio, 0.0),
        failing_ratio=failing_ratio,
        failed_count=failed_count,
        total_count=total_count,
        samples=build_samples(series, dup_mask, sample_rows),
    )


@register_check("range")
def check_range(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> RuleResult:
    total_count = int(series.shape[0])
    values = pd.to_numeric(series, errors="coerce")
    fail_mask = values.isna()
    if "min" in params and params["min"] is not None:
        fail_mask |= values < float(params["min"])
    if "max" in params and params["max"] is not None:
        fail_mask |= values > float(params["max"])
    failed_count = int(fail_mask.sum())
    failing_ratio = failed_count / total_count if total_count else 0.0
    return RuleResult(
        rule_id=f"range:{column}",
        column=column,
        status=_status(failing_ratio, 0.0),
        failing_ratio=failing_ratio,
        failed_count=failed_count,
        total_count=total_count,
        samples=build_samples(series, fail_mask, sample_rows),
    )


@register_check("allowed_values")
def check_allowed_values(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> RuleResult:
    allowed = set(params.get("values", []))
    total_count = int(series.shape[0])
    fail_mask = ~series.isin(allowed)
    failed_count = int(fail_mask.sum())
    failing_ratio = failed_count / total_count if total_count else 0.0
    return RuleResult(
        rule_id=f"allowed_values:{column}",
        column=column,
        status=_status(failing_ratio, 0.0),
        failing_ratio=failing_ratio,
        failed_count=failed_count,
        total_count=total_count,
        samples=build_samples(series, fail_mask, sample_rows),
    )
