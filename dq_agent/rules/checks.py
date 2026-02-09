"""Rule checks."""

from __future__ import annotations

import re
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


@register_check("string_noise")
def check_string_noise(
    column: str,
    series: pd.Series,
    params: Dict[str, Any],
    sample_rows: int,
) -> RuleResult:
    """Detect "string noise" by simple substring / regex pattern matching.

    This is intentionally lightweight and configurable.

    Params:
      - contains: list[str]      # literal substrings (NOT regex)
      - regex: list[str]         # regex patterns
      - ignore_case: bool        # default False
      - strip: bool              # default True
      - treat_empty_as_null: bool  # default True
      - max_rate: float          # tolerated noisy rate, default 0.0
    """

    contains = params.get("contains") or []
    regex = params.get("regex") or []
    ignore_case = bool(params.get("ignore_case", False))
    strip = bool(params.get("strip", True))
    treat_empty_as_null = bool(params.get("treat_empty_as_null", True))
    max_rate = float(params.get("max_rate", 0.0))

    total_count = int(series.shape[0])

    # No patterns configured => always PASS (opt-in check).
    if not contains and not regex:
        return RuleResult(
            rule_id=f"string_noise:{column}",
            column=column,
            status="PASS",
            failing_ratio=0.0,
            failed_count=0,
            total_count=total_count,
            samples=[],
        )

    # Normalize into pandas' string dtype; keep nulls as <NA>
    s = series.astype("string")
    if strip:
        s = s.str.strip()
    if treat_empty_as_null:
        s = s.replace("", pd.NA)

    non_null = s.notna()
    s2 = s.fillna("")

    # Build a boolean mask for any pattern matches.
    mask = pd.Series(False, index=s2.index)
    # literal substrings: escape into regex
    for sub in contains:
        if sub is None:
            continue
        sub = str(sub)
        if sub == "":
            continue
        mask |= s2.str.contains(re.escape(sub), regex=True, case=not ignore_case, na=False)

    # regex patterns
    flags = re.IGNORECASE if ignore_case else 0
    for pat in regex:
        if pat is None:
            continue
        pat = str(pat)
        if pat == "":
            continue
        mask |= s2.str.contains(pat, regex=True, flags=flags, na=False)

    fail_mask = non_null & mask
    failed_count = int(fail_mask.sum())
    failing_ratio = failed_count / total_count if total_count else 0.0

    return RuleResult(
        rule_id=f"string_noise:{column}",
        column=column,
        status=_status(failing_ratio, max_rate),
        failing_ratio=failing_ratio,
        failed_count=failed_count,
        total_count=total_count,
        samples=build_samples(series, fail_mask, sample_rows),
    )
