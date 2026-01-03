"""Rule engine interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dq_agent.config import Config


@dataclass
class RuleResult:
    rule_id: str
    column: str
    status: str
    failing_ratio: float
    failed_count: int
    total_count: int
    samples: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CheckFn = Callable[[str, pd.Series, Dict[str, Any], int], RuleResult]
_CHECK_REGISTRY: Dict[str, CheckFn] = {}


def register_check(name: str) -> Callable[[CheckFn], CheckFn]:
    def decorator(func: CheckFn) -> CheckFn:
        _CHECK_REGISTRY[name] = func
        return func

    return decorator


def get_check(name: str) -> Optional[CheckFn]:
    return _CHECK_REGISTRY.get(name)


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def build_samples(series: pd.Series, mask: Iterable[bool], sample_rows: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, value in series[mask].head(sample_rows).items():
        if isinstance(idx, (int, np.integer)):
            row_index: Any = int(idx)
        else:
            row_index = str(idx)
        samples.append({"row_index": row_index, "value": _normalize_value(value)})
    return samples


def run_rules(df: pd.DataFrame, cfg: Config) -> List[RuleResult]:
    results: List[RuleResult] = []
    sample_rows = cfg.report.sample_rows
    for column_name, column_cfg in cfg.columns.items():
        if column_name not in df.columns:
            continue
        for check_entry in column_cfg.checks:
            if not isinstance(check_entry, dict) or len(check_entry) != 1:
                continue
            check_name, raw_params = next(iter(check_entry.items()))
            params = raw_params if isinstance(raw_params, dict) else {}
            check = get_check(check_name)
            if check is None:
                continue
            results.append(check(column_name, df[column_name], params, sample_rows))
    return results
