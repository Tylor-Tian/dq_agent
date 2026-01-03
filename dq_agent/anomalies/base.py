"""Detector interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dq_agent.config import Config


@dataclass
class AnomalyResult:
    anomaly_id: str
    column: str
    status: str
    metric: Dict[str, Any]
    threshold: Dict[str, Any]
    samples: List[Dict[str, Any]]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AnomalyFn = Callable[[str, pd.Series, Dict[str, Any], int], AnomalyResult]
_ANOMALY_REGISTRY: Dict[str, AnomalyFn] = {}


def register_anomaly(name: str) -> Callable[[AnomalyFn], AnomalyFn]:
    def decorator(func: AnomalyFn) -> AnomalyFn:
        _ANOMALY_REGISTRY[name] = func
        return func

    return decorator


def get_anomaly(name: str) -> Optional[AnomalyFn]:
    return _ANOMALY_REGISTRY.get(name)


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


def run_anomalies(df: pd.DataFrame, cfg: Config) -> List[AnomalyResult]:
    results: List[AnomalyResult] = []
    sample_rows = cfg.report.sample_rows
    for column_name, column_cfg in cfg.columns.items():
        if column_name not in df.columns:
            continue
        for anomaly_entry in column_cfg.anomalies:
            if not isinstance(anomaly_entry, dict) or len(anomaly_entry) != 1:
                continue
            anomaly_name, raw_params = next(iter(anomaly_entry.items()))
            params = raw_params if isinstance(raw_params, dict) else {}
            anomaly = get_anomaly(anomaly_name)
            if anomaly is None:
                continue
            results.append(anomaly(column_name, df[column_name], params, sample_rows))
    return results
