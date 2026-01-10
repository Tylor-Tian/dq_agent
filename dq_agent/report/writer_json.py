from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from dq_agent.contract import ContractIssue
from dq_agent.anomalies import AnomalyResult
from dq_agent.rules import RuleResult


TIMING_KEYS: tuple[str, ...] = ("load", "contract", "rules", "anomalies", "report", "total")


def _count_failed(items: Iterable[dict], status_key: str = "status") -> int:
    return sum(1 for item in items if item.get(status_key) == "FAIL")


def write_report_json(
    *,
    output_dir: Path,
    data_path: Path,
    config_path: Path,
    rows: int,
    cols: int,
    contract_issues: List[ContractIssue],
    rule_results: Optional[List[RuleResult]] = None,
    anomalies: Optional[List[AnomalyResult]] = None,
    observability_timing_ms: Optional[dict[str, float]] = None,
) -> Path:
    run_id = uuid.uuid4().hex
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    started_at = datetime.now(timezone.utc)

    rule_payloads = [result.to_dict() for result in (rule_results or [])]
    anomaly_payloads = [result.to_dict() for result in (anomalies or [])]
    timing_ms = {key: 0.0 for key in TIMING_KEYS}
    if observability_timing_ms is not None:
        timing_ms.update(
            {key: round(value, 3) for key, value in observability_timing_ms.items() if key in timing_ms}
        )

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "data_path": str(data_path),
            "config_path": str(config_path),
            "format": "json",
        },
        "summary": {
            "rows": rows,
            "cols": cols,
            "issue_counts": {
                "error": sum(1 for issue in contract_issues if issue.severity == "error"),
                "warn": sum(1 for issue in contract_issues if issue.severity == "warn"),
                "info": sum(1 for issue in contract_issues if issue.severity == "info"),
            },
        },
        "contract_issues": [issue.to_dict() for issue in contract_issues],
        "rule_results": rule_payloads,
        "anomalies": anomaly_payloads,
        "fix_actions": [],
        "observability": {
            "timing_ms": timing_ms,
            "counts": {
                "rules_total": len(rule_payloads),
                "rules_failed": _count_failed(rule_payloads),
                "anomalies_total": len(anomaly_payloads),
                "anomalies_failed": _count_failed(anomaly_payloads),
            },
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
