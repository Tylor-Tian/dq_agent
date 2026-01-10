from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dq_agent.contract import ContractIssue
from dq_agent.anomalies import AnomalyResult
from dq_agent.rules import RuleResult


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
        "rule_results": [result.to_dict() for result in (rule_results or [])],
        "anomalies": [result.to_dict() for result in (anomalies or [])],
    }

    if observability_timing_ms is not None:
        report["observability"] = {
            "timing_ms": {
                key: round(value, 3) for key, value in observability_timing_ms.items()
            }
        }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
