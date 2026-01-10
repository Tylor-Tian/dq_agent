"""Markdown report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _format_kv_rows(rows: Iterable[tuple[str, Any]]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _format_status_list(items: list[dict[str, Any]], id_key: str) -> list[str]:
    lines: list[str] = []
    for item in items:
        status = item.get("status", "UNKNOWN")
        item_id = item.get(id_key, "unknown")
        column = item.get("column", "")
        column_suffix = f" ({column})" if column else ""
        lines.append(f"- {item_id}{column_suffix}: {status}")
    return lines


def write_report_md(report: dict[str, Any], path: Path) -> Path:
    summary = report.get("summary", {})
    issue_counts = summary.get("issue_counts", {})
    rule_results = report.get("rule_results", [])
    anomalies = report.get("anomalies", [])

    sections = [
        "# Summary",
        _format_kv_rows(
            [
                ("Run ID", report.get("run_id", "")),
                ("Rows", summary.get("rows", "")),
                ("Cols", summary.get("cols", "")),
                ("Contract Errors", issue_counts.get("error", 0)),
                ("Contract Warnings", issue_counts.get("warn", 0)),
                ("Contract Info", issue_counts.get("info", 0)),
            ]
        ),
        "",
        "# Rule Results",
    ]
    sections.extend(_format_status_list(rule_results, "rule_id") or ["- No rules executed."])
    sections.extend(["", "# Anomalies"])
    sections.extend(_format_status_list(anomalies, "anomaly_id") or ["- No anomalies detected."])

    path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
    return path
