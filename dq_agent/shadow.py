from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


SHADOW_DIFF_SCHEMA_VERSION = 1


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_shadow_diff(diff: Dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _issue_counts(report: Dict[str, Any]) -> Dict[str, int]:
    summary = report.get("summary") or {}
    issue_counts = summary.get("issue_counts") or {}
    return {
        "error": int(issue_counts.get("error", 0) or 0),
        "warn": int(issue_counts.get("warn", 0) or 0),
        "info": int(issue_counts.get("info", 0) or 0),
    }


def _counts(report: Dict[str, Any]) -> Dict[str, int]:
    return {
        "contract_issues": len(report.get("contract_issues") or []),
        "rule_results": len(report.get("rule_results") or []),
        "anomalies": len(report.get("anomalies") or []),
        "fix_actions": len(report.get("fix_actions") or []),
    }


def _rule_key(rule: Dict[str, Any]) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    rule_id = rule.get("rule_id")
    column = rule.get("column")
    if rule_id is not None and column is not None:
        return (str(rule_id), str(column)), {"rule_id": str(rule_id), "column": str(column)}
    if rule_id is not None:
        return (str(rule_id),), {"rule_id": str(rule_id)}
    if column is not None:
        return (str(column),), {"column": str(column)}
    return ("unknown",), {"rule_id": "unknown"}


def _anomaly_key(anomaly: Dict[str, Any]) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    anomaly_id = anomaly.get("anomaly_id")
    column = anomaly.get("column")
    if anomaly_id is not None and column is not None:
        return (str(anomaly_id), str(column)), {"anomaly_id": str(anomaly_id), "column": str(column)}
    metric = anomaly.get("metric") or {}
    threshold = anomaly.get("threshold") or {}
    anomaly_type = None
    if isinstance(metric, dict):
        anomaly_type = metric.get("type")
    if anomaly_type is None and isinstance(threshold, dict):
        anomaly_type = threshold.get("type")
    type_value = str(anomaly_type) if anomaly_type is not None else "unknown"
    column_value = str(column) if column is not None else "unknown"
    return (column_value, type_value), {"column": column_value, "type": type_value}


def _diff_items(
    baseline_items: Iterable[Dict[str, Any]],
    candidate_items: Iterable[Dict[str, Any]],
    *,
    key_fn,
) -> Dict[str, list[Dict[str, str]]]:
    baseline_map: Dict[Tuple[str, ...], str] = {}
    baseline_repr: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for item in baseline_items:
        key, payload = key_fn(item)
        baseline_map[key] = str(item.get("status", ""))
        baseline_repr[key] = payload

    candidate_map: Dict[Tuple[str, ...], str] = {}
    candidate_repr: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for item in candidate_items:
        key, payload = key_fn(item)
        candidate_map[key] = str(item.get("status", ""))
        candidate_repr[key] = payload

    baseline_keys = set(baseline_map.keys())
    candidate_keys = set(candidate_map.keys())
    added_keys = sorted(candidate_keys - baseline_keys)
    removed_keys = sorted(baseline_keys - candidate_keys)
    status_changed_keys = sorted(
        key for key in baseline_keys & candidate_keys if baseline_map.get(key) != candidate_map.get(key)
    )

    def _format(keys: Iterable[Tuple[str, ...]]) -> list[Dict[str, str]]:
        formatted = []
        for key in keys:
            if key in candidate_repr:
                formatted.append(candidate_repr[key])
            elif key in baseline_repr:
                formatted.append(baseline_repr[key])
        return formatted

    return {
        "added": _format(added_keys),
        "removed": _format(removed_keys),
        "status_changed": _format(status_changed_keys),
    }


def build_shadow_diff(
    *,
    baseline_report: Dict[str, Any],
    candidate_report: Dict[str, Any],
    shadow_run_id: str,
) -> Dict[str, Any]:
    baseline_issue_counts = _issue_counts(baseline_report)
    candidate_issue_counts = _issue_counts(candidate_report)
    baseline_counts = _counts(baseline_report)
    candidate_counts = _counts(candidate_report)

    issue_counts_delta = {
        key: candidate_issue_counts.get(key, 0) - baseline_issue_counts.get(key, 0)
        for key in ("error", "warn", "info")
    }

    rules_changed = _diff_items(
        baseline_report.get("rule_results") or [],
        candidate_report.get("rule_results") or [],
        key_fn=_rule_key,
    )
    anomalies_changed = _diff_items(
        baseline_report.get("anomalies") or [],
        candidate_report.get("anomalies") or [],
        key_fn=_anomaly_key,
    )

    return {
        "schema_version": SHADOW_DIFF_SCHEMA_VERSION,
        "shadow_run_id": shadow_run_id,
        "baseline": {
            "run_id": baseline_report.get("run_id"),
            "status": baseline_report.get("status"),
            "issue_counts": baseline_issue_counts,
            "counts": baseline_counts,
        },
        "candidate": {
            "run_id": candidate_report.get("run_id"),
            "status": candidate_report.get("status"),
            "issue_counts": candidate_issue_counts,
            "counts": candidate_counts,
        },
        "diff_summary": {
            "status_changed": baseline_report.get("status") != candidate_report.get("status"),
            "issue_counts_delta": issue_counts_delta,
            "contract_issues_delta": candidate_counts["contract_issues"] - baseline_counts["contract_issues"],
            "rule_results_delta": candidate_counts["rule_results"] - baseline_counts["rule_results"],
            "anomalies_delta": candidate_counts["anomalies"] - baseline_counts["anomalies"],
            "fix_actions_delta": candidate_counts["fix_actions"] - baseline_counts["fix_actions"],
            "rules_changed": {
                "added": rules_changed["added"],
                "removed": rules_changed["removed"],
                "status_changed": rules_changed["status_changed"],
            },
            "anomalies_changed": {
                "added": anomalies_changed["added"],
                "removed": anomalies_changed["removed"],
            },
        },
    }
