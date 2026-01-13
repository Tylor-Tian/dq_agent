from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dq_agent.guardrails import GuardrailsState
from dq_agent.run_record_schema import RunRecordModel

@dataclass(frozen=True)
class RunRecord:
    data: Dict[str, Any]


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_sha() -> Optional[str]:
    env_sha = os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def get_dq_agent_version() -> str:
    try:
        return metadata.version("dq_agent")
    except metadata.PackageNotFoundError:
        return "0.1.0"
    except Exception:
        return "0.1.0"


def write_run_record(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    command: str,
    argv: list[str],
    data_path: Optional[Path],
    config_path: Optional[Path],
    output_dir: Optional[Path],
    report_json_path: Path,
    report_md_path: Optional[Path],
    guardrails: GuardrailsState,
) -> Path:
    run_dir = report_json_path.parent
    run_record_path = run_dir / "run_record.json"
    data_sha = sha256_path(data_path) if data_path and data_path.exists() else None
    config_sha = sha256_path(config_path) if config_path and config_path.exists() else None
    report_json_sha = sha256_path(report_json_path) if report_json_path.exists() else None
    report_md_sha = sha256_path(report_md_path) if report_md_path and report_md_path.exists() else None

    record = RunRecordModel(
        schema_version=1,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        command=command,
        argv=argv,
        input={
            "data_path": str(data_path) if data_path else None,
            "config_path": str(config_path) if config_path else None,
            "output_dir": str(output_dir) if output_dir else None,
        },
        fingerprints={
            "data_sha256": data_sha,
            "config_sha256": config_sha,
            "git_sha": get_git_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dq_agent_version": get_dq_agent_version(),
        },
        outputs={
            "report_json_path": str(report_json_path),
            "report_md_path": str(report_md_path) if report_md_path else None,
            "report_json_sha256": report_json_sha,
            "report_md_sha256": report_md_sha,
        },
        guardrails=guardrails,
    )
    run_record_path.write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_record_path


def load_run_record(path: Path) -> RunRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = RunRecordModel.model_validate(data)
    return RunRecord(data=model.model_dump(mode="json"))


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, val in value.items():
            if key in {"run_id", "started_at", "finished_at", "duration_ms"}:
                continue
            if key in {"output_dir", "run_dir"}:
                continue
            if key == "timing_ms" or key.endswith("timing_ms"):
                continue
            cleaned[key] = _canonicalize(val)
        return cleaned
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def canonicalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return _canonicalize(report)


def diff_summary(old: Any, new: Any) -> Dict[str, Any]:
    if isinstance(old, dict) and isinstance(new, dict):
        changed: list[str] = []
        keys = set(old.keys()) | set(new.keys())
        for key in sorted(keys):
            if key not in old or key not in new or old[key] != new[key]:
                changed.append(key)
        return {"changed_top_level_keys": changed}
    return {"changed": True}


def compare_reports(old_report: Dict[str, Any], new_report: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    old_canon = canonicalize_report(old_report)
    new_canon = canonicalize_report(new_report)
    same = old_canon == new_canon
    summary = {} if same else diff_summary(old_canon, new_canon)
    return same, summary
