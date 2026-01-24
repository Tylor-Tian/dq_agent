import json
import subprocess
import sys
from pathlib import Path

import yaml

from dq_agent.demo.generate_demo_data import generate_demo_data


def _write_candidate_config(tmp_path: Path, baseline_config: Path) -> Path:
    payload = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))
    payload.setdefault("columns", {})
    payload["columns"]["missing_column"] = {"type": "string", "required": True}
    candidate_path = tmp_path / "candidate_rules.yml"
    candidate_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return candidate_path


def test_shadow_writes_diff_and_reports(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    baseline_config = repo_root / "dq_agent" / "resources" / "demo_rules.yml"
    data_path = generate_demo_data(tmp_path)
    candidate_config = _write_candidate_config(tmp_path, baseline_config)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "shadow",
            "--data",
            str(data_path),
            "--baseline-config",
            str(baseline_config),
            "--candidate-config",
            str(candidate_config),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    payload = json.loads(result.stdout)
    shadow_diff_path = Path(payload["shadow_diff_path"])
    baseline_report_path = Path(payload["baseline_report_json_path"])
    candidate_report_path = Path(payload["candidate_report_json_path"])

    assert shadow_diff_path.exists(), "shadow_diff.json missing"
    assert baseline_report_path.exists(), "baseline report.json missing"
    assert candidate_report_path.exists(), "candidate report.json missing"

    diff = json.loads(shadow_diff_path.read_text(encoding="utf-8"))
    assert diff["schema_version"] == 1
    summary = diff["diff_summary"]
    assert summary["issue_counts_delta"]["error"] > 0


def test_shadow_fail_on_regression(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    baseline_config = repo_root / "dq_agent" / "resources" / "demo_rules.yml"
    data_path = generate_demo_data(tmp_path)
    candidate_config = _write_candidate_config(tmp_path, baseline_config)
    output_dir = tmp_path / "shadow_regression"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "shadow",
            "--data",
            str(data_path),
            "--baseline-config",
            str(baseline_config),
            "--candidate-config",
            str(candidate_config),
            "--output-dir",
            str(output_dir),
            "--fail-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, f"stdout={result.stdout} stderr={result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["error"]["type"] == "regression"
    shadow_diff_path = Path(payload["shadow_diff_path"])
    assert shadow_diff_path.exists(), "shadow_diff.json missing on regression"
