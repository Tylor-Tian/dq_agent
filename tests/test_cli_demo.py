import json
import subprocess
import sys
from pathlib import Path


def test_cli_demo_runs_and_writes_reports(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "demo", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    run_record_path = Path(payload["run_record_path"])
    assert run_record_path.exists(), f"run_record.json missing. stdout={result.stdout} stderr={result.stderr}"
    report_files = list(tmp_path.glob("*/report.json"))
    assert report_files, f"No report.json found. stdout={result.stdout} stderr={result.stderr}"
    report_md_files = list(tmp_path.glob("*/report.md"))
    assert report_md_files, f"No report.md found. stdout={result.stdout} stderr={result.stderr}"
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    required_keys = {
        "schema_version",
        "run_id",
        "started_at",
        "finished_at",
        "input",
        "summary",
        "contract_issues",
        "rule_results",
        "anomalies",
        "fix_actions",
        "observability",
    }
    assert required_keys.issubset(report.keys())
    assert report["schema_version"] == 1
    assert "run_id" in report
    assert "summary" in report
    assert "contract_issues" in report
    assert "rule_results" in report
    assert "anomalies" in report
    assert report["rule_results"], "Expected rule_results to be populated"
    assert isinstance(report["anomalies"], list)
    timing_ms = report["observability"]["timing_ms"]
    for key in ("load", "contract", "rules", "anomalies", "report", "total"):
        assert key in timing_ms
    assert any(result["status"] == "FAIL" for result in report["rule_results"])
    assert any(result["status"] == "FAIL" for result in report["anomalies"])
    report_md = report_md_files[0].read_text(encoding="utf-8")
    assert "Summary" in report_md
    assert "Rule Results" in report_md
    assert "Anomalies" in report_md


def test_cli_demo_fail_on_error_exits_with_2(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "demo",
            "--output-dir",
            str(tmp_path),
            "--fail-on",
            "ERROR",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, f"stdout={result.stdout} stderr={result.stderr}"
