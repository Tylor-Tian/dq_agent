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
    report_files = list(tmp_path.glob("*/report.json"))
    assert report_files, f"No report.json found. stdout={result.stdout} stderr={result.stderr}"
    report_md_files = list(tmp_path.glob("*/report.md"))
    assert report_md_files, f"No report.md found. stdout={result.stdout} stderr={result.stderr}"
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert "run_id" in report
    assert "summary" in report
    assert "contract_issues" in report
    assert "rule_results" in report
    assert "anomalies" in report
    assert report["rule_results"], "Expected rule_results to be populated"
    assert isinstance(report["anomalies"], list)
    assert any(result["status"] == "FAIL" for result in report["rule_results"])
    assert any(result["status"] == "FAIL" for result in report["anomalies"])
    report_md = report_md_files[0].read_text(encoding="utf-8")
    assert "Summary" in report_md
    assert "Rule Results" in report_md
    assert "Anomalies" in report_md
