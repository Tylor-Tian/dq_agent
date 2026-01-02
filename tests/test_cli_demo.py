import json
import subprocess
import sys
from pathlib import Path


def test_cli_demo_runs_and_writes_report_json(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "demo", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report_files = list(tmp_path.glob("*/report.json"))
    assert report_files, f"No report.json found. stdout={result.stdout} stderr={result.stderr}"
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert "run_id" in report
    assert "summary" in report
    assert "contract_issues" in report
