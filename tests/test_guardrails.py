import json
import subprocess
import sys
from pathlib import Path


def test_guardrail_max_rows_trips(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "demo",
            "--output-dir",
            str(tmp_path),
            "--max-rows",
            "10",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, f"stdout={result.stdout} stderr={result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["error"]["type"] == "guardrail_violation"
    assert payload["error"]["code"] == "max_rows"

    report_path = Path(payload["report_json_path"])
    run_record_path = Path(payload["run_record_path"])
    assert report_path.exists()
    assert run_record_path.exists()

    report_validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "validate",
            "--kind",
            "report",
            "--path",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )
    assert report_validation.returncode == 0, report_validation.stderr

    run_record_validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "validate",
            "--kind",
            "run_record",
            "--path",
            str(run_record_path),
        ],
        capture_output=True,
        text=True,
    )
    assert run_record_validation.returncode == 0, run_record_validation.stderr
