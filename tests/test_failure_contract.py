import json
import subprocess
import sys
from pathlib import Path


def test_guardrail_failure_writes_error_artifacts(tmp_path: Path):
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
    error = payload["error"]
    assert error["type"] == "guardrail_violation"
    assert error["code"] == "max_rows"
    report_path = Path(payload["report_json_path"])
    run_record_path = Path(payload["run_record_path"])
    assert report_path.exists(), "report.json missing for guardrail failure"
    assert run_record_path.exists(), "run_record.json missing for guardrail failure"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAILED_GUARDRAIL"
    assert report.get("error"), "report.error should be populated on guardrail failure"

    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    assert run_record.get("error"), "run_record.error should be populated on guardrail failure"

    subprocess.run(
        [sys.executable, "-m", "dq_agent", "validate", "--kind", "report", "--path", str(report_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "dq_agent", "replay", "--run-record", str(run_record_path), "--strict"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_io_failure_writes_run_record(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "dq_agent" / "resources" / "demo_rules.yml"
    missing_data = tmp_path / "missing.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "run",
            "--data",
            str(missing_data),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, f"stdout={result.stdout} stderr={result.stderr}"
    payload = json.loads(result.stdout)
    error = payload["error"]
    assert error["type"] == "io_error"
    run_record_path = Path(payload["run_record_path"])
    assert run_record_path.exists(), "run_record.json missing for IO failure"
    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
