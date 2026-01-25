import json
import subprocess
import sys
from pathlib import Path


def test_schema_command_outputs_json() -> None:
    report_schema = subprocess.run(
        [sys.executable, "-m", "dq_agent", "schema", "--kind", "report"],
        check=True,
        capture_output=True,
        text=True,
    )
    json.loads(report_schema.stdout)

    run_record_schema = subprocess.run(
        [sys.executable, "-m", "dq_agent", "schema", "--kind", "run_record"],
        check=True,
        capture_output=True,
        text=True,
    )
    json.loads(run_record_schema.stdout)

    checkpoint_schema = subprocess.run(
        [sys.executable, "-m", "dq_agent", "schema", "--kind", "checkpoint"],
        check=True,
        capture_output=True,
        text=True,
    )
    json.loads(checkpoint_schema.stdout)


def test_validate_demo_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "demo", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    report_path = Path(payload["report_json_path"])
    run_record_path = Path(payload["run_record_path"])
    checkpoint_path = Path(payload["checkpoint_path"])

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

    checkpoint_validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "validate",
            "--kind",
            "checkpoint",
            "--path",
            str(checkpoint_path),
        ],
        capture_output=True,
        text=True,
    )
    assert checkpoint_validation.returncode == 0, checkpoint_validation.stderr
