import json
import subprocess
import sys
from pathlib import Path


def _run_demo(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "demo",
            "--output-dir",
            str(tmp_path),
            "--idempotency-key",
            "stable-key",
            "--idempotency-mode",
            mode,
        ],
        capture_output=True,
        text=True,
    )


def _validate_artifacts(report_path: Path, run_record_path: Path) -> None:
    report_result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "validate", "--kind", "report", "--path", str(report_path)],
        capture_output=True,
        text=True,
    )
    assert report_result.returncode == 0, report_result.stderr
    run_record_result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "validate", "--kind", "run_record", "--path", str(run_record_path)],
        capture_output=True,
        text=True,
    )
    assert run_record_result.returncode == 0, run_record_result.stderr


def test_demo_idempotency_reuse(tmp_path: Path) -> None:
    first = _run_demo(tmp_path, "reuse")
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    second = _run_demo(tmp_path, "reuse")
    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert first_payload["report_json_path"] == second_payload["report_json_path"]
    assert first_payload["run_record_path"] == second_payload["run_record_path"]
    report_files = list(tmp_path.glob("*/report.json"))
    assert len(report_files) == 1


def test_demo_idempotency_fail_mode(tmp_path: Path) -> None:
    first = _run_demo(tmp_path, "reuse")
    assert first.returncode == 0, first.stderr
    second = _run_demo(tmp_path, "fail")
    assert second.returncode == 2, second.stderr
    payload = json.loads(second.stdout)
    error = payload["error"]
    assert error["type"] == "idempotency_conflict"
    report_path = Path(payload["report_json_path"])
    run_record_path = Path(payload["run_record_path"])
    assert report_path.exists()
    assert run_record_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error"]
    _validate_artifacts(report_path, run_record_path)


def test_demo_idempotency_overwrite(tmp_path: Path) -> None:
    first = _run_demo(tmp_path, "reuse")
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    second = _run_demo(tmp_path, "overwrite")
    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert first_payload["report_json_path"] == second_payload["report_json_path"]
    assert first_payload["run_record_path"] == second_payload["run_record_path"]
    report_path = Path(second_payload["report_json_path"])
    run_record_path = Path(second_payload["run_record_path"])
    _validate_artifacts(report_path, run_record_path)
