import json
import subprocess
import sys
from pathlib import Path

from dq_agent.checkpoint_schema import CheckpointModel


def test_resume_repairs_missing_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "demo", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    report_path = Path(payload["report_json_path"])
    run_record_path = Path(payload["run_record_path"])
    run_dir = report_path.parent
    report_md_path = run_dir / "report.md"
    checkpoint_path = run_dir / "checkpoint.json"

    assert report_md_path.exists()
    report_md_path.unlink()

    resume = subprocess.run(
        [sys.executable, "-m", "dq_agent", "resume", "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
    )
    assert resume.returncode == 0, resume.stderr
    assert report_md_path.exists()

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

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    CheckpointModel.model_validate(checkpoint_payload)
