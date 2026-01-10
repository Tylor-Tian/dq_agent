import json
import subprocess
import sys
from pathlib import Path


def test_cli_replay_round_trip(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("id,value\n1,foo\n2,bar\n", encoding="utf-8")

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
version: 1

dataset:
  name: replay_demo

columns:
  id:
    type: string
    required: true
    checks:
      - not_null: { max_null_rate: 0.0 }
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "run",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    run_record_path = Path(payload["run_record_path"])
    assert run_record_path.exists(), f"run_record.json missing. stdout={result.stdout} stderr={result.stderr}"

    replay_dir = tmp_path / "replay_artifacts"
    replay_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent",
            "replay",
            "--run-record",
            str(run_record_path),
            "--output-dir",
            str(replay_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
    )
    assert replay_result.returncode == 0, (
        f"stdout={replay_result.stdout} stderr={replay_result.stderr}"
    )
    summary = json.loads(replay_result.stdout)
    assert summary["same"] is True
    new_report_path = Path(summary["new_report_json_path"])
    assert new_report_path.exists()
