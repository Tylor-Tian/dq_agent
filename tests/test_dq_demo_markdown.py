import json
import subprocess
import sys
from pathlib import Path


def test_dq_demo_generates_expected_markdown(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent.cli",
            "demo",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    out = Path(payload["report_md_path"])
    text = out.read_text(encoding="utf-8")
    assert "# Summary" in text
    assert "Contract Errors" in text
    assert "# Rule Results" in text
    assert "# Anomalies" in text
