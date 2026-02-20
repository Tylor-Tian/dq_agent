import subprocess
import sys
from pathlib import Path


def test_dq_demo_generates_expected_markdown(tmp_path: Path):
    out = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dq_agent.cli",
            "demo",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "# Data Quality Report" in text
    assert "Score: **" in text
    assert "**100**/100" not in text
    assert "## Top Violations" in text
    violations = [line for line in text.splitlines() if line.startswith("- ") and "hits" in line]
    assert len(violations) >= 2
    assert "## Samples" in text
    sample_lines = [line for line in text.splitlines() if line.startswith('{"')]
    assert len(sample_lines) >= 2
    assert "## Advice" in text
    advice_lines = [
        line
        for line in text.splitlines()
        if line.startswith("- ") and ("validate" in line.lower() or "fix" in line.lower() or "de-duplicate" in line.lower())
    ]
    assert len(advice_lines) >= 2
