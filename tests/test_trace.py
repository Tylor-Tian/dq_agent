import json
import subprocess
import sys
from pathlib import Path


def _load_trace(trace_path: Path) -> list[dict]:
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "trace.jsonl is empty"
    return [json.loads(line) for line in lines]


def _assert_trace_events(events: list[dict]) -> None:
    run_ids = {event["run_id"] for event in events}
    assert len(run_ids) == 1
    seqs = [event["seq"] for event in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    assert all(curr > prev for prev, curr in zip(seqs, seqs[1:]))
    for event in events:
        assert isinstance(event["t_ms"], int)
        assert event["t_ms"] >= 0
    assert any(event["event"] == "run_start" for event in events)
    assert any(event["event"] == "run_end" for event in events)


def test_trace_written_for_success(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dq_agent", "demo", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    trace_path = Path(payload["trace_path"])
    assert trace_path.exists()

    events = _load_trace(trace_path)
    _assert_trace_events(events)
    stage_events = [event for event in events if event["event"].startswith("stage_")]
    assert len(stage_events) >= 2


def test_trace_written_for_failure(tmp_path: Path) -> None:
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
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    trace_path = Path(payload["trace_path"])
    assert trace_path.exists()

    events = _load_trace(trace_path)
    _assert_trace_events(events)
    run_end = next(event for event in events if event["event"] == "run_end")
    assert run_end["status"].startswith("FAILED")
    assert run_end["error"]
