"""Checkpoint helpers for resumable runs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dq_agent.checkpoint_schema import CheckpointArtifacts, CheckpointInput, CheckpointModel, CheckpointStage
from dq_agent.errors import AgentError
from dq_agent.guardrails import GuardrailsConfig

STAGE_ORDER: list[str] = [
    "config",
    "load",
    "guardrails",
    "contract",
    "rules",
    "anomalies",
    "report_json",
    "report_md",
    "run_record",
    "trace",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_stage_list(stages: Iterable[str]) -> list[CheckpointStage]:
    return [CheckpointStage(name=stage, status="PENDING") for stage in stages]


def build_checkpoint(
    *,
    run_id: str,
    command: str,
    argv: list[str],
    data_path: Optional[Path],
    config_path: Optional[Path],
    output_dir: Optional[Path],
    guardrails: GuardrailsConfig,
    fail_on: Optional[str],
    idempotency_key: Optional[str],
    idempotency_mode: Optional[str],
    report_path: Optional[Path],
    report_md_path: Optional[Path],
    run_record_path: Optional[Path],
    trace_path: Optional[Path],
    checkpoint_path: Path,
    status: str = "RUNNING",
) -> CheckpointModel:
    now = _now_iso()
    return CheckpointModel(
        schema_version=1,
        run_id=run_id,
        status=status,
        command=command,
        argv=argv,
        input=CheckpointInput(
            data_path=str(data_path) if data_path else None,
            config_path=str(config_path) if config_path else None,
            output_dir=str(output_dir) if output_dir else None,
            idempotency_key=idempotency_key,
            idempotency_mode=idempotency_mode,
            fail_on=fail_on,
            guardrails=guardrails,
        ),
        stages=_build_stage_list(STAGE_ORDER),
        artifacts=CheckpointArtifacts(
            report_json_path=str(report_path) if report_path else None,
            report_md_path=str(report_md_path) if report_md_path else None,
            run_record_path=str(run_record_path) if run_record_path else None,
            trace_path=str(trace_path) if trace_path else None,
            checkpoint_path=str(checkpoint_path),
        ),
        started_at=now,
        updated_at=now,
        finished_at=None,
    )


def write_checkpoint_atomic(checkpoint: CheckpointModel, path: Path) -> None:
    payload = checkpoint.model_dump(mode="json")
    CheckpointModel.model_validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


class CheckpointTracker:
    def __init__(
        self,
        *,
        checkpoint: CheckpointModel,
        checkpoint_path: Path,
        start_time: float,
    ) -> None:
        self._checkpoint = checkpoint
        self._checkpoint_path = checkpoint_path
        self._start_time = start_time
        self._stage_index = {stage.name: stage for stage in checkpoint.stages}
        self._write()

    @property
    def checkpoint(self) -> CheckpointModel:
        return self._checkpoint

    def _elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    def _write(self) -> None:
        self._checkpoint.updated_at = _now_iso()
        write_checkpoint_atomic(self._checkpoint, self._checkpoint_path)

    def mark_stage_start(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        stage_state = self._stage_index.get(stage)
        if not stage_state:
            return
        stage_state.status = "RUNNING"
        stage_state.t_start_ms = self._elapsed_ms()
        stage_state.details = details
        stage_state.error = None
        self._write()

    def mark_stage_end(
        self,
        stage: str,
        *,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[AgentError] = None,
    ) -> None:
        stage_state = self._stage_index.get(stage)
        if not stage_state:
            return
        stage_state.status = status
        stage_state.t_end_ms = self._elapsed_ms()
        if stage_state.t_start_ms is not None:
            stage_state.duration_ms = stage_state.t_end_ms - stage_state.t_start_ms
        if details is not None:
            stage_state.details = details
        stage_state.error = error
        self._write()

    def mark_stage_skipped(self, stage: str) -> None:
        stage_state = self._stage_index.get(stage)
        if not stage_state:
            return
        if stage_state.status == "PENDING":
            stage_state.status = "SKIPPED"
        self._write()

    def finalize(
        self,
        *,
        status: str,
        error: Optional[AgentError] = None,
        trace_path: Optional[Path] = None,
    ) -> None:
        for stage_state in self._checkpoint.stages:
            if stage_state.status == "PENDING":
                stage_state.status = "SKIPPED"
        if trace_path is not None:
            trace_stage = self._stage_index.get("trace")
            if trace_stage and trace_stage.status in {"PENDING", "SKIPPED"}:
                trace_stage.status = "OK"
                trace_stage.t_end_ms = self._elapsed_ms()
                if trace_stage.t_start_ms is None:
                    trace_stage.t_start_ms = trace_stage.t_end_ms
                trace_stage.duration_ms = 0
                trace_stage.details = {"trace_path": str(trace_path)}
        if error is not None:
            for stage_state in self._checkpoint.stages:
                if stage_state.status == "FAILED":
                    stage_state.error = error
                    break
        self._checkpoint.status = status
        self._checkpoint.finished_at = _now_iso()
        self._write()
