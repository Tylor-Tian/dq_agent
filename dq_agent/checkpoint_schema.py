"""Checkpoint schema for resumable runs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from dq_agent.errors import AgentError
from dq_agent.guardrails import GuardrailsConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckpointInput(StrictModel):
    data_path: Optional[str] = None
    config_path: Optional[str] = None
    output_dir: Optional[str] = None
    idempotency_key: Optional[str] = None
    idempotency_mode: Optional[str] = None
    fail_on: Optional[str] = None
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)


class CheckpointStage(StrictModel):
    name: str
    status: Literal["PENDING", "RUNNING", "OK", "FAILED", "SKIPPED"] = Field(default="PENDING")
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[AgentError] = None


class CheckpointArtifacts(StrictModel):
    report_json_path: Optional[str] = None
    report_md_path: Optional[str] = None
    run_record_path: Optional[str] = None
    trace_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


class CheckpointModel(StrictModel):
    schema_version: Literal[1] = Field(default=1)
    run_id: str
    status: Literal["RUNNING", "SUCCESS", "FAILED_GUARDRAIL", "FAILED"] = Field(default="RUNNING")
    command: str
    argv: List[str]
    input: CheckpointInput
    stages: List[CheckpointStage]
    artifacts: CheckpointArtifacts
    started_at: str
    updated_at: str
    finished_at: Optional[str] = None
