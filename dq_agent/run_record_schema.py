"""Run record schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dq_agent.errors import AgentError
from dq_agent.guardrails import GuardrailsState

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRecordInput(StrictModel):
    data_path: Optional[str] = None
    config_path: Optional[str] = None
    output_dir: Optional[str] = None


class RunRecordFingerprints(StrictModel):
    data_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    git_sha: Optional[str] = None
    python_version: str
    platform: str
    dq_agent_version: str


class RunRecordOutputs(StrictModel):
    report_json_path: Optional[str] = None
    report_md_path: Optional[str] = None
    report_json_sha256: Optional[str] = None
    report_md_sha256: Optional[str] = None
    trace_path: Optional[str] = None


class RunRecordModel(StrictModel):
    schema_version: Literal[1] = Field(default=1)
    run_id: str
    status: Literal["SUCCESS", "FAILED_GUARDRAIL", "FAILED"] = Field(default="SUCCESS")
    error: Optional[AgentError] = None
    started_at: datetime
    finished_at: datetime
    command: str
    argv: List[str]
    input: RunRecordInput
    fingerprints: RunRecordFingerprints
    outputs: RunRecordOutputs
    guardrails: GuardrailsState

    @model_validator(mode="after")
    def _validate_error_presence(self) -> "RunRecordModel":
        if self.status == "SUCCESS":
            if self.error is not None:
                raise ValueError("error must be omitted for successful run records")
            if not self.outputs.report_json_path:
                raise ValueError("report_json_path is required for successful run records")
        elif self.error is None:
            raise ValueError("error must be set for failed run records")
        return self
