"""Run record schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    report_json_path: str
    report_md_path: Optional[str] = None
    report_json_sha256: Optional[str] = None
    report_md_sha256: Optional[str] = None


class RunRecordModel(StrictModel):
    schema_version: Literal[1] = Field(default=1)
    run_id: str
    started_at: datetime
    finished_at: datetime
    command: str
    argv: List[str]
    input: RunRecordInput
    fingerprints: RunRecordFingerprints
    outputs: RunRecordOutputs
