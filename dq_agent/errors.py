"""Typed error models for failures."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentErrorType(str, Enum):
    guardrail_violation = "guardrail_violation"
    io_error = "io_error"
    config_error = "config_error"
    schema_validation_error = "schema_validation_error"
    internal_error = "internal_error"


class AgentError(StrictModel):
    type: AgentErrorType
    code: str
    message: str
    is_retryable: bool
    suggested_next_step: str
    details: Optional[dict[str, Any]] = None
