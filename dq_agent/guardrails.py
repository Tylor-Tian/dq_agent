"""Guardrails for resource usage limits."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Number = Union[int, float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuardrailsConfig(StrictModel):
    max_input_mb: Optional[int] = None
    max_rows: Optional[int] = None
    max_cols: Optional[int] = None
    max_rules: Optional[int] = None
    max_anomalies: Optional[int] = None
    max_wall_time_s: Optional[float] = None


class GuardrailViolation(StrictModel):
    code: str
    message: str
    limit: Optional[Number] = None
    observed: Optional[Number] = None


class GuardrailsObserved(StrictModel):
    input_mb: Optional[float] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    rules: Optional[int] = None
    anomalies: Optional[int] = None
    wall_time_s: Optional[float] = None


class GuardrailsState(StrictModel):
    limits: GuardrailsConfig
    violations: list[GuardrailViolation] = Field(default_factory=list)
    observed: Optional[GuardrailsObserved] = None


class GuardrailError(RuntimeError):
    def __init__(self, violation: GuardrailViolation) -> None:
        super().__init__(violation.message)
        self.violation = violation


def enforce_guardrails(
    *,
    code: str,
    limit: Optional[Number],
    observed: Optional[Number],
    message: str,
) -> None:
    if limit is None or observed is None:
        return
    if observed > limit:
        raise GuardrailError(
            GuardrailViolation(
                code=code,
                message=message,
                limit=limit,
                observed=observed,
            )
        )
