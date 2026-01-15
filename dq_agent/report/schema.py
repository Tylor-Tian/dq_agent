"""Report schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dq_agent.errors import AgentError
from dq_agent.guardrails import GuardrailsState

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportInput(StrictModel):
    data_path: str
    config_path: str
    format: str


class IssueCounts(StrictModel):
    error: int
    warn: int
    info: int


class Summary(StrictModel):
    rows: int
    cols: int
    issue_counts: IssueCounts


class ContractIssue(StrictModel):
    issue_id: str
    severity: Literal["error", "warn", "info"]
    message: str
    column: Optional[str] = None


class Sample(StrictModel):
    row_index: Union[int, str]
    value: Any | None = None
    z: Optional[float] = None


class RuleResult(StrictModel):
    rule_id: str
    column: str
    status: Literal["PASS", "FAIL"]
    failing_ratio: float
    failed_count: int
    total_count: int
    samples: List[Sample]


class AnomalyResult(StrictModel):
    anomaly_id: str
    column: str
    status: Literal["PASS", "FAIL"]
    metric: Dict[str, Any]
    threshold: Dict[str, Any]
    samples: List[Sample]
    explanation: str


class TimingMs(StrictModel):
    load: float = 0.0
    contract: float = 0.0
    rules: float = 0.0
    anomalies: float = 0.0
    report: float = 0.0
    total: float = 0.0


class ObservabilityCounts(StrictModel):
    rules_total: int
    rules_failed: int
    anomalies_total: int
    anomalies_failed: int


class Observability(StrictModel):
    timing_ms: TimingMs
    counts: ObservabilityCounts


class Report(StrictModel):
    schema_version: Literal[1] = Field(default=1)
    run_id: str
    status: Literal["SUCCESS", "FAILED_GUARDRAIL", "FAILED"] = Field(default="SUCCESS")
    error: Optional[AgentError] = None
    started_at: datetime
    finished_at: datetime
    input: ReportInput
    summary: Summary
    contract_issues: List[ContractIssue]
    rule_results: List[RuleResult]
    anomalies: List[AnomalyResult]
    fix_actions: List[Dict[str, Any]]
    observability: Observability
    guardrails: GuardrailsState

    @model_validator(mode="after")
    def _validate_error_presence(self) -> "Report":
        if self.status == "SUCCESS":
            if self.error is not None:
                raise ValueError("error must be omitted for successful reports")
        elif self.error is None:
            raise ValueError("error must be set for failed reports")
        return self
