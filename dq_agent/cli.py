import argparse
import hashlib
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from dq_agent.anomalies import run_anomalies
from dq_agent.anomalies.base import get_anomaly
from dq_agent.checkpoint import CheckpointTracker, build_checkpoint, write_checkpoint_atomic
from dq_agent.checkpoint_schema import CheckpointModel
from dq_agent.config import load_config
from dq_agent.contract import validate_contract
from dq_agent.demo.generate_demo_data import generate_demo_data
from dq_agent.errors import AgentError, AgentErrorType
from dq_agent.guardrails import (
    GuardrailError,
    GuardrailsConfig,
    GuardrailsObserved,
    GuardrailsState,
    enforce_guardrails,
)
from dq_agent.loader import load_table
from dq_agent.report.schema import Report
from dq_agent.report.writer_json import build_report_model, write_report_json
from dq_agent.report.writer_md import write_report_md
from dq_agent.rules import run_rules
from dq_agent.rules.base import get_check
from dq_agent.run_record import compare_reports, load_run_record, sha256_path, write_run_record
from dq_agent.run_record_schema import RunRecordModel
from dq_agent.shadow import build_shadow_diff, load_report, write_shadow_diff
from dq_agent.trace import Tracer
from dq_agent.runner import run_job as run_demo_job


class FailOn(str, Enum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"


class SchemaKind(str, Enum):
    report = "report"
    run_record = "run_record"
    checkpoint = "checkpoint"


class IdempotencyMode(str, Enum):
    reuse = "reuse"
    overwrite = "overwrite"
    fail = "fail"


def _get_schema_model(kind: SchemaKind):
    if kind == SchemaKind.report:
        return Report
    if kind == SchemaKind.run_record:
        return RunRecordModel
    if kind == SchemaKind.checkpoint:
        return CheckpointModel
    raise ValueError(f"Unsupported schema kind: {kind}")


app = typer.Typer(add_completion=False)


@dataclass
class RunExecutionResult:
    report_path: Optional[Path]
    report_md_path: Optional[Path]
    run_record_path: Optional[Path]
    trace_path: Path
    checkpoint_path: Optional[Path]
    issues: list
    rule_results: list
    anomaly_results: list
    status: str
    error: Optional[AgentError]


@dataclass(frozen=True)
class IdempotencyContext:
    key: str
    run_id: str
    run_dir: Path
    report_path: Path
    report_md_path: Path
    run_record_path: Path
    trace_path: Path
    checkpoint_path: Path


def _should_fail(
    *,
    fail_on: Optional[FailOn],
    contract_issues: list,
    rule_results: list,
    anomalies: list,
) -> bool:
    if fail_on is None:
        return False
    severity_rank = {"INFO": 0, "WARN": 1, "ERROR": 2}
    threshold = severity_rank[fail_on.value]
    severities = [issue.severity.upper() for issue in contract_issues]
    severities.extend("ERROR" for result in rule_results if result.status == "FAIL")
    severities.extend("ERROR" for result in anomalies if result.status == "FAIL")
    return any(severity_rank.get(severity, 0) >= threshold for severity in severities)


def _count_planned_rules(df, cfg) -> int:
    planned = 0
    for column_name, column_cfg in cfg.columns.items():
        if column_name not in df.columns:
            continue
        for check_entry in column_cfg.checks:
            if not isinstance(check_entry, dict) or len(check_entry) != 1:
                continue
            check_name = next(iter(check_entry.keys()))
            if get_check(check_name) is None:
                continue
            planned += 1
    return planned


def _count_planned_anomalies(df, cfg) -> int:
    planned = 0
    for column_name, column_cfg in cfg.columns.items():
        if column_name not in df.columns:
            continue
        for anomaly_entry in column_cfg.anomalies:
            if not isinstance(anomaly_entry, dict) or len(anomaly_entry) != 1:
                continue
            anomaly_name = next(iter(anomaly_entry.keys()))
            if get_anomaly(anomaly_name) is None:
                continue
            planned += 1
    return planned


def _wall_time_guardrail(*, limits: GuardrailsConfig, started_at: float, enforce: bool) -> None:
    if not enforce or limits.max_wall_time_s is None:
        return
    elapsed = time.perf_counter() - started_at
    enforce_guardrails(
        code="max_wall_time_s",
        limit=limits.max_wall_time_s,
        observed=elapsed,
        message=f"Wall time {elapsed:.3f}s exceeds limit {limits.max_wall_time_s}s.",
    )


def _make_guardrail_error(error: GuardrailError) -> AgentError:
    violation = error.violation
    return AgentError(
        type=AgentErrorType.guardrail_violation,
        code=violation.code,
        message=violation.message,
        is_retryable=False,
        suggested_next_step="Adjust guardrail limits or reduce input size before retrying.",
        details={"limit": violation.limit, "observed": violation.observed},
    )


def _make_io_error(
    code: str,
    message: str,
    path: Optional[Path],
    details: Optional[dict] = None,
) -> AgentError:
    payload = {"path": str(path)} if path is not None else {}
    if details:
        payload.update(details)
    return AgentError(
        type=AgentErrorType.io_error,
        code=code,
        message=message,
        is_retryable=False,
        suggested_next_step="Verify the file exists and is readable, then retry.",
        details=payload or None,
    )


def _make_config_error(code: str, message: str, path: Optional[Path], details: Optional[dict] = None) -> AgentError:
    payload = {"path": str(path)} if path is not None else {}
    if details:
        payload.update(details)
    return AgentError(
        type=AgentErrorType.config_error,
        code=code,
        message=message,
        is_retryable=False,
        suggested_next_step="Fix the config file format or schema, then retry.",
        details=payload or None,
    )


def _make_schema_validation_error(code: str, details: dict) -> AgentError:
    return AgentError(
        type=AgentErrorType.schema_validation_error,
        code=code,
        message="Schema validation failed.",
        is_retryable=False,
        suggested_next_step="Inspect the schema validation errors and fix the output.",
        details=details,
    )


def _make_internal_error(exc: Exception) -> AgentError:
    return AgentError(
        type=AgentErrorType.internal_error,
        code="exception",
        message="Unexpected error during execution.",
        is_retryable=False,
        suggested_next_step="Check logs for details and retry if the issue is resolved.",
        details={"exception": exc.__class__.__name__, "message": str(exc)},
    )


def _make_regression_error(
    *,
    baseline_status: str,
    candidate_status: str,
    baseline_issue_counts: dict,
    candidate_issue_counts: dict,
) -> AgentError:
    return AgentError(
        type=AgentErrorType.regression,
        code="shadow_regression",
        message="Candidate config regressed compared to baseline.",
        is_retryable=False,
        suggested_next_step="Inspect shadow_diff.json and adjust the candidate config before promoting.",
        details={
            "baseline_status": baseline_status,
            "candidate_status": candidate_status,
            "baseline_issue_counts": baseline_issue_counts,
            "candidate_issue_counts": candidate_issue_counts,
        },
    )


def _exit_code_for_error(error: AgentError) -> int:
    if error.type in {
        AgentErrorType.guardrail_violation,
        AgentErrorType.schema_validation_error,
        AgentErrorType.idempotency_conflict,
        AgentErrorType.regression,
    }:
        return 2
    return 1


def _make_idempotency_conflict_error(*, key: str, run_dir: Path) -> AgentError:
    return AgentError(
        type=AgentErrorType.idempotency_conflict,
        code="idempotency_conflict",
        message="Idempotency key already has artifacts.",
        is_retryable=False,
        suggested_next_step="Use --idempotency-mode reuse or overwrite to continue.",
        details={"idempotency_key": key, "run_dir": str(run_dir)},
    )


def _idempotency_run_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _build_idempotency_context(*, key: str, output_dir: Path) -> IdempotencyContext:
    run_id = _idempotency_run_id(key)
    run_dir = output_dir / run_id
    return IdempotencyContext(
        key=key,
        run_id=run_id,
        run_dir=run_dir,
        report_path=run_dir / "report.json",
        report_md_path=run_dir / "report.md",
        run_record_path=run_dir / "run_record.json",
        trace_path=run_dir / "trace.jsonl",
        checkpoint_path=run_dir / "checkpoint.json",
    )


def _idempotency_artifacts_exist(ctx: IdempotencyContext) -> bool:
    return ctx.report_path.exists() and ctx.run_record_path.exists()


def _emit_success_payload(
    *,
    report_json_path: Path,
    report_md_path: Optional[Path],
    run_record_path: Path,
    trace_path: Optional[Path],
    checkpoint_path: Optional[Path],
) -> None:
    payload: dict[str, object] = {
        "report_json_path": str(report_json_path),
        "run_record_path": str(run_record_path),
    }
    if report_md_path is not None:
        payload["report_md_path"] = str(report_md_path)
    if trace_path is not None:
        payload["trace_path"] = str(trace_path)
    if checkpoint_path is not None:
        payload["checkpoint_path"] = str(checkpoint_path)
    typer.echo(json.dumps(payload, ensure_ascii=False))


def _emit_idempotency_reuse(ctx: IdempotencyContext) -> None:
    report_md_path = ctx.report_md_path if ctx.report_md_path.exists() else None
    trace_path = ctx.trace_path if ctx.trace_path.exists() else None
    checkpoint_path = ctx.checkpoint_path if ctx.checkpoint_path.exists() else None
    _emit_success_payload(
        report_json_path=ctx.report_path,
        report_md_path=report_md_path,
        run_record_path=ctx.run_record_path,
        trace_path=trace_path,
        checkpoint_path=checkpoint_path,
    )


def _write_idempotency_failure(
    *,
    ctx: IdempotencyContext,
    data_path: Path,
    config_path: Path,
    output_dir: Path,
    command: str,
    argv: list[str],
    guardrails: GuardrailsConfig,
) -> RunExecutionResult:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    error = _make_idempotency_conflict_error(key=ctx.key, run_dir=ctx.run_dir)
    now = datetime.now(timezone.utc)
    guardrails_state = GuardrailsState(
        limits=guardrails,
        violations=[],
        observed=GuardrailsObserved(),
    )
    checkpoint = build_checkpoint(
        run_id=ctx.run_id,
        command=command,
        argv=argv,
        data_path=data_path,
        config_path=config_path,
        output_dir=output_dir,
        guardrails=guardrails,
        fail_on=None,
        idempotency_key=ctx.key,
        idempotency_mode="fail",
        report_path=ctx.report_path,
        report_md_path=ctx.report_md_path,
        run_record_path=ctx.run_record_path,
        trace_path=ctx.trace_path,
        checkpoint_path=ctx.checkpoint_path,
        status="FAILED",
    )
    for stage_state in checkpoint.stages:
        stage_state.status = "SKIPPED"
    checkpoint.finished_at = now.isoformat()
    checkpoint.updated_at = now.isoformat()
    write_checkpoint_atomic(checkpoint, ctx.checkpoint_path)
    report_model = build_report_model(
        run_id=ctx.run_id,
        started_at=now,
        finished_at=now,
        data_path=data_path,
        config_path=config_path,
        rows=0,
        cols=0,
        contract_issues=[],
        rule_results=[],
        anomalies=[],
        observability_timing_ms={},
        status="FAILED",
        guardrails=guardrails_state,
        error=error,
        trace_file=None,
    )
    write_report_json(report_model, ctx.report_path)
    run_record_path = write_run_record(
        run_id=ctx.run_id,
        started_at=now,
        finished_at=now,
        command=command,
        argv=argv,
        data_path=data_path,
        config_path=config_path,
        output_dir=output_dir,
        run_dir=ctx.run_dir,
        report_json_path=ctx.report_path,
        report_md_path=None,
        trace_path=None,
        guardrails=guardrails_state,
        status="FAILED",
        error=error,
    )
    return RunExecutionResult(
        report_path=ctx.report_path,
        report_md_path=None,
        run_record_path=run_record_path,
        trace_path=ctx.trace_path,
        checkpoint_path=ctx.checkpoint_path,
        issues=[],
        rule_results=[],
        anomaly_results=[],
        status="FAILED",
        error=error,
    )


def _emit_failure_payload(
    *,
    error: AgentError,
    report_json_path: Optional[Path],
    report_md_path: Optional[Path],
    run_record_path: Optional[Path],
    trace_path: Optional[Path],
    checkpoint_path: Optional[Path],
) -> None:
    payload: dict[str, object] = {"error": error.model_dump(mode="json")}
    if report_json_path is not None:
        payload["report_json_path"] = str(report_json_path)
    if report_md_path is not None:
        payload["report_md_path"] = str(report_md_path)
    if run_record_path is not None:
        payload["run_record_path"] = str(run_record_path)
    if trace_path is not None:
        payload["trace_path"] = str(trace_path)
    if checkpoint_path is not None:
        payload["checkpoint_path"] = str(checkpoint_path)
    typer.echo(json.dumps(payload, ensure_ascii=False))


def _emit_shadow_payload(
    *,
    shadow_run_id: str,
    shadow_dir: Path,
    baseline_report_json_path: Path,
    candidate_report_json_path: Path,
    shadow_diff_path: Path,
    error: Optional[AgentError],
) -> None:
    payload: dict[str, object] = {
        "shadow_run_id": shadow_run_id,
        "shadow_dir": str(shadow_dir),
        "baseline_report_json_path": str(baseline_report_json_path),
        "candidate_report_json_path": str(candidate_report_json_path),
        "shadow_diff_path": str(shadow_diff_path),
        "error": error.model_dump(mode="json") if error else None,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))


def _extract_report_error(report: dict) -> Optional[AgentError]:
    error_payload = report.get("error")
    if not error_payload:
        return None
    return AgentError.model_validate(error_payload)


def _ensure_report_md(report_path: Path) -> Optional[Path]:
    if not report_path.exists():
        return None
    report_md_path = report_path.with_name("report.md")
    if report_md_path.exists():
        return report_md_path
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    write_report_md(report_payload, report_md_path)
    return report_md_path


def _load_report_payload(
    report_path: Path,
    *,
    error_override: Optional[AgentError] = None,
) -> tuple[dict, Optional[AgentError]]:
    report_payload = load_report(report_path)
    error = error_override or _extract_report_error(report_payload)
    return report_payload, error


class ExecutionError(RuntimeError):
    def __init__(self, error: AgentError) -> None:
        super().__init__(error.message)
        self.error = error


def _execute_run(
    *,
    data_path: Path,
    config_path: Path,
    output_dir: Path,
    command: str,
    argv: list[str],
    guardrails: GuardrailsConfig,
    enforce_wall_time: bool,
    run_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
    fail_on: Optional[FailOn] = None,
    idempotency_key: Optional[str] = None,
    idempotency_mode: Optional[IdempotencyMode] = None,
) -> RunExecutionResult:
    run_id = run_id or uuid.uuid4().hex
    run_started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    guardrails_observed = GuardrailsObserved()
    issues: list = []
    rule_results: list = []
    anomaly_results: list = []
    df = None

    run_dir = run_dir or output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    run_record_path = run_dir / "run_record.json"
    trace_path = run_dir / "trace.jsonl"
    checkpoint_path = run_dir / "checkpoint.json"
    report_md_target_path = run_dir / "report.md"
    checkpoint = build_checkpoint(
        run_id=run_id,
        command=command,
        argv=argv,
        data_path=data_path,
        config_path=config_path,
        output_dir=output_dir,
        guardrails=guardrails,
        fail_on=fail_on.value if fail_on else None,
        idempotency_key=idempotency_key,
        idempotency_mode=idempotency_mode.value if idempotency_mode else None,
        report_path=report_path,
        report_md_path=report_md_target_path,
        run_record_path=run_record_path,
        trace_path=trace_path,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_tracker = CheckpointTracker(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        start_time=total_start,
    )
    tracer = Tracer(run_id=run_id, trace_path=trace_path, start_time=total_start)
    report_written = False
    report_md_path: Optional[Path] = None
    current_stage: Optional[str] = None

    def _stage_start(stage: str, details: Optional[dict] = None) -> None:
        nonlocal current_stage
        current_stage = stage
        tracer.emit(event="stage_start", stage=stage, details=details)
        checkpoint_tracker.mark_stage_start(stage, details)

    def _stage_end(
        status: str = "OK",
        details: Optional[dict] = None,
        error: Optional[AgentError] = None,
    ) -> None:
        nonlocal current_stage
        if current_stage is None:
            return
        error_payload = error.model_dump(mode="json") if error else None
        checkpoint_status = "OK" if status == "OK" else "FAILED"
        tracer.emit(
            event="stage_end",
            stage=current_stage,
            status=status,
            details=details,
            error=error_payload,
        )
        checkpoint_tracker.mark_stage_end(
            current_stage,
            status=checkpoint_status,
            details=details,
            error=error,
        )
        current_stage = None

    try:
        tracer.emit(
            event="run_start",
            status="OK",
            details={
                "command": command,
                "argv": argv,
                "data_path": str(data_path),
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "guardrails": guardrails.model_dump(mode="json"),
            },
        )
        if data_path.exists():
            input_mb = data_path.stat().st_size / (1024 * 1024)
            guardrails_observed.input_mb = round(input_mb, 3)
            enforce_guardrails(
                code="max_input_mb",
                limit=guardrails.max_input_mb,
                observed=guardrails_observed.input_mb,
                message=f"Input size {guardrails_observed.input_mb} MB exceeds limit {guardrails.max_input_mb} MB.",
            )
        _wall_time_guardrail(limits=guardrails, started_at=total_start, enforce=enforce_wall_time)

        _stage_start("config", {"config_path": str(config_path)})
        start = time.perf_counter()
        try:
            cfg = load_config(config_path)
        except FileNotFoundError:
            raise ExecutionError(_make_io_error("config_not_found", "Config file not found.", config_path))
        except ValidationError as exc:
            raise ExecutionError(
                _make_config_error("invalid_config", "Config failed schema validation.", config_path, {"errors": exc.errors()})
            )
        except ValueError as exc:
            raise ExecutionError(_make_config_error("invalid_config", "Config failed to parse.", config_path, {"message": str(exc)}))
        except Exception as exc:
            raise ExecutionError(
                _make_config_error(
                    "config_read_failed",
                    "Failed to load config.",
                    config_path,
                    {"exception": exc.__class__.__name__, "message": str(exc)},
                )
            )
        _stage_end(status="OK")

        _stage_start("load", {"data_path": str(data_path)})
        try:
            df = load_table(data_path)
        except FileNotFoundError:
            raise ExecutionError(_make_io_error("data_not_found", "Data file not found.", data_path))
        except ValueError as exc:
            raise ExecutionError(
                _make_io_error(
                    "unsupported_data_format",
                    "Unsupported data format.",
                    data_path,
                    {"message": str(exc)},
                )
            )
        except Exception as exc:
            raise ExecutionError(
                _make_io_error(
                    "data_read_failed",
                    "Failed to read data file.",
                    data_path,
                    {"exception": exc.__class__.__name__, "message": str(exc)},
                )
            )
        _stage_end(
            status="OK",
            details={
                "rows": int(len(df.index)),
                "cols": int(len(df.columns)),
            },
        )

        timings["load"] = (time.perf_counter() - start) * 1000
        _wall_time_guardrail(limits=guardrails, started_at=total_start, enforce=enforce_wall_time)

        _stage_start("guardrails")
        guardrails_observed.rows = int(len(df.index))
        guardrails_observed.cols = int(len(df.columns))
        enforce_guardrails(
            code="max_rows",
            limit=guardrails.max_rows,
            observed=guardrails_observed.rows,
            message=f"Row count {guardrails_observed.rows} exceeds limit {guardrails.max_rows}.",
        )
        enforce_guardrails(
            code="max_cols",
            limit=guardrails.max_cols,
            observed=guardrails_observed.cols,
            message=f"Column count {guardrails_observed.cols} exceeds limit {guardrails.max_cols}.",
        )

        guardrails_observed.rules = _count_planned_rules(df, cfg)
        guardrails_observed.anomalies = _count_planned_anomalies(df, cfg)
        enforce_guardrails(
            code="max_rules",
            limit=guardrails.max_rules,
            observed=guardrails_observed.rules,
            message=f"Planned rules {guardrails_observed.rules} exceeds limit {guardrails.max_rules}.",
        )
        enforce_guardrails(
            code="max_anomalies",
            limit=guardrails.max_anomalies,
            observed=guardrails_observed.anomalies,
            message=f"Planned anomalies {guardrails_observed.anomalies} exceeds limit {guardrails.max_anomalies}.",
        )
        _stage_end(
            status="OK",
            details=guardrails_observed.model_dump(mode="json"),
        )
        _wall_time_guardrail(limits=guardrails, started_at=total_start, enforce=enforce_wall_time)

        start = time.perf_counter()
        _stage_start("contract")
        issues = validate_contract(df, cfg)
        timings["contract"] = (time.perf_counter() - start) * 1000
        _stage_end(status="OK", details={"issues": len(issues)})

        start = time.perf_counter()
        _stage_start("rules")
        rule_results = run_rules(df, cfg)
        timings["rules"] = (time.perf_counter() - start) * 1000
        _stage_end(status="OK", details={"results": len(rule_results)})

        start = time.perf_counter()
        _stage_start("anomalies")
        anomaly_results = run_anomalies(df, cfg)
        timings["anomalies"] = (time.perf_counter() - start) * 1000
        _stage_end(status="OK", details={"results": len(anomaly_results)})

        report_start = time.perf_counter()
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        report_model = build_report_model(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=datetime.now(timezone.utc),
            data_path=data_path,
            config_path=config_path,
            rows=len(df.index),
            cols=len(df.columns),
            contract_issues=issues,
            rule_results=rule_results,
            anomalies=anomaly_results,
            observability_timing_ms=timings,
            status="SUCCESS",
            guardrails=guardrails_state,
            trace_file=trace_path.name,
        )
        _stage_start("report_json", {"report_json_path": str(report_path)})
        write_report_json(report_model, report_path)
        _stage_end(status="OK")
        report_written = True
        report_md_path = report_path.with_name("report.md")
        _stage_start("report_md", {"report_md_path": str(report_md_path)})
        write_report_md(report_model.model_dump(mode="json"), report_md_path)
        _stage_end(status="OK")
        timings["report"] = (time.perf_counter() - report_start) * 1000
        timings["total"] = (time.perf_counter() - total_start) * 1000
        report_model.observability.timing_ms.report = round(timings["report"], 3)
        report_model.observability.timing_ms.total = round(timings["total"], 3)
        report_model.finished_at = datetime.now(timezone.utc)
        report_model.trace_file = trace_path.name
        write_report_json(report_model, report_path)
        run_finished_at = datetime.now(timezone.utc)
        _stage_start("run_record")
        run_record_path = write_run_record(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=run_finished_at,
            command=command,
            argv=argv,
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir,
            run_dir=run_dir,
            report_json_path=report_path,
            report_md_path=report_md_path,
            trace_path=trace_path,
            guardrails=guardrails_state,
            status="SUCCESS",
            error=None,
        )
        _stage_end(status="OK")
        tracer.emit(event="run_end", status="SUCCESS")
        checkpoint_tracker.finalize(status="SUCCESS", trace_path=trace_path)
        return RunExecutionResult(
            report_path=report_path,
            report_md_path=report_md_path,
            run_record_path=run_record_path,
            trace_path=trace_path,
            checkpoint_path=checkpoint_path,
            issues=issues,
            rule_results=rule_results,
            anomaly_results=anomaly_results,
            status="SUCCESS",
            error=None,
        )
    except GuardrailError as exc:
        error = _make_guardrail_error(exc)
        _stage_end(status="FAILED_GUARDRAIL", error=error)
        rows = int(len(df.index)) if df is not None else 0
        cols = int(len(df.columns)) if df is not None else 0
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[exc.violation],
            observed=guardrails_observed,
        )
        report_model = build_report_model(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=datetime.now(timezone.utc),
            data_path=data_path,
            config_path=config_path,
            rows=rows,
            cols=cols,
            contract_issues=[],
            rule_results=[],
            anomalies=[],
            observability_timing_ms=timings,
            status="FAILED_GUARDRAIL",
            guardrails=guardrails_state,
            error=error,
            trace_file=trace_path.name,
        )
        _stage_start("report_json", {"report_json_path": str(report_path)})
        write_report_json(report_model, report_path)
        _stage_end(status="OK")
        report_written = True
        report_md_path = report_path.with_name("report.md")
        _stage_start("report_md", {"report_md_path": str(report_md_path)})
        write_report_md(report_model.model_dump(mode="json"), report_md_path)
        _stage_end(status="OK")
        run_finished_at = datetime.now(timezone.utc)
        _stage_start("run_record")
        run_record_path = write_run_record(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=run_finished_at,
            command=command,
            argv=argv,
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir,
            run_dir=run_dir,
            report_json_path=report_path,
            report_md_path=report_md_path,
            trace_path=trace_path,
            guardrails=guardrails_state,
            status="FAILED_GUARDRAIL",
            error=error,
        )
        _stage_end(status="OK")
        tracer.emit(event="run_end", status="FAILED_GUARDRAIL", error=error.model_dump(mode="json"))
        checkpoint_tracker.finalize(status="FAILED_GUARDRAIL", error=error, trace_path=trace_path)
        return RunExecutionResult(
            report_path=report_path,
            report_md_path=report_md_path,
            run_record_path=run_record_path,
            trace_path=trace_path,
            checkpoint_path=checkpoint_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED_GUARDRAIL",
            error=error,
        )
    except ExecutionError as exc:
        error = exc.error
        _stage_end(status="FAILED", error=error)
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        rows = int(len(df.index)) if df is not None else 0
        cols = int(len(df.columns)) if df is not None else 0
        if not report_written:
            report_model = build_report_model(
                run_id=run_id,
                started_at=run_started_at,
                finished_at=datetime.now(timezone.utc),
                data_path=data_path,
                config_path=config_path,
                rows=rows,
                cols=cols,
                contract_issues=issues,
                rule_results=rule_results,
                anomalies=anomaly_results,
                observability_timing_ms=timings,
                status="FAILED",
                guardrails=guardrails_state,
                error=error,
                trace_file=trace_path.name,
            )
            _stage_start("report_json", {"report_json_path": str(report_path)})
            write_report_json(report_model, report_path)
            _stage_end(status="OK")
            report_written = True
            report_md_path = report_path.with_name("report.md")
            _stage_start("report_md", {"report_md_path": str(report_md_path)})
            write_report_md(report_model.model_dump(mode="json"), report_md_path)
            _stage_end(status="OK")
        run_finished_at = datetime.now(timezone.utc)
        _stage_start("run_record")
        run_record_path = write_run_record(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=run_finished_at,
            command=command,
            argv=argv,
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir,
            run_dir=run_dir,
            report_json_path=report_path if report_written else None,
            report_md_path=report_md_path,
            trace_path=trace_path,
            guardrails=guardrails_state,
            status="FAILED",
            error=error,
        )
        _stage_end(status="OK")
        tracer.emit(event="run_end", status="FAILED", error=error.model_dump(mode="json"))
        checkpoint_tracker.finalize(status="FAILED", error=error, trace_path=trace_path)
        return RunExecutionResult(
            report_path=report_path if report_written else None,
            report_md_path=report_md_path,
            run_record_path=run_record_path,
            trace_path=trace_path,
            checkpoint_path=checkpoint_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED",
            error=error,
        )
    except ValidationError as exc:
        error = _make_schema_validation_error("pydantic_validation", {"errors": exc.errors()})
        _stage_end(status="FAILED", error=error)
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        run_finished_at = datetime.now(timezone.utc)
        _stage_start("run_record")
        run_record_path = write_run_record(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=run_finished_at,
            command=command,
            argv=argv,
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir,
            run_dir=run_dir,
            report_json_path=report_path if report_written else None,
            report_md_path=None,
            trace_path=trace_path,
            guardrails=guardrails_state,
            status="FAILED",
            error=error,
        )
        _stage_end(status="OK")
        tracer.emit(event="run_end", status="FAILED", error=error.model_dump(mode="json"))
        checkpoint_tracker.finalize(status="FAILED", error=error, trace_path=trace_path)
        return RunExecutionResult(
            report_path=report_path if report_written else None,
            report_md_path=None,
            run_record_path=run_record_path,
            trace_path=trace_path,
            checkpoint_path=checkpoint_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED",
            error=error,
        )
    except Exception as exc:
        error = _make_internal_error(exc)
        _stage_end(status="FAILED", error=error)
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        run_finished_at = datetime.now(timezone.utc)
        _stage_start("run_record")
        run_record_path = write_run_record(
            run_id=run_id,
            started_at=run_started_at,
            finished_at=run_finished_at,
            command=command,
            argv=argv,
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir,
            run_dir=run_dir,
            report_json_path=report_path if report_written else None,
            report_md_path=None,
            trace_path=trace_path,
            guardrails=guardrails_state,
            status="FAILED",
            error=error,
        )
        _stage_end(status="OK")
        tracer.emit(event="run_end", status="FAILED", error=error.model_dump(mode="json"))
        checkpoint_tracker.finalize(status="FAILED", error=error, trace_path=trace_path)
        return RunExecutionResult(
            report_path=report_path if report_written else None,
            report_md_path=None,
            run_record_path=run_record_path,
            trace_path=trace_path,
            checkpoint_path=checkpoint_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED",
            error=error,
        )


@app.command()
def run(
    data: Path = typer.Option(..., "--data", help="Path to CSV/Parquet data"),
    config: Path = typer.Option(..., "--config", help="Path to YAML/JSON config"),
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    idempotency_mode: IdempotencyMode = typer.Option(
        IdempotencyMode.reuse,
        "--idempotency-mode",
        case_sensitive=False,
    ),
    max_input_mb: Optional[int] = typer.Option(None, "--max-input-mb"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows"),
    max_cols: Optional[int] = typer.Option(None, "--max-cols"),
    max_rules: Optional[int] = typer.Option(None, "--max-rules"),
    max_anomalies: Optional[int] = typer.Option(None, "--max-anomalies"),
    max_wall_time_s: Optional[float] = typer.Option(None, "--max-wall-time-s"),
    fail_on: Optional[FailOn] = typer.Option(
        None,
        "--fail-on",
        case_sensitive=False,
        help="Exit with code 2 when issues/anomalies meet or exceed this severity.",
    ),
) -> None:
    """Run data quality checks against a dataset."""
    guardrails = GuardrailsConfig(
        max_input_mb=max_input_mb,
        max_rows=max_rows,
        max_cols=max_cols,
        max_rules=max_rules,
        max_anomalies=max_anomalies,
        max_wall_time_s=max_wall_time_s,
    )
    idempotency_ctx = None
    if idempotency_key:
        idempotency_ctx = _build_idempotency_context(key=idempotency_key, output_dir=output_dir)
        if _idempotency_artifacts_exist(idempotency_ctx):
            if idempotency_mode == IdempotencyMode.reuse:
                _emit_idempotency_reuse(idempotency_ctx)
                return
            if idempotency_mode == IdempotencyMode.fail:
                result = _write_idempotency_failure(
                    ctx=idempotency_ctx,
                    data_path=data,
                    config_path=config,
                    output_dir=output_dir,
                    command="run",
                    argv=sys.argv,
                    guardrails=guardrails,
                )
                _emit_failure_payload(
                    error=result.error,
                    report_json_path=result.report_path,
                    report_md_path=result.report_md_path,
                    run_record_path=result.run_record_path,
                    trace_path=None,
                    checkpoint_path=result.checkpoint_path,
                )
                raise typer.Exit(code=_exit_code_for_error(result.error))
    result = _execute_run(
        data_path=data,
        config_path=config,
        output_dir=output_dir,
        command="run",
        argv=sys.argv,
        guardrails=guardrails,
        enforce_wall_time=True,
        run_id=idempotency_ctx.run_id if idempotency_ctx else None,
        run_dir=idempotency_ctx.run_dir if idempotency_ctx else None,
        fail_on=fail_on,
        idempotency_key=idempotency_key,
        idempotency_mode=idempotency_mode,
    )
    if result.error is not None:
        _emit_failure_payload(
            error=result.error,
            report_json_path=result.report_path,
            report_md_path=result.report_md_path,
            run_record_path=result.run_record_path,
            trace_path=result.trace_path,
            checkpoint_path=result.checkpoint_path,
        )
        raise typer.Exit(code=_exit_code_for_error(result.error))
    _emit_success_payload(
        report_json_path=result.report_path,
        report_md_path=result.report_md_path,
        run_record_path=result.run_record_path,
        trace_path=result.trace_path,
        checkpoint_path=result.checkpoint_path,
    )
    if _should_fail(
        fail_on=fail_on,
        contract_issues=result.issues,
        rule_results=result.rule_results,
        anomalies=result.anomaly_results,
    ):
        raise typer.Exit(code=2)


@app.command()
def demo(
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
    seed: Optional[int] = typer.Option(42, "--seed"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    idempotency_mode: IdempotencyMode = typer.Option(
        IdempotencyMode.reuse,
        "--idempotency-mode",
        case_sensitive=False,
    ),
    max_input_mb: Optional[int] = typer.Option(None, "--max-input-mb"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows"),
    max_cols: Optional[int] = typer.Option(None, "--max-cols"),
    max_rules: Optional[int] = typer.Option(None, "--max-rules"),
    max_anomalies: Optional[int] = typer.Option(None, "--max-anomalies"),
    max_wall_time_s: Optional[float] = typer.Option(None, "--max-wall-time-s"),
    fail_on: Optional[FailOn] = typer.Option(
        None,
        "--fail-on",
        case_sensitive=False,
        help="Exit with code 2 when issues/anomalies meet or exceed this severity.",
    ),
) -> None:
    """Generate demo data and run the contract checks."""
    guardrails = GuardrailsConfig(
        max_input_mb=max_input_mb,
        max_rows=max_rows,
        max_cols=max_cols,
        max_rules=max_rules,
        max_anomalies=max_anomalies,
        max_wall_time_s=max_wall_time_s,
    )
    demo_dir = output_dir / "demo"
    data_path = demo_dir / "orders.parquet"
    config_path = Path(__file__).parent / "resources" / "demo_rules.yml"
    idempotency_ctx = None
    if idempotency_key:
        idempotency_ctx = _build_idempotency_context(key=idempotency_key, output_dir=output_dir)
        if _idempotency_artifacts_exist(idempotency_ctx):
            if idempotency_mode == IdempotencyMode.reuse:
                _emit_idempotency_reuse(idempotency_ctx)
                return
            if idempotency_mode == IdempotencyMode.fail:
                result = _write_idempotency_failure(
                    ctx=idempotency_ctx,
                    data_path=data_path,
                    config_path=config_path,
                    output_dir=output_dir,
                    command="demo",
                    argv=sys.argv,
                    guardrails=guardrails,
                )
                _emit_failure_payload(
                    error=result.error,
                    report_json_path=result.report_path,
                    report_md_path=result.report_md_path,
                    run_record_path=result.run_record_path,
                    trace_path=None,
                    checkpoint_path=result.checkpoint_path,
                )
                raise typer.Exit(code=_exit_code_for_error(result.error))

    demo_dir.mkdir(parents=True, exist_ok=True)
    data_path = generate_demo_data(demo_dir, seed=seed)
    result = _execute_run(
        data_path=data_path,
        config_path=config_path,
        output_dir=output_dir,
        command="demo",
        argv=sys.argv,
        guardrails=guardrails,
        enforce_wall_time=True,
        run_id=idempotency_ctx.run_id if idempotency_ctx else None,
        run_dir=idempotency_ctx.run_dir if idempotency_ctx else None,
        fail_on=fail_on,
        idempotency_key=idempotency_key,
        idempotency_mode=idempotency_mode,
    )
    if result.error is not None:
        _emit_failure_payload(
            error=result.error,
            report_json_path=result.report_path,
            report_md_path=result.report_md_path,
            run_record_path=result.run_record_path,
            trace_path=result.trace_path,
            checkpoint_path=result.checkpoint_path,
        )
        raise typer.Exit(code=_exit_code_for_error(result.error))
    _emit_success_payload(
        report_json_path=result.report_path,
        report_md_path=result.report_md_path,
        run_record_path=result.run_record_path,
        trace_path=result.trace_path,
        checkpoint_path=result.checkpoint_path,
    )
    if _should_fail(
        fail_on=fail_on,
        contract_issues=result.issues,
        rule_results=result.rule_results,
        anomalies=result.anomaly_results,
    ):
        raise typer.Exit(code=2)


@app.command()
def shadow(
    data: Path = typer.Option(..., "--data", help="Path to CSV/Parquet data"),
    baseline_config: Path = typer.Option(..., "--baseline-config", help="Path to baseline config"),
    candidate_config: Path = typer.Option(..., "--candidate-config", help="Path to candidate config"),
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    idempotency_mode: IdempotencyMode = typer.Option(
        IdempotencyMode.reuse,
        "--idempotency-mode",
        case_sensitive=False,
    ),
    max_input_mb: Optional[int] = typer.Option(None, "--max-input-mb"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows"),
    max_cols: Optional[int] = typer.Option(None, "--max-cols"),
    max_rules: Optional[int] = typer.Option(None, "--max-rules"),
    max_anomalies: Optional[int] = typer.Option(None, "--max-anomalies"),
    max_wall_time_s: Optional[float] = typer.Option(None, "--max-wall-time-s"),
) -> None:
    """Run baseline and candidate configs on the same data and compare outputs."""
    guardrails = GuardrailsConfig(
        max_input_mb=max_input_mb,
        max_rows=max_rows,
        max_cols=max_cols,
        max_rules=max_rules,
        max_anomalies=max_anomalies,
        max_wall_time_s=max_wall_time_s,
    )
    shadow_run_id = _idempotency_run_id(idempotency_key) if idempotency_key else uuid.uuid4().hex
    shadow_dir = output_dir / shadow_run_id
    baseline_root = shadow_dir / "baseline"
    candidate_root = shadow_dir / "candidate"
    shadow_diff_path = shadow_dir / "shadow_diff.json"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)
    candidate_root.mkdir(parents=True, exist_ok=True)

    baseline_ctx = None
    candidate_ctx = None
    baseline_exists = False
    candidate_exists = False
    if idempotency_key:
        baseline_ctx = _build_idempotency_context(key=f"{idempotency_key}:baseline", output_dir=baseline_root)
        candidate_ctx = _build_idempotency_context(key=f"{idempotency_key}:candidate", output_dir=candidate_root)
        baseline_exists = _idempotency_artifacts_exist(baseline_ctx)
        candidate_exists = _idempotency_artifacts_exist(candidate_ctx)
        any_existing = baseline_exists or candidate_exists or shadow_diff_path.exists()
        if idempotency_mode == IdempotencyMode.fail and any_existing:
            error = _make_idempotency_conflict_error(key=idempotency_key, run_dir=shadow_dir)
            _emit_shadow_payload(
                shadow_run_id=shadow_run_id,
                shadow_dir=shadow_dir,
                baseline_report_json_path=baseline_ctx.report_path,
                candidate_report_json_path=candidate_ctx.report_path,
                shadow_diff_path=shadow_diff_path,
                error=error,
            )
            raise typer.Exit(code=_exit_code_for_error(error))

    def _run_or_reuse(
        *,
        label: str,
        config_path: Path,
        output_base: Path,
        ctx: Optional[IdempotencyContext],
        reuse_allowed: bool,
    ) -> tuple[dict, Path, Optional[AgentError]]:
        if ctx and reuse_allowed and _idempotency_artifacts_exist(ctx):
            report_path = ctx.report_path
            _ensure_report_md(report_path)
            report_payload, error_payload = _load_report_payload(report_path)
            return report_payload, report_path, error_payload
        result = _execute_run(
            data_path=data,
            config_path=config_path,
            output_dir=output_base,
            command=f"shadow:{label}",
            argv=sys.argv,
            guardrails=guardrails,
            enforce_wall_time=True,
            run_id=ctx.run_id if ctx else None,
            run_dir=ctx.run_dir if ctx else None,
            idempotency_key=ctx.key if ctx else None,
            idempotency_mode=idempotency_mode,
        )
        if result.report_path is None:
            raise RuntimeError("Missing report.json for shadow run.")
        _ensure_report_md(result.report_path)
        report_payload, error_payload = _load_report_payload(result.report_path, error_override=result.error)
        return report_payload, result.report_path, error_payload

    reuse_allowed = idempotency_mode == IdempotencyMode.reuse
    baseline_report, baseline_report_path, baseline_error = _run_or_reuse(
        label="baseline",
        config_path=baseline_config,
        output_base=baseline_root,
        ctx=baseline_ctx,
        reuse_allowed=reuse_allowed,
    )
    candidate_report, candidate_report_path, candidate_error = _run_or_reuse(
        label="candidate",
        config_path=candidate_config,
        output_base=candidate_root,
        ctx=candidate_ctx,
        reuse_allowed=reuse_allowed,
    )

    shadow_diff = build_shadow_diff(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        shadow_run_id=shadow_run_id,
    )
    write_shadow_diff(shadow_diff, shadow_diff_path)

    baseline_issue_counts = (baseline_report.get("summary") or {}).get("issue_counts") or {}
    candidate_issue_counts = (candidate_report.get("summary") or {}).get("issue_counts") or {}
    baseline_error_count = int(baseline_issue_counts.get("error", 0) or 0)
    candidate_error_count = int(candidate_issue_counts.get("error", 0) or 0)
    baseline_status = str(baseline_report.get("status", "UNKNOWN"))
    candidate_status = str(candidate_report.get("status", "UNKNOWN"))
    regression = (
        (baseline_status == "SUCCESS" and candidate_status.startswith("FAILED"))
        or candidate_error_count > baseline_error_count
    )

    error: Optional[AgentError] = None
    exit_code = 0
    if fail_on_regression and regression:
        error = _make_regression_error(
            baseline_status=baseline_status,
            candidate_status=candidate_status,
            baseline_issue_counts=baseline_issue_counts,
            candidate_issue_counts=candidate_issue_counts,
        )
        exit_code = 2
    elif baseline_error is not None:
        error = baseline_error
        exit_code = _exit_code_for_error(error)
    elif candidate_error is not None:
        error = candidate_error
        exit_code = _exit_code_for_error(error)

    _emit_shadow_payload(
        shadow_run_id=shadow_run_id,
        shadow_dir=shadow_dir,
        baseline_report_json_path=baseline_report_path,
        candidate_report_json_path=candidate_report_path,
        shadow_diff_path=shadow_diff_path,
        error=error,
    )
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.command()
def replay(
    run_record: Path = typer.Option(..., "--run-record", help="Path to run_record.json"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    """Replay a run from a run_record.json and compare reports."""
    try:
        record = load_run_record(run_record).data
    except Exception as exc:
        typer.echo(f"Invalid run record: {exc}", err=True)
        raise typer.Exit(code=1)

    input_section = record.get("input") or {}
    data_path_raw = input_section.get("data_path")
    config_path_raw = input_section.get("config_path")
    if not data_path_raw or not config_path_raw:
        typer.echo("Run record missing data_path or config_path.", err=True)
        raise typer.Exit(code=1)

    data_path = Path(data_path_raw)
    config_path = Path(config_path_raw)
    if not data_path.exists():
        typer.echo(f"Missing data file: {data_path}", err=True)
        raise typer.Exit(code=1)
    if not config_path.exists():
        typer.echo(f"Missing config file: {config_path}", err=True)
        raise typer.Exit(code=1)

    fingerprints = record.get("fingerprints") or {}
    expected_data_sha = fingerprints.get("data_sha256")
    expected_config_sha = fingerprints.get("config_sha256")
    actual_data_sha = sha256_path(data_path)
    actual_config_sha = sha256_path(config_path)
    if expected_data_sha and actual_data_sha != expected_data_sha:
        typer.echo(
            f"Warning: data sha256 mismatch (expected {expected_data_sha}, got {actual_data_sha}).",
            err=True,
        )
    if expected_config_sha and actual_config_sha != expected_config_sha:
        typer.echo(
            f"Warning: config sha256 mismatch (expected {expected_config_sha}, got {actual_config_sha}).",
            err=True,
        )

    resolved_output_dir = output_dir
    if resolved_output_dir is None:
        recorded_output_dir = input_section.get("output_dir")
        if not recorded_output_dir:
            typer.echo("Run record missing output_dir and none provided.", err=True)
            raise typer.Exit(code=1)
        resolved_output_dir = Path(recorded_output_dir)

    outputs = record.get("outputs") or {}
    old_report_json_raw = outputs.get("report_json_path")
    if not old_report_json_raw:
        typer.echo("Run record missing report_json_path.", err=True)
        raise typer.Exit(code=1)
    old_report_json_path = Path(old_report_json_raw)
    if not old_report_json_path.exists():
        typer.echo(f"Missing original report.json: {old_report_json_path}", err=True)
        raise typer.Exit(code=1)

    guardrails_section = record.get("guardrails") or {}
    guardrails_limits = guardrails_section.get("limits") or {}
    guardrails = GuardrailsConfig.model_validate(guardrails_limits)

    new_report_path = _execute_run(
        data_path=data_path,
        config_path=config_path,
        output_dir=resolved_output_dir,
        command="replay",
        argv=sys.argv,
        guardrails=guardrails,
        enforce_wall_time=False,
    ).report_path

    try:
        old_report = json.loads(old_report_json_path.read_text(encoding="utf-8"))
        new_report = json.loads(new_report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"Failed to load report JSON: {exc}", err=True)
        raise typer.Exit(code=1)

    same, summary = compare_reports(old_report, new_report)
    result = {
        "same": same,
        "old_run_id": record.get("run_id"),
        "new_run_id": new_report.get("run_id"),
        "old_report_json_path": str(old_report_json_path),
        "new_report_json_path": str(new_report_path),
        "diff_summary": summary,
    }
    typer.echo(json.dumps(result, ensure_ascii=False))
    if strict and not same:
        raise typer.Exit(code=2)


def _find_checkpoint_error(checkpoint: CheckpointModel) -> Optional[AgentError]:
    for stage in checkpoint.stages:
        if stage.error is not None:
            return stage.error
    return None


def _load_checkpoint(path: Path) -> CheckpointModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CheckpointModel.model_validate(payload)


@app.command()
def resume(
    run_dir: Path = typer.Option(..., "--run-dir", help="Path to the existing run directory"),
    force: bool = typer.Option(False, "--force", help="Force rerun even for terminal failures"),
) -> None:
    """Resume a run using an existing checkpoint.json."""
    checkpoint_path = run_dir / "checkpoint.json"
    try:
        checkpoint = _load_checkpoint(checkpoint_path)
    except FileNotFoundError as exc:
        typer.echo(f"Missing checkpoint: {exc}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        typer.echo(f"Invalid JSON in checkpoint: {exc}", err=True)
        raise typer.Exit(code=1)
    except ValidationError as exc:
        typer.echo(f"Checkpoint validation failed: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        typer.echo(f"Unexpected error reading checkpoint: {exc}", err=True)
        raise typer.Exit(code=1)

    report_path = Path(checkpoint.artifacts.report_json_path or run_dir / "report.json")
    report_md_path = Path(checkpoint.artifacts.report_md_path or run_dir / "report.md")
    run_record_path = Path(checkpoint.artifacts.run_record_path or run_dir / "run_record.json")
    trace_path = Path(checkpoint.artifacts.trace_path or run_dir / "trace.jsonl")

    report_payload: Optional[dict] = None
    report_model: Optional[Report] = None
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            report_model = Report.model_validate(report_payload)
        except (json.JSONDecodeError, ValidationError):
            report_payload = None
            report_model = None

    if checkpoint.status == "FAILED_GUARDRAIL" and not force:
        error = _find_checkpoint_error(checkpoint)
        if report_model is not None and report_model.error is not None:
            error = report_model.error
        if error is None:
            error = AgentError(
                type=AgentErrorType.guardrail_violation,
                code="guardrail_failure",
                message="Run failed guardrails; resume requires --force to rerun.",
                is_retryable=False,
                suggested_next_step="Adjust guardrail limits or re-run with --force to continue.",
                details={"run_dir": str(run_dir)},
            )
        _emit_failure_payload(
            error=error,
            report_json_path=report_path if report_path.exists() else None,
            report_md_path=report_md_path if report_md_path.exists() else None,
            run_record_path=run_record_path if run_record_path.exists() else None,
            trace_path=trace_path if trace_path.exists() else None,
            checkpoint_path=checkpoint_path,
        )
        raise typer.Exit(code=2)

    if report_model is None:
        data_path = checkpoint.input.data_path
        config_path = checkpoint.input.config_path
        if not data_path or not config_path:
            typer.echo("Checkpoint missing data_path or config_path; cannot resume.", err=True)
            raise typer.Exit(code=1)
        data_path_obj = Path(data_path)
        config_path_obj = Path(config_path)
        if not data_path_obj.exists():
            typer.echo(f"Missing data file: {data_path_obj}", err=True)
            raise typer.Exit(code=1)
        if not config_path_obj.exists():
            typer.echo(f"Missing config file: {config_path_obj}", err=True)
            raise typer.Exit(code=1)
        fail_on = None
        if checkpoint.input.fail_on:
            try:
                fail_on = FailOn(checkpoint.input.fail_on)
            except ValueError:
                fail_on = None
        idempotency_mode = None
        if checkpoint.input.idempotency_mode:
            try:
                idempotency_mode = IdempotencyMode(checkpoint.input.idempotency_mode)
            except ValueError:
                idempotency_mode = None
        result = _execute_run(
            data_path=data_path_obj,
            config_path=config_path_obj,
            output_dir=Path(checkpoint.input.output_dir or run_dir.parent),
            command=checkpoint.command,
            argv=checkpoint.argv,
            guardrails=checkpoint.input.guardrails,
            enforce_wall_time=True,
            run_id=checkpoint.run_id,
            run_dir=run_dir,
            fail_on=fail_on,
            idempotency_key=checkpoint.input.idempotency_key,
            idempotency_mode=idempotency_mode,
        )
        if result.error is not None:
            _emit_failure_payload(
                error=result.error,
                report_json_path=result.report_path,
                report_md_path=result.report_md_path,
                run_record_path=result.run_record_path,
                trace_path=result.trace_path,
                checkpoint_path=result.checkpoint_path,
            )
            raise typer.Exit(code=_exit_code_for_error(result.error))
        _emit_success_payload(
            report_json_path=result.report_path,
            report_md_path=result.report_md_path,
            run_record_path=result.run_record_path,
            trace_path=result.trace_path,
            checkpoint_path=result.checkpoint_path,
        )
        if _should_fail(
            fail_on=fail_on,
            contract_issues=result.issues,
            rule_results=result.rule_results,
            anomalies=result.anomaly_results,
        ):
            raise typer.Exit(code=2)
        return

    repaired = False
    if not report_md_path.exists() and report_payload is not None:
        write_report_md(report_payload, report_md_path)
        repaired = True

    if not run_record_path.exists():
        data_path = Path(checkpoint.input.data_path) if checkpoint.input.data_path else None
        config_path = Path(checkpoint.input.config_path) if checkpoint.input.config_path else None
        output_dir = Path(checkpoint.input.output_dir) if checkpoint.input.output_dir else run_dir.parent
        guardrails_state = GuardrailsState.model_validate(report_payload.get("guardrails") if report_payload else {})
        run_record_path = write_run_record(
            run_id=checkpoint.run_id,
            started_at=report_model.started_at,
            finished_at=report_model.finished_at,
            command=checkpoint.command,
            argv=checkpoint.argv,
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir,
            run_dir=run_dir,
            report_json_path=report_path,
            report_md_path=report_md_path if report_md_path.exists() else None,
            trace_path=trace_path if trace_path.exists() else None,
            guardrails=guardrails_state,
            status=report_model.status,
            error=report_model.error,
        )
        repaired = True

    if repaired:
        now = datetime.now(timezone.utc).isoformat()
        checkpoint.updated_at = now
        if checkpoint.finished_at is None:
            checkpoint.finished_at = now
        checkpoint.status = report_model.status
        checkpoint.artifacts.report_json_path = str(report_path)
        checkpoint.artifacts.report_md_path = str(report_md_path) if report_md_path.exists() else None
        checkpoint.artifacts.run_record_path = str(run_record_path) if run_record_path.exists() else None
        checkpoint.artifacts.trace_path = str(trace_path) if trace_path.exists() else None
        for stage in checkpoint.stages:
            if stage.name == "report_json" and report_path.exists():
                stage.status = "OK"
            if stage.name == "report_md" and report_md_path.exists():
                stage.status = "OK"
            if stage.name == "run_record" and run_record_path.exists():
                stage.status = "OK"
            if stage.name == "trace" and trace_path.exists():
                stage.status = "OK"
        write_checkpoint_atomic(checkpoint, checkpoint_path)

    _emit_success_payload(
        report_json_path=report_path,
        report_md_path=report_md_path if report_md_path.exists() else None,
        run_record_path=run_record_path,
        trace_path=trace_path if trace_path.exists() else None,
        checkpoint_path=checkpoint_path,
    )


@app.command()
def schema(
    kind: SchemaKind = typer.Option(..., "--kind"),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Print JSON Schema for report, run_record, or checkpoint."""
    model = _get_schema_model(kind)
    schema_payload = model.model_json_schema()
    output = json.dumps(schema_payload, ensure_ascii=False, indent=2)
    if out is None:
        typer.echo(output)
        return
    out.write_text(output, encoding="utf-8")


@app.command()
def validate(
    kind: SchemaKind = typer.Option(..., "--kind"),
    path: Path = typer.Option(..., "--path"),
) -> None:
    """Validate a JSON file against the report, run_record, or checkpoint schema."""
    model = _get_schema_model(kind)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model.model_validate(payload)
    except FileNotFoundError as exc:
        typer.echo(f"Failed to read {path}: {exc}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        typer.echo(f"Invalid JSON in {path}: {exc}", err=True)
        raise typer.Exit(code=1)
    except ValidationError as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        typer.echo(f"Unexpected error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo("ok")


def _resolve_demo_input(inp: str | None, out: str) -> str:
    if inp:
        return inp
    example = Path("examples/orders.csv")
    if example.exists():
        return str(example)
    fallback = Path(out).parent / "orders.csv"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(
        "order_id,amount,email\n1,10,a@example.com\n1,-2,bad\n2,,x@example.com\n",
        encoding="utf-8",
    )
    return str(fallback)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--in", dest="inp", default=None)
    demo_parser.add_argument("--out", dest="out", default="artifacts/demo/report.md")
    demo_parser.add_argument("--config", dest="config", default=None)
    demo_parser.add_argument("--limit", dest="limit", type=int, default=0)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--in", dest="inp", required=True)
    run_parser.add_argument("--out", dest="out", required=True)
    run_parser.add_argument("--config", dest="config", default="dq.yaml")
    run_parser.add_argument("--limit", dest="limit", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "demo":
        inp = _resolve_demo_input(args.inp, args.out)
        config = args.config or "dq.yaml"
        run_demo_job(inp=inp, out=args.out, limit=args.limit, config=config)
        print(args.out)
        return 0

    run_demo_job(inp=args.inp, out=args.out, limit=args.limit, config=args.config)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
