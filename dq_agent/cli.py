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


class FailOn(str, Enum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"


class SchemaKind(str, Enum):
    report = "report"
    run_record = "run_record"


def _get_schema_model(kind: SchemaKind):
    if kind == SchemaKind.report:
        return Report
    if kind == SchemaKind.run_record:
        return RunRecordModel
    raise ValueError(f"Unsupported schema kind: {kind}")


app = typer.Typer(add_completion=False)


@dataclass
class RunExecutionResult:
    report_path: Optional[Path]
    report_md_path: Optional[Path]
    run_record_path: Optional[Path]
    issues: list
    rule_results: list
    anomaly_results: list
    status: str
    error: Optional[AgentError]


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


def _exit_code_for_error(error: AgentError) -> int:
    if error.type in {AgentErrorType.guardrail_violation, AgentErrorType.schema_validation_error}:
        return 2
    return 1


def _emit_failure_payload(
    *,
    error: AgentError,
    report_json_path: Optional[Path],
    report_md_path: Optional[Path],
    run_record_path: Optional[Path],
) -> None:
    payload: dict[str, object] = {"error": error.model_dump(mode="json")}
    if report_json_path is not None:
        payload["report_json_path"] = str(report_json_path)
    if report_md_path is not None:
        payload["report_md_path"] = str(report_md_path)
    if run_record_path is not None:
        payload["run_record_path"] = str(run_record_path)
    typer.echo(json.dumps(payload, ensure_ascii=False))


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
) -> RunExecutionResult:
    run_id = uuid.uuid4().hex
    run_started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    guardrails_observed = GuardrailsObserved()
    issues: list = []
    rule_results: list = []
    anomaly_results: list = []
    df = None

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_md_path: Optional[Path] = None
    report_written = False

    try:
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

        timings["load"] = (time.perf_counter() - start) * 1000
        _wall_time_guardrail(limits=guardrails, started_at=total_start, enforce=enforce_wall_time)

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
        _wall_time_guardrail(limits=guardrails, started_at=total_start, enforce=enforce_wall_time)

        start = time.perf_counter()
        issues = validate_contract(df, cfg)
        timings["contract"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        rule_results = run_rules(df, cfg)
        timings["rules"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        anomaly_results = run_anomalies(df, cfg)
        timings["anomalies"] = (time.perf_counter() - start) * 1000

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
        )
        write_report_json(report_model, report_path)
        report_written = True
        report_md_path = report_path.with_name("report.md")
        write_report_md(report_model.model_dump(mode="json"), report_md_path)
        timings["report"] = (time.perf_counter() - report_start) * 1000
        timings["total"] = (time.perf_counter() - total_start) * 1000
        report_model.observability.timing_ms.report = round(timings["report"], 3)
        report_model.observability.timing_ms.total = round(timings["total"], 3)
        report_model.finished_at = datetime.now(timezone.utc)
        write_report_json(report_model, report_path)
        run_finished_at = datetime.now(timezone.utc)
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
            guardrails=guardrails_state,
            status="SUCCESS",
            error=None,
        )
        return RunExecutionResult(
            report_path=report_path,
            report_md_path=report_md_path,
            run_record_path=run_record_path,
            issues=issues,
            rule_results=rule_results,
            anomaly_results=anomaly_results,
            status="SUCCESS",
            error=None,
        )
    except GuardrailError as exc:
        error = _make_guardrail_error(exc)
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
        )
        write_report_json(report_model, report_path)
        report_written = True
        run_finished_at = datetime.now(timezone.utc)
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
            report_md_path=None,
            guardrails=guardrails_state,
            status="FAILED_GUARDRAIL",
            error=error,
        )
        return RunExecutionResult(
            report_path=report_path,
            report_md_path=None,
            run_record_path=run_record_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED_GUARDRAIL",
            error=error,
        )
    except ExecutionError as exc:
        error = exc.error
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        run_finished_at = datetime.now(timezone.utc)
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
            guardrails=guardrails_state,
            status="FAILED",
            error=error,
        )
        return RunExecutionResult(
            report_path=report_path if report_written else None,
            report_md_path=None,
            run_record_path=run_record_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED",
            error=error,
        )
    except ValidationError as exc:
        error = _make_schema_validation_error("pydantic_validation", {"errors": exc.errors()})
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        run_finished_at = datetime.now(timezone.utc)
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
            guardrails=guardrails_state,
            status="FAILED",
            error=error,
        )
        return RunExecutionResult(
            report_path=report_path if report_written else None,
            report_md_path=None,
            run_record_path=run_record_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            status="FAILED",
            error=error,
        )
    except Exception as exc:
        error = _make_internal_error(exc)
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[],
            observed=guardrails_observed,
        )
        run_finished_at = datetime.now(timezone.utc)
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
            guardrails=guardrails_state,
            status="FAILED",
            error=error,
        )
        return RunExecutionResult(
            report_path=report_path if report_written else None,
            report_md_path=None,
            run_record_path=run_record_path,
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
    result = _execute_run(
        data_path=data,
        config_path=config,
        output_dir=output_dir,
        command="run",
        argv=sys.argv,
        guardrails=guardrails,
        enforce_wall_time=True,
    )
    if result.error is not None:
        _emit_failure_payload(
            error=result.error,
            report_json_path=result.report_path,
            report_md_path=result.report_md_path,
            run_record_path=result.run_record_path,
        )
        raise typer.Exit(code=_exit_code_for_error(result.error))
    typer.echo(
        json.dumps(
            {
                "report_json_path": str(result.report_path),
                "report_md_path": str(result.report_md_path),
                "run_record_path": str(result.run_record_path),
            },
            ensure_ascii=False,
        )
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
    demo_dir = output_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    data_path = generate_demo_data(demo_dir, seed=seed)

    config_path = Path(__file__).parent / "resources" / "demo_rules.yml"
    guardrails = GuardrailsConfig(
        max_input_mb=max_input_mb,
        max_rows=max_rows,
        max_cols=max_cols,
        max_rules=max_rules,
        max_anomalies=max_anomalies,
        max_wall_time_s=max_wall_time_s,
    )
    result = _execute_run(
        data_path=data_path,
        config_path=config_path,
        output_dir=output_dir,
        command="demo",
        argv=sys.argv,
        guardrails=guardrails,
        enforce_wall_time=True,
    )
    if result.error is not None:
        _emit_failure_payload(
            error=result.error,
            report_json_path=result.report_path,
            report_md_path=result.report_md_path,
            run_record_path=result.run_record_path,
        )
        raise typer.Exit(code=_exit_code_for_error(result.error))
    typer.echo(
        json.dumps(
            {
                "report_json_path": str(result.report_path),
                "report_md_path": str(result.report_md_path),
                "run_record_path": str(result.run_record_path),
            },
            ensure_ascii=False,
        )
    )
    if _should_fail(
        fail_on=fail_on,
        contract_issues=result.issues,
        rule_results=result.rule_results,
        anomalies=result.anomaly_results,
    ):
        raise typer.Exit(code=2)


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


@app.command()
def schema(
    kind: SchemaKind = typer.Option(..., "--kind"),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Print JSON Schema for report or run_record."""
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
    """Validate a JSON file against the report or run_record schema."""
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
