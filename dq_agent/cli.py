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
    report_path: Path
    report_md_path: Optional[Path]
    run_record_path: Path
    issues: list
    rule_results: list
    anomaly_results: list
    guardrail_violation: Optional[dict]
    status: str


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
        cfg = load_config(config_path)
        df = load_table(data_path)
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
            report_json_path=report_path,
            report_md_path=report_md_path,
            guardrails=guardrails_state,
        )
        return RunExecutionResult(
            report_path=report_path,
            report_md_path=report_md_path,
            run_record_path=run_record_path,
            issues=issues,
            rule_results=rule_results,
            anomaly_results=anomaly_results,
            guardrail_violation=None,
            status="SUCCESS",
        )
    except GuardrailError as exc:
        guardrail_violation = exc.violation
        rows = int(len(df.index)) if df is not None else 0
        cols = int(len(df.columns)) if df is not None else 0
        guardrails_state = GuardrailsState(
            limits=guardrails,
            violations=[guardrail_violation],
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
        )
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
            report_json_path=report_path,
            report_md_path=None,
            guardrails=guardrails_state,
        )
        return RunExecutionResult(
            report_path=report_path,
            report_md_path=None,
            run_record_path=run_record_path,
            issues=[],
            rule_results=[],
            anomaly_results=[],
            guardrail_violation=guardrail_violation.model_dump(mode="json"),
            status="FAILED_GUARDRAIL",
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
    if result.guardrail_violation is not None:
        payload = {
            "error": {
                "type": "guardrail_violation",
                "code": result.guardrail_violation.get("code"),
                "message": result.guardrail_violation.get("message"),
            },
            "report_json_path": str(result.report_path),
            "run_record_path": str(result.run_record_path),
        }
        typer.echo(json.dumps(payload, ensure_ascii=False))
        raise typer.Exit(code=2)
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
    if result.guardrail_violation is not None:
        payload = {
            "error": {
                "type": "guardrail_violation",
                "code": result.guardrail_violation.get("code"),
                "message": result.guardrail_violation.get("message"),
            },
            "report_json_path": str(result.report_path),
            "run_record_path": str(result.run_record_path),
        }
        typer.echo(json.dumps(payload, ensure_ascii=False))
        raise typer.Exit(code=2)
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
