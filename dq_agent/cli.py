import json
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from dq_agent.config import load_config
from dq_agent.contract import validate_contract
from dq_agent.demo.generate_demo_data import generate_demo_data
from dq_agent.loader import load_table
from dq_agent.report.writer_md import write_report_md
from dq_agent.report.writer_json import write_report_json
from dq_agent.anomalies import run_anomalies
from dq_agent.rules import run_rules

class FailOn(str, Enum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"


app = typer.Typer(add_completion=False)


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


@app.command()
def run(
    data: Path = typer.Option(..., "--data", help="Path to CSV/Parquet data"),
    config: Path = typer.Option(..., "--config", help="Path to YAML/JSON config"),
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
    fail_on: Optional[FailOn] = typer.Option(
        None,
        "--fail-on",
        case_sensitive=False,
        help="Exit with code 2 when issues/anomalies meet or exceed this severity.",
    ),
) -> None:
    """Run data quality checks against a dataset."""
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    start = time.perf_counter()
    cfg = load_config(config)
    df = load_table(data)
    timings["load"] = (time.perf_counter() - start) * 1000

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
    report_path = write_report_json(
        output_dir=output_dir,
        data_path=data,
        config_path=config,
        rows=len(df.index),
        cols=len(df.columns),
        contract_issues=issues,
        rule_results=rule_results,
        anomalies=anomaly_results,
        observability_timing_ms=timings,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_md_path = report_path.with_name("report.md")
    write_report_md(report, report_md_path)
    timings["report"] = (time.perf_counter() - report_start) * 1000
    timings["total"] = (time.perf_counter() - total_start) * 1000
    report.setdefault("observability", {}).setdefault("timing_ms", {})["report"] = round(
        timings["report"], 3
    )
    report.setdefault("observability", {}).setdefault("timing_ms", {})["total"] = round(
        timings["total"], 3
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "report_json_path": str(report_path),
                "report_md_path": str(report_md_path),
            },
            ensure_ascii=False,
        )
    )
    if _should_fail(
        fail_on=fail_on,
        contract_issues=issues,
        rule_results=rule_results,
        anomalies=anomaly_results,
    ):
        raise typer.Exit(code=2)


@app.command()
def demo(
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
    seed: Optional[int] = typer.Option(42, "--seed"),
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
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    start = time.perf_counter()
    cfg = load_config(config_path)
    df = load_table(data_path)
    timings["load"] = (time.perf_counter() - start) * 1000

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
    report_path = write_report_json(
        output_dir=output_dir,
        data_path=data_path,
        config_path=config_path,
        rows=len(df.index),
        cols=len(df.columns),
        contract_issues=issues,
        rule_results=rule_results,
        anomalies=anomaly_results,
        observability_timing_ms=timings,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_md_path = report_path.with_name("report.md")
    write_report_md(report, report_md_path)
    timings["report"] = (time.perf_counter() - report_start) * 1000
    timings["total"] = (time.perf_counter() - total_start) * 1000
    report.setdefault("observability", {}).setdefault("timing_ms", {})["report"] = round(
        timings["report"], 3
    )
    report.setdefault("observability", {}).setdefault("timing_ms", {})["total"] = round(
        timings["total"], 3
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "report_json_path": str(report_path),
                "report_md_path": str(report_md_path),
            },
            ensure_ascii=False,
        )
    )
    if _should_fail(
        fail_on=fail_on,
        contract_issues=issues,
        rule_results=rule_results,
        anomalies=anomaly_results,
    ):
        raise typer.Exit(code=2)
