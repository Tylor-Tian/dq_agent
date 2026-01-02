import json
from pathlib import Path
from typing import Optional

import typer

from dq_agent.config import load_config
from dq_agent.contract import validate_contract
from dq_agent.demo.generate_demo_data import generate_demo_data
from dq_agent.loader import load_table
from dq_agent.report.writer_json import write_report_json

app = typer.Typer(add_completion=False)


@app.command()
def run(
    data: Path = typer.Option(..., "--data", exists=True, help="Path to CSV/Parquet data"),
    config: Path = typer.Option(..., "--config", exists=True, help="Path to YAML/JSON config"),
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
) -> None:
    """Run data quality checks against a dataset."""
    cfg = load_config(config)
    df = load_table(data)
    issues = validate_contract(df, cfg)
    report_path = write_report_json(
        output_dir=output_dir,
        data_path=data,
        config_path=config,
        rows=len(df.index),
        cols=len(df.columns),
        contract_issues=issues,
    )
    typer.echo(json.dumps({"report_path": str(report_path)}, ensure_ascii=False))


@app.command()
def demo(
    output_dir: Path = typer.Option(Path("artifacts"), "--output-dir"),
    seed: Optional[int] = typer.Option(42, "--seed"),
) -> None:
    """Generate demo data and run the contract checks."""
    demo_dir = output_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    data_path = generate_demo_data(demo_dir, seed=seed)

    config_path = Path(__file__).parent / "resources" / "demo_rules.yml"
    cfg = load_config(config_path)
    df = load_table(data_path)
    issues = validate_contract(df, cfg)
    report_path = write_report_json(
        output_dir=output_dir,
        data_path=data_path,
        config_path=config_path,
        rows=len(df.index),
        cols=len(df.columns),
        contract_issues=issues,
    )
    typer.echo(json.dumps({"report_path": str(report_path)}, ensure_ascii=False))
