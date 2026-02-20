from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from dq_agent.reporters.markdown import to_markdown


def _read_table(inp: str) -> pd.DataFrame:
    path = Path(inp)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def _load_config(config: str | None) -> dict[str, Any]:
    cfg_path = Path(config) if config else Path("dq.yaml")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _advice_for(rule_type: str, column: str) -> str:
    if rule_type == "unique":
        return f"De-duplicate {column} and enforce uniqueness upstream."
    if rule_type == "min_value":
        return f"Fix negative values in {column} and add source validation."
    if rule_type == "regex":
        return f"Normalize {column} and validate format before ingestion."
    if rule_type == "null_ratio":
        return "Backfill required fields and add null checks in ETL."
    return "Review this rule and upstream data quality controls."


def run_job(inp: str, out: str, limit: int = 0, config: str | None = None) -> dict[str, Any]:
    df = _read_table(inp)
    cfg = _load_config(config)

    rules: list[dict[str, Any]] = []
    for check in cfg.get("checks", []):
        rid = check.get("id", "unknown_rule")
        rtype = check.get("type", "unknown")
        column = check.get("column", "")
        hits = 0
        detail = ""

        if rtype == "unique":
            hits = int(df[column].duplicated(keep=False).sum())
        elif rtype == "min_value":
            threshold = check.get("gte", 0)
            series = pd.to_numeric(df[column], errors="coerce")
            hits = int((series < threshold).fillna(False).sum())
        elif rtype == "regex":
            pattern = re.compile(check.get("pattern", ".*"))
            series = df[column].dropna().astype(str)
            hits = int((~series.str.match(pattern)).sum())
        elif rtype == "null_ratio":
            threshold = float(check.get("lte", 0.0))
            if column == "*":
                ratios = df.isna().mean()
                bad = ratios[ratios > threshold]
                hits = int(len(bad))
                if hits:
                    detail = ", ".join(f"{name}={value:.2%}" for name, value in bad.items())
            else:
                ratio = float(df[column].isna().mean())
                hits = 1 if ratio > threshold else 0
                if hits:
                    detail = f"{column}={ratio:.2%}"

        rules.append(
            {
                "id": rid,
                "type": rtype,
                "hits": hits,
                "detail": detail,
                "advice": _advice_for(rtype, column),
            }
        )

    scoring = cfg.get("scoring", {})
    base = int(scoring.get("base", 100))
    penalties = scoring.get("penalties", {})
    score = base - sum(int(penalties.get(rule["id"], 0)) for rule in rules if rule["hits"] > 0)
    score = max(0, min(base, score))

    failing = [rule for rule in rules if rule["hits"] > 0]
    if limit and limit > 0:
        failing = failing[:limit]
    samples = [
        row.to_json(force_ascii=False)
        for _, row in df.head(max(2, limit or 3)).iterrows()
    ]

    result = {
        "meta": {
            "input": inp,
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "config": config or "dq.yaml",
        },
        "score": score,
        "rules": rules,
        "samples": samples,
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_markdown(result), encoding="utf-8")
    return result
