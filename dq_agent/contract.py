from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd

from dq_agent.config import Config


@dataclass
class ContractIssue:
    issue_id: str
    severity: str
    message: str
    column: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _type_issue(column: str, expected_type: str) -> ContractIssue:
    return ContractIssue(
        issue_id="type_mismatch",
        severity="error",
        message=f"Column '{column}' failed to parse as {expected_type}.",
        column=column,
    )


def _required_issue(column: str) -> ContractIssue:
    return ContractIssue(
        issue_id="missing_required_column",
        severity="error",
        message=f"Required column '{column}' is missing.",
        column=column,
    )


def _can_parse(series: pd.Series, expected_type: str) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True
    if expected_type == "string":
        return True
    if expected_type == "int":
        parsed = pd.to_numeric(non_null, errors="coerce", downcast="integer")
        return parsed.notna().all()
    if expected_type == "float":
        parsed = pd.to_numeric(non_null, errors="coerce")
        return parsed.notna().all()
    if expected_type == "bool":
        try:
            non_null.astype("boolean")
        except (TypeError, ValueError):
            return False
        return True
    if expected_type == "datetime":
        parsed = pd.to_datetime(non_null, errors="coerce")
        return parsed.notna().all()
    return True


def validate_contract(df: pd.DataFrame, cfg: Config) -> List[ContractIssue]:
    issues: List[ContractIssue] = []
    for column_name, column_cfg in cfg.columns.items():
        if column_cfg.required and column_name not in df.columns:
            issues.append(_required_issue(column_name))
            continue
        if column_name not in df.columns:
            continue
        if column_cfg.type:
            expected_type = column_cfg.type.lower()
            if not _can_parse(df[column_name], expected_type):
                issues.append(_type_issue(column_name, expected_type))

    for key in cfg.dataset.primary_key:
        if key not in df.columns:
            issues.append(_required_issue(key))
        elif df[key].isna().any():
            issues.append(
                ContractIssue(
                    issue_id="primary_key_null",
                    severity="error",
                    message=f"Primary key column '{key}' contains nulls.",
                    column=key,
                )
            )

    return issues
