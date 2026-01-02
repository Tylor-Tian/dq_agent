from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ColumnConfig(BaseModel):
    type: Optional[str] = None
    required: bool = False
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    name: Optional[str] = None
    primary_key: List[str] = Field(default_factory=list)
    time_column: Optional[str] = None
    expected_row_count: Optional[Dict[str, Any]] = None


class ReportConfig(BaseModel):
    sample_rows: int = 20


class Config(BaseModel):
    version: int = 1
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    columns: Dict[str, ColumnConfig] = Field(default_factory=dict)
    report: ReportConfig = Field(default_factory=ReportConfig)


def _parse_config(data: Dict[str, Any]) -> Config:
    if hasattr(Config, "model_validate"):
        return Config.model_validate(data)
    return Config.parse_obj(data)


def load_config(path: Path) -> Config:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(content)
    elif path.suffix.lower() == ".json":
        data = json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return _parse_config(data)
