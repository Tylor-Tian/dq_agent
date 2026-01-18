from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1)
    run_id: str
    seq: int
    t_ms: int
    event: str
    stage: Optional[str] = None
    status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class Tracer:
    def __init__(self, *, run_id: str, trace_path: Path, start_time: Optional[float] = None) -> None:
        self.run_id = run_id
        self.trace_path = trace_path
        self._start_time = start_time if start_time is not None else time.perf_counter()
        self._seq = 0

    def emit(
        self,
        *,
        event: str,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        self._seq += 1
        t_ms = int((time.perf_counter() - self._start_time) * 1000)
        payload = TraceEvent(
            schema_version=1,
            run_id=self.run_id,
            seq=self._seq,
            t_ms=t_ms,
            event=event,
            stage=stage,
            status=status,
            details=details,
            error=error,
        )
        line = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return payload
