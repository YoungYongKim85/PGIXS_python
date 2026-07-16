"""JSON report export for workflow provenance and diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WorkflowReport:
    measurement_mode: str
    input_files: list[str]
    output_files: dict[str, str | None]
    options: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def save_workflow_report_json(file_path: str | Path, report: WorkflowReport) -> Path:
    """Save a human-readable JSON workflow report."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measurement_mode": report.measurement_mode,
        "input_files": report.input_files,
        "output_files": report.output_files,
        "options": _json_safe(report.options),
        "calibration": _json_safe(report.calibration),
        "diagnostics": _json_safe(report.diagnostics),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
