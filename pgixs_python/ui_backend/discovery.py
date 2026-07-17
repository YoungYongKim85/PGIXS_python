"""Discovery/status helpers for the future PGIXS local web UI."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from pgixs_python import __version__
from pgixs_python.workflows.config_schema import build_workflow_schema


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_WORKFLOWS = ("batch", "gixs", "trxs", "export-1d")


@dataclass(frozen=True)
class UIWorkflowInfo:
    workflow: str
    title: str
    schema: dict[str, object]


@dataclass(frozen=True)
class UIBackendStatus:
    version: str
    python_version: str
    python_ok: bool
    platform: str
    root: str
    workflows: tuple[str, ...]


def get_workflow_info(workflow: str) -> UIWorkflowInfo:
    """Return title/schema metadata for one workflow."""

    workflow_key = normalize_workflow(workflow)
    schema = build_workflow_schema(workflow_key)
    return UIWorkflowInfo(
        workflow=workflow_key,
        title=str(schema.get("title", workflow_key)),
        schema=schema,
    )


def list_workflows() -> list[UIWorkflowInfo]:
    """Return all workflow schemas needed by the UI form layer."""

    return [get_workflow_info(workflow) for workflow in SUPPORTED_WORKFLOWS]


def get_backend_status() -> UIBackendStatus:
    """Return a compact backend status suitable for a UI status panel."""

    return UIBackendStatus(
        version=__version__,
        python_version=sys.version.split()[0],
        python_ok=(sys.version_info[:2] == (3, 10)),
        platform=platform.platform(),
        root=str(ROOT),
        workflows=SUPPORTED_WORKFLOWS,
    )


def normalize_workflow(workflow: str) -> str:
    normalized = workflow.strip().lower().replace("_", "-")
    if normalized not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"Unsupported workflow: {workflow}")
    return normalized
