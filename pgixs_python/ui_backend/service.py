"""Thin workflow service contract for the future local web UI."""

from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pgixs_python.workflows.common import parse_workflow_args


WorkflowRunner = Callable[[list[str] | None], int]
ParserBuilder = Callable[[], argparse.ArgumentParser]


@dataclass
class UIWorkflowRequest:
    """UI-facing description of one workflow request."""

    workflow: str
    config: str | Path | None = None
    args: list[str] = field(default_factory=list)
    label: str = ""


@dataclass
class UIWorkflowResult:
    """UI-facing result for validation or processing."""

    workflow: str
    ok: bool
    exit_code: int
    message: str = ""
    stdout: str = ""
    stderr: str = ""


def validate_workflow(request: UIWorkflowRequest, parsers: dict[str, ParserBuilder] | None = None) -> UIWorkflowResult:
    """Validate one workflow request without loading detector data."""

    try:
        workflow = normalize_workflow(request.workflow)
        parser_builder = (parsers or default_parsers())[workflow]
    except Exception as exc:  # noqa: BLE001 - UI layer must report invalid requests.
        return UIWorkflowResult(workflow=request.workflow, ok=False, exit_code=1, message=str(exc))
    argv = build_workflow_argv(request)

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            parse_workflow_args(parser_builder(), argv)
    except SystemExit as exc:
        exit_code, message = system_exit_result(exc)
        message = message or stderr.getvalue().strip() or stdout.getvalue().strip()
        return UIWorkflowResult(
            workflow=workflow,
            ok=(exit_code == 0),
            exit_code=exit_code,
            message=message,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )
    except Exception as exc:  # noqa: BLE001 - UI layer must report validation failures.
        message = str(exc) or stderr.getvalue().strip() or stdout.getvalue().strip()
        return UIWorkflowResult(workflow=workflow, ok=False, exit_code=1, message=message)

    return UIWorkflowResult(
        workflow=workflow,
        ok=True,
        exit_code=0,
        message="validation ok",
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def run_workflow(request: UIWorkflowRequest, runners: dict[str, WorkflowRunner] | None = None) -> UIWorkflowResult:
    """Run one workflow request and capture output for the UI log panel."""

    try:
        workflow = normalize_workflow(request.workflow)
        runner = (runners or default_runners())[workflow]
    except Exception as exc:  # noqa: BLE001 - UI layer must report invalid requests.
        return UIWorkflowResult(workflow=request.workflow, ok=False, exit_code=1, message=str(exc))
    argv = build_workflow_argv(request)

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = int(runner(argv) or 0)
    except SystemExit as exc:
        exit_code, message = system_exit_result(exc)
        message = message or stderr.getvalue().strip() or stdout.getvalue().strip()
    except Exception as exc:  # noqa: BLE001 - UI layer must report backend failures.
        exit_code = 1
        message = str(exc) or stderr.getvalue().strip() or stdout.getvalue().strip()
    else:
        message = stdout.getvalue().strip()

    return UIWorkflowResult(
        workflow=workflow,
        ok=(exit_code == 0),
        exit_code=exit_code,
        message=message,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def build_workflow_argv(request: UIWorkflowRequest) -> list[str]:
    """Build CLI-compatible argv from a UI request."""

    argv: list[str] = []
    if request.config:
        argv.extend(["--config", str(request.config)])
    argv.extend(list(request.args or []))
    return argv


def normalize_workflow(workflow: str) -> str:
    normalized = workflow.strip().lower().replace("_", "-")
    if normalized not in {"gixs", "trxs", "export-1d"}:
        raise ValueError(f"Unsupported workflow: {workflow}")
    return normalized


def default_parsers() -> dict[str, ParserBuilder]:
    from pgixs_python.workflows.export_1d_from_h5 import build_parser as build_export_parser
    from pgixs_python.workflows.gixs_process import build_parser as build_gixs_parser
    from pgixs_python.workflows.trxs_process import build_parser as build_trxs_parser

    return {
        "gixs": build_gixs_parser,
        "trxs": build_trxs_parser,
        "export-1d": build_export_parser,
    }


def default_runners() -> dict[str, WorkflowRunner]:
    from pgixs_python.workflows.export_1d_from_h5 import main as export_main
    from pgixs_python.workflows.gixs_process import main as gixs_main
    from pgixs_python.workflows.trxs_process import main as trxs_main

    return {
        "gixs": gixs_main,
        "trxs": trxs_main,
        "export-1d": export_main,
    }


def system_exit_result(exc: SystemExit) -> tuple[int, str]:
    if exc.code is None:
        return 0, ""
    if isinstance(exc.code, int):
        return int(exc.code), ""
    return 1, str(exc.code)
