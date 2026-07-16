"""Run multiple PGIXS workflow configs from one batch file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


WorkflowRunner = Callable[[list[str] | None], int]


@dataclass
class BatchJob:
    workflow: str
    config: str | None = None
    args: list[str] | None = None
    label: str = ""


@dataclass
class BatchJobResult:
    job: BatchJob
    ok: bool
    exit_code: int
    message: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PGIXS workflow batch from a JSON job file.")
    parser.add_argument("--batch", required=False, help="Batch JSON file containing a jobs array.")
    parser.add_argument("--manifest", required=False, help="Shared JSONL manifest path for TrXS/GIXS jobs.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed job instead of continuing through the batch.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.batch:
        parser.exit(status=0, message="Batch workflow runner is ready; pass --batch jobs.json.\n")

    jobs = load_batch_jobs(args.batch)
    results = run_batch(jobs, shared_manifest=args.manifest, stop_on_error=bool(args.stop_on_error))
    failed = [result for result in results if not result.ok]
    message = format_batch_summary(results)
    parser.exit(status=1 if failed else 0, message=message)
    return 0


def load_batch_jobs(file_path: str | Path) -> list[BatchJob]:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(raw_jobs, list):
        raise ValueError("Batch JSON must be an array or an object with a 'jobs' array.")
    jobs: list[BatchJob] = []
    for index, item in enumerate(raw_jobs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Batch job {index} must be an object.")
        workflow = str(item.get("workflow", "")).lower().replace("_", "-")
        if workflow not in {"trxs", "gixs", "export-1d"}:
            raise ValueError(f"Batch job {index} has unsupported workflow: {workflow}")
        config = item.get("config")
        raw_args = item.get("args", [])
        if raw_args is None:
            raw_args = []
        if not isinstance(raw_args, list) or not all(isinstance(value, str) for value in raw_args):
            raise ValueError(f"Batch job {index} args must be a list of strings.")
        jobs.append(
            BatchJob(
                workflow=workflow,
                config=None if config is None else str(config),
                args=list(raw_args),
                label=str(item.get("label", f"job-{index}")),
            )
        )
    return jobs


def run_batch(
    jobs: list[BatchJob],
    shared_manifest: str | None = None,
    stop_on_error: bool = False,
    runners: dict[str, WorkflowRunner] | None = None,
) -> list[BatchJobResult]:
    if runners is None:
        runners = _default_runners()
    results: list[BatchJobResult] = []
    for job in jobs:
        argv = build_job_argv(job, shared_manifest=shared_manifest)
        runner = runners[job.workflow]
        exit_code = 0
        message = ""
        try:
            exit_code = int(runner(argv) or 0)
        except SystemExit as exc:
            exit_code, message = _system_exit_result(exc)
        except Exception as exc:  # noqa: BLE001 - batch summary must preserve per-job failures.
            exit_code = 1
            message = str(exc)
        result = BatchJobResult(job=job, ok=(exit_code == 0), exit_code=exit_code, message=message)
        results.append(result)
        if stop_on_error and not result.ok:
            break
    return results


def build_job_argv(job: BatchJob, shared_manifest: str | None = None) -> list[str]:
    argv: list[str] = []
    if job.config:
        argv.extend(["--config", job.config])
    argv.extend(job.args or [])
    if shared_manifest and job.workflow in {"trxs", "gixs"} and "--manifest" not in argv:
        argv.extend(["--manifest", shared_manifest])
    return argv


def format_batch_summary(results: list[BatchJobResult]) -> str:
    lines = ["PGIXS batch summary"]
    for index, result in enumerate(results, start=1):
        status = "OK" if result.ok else f"FAIL({result.exit_code})"
        label = result.job.label or f"job-{index}"
        suffix = f" - {result.message}" if result.message else ""
        lines.append(f"  {index}. {label} [{result.job.workflow}]: {status}{suffix}")
    failed = sum(1 for result in results if not result.ok)
    lines.append(f"completed: {len(results)} jobs, failed: {failed}")
    return "\n".join(lines) + "\n"


def _default_runners() -> dict[str, WorkflowRunner]:
    from pgixs_python.workflows.export_1d_from_h5 import main as export_1d_main
    from pgixs_python.workflows.gixs_process import main as gixs_main
    from pgixs_python.workflows.trxs_process import main as trxs_main

    return {
        "trxs": trxs_main,
        "gixs": gixs_main,
        "export-1d": export_1d_main,
    }


def _system_exit_result(exc: SystemExit) -> tuple[int, str]:
    if exc.code is None:
        return 0, ""
    if isinstance(exc.code, int):
        return int(exc.code), ""
    return 1, str(exc.code)


__all__ = [
    "BatchJob",
    "BatchJobResult",
    "build_job_argv",
    "build_parser",
    "format_batch_summary",
    "load_batch_jobs",
    "main",
    "run_batch",
]


if __name__ == "__main__":
    raise SystemExit(main())
