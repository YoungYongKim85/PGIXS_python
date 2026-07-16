"""Command-line scaffold for GIXS Tab1 processing."""

from __future__ import annotations

import argparse

from pgixs_python.workflows.common import (
    add_calibration_arguments,
    add_h5_input_arguments,
    format_h5_series_summary,
    format_qmap_summary,
    load_h5_input_stack,
    load_workflow_qmap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process GIXS raw data with PGIXS Tab1 Python core.")
    add_h5_input_arguments(parser)
    add_calibration_arguments(parser, include_incident_angle=True)
    parser.add_argument("--out", required=False, help="Output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input:
        parser.exit(status=0, message="GIXS workflow scaffold is ready; pass --input to load an H5 stack.\n")
    result = load_h5_input_stack(args)
    qmap = load_workflow_qmap(args, result.stack.shape[1:3], mode="gixs")
    parser.exit(status=0, message=format_h5_series_summary("GIXS", result) + format_qmap_summary(qmap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
