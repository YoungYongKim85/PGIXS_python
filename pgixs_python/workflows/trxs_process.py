"""Command-line scaffold for TrXS Tab1 processing."""

from __future__ import annotations

import argparse

from pgixs_python.workflows.common import (
    add_calibration_arguments,
    add_h5_input_arguments,
    add_output_arguments,
    add_processing_arguments,
    export_workflow_1d,
    export_workflow_mat,
    format_1d_export_summary,
    format_h5_series_summary,
    format_mat_export_summary,
    format_output_summary,
    format_processing_summary,
    format_qmap_summary,
    format_report_summary,
    load_h5_input_stack,
    load_workflow_qmap,
    process_workflow_stack,
    save_workflow_report,
    save_workflow_two_d_info,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process TrXS raw data with PGIXS Tab1 Python core.")
    add_h5_input_arguments(parser)
    add_calibration_arguments(parser)
    add_processing_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input:
        parser.exit(status=0, message="TrXS workflow scaffold is ready; pass --input to load an H5 stack.\n")
    result = load_h5_input_stack(args)
    qmap = load_workflow_qmap(args, result.stack.shape[1:3], mode="trxs")
    processed = process_workflow_stack(args, result, qmap)
    output_path = save_workflow_two_d_info(args, "TrXS", processed.stack_result, qmap, processed.diagnostics)
    mat_export = export_workflow_mat(args, "TrXS", processed.stack_result, qmap, processed.diagnostics)
    export_1d = export_workflow_1d(args, "TrXS", processed.stack_result, qmap)
    report = save_workflow_report(args, "TrXS", result, processed.stack_result, qmap, processed, output_path, mat_export, export_1d)
    parser.exit(
        status=0,
        message=(
            format_h5_series_summary("TrXS", result)
            + format_qmap_summary(qmap)
            + format_processing_summary(processed)
            + format_output_summary(output_path)
            + format_mat_export_summary(mat_export)
            + format_1d_export_summary(export_1d)
            + format_report_summary(report)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
