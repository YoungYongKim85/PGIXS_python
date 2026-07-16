"""Shared command-line helpers for PGIXS Tab1 workflows."""

from __future__ import annotations

from pathlib import Path

from pgixs_python.io.h5_loader import DEFAULT_DETECTOR_DATASET
from pgixs_python.io.h5_series import H5SeriesLoadResult, load_h5_series


def add_h5_input_arguments(parser) -> None:
    parser.add_argument("--input", required=False, help="Input H5 detector file or folder.")
    parser.add_argument("--dataset", default=DEFAULT_DETECTOR_DATASET, help="H5 detector dataset path.")
    parser.add_argument("--metadata-mode", default="", help="Metadata scan mode: Temperature, LowTemp., Angle, or blank.")
    parser.add_argument("--reference-timestamp", type=float, default=None, help="Optional timestamp zero-point.")
    parser.add_argument("--retry-count", type=int, default=1, help="Frame load retry count before NaN fallback.")


def collect_h5_inputs(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Only H5/HDF5 inputs are wired into the workflow scaffold: {path}")
        return [path]
    if path.is_dir():
        files = sorted([*path.glob("*.h5"), *path.glob("*.hdf5")])
        if not files:
            raise FileNotFoundError(f"No .h5/.hdf5 files found in input folder: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def load_h5_input_stack(args) -> H5SeriesLoadResult:
    files = collect_h5_inputs(args.input)
    return load_h5_series(
        files,
        dataset_path=args.dataset,
        retry_count=args.retry_count,
        metadata_mode=args.metadata_mode,
        reference_timestamp=args.reference_timestamp,
    )


def format_h5_series_summary(label: str, result: H5SeriesLoadResult) -> str:
    return (
        f"{label} H5 input loaded\n"
        f"  frames: {result.stack.shape[0]}\n"
        f"  image shape: {result.stack.shape[1]} x {result.stack.shape[2]}\n"
        f"  retry failed frames: {result.retry_failed_frames.tolist()}\n"
        f"  metadata failed frames: {result.metadata_failed_frames.tolist()}\n"
    )
