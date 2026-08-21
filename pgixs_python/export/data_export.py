"""Tab1 1D data export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OneDExport:
    x: np.ndarray
    intensity: np.ndarray
    std: np.ndarray | None = None
    counts: np.ndarray | None = None
    coverage: np.ndarray | None = None
    x_label: str = "q"
    intensity_label: str = "intensity"


def save_1d_data(file_path: str | Path, data: OneDExport, include_diagnostics: bool = False) -> Path:
    """Save 1D curve data as CSV or whitespace DAT.

    The default exported columns intentionally stay close to the MATLAB
    OneDplot cleanup request: q, intensity, and optional standard deviation.
    Counts/coverage diagnostics are opt-in because they are useful for gap
    analysis but noisy for routine publication data.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table, columns = build_1d_table(data, include_diagnostics=include_diagnostics)
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    header = delimiter.join(columns)
    np.savetxt(path, table, delimiter=delimiter, header=header, comments="", fmt="%.10g")
    return path


def build_1d_table(data: OneDExport, include_diagnostics: bool = False) -> tuple[np.ndarray, list[str]]:
    """Build a numeric table for 1D data export."""

    x = _as_1d(data.x, "x")
    intensity = _as_1d(data.intensity, "intensity")
    if x.size != intensity.size:
        raise ValueError("x and intensity must have the same length.")

    columns = [data.x_label, data.intensity_label]
    arrays = [x, intensity]

    if data.std is not None:
        std = _as_1d(data.std, "std")
        if std.size != x.size:
            raise ValueError("std must have the same length as x.")
        columns.append("standard_deviation")
        arrays.append(std)

    if include_diagnostics:
        if data.counts is not None:
            counts = _as_1d(data.counts, "counts")
            if counts.size != x.size:
                raise ValueError("counts must have the same length as x.")
            columns.append("valid_count")
            arrays.append(counts)
        if data.coverage is not None:
            coverage = _as_1d(data.coverage, "coverage")
            if coverage.size != x.size:
                raise ValueError("coverage must have the same length as x.")
            columns.append("coverage")
            arrays.append(coverage)

    return np.column_stack(arrays).astype(np.float32), columns


def save_series_data(
    file_path: str | Path,
    x_axis: np.ndarray,
    x_label: str,
    matrix: np.ndarray,
    column_labels: list[str],
) -> Path:
    """Save a GIXS2-style series curve: one shared axis column plus one
    intensity column per series frame (Save 1D/Save Azi/Save Integ).

    `matrix` is `(n_frames, n_points)` (the shape `series_processing.py`'s
    `SeriesCurveResult.intensity` and `process_roi_stack` both produce) --
    transposed here so each frame becomes its own column, matching the
    MATLAB save layout (one Q/angle/X-axis column, N frame columns).
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = _as_1d(x_axis, "x_axis")
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != x.size:
        raise ValueError(f"matrix must be (n_frames, {x.size}); got {mat.shape}.")
    if len(column_labels) != mat.shape[0]:
        raise ValueError(f"column_labels must have {mat.shape[0]} entries; got {len(column_labels)}.")

    table = np.column_stack([x, mat.T]).astype(np.float32)
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    # A label containing the delimiter itself (e.g. the X-axis label
    # "Time, s") would silently split into an extra header field once
    # written out -- np.savetxt's header is a plain joined string with no
    # quoting, so every column after it reads one column short (reported:
    # ROI1..ROI5 each shifted one column left of their actual data).
    columns = [label.replace(delimiter, " ") if delimiter in label else label for label in (x_label, *column_labels)]
    header = delimiter.join(columns)
    np.savetxt(path, table, delimiter=delimiter, header=header, comments="", fmt="%.10g")
    return path


def _as_1d(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr
