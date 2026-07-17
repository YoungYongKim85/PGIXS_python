"""Preview helpers for local web UI result inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.averaging import summarize_coverage
from pgixs_python.io.two_d_info import load_two_d_info_h5


def two_d_info_preview_payload(file_path: str | Path, max_points: int = 64) -> dict[str, Any]:
    """Load a 2D-info H5 file and return a compact JSON-friendly preview."""

    path = Path(file_path)
    info = load_two_d_info_h5(path)
    image = np.asarray(info.data2d_q, dtype=np.float32)
    q_map = np.asarray(info.q_map, dtype=np.float32)
    finite = np.isfinite(image)
    finite_q = np.isfinite(q_map)
    return {
        "ok": True,
        "path": str(path),
        "measurement_mode": info.measurement_mode,
        "detector": info.detector,
        "shape": [int(value) for value in image.shape],
        "finite_fraction": _finite_fraction(image),
        "nan_fraction": float(1.0 - _finite_fraction(image)),
        "intensity": _array_stats(image, finite),
        "q": _array_stats(q_map, finite_q),
        "calibration": {
            "beam_position": _array_to_list(info.beam_position),
            "wavelength": float(info.wavelength),
            "sdd": float(info.sdd),
            "pixel": float(info.pixel),
            "energy_kev": None if info.energy_kev is None else float(info.energy_kev),
            "incident_angle": None if info.incident_angle is None else float(info.incident_angle),
        },
        "source_files": list(info.source_files),
        "retry_failed_frames": _array_to_list(info.retry_failed_frames),
        "metadata_failed_frames": _array_to_list(info.metadata_failed_frames),
        "preview": {
            "data2d_q": _downsample_for_json(image, max_points=max_points),
            "q_map": _downsample_for_json(q_map, max_points=max_points),
        },
    }


def one_d_export_preview_payload(file_path: str | Path, max_points: int = 600) -> dict[str, Any]:
    """Load a CSV/DAT 1D export and return a compact curve preview."""

    path = Path(file_path)
    table = _load_1d_table(path)
    q = _column_by_alias(table, ("q", "x"))
    intensity = _column_by_alias(table, ("intensity", "i", "data"))
    std = _optional_column_by_alias(table, ("standard_deviation", "std", "sigma"))
    coverage = _optional_column_by_alias(table, ("coverage", "coverage_fraction"))
    finite = np.isfinite(q) & np.isfinite(intensity)
    if not np.any(finite):
        raise ValueError(f"1D curve has no finite q/intensity pairs: {path}")
    q_valid = q[finite]
    intensity_valid = intensity[finite]
    payload: dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "point_count": int(q.size),
        "finite_count": int(np.count_nonzero(finite)),
        "q": _array_stats(q_valid, np.isfinite(q_valid)),
        "intensity": _array_stats(intensity_valid, np.isfinite(intensity_valid)),
        "has_standard_deviation": std is not None,
        "has_coverage": coverage is not None,
        "preview": {
            "q": _downsample_1d_for_json(q_valid, max_points=max_points),
            "intensity": _downsample_1d_for_json(intensity_valid, max_points=max_points),
        },
    }
    if std is not None:
        std_valid = std[finite]
        payload["standard_deviation"] = _array_stats(std_valid, np.isfinite(std_valid))
    if coverage is not None:
        coverage_valid = coverage[finite]
        payload["coverage"] = _array_stats(coverage_valid, np.isfinite(coverage_valid))
        payload["coverage_diagnostics"] = _coverage_diagnostics(coverage_valid)
        payload["preview"]["coverage"] = _downsample_1d_for_json(coverage_valid, max_points=max_points)
    return payload


def preview_error_payload(file_path: str | Path, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "path": str(file_path),
        "message": message,
    }


def _array_stats(values: np.ndarray, finite: np.ndarray) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(finite))
    if count == 0:
        return {"finite_count": 0, "min": None, "max": None, "median": None}
    selected = values[finite]
    return {
        "finite_count": count,
        "min": float(np.nanmin(selected)),
        "max": float(np.nanmax(selected)),
        "median": float(np.nanmedian(selected)),
    }


def _finite_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(values)) / values.size)


def _downsample_for_json(values: np.ndarray, max_points: int = 64) -> list[list[float | None]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Preview array must be 2D.")
    step_y = max(1, int(np.ceil(array.shape[0] / max(1, int(max_points)))))
    step_x = max(1, int(np.ceil(array.shape[1] / max(1, int(max_points)))))
    sampled = array[::step_y, ::step_x]
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in sampled
    ]


def _downsample_1d_for_json(values: np.ndarray, max_points: int = 600) -> list[float | None]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    step = max(1, int(np.ceil(array.size / max(1, int(max_points)))))
    sampled = array[::step]
    return [None if not np.isfinite(value) else float(value) for value in sampled]


def _load_1d_table(path: Path) -> np.ndarray:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        table = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=np.float32, encoding="utf-8")
    except ValueError:
        table = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32, encoding="utf-8")
    if getattr(table, "dtype", None) is not None and table.dtype.names:
        names = {_normalize_column_name(name) for name in table.dtype.names}
        if not names.intersection({"q", "x"}) or not names.intersection({"intensity", "i", "data"}):
            return _load_1d_numeric_table(path, delimiter=delimiter)
        return np.atleast_1d(table)
    return _numeric_array_to_structured(table, path)


def _load_1d_numeric_table(path: Path, delimiter: str | None) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32, encoding="utf-8")
    return _numeric_array_to_structured(table, path)


def _numeric_array_to_structured(table: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(table, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] < 2:
        raise ValueError(f"1D curve file must contain at least q and intensity columns: {path}")
    structured = np.zeros(array.shape[0], dtype=[("q", "f4"), ("intensity", "f4")])
    structured["q"] = array[:, 0]
    structured["intensity"] = array[:, 1]
    return structured


def _column_by_alias(table: np.ndarray, aliases: tuple[str, ...]) -> np.ndarray:
    column = _optional_column_by_alias(table, aliases)
    if column is None:
        raise ValueError(f"1D curve is missing required column: {aliases[0]}")
    return column


def _optional_column_by_alias(table: np.ndarray, aliases: tuple[str, ...]) -> np.ndarray | None:
    names = table.dtype.names or ()
    lookup = {_normalize_column_name(name): name for name in names}
    for alias in aliases:
        name = lookup.get(_normalize_column_name(alias))
        if name is not None:
            return np.asarray(table[name], dtype=np.float32).reshape(-1)
    return None


def _normalize_column_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _coverage_diagnostics(coverage: np.ndarray) -> dict[str, Any]:
    summary = summarize_coverage(np.asarray(coverage, dtype=np.float32), low_threshold=0.15)
    low_bins = [int(value) for value in np.asarray(summary["coverage_low_bins"], dtype=np.int64).reshape(-1)]
    drop_bins = [int(value) for value in np.asarray(summary["coverage_drop_bins"], dtype=np.int64).reshape(-1)]
    warning_count = len(low_bins) + len(drop_bins)
    if warning_count:
        message = f"Coverage warning: {len(low_bins)} low bins, {len(drop_bins)} abrupt drops."
    else:
        message = "Coverage looks stable."
    return {
        "coverage_min": _json_float(summary["coverage_min"]),
        "coverage_median": _json_float(summary["coverage_median"]),
        "coverage_low_bins": low_bins,
        "coverage_drop_bins": drop_bins,
        "warning_count": int(warning_count),
        "message": message,
    }


def _json_float(value: object) -> float | None:
    number = float(value)
    return None if not np.isfinite(number) else number


def _array_to_list(values: np.ndarray | None) -> list[float | int] | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.size == 0:
        return []
    return [item.item() if hasattr(item, "item") else item for item in array.reshape(-1)]


__all__ = [
    "one_d_export_preview_payload",
    "preview_error_payload",
    "two_d_info_preview_payload",
]
