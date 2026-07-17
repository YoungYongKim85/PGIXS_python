"""Preview helpers for local web UI result inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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


def _array_to_list(values: np.ndarray | None) -> list[float | int] | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.size == 0:
        return []
    return [item.item() if hasattr(item, "item") else item for item in array.reshape(-1)]


__all__ = [
    "preview_error_payload",
    "two_d_info_preview_payload",
]
