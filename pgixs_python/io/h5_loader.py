"""H5 detector frame loading for PGIXS Tab1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DETECTOR_DATASET = "/entry/data/data"


@dataclass
class H5RetryReport:
    ok: bool
    attempts: int
    first_message: str = ""
    message: str = ""


def load_h5_frame(
    file_path: str | Path,
    dataset_path: str = DEFAULT_DETECTOR_DATASET,
    mask: np.ndarray | None = None,
    saturated_value: float = np.nan,
) -> np.ndarray:
    """Load one detector frame from a PGIXS H5 file.

    The returned array follows MATLAB `loadPGIXSH5Frame`: read the first
    detector frame, transpose it, convert to `float32`, optionally apply a mask,
    and replace saturated values above `1e9`.
    """

    h5py = _import_h5py()
    path = Path(file_path)
    with h5py.File(path, "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"H5 detector dataset not found: {dataset_path}")
        dataset = h5[dataset_path]
        raw = _read_first_frame(dataset)

    if raw.ndim != 2:
        raise ValueError(f"Could not reduce H5 detector dataset to one 2D frame: {path}")

    data = np.asarray(raw.T, dtype=np.float32)
    if mask is not None:
        data = data * np.asarray(mask, dtype=np.float32)
    data[data > 1e9] = np.float32(saturated_value)
    return data


def load_h5_frame_with_retry(
    file_path: str | Path,
    dataset_path: str = DEFAULT_DETECTOR_DATASET,
    mask: np.ndarray | None = None,
    saturated_value: float = np.nan,
    fallback_size: tuple[int, int] | None = None,
    retry_count: int = 1,
) -> tuple[np.ndarray, H5RetryReport]:
    """Load one H5 frame, retrying before returning a NaN fallback image."""

    if fallback_size is None and mask is not None:
        fallback_size = tuple(np.asarray(mask).shape)  # type: ignore[assignment]

    first_message = ""
    message = ""
    for attempt in range(1, int(retry_count) + 2):
        try:
            data = load_h5_frame(file_path, dataset_path, mask, saturated_value)
            return data, H5RetryReport(ok=True, attempts=attempt, first_message=first_message, message="")
        except Exception as exc:  # noqa: BLE001 - preserve failure message for diagnostics.
            message = str(exc)
            if not first_message:
                first_message = message

    if fallback_size is None:
        raise RuntimeError(f"H5 frame load failed after {retry_count + 1} attempts: {message}")

    fallback = np.full(fallback_size, np.nan, dtype=np.float32)
    return fallback, H5RetryReport(ok=False, attempts=retry_count + 1, first_message=first_message, message=message)


def _read_first_frame(dataset: Any) -> np.ndarray:
    shape = tuple(int(v) for v in dataset.shape)
    if len(shape) < 2:
        raise ValueError("H5 detector dataset must have at least two dimensions.")
    if len(shape) == 2:
        return np.asarray(dataset[()])

    # EIGER stacks commonly appear as (frame, z, y) in h5py, while some files
    # are stored as (z, y, frame). Choose the smallest edge as the frame axis
    # when it is clearly not an image dimension; otherwise match MATLAB's
    # first-frame behavior on the last extra dimension.
    if len(shape) == 3 and shape[0] < shape[1] and shape[0] < shape[2]:
        return np.asarray(dataset[0, :, :])

    index = [slice(None), slice(None)] + [0] * (len(shape) - 2)
    return np.asarray(dataset[tuple(index)])


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for PGIXS H5 detector frame loading.") from exc
    return h5py

