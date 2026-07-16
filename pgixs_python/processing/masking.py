"""Detector mask helpers for PGIXS processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetectorMaskReport:
    fallback_used: bool = False
    message: str = ""
    mask_file: str = ""


@dataclass
class DetectorMask:
    mask: np.ndarray | None
    pixel_size_mm: float
    report: DetectorMaskReport


def apply_detector_mask(data: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Apply a detector mask while preserving masked pixels as NaN."""

    out = np.asarray(data, dtype=np.float32).copy()
    if mask is None:
        return out

    mask_array = np.asarray(mask, dtype=np.float32)
    if out.shape != mask_array.shape:
        raise ValueError(f"Data shape {out.shape} does not match mask shape {mask_array.shape}.")

    out *= mask_array
    out[mask_array <= 0] = np.nan
    return out


def mask_zero_to_nan(mask: np.ndarray) -> np.ndarray:
    """Return a float32 mask where non-positive pixels become NaN."""

    out = np.asarray(mask, dtype=np.float32).copy()
    out[out <= 0] = np.nan
    return out


def detector_mask_for_type(detector_type: str, mask_dir: str | Path | None = None) -> DetectorMask:
    """Return detector default mask and pixel size.

    MAT mask files are optional in the Python port. If a legacy mask file cannot
    be read, the helper returns an all-one fallback with the MATLAB detector
    dimensions so workflows can still run and report the fallback explicitly.
    """

    detector = str(detector_type)
    if detector == "Eiger4M":
        return _mask_from_file_or_default("EigerMask.mat", (2167, 2070), 0.075, mask_dir)
    if detector in {"MarCCD", "SX165"}:
        return _mask_from_file_or_default("SX165mask.mat", (2048, 2048), 0.079, mask_dir)
    if detector == "MX170HS":
        return DetectorMask(np.ones((1920, 1920), dtype=np.float32), 0.0886, DetectorMaskReport())
    if detector in {"PI-CCD", "PI_CCD"}:
        return DetectorMask(np.ones((1042, 1042), dtype=np.float32), 0.1152, DetectorMaskReport())
    if detector == "MAR345":
        return DetectorMask(np.ones((3450, 3450), dtype=np.float32), 0.1, DetectorMaskReport())

    return DetectorMask(
        mask=None,
        pixel_size_mm=0.075,
        report=DetectorMaskReport(fallback_used=True, message=f"Unknown detector type: {detector}"),
    )


def _mask_from_file_or_default(
    file_name: str,
    fallback_size: tuple[int, int],
    pixel_size_mm: float,
    mask_dir: str | Path | None,
) -> DetectorMask:
    report = DetectorMaskReport(fallback_used=False, message="", mask_file=file_name)
    path = Path(mask_dir) / file_name if mask_dir is not None else Path(file_name)

    try:
        mask = _read_mat_mask(path)
    except Exception as exc:  # noqa: BLE001 - return MATLAB-compatible fallback plus diagnostic.
        mask = np.ones(fallback_size, dtype=np.float32)
        report.fallback_used = True
        report.message = str(exc)

    return DetectorMask(mask_zero_to_nan(mask), pixel_size_mm, report)


def _read_mat_mask(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to read legacy MATLAB mask files.") from exc

    data: dict[str, Any] = loadmat(path)
    if "Mask" not in data:
        raise KeyError(f"MAT mask file does not contain 'Mask': {path}")
    return np.asarray(data["Mask"], dtype=np.float32)
