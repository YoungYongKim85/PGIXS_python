"""Detector mask helpers for PGIXS processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DATA_DIR = Path(__file__).parent / "data"


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


def dilate_gap_mask(mask: np.ndarray, n_pad: int = 3) -> np.ndarray:
    """Grow a detector gap/dead-pixel mask by `n_pad` pixels.

    Port of MATLAB `dilatePGIXSGapMask.m`: 4-neighbour (row/column) dilation,
    matching the axis-aligned Eiger module gap geometry. Returns True wherever
    a pixel is itself a gap or sits within `n_pad` pixels of one.

    Real detector pixels bordering a gap can carry biased counts (partial
    charge collection / double-pixel summation), so the 1D reductions drop
    that shoulder. The 2D image itself is deliberately left alone -- MATLAB
    applies this only inside its averaging paths, never to the displayed
    frame."""

    grown = ~np.isfinite(np.asarray(mask, dtype=np.float64)) | (np.asarray(mask, dtype=np.float64) <= 0)
    for _ in range(max(0, int(n_pad))):
        padded = grown.copy()
        padded[1:, :] |= grown[:-1, :]
        padded[:-1, :] |= grown[1:, :]
        padded[:, 1:] |= grown[:, :-1]
        padded[:, :-1] |= grown[:, 1:]
        grown = padded
    return grown


def detector_mask_for_type(detector_type: str, mask_dir: str | Path | None = None) -> DetectorMask:
    """Return detector default mask and pixel size.

    MAT mask files are optional in the Python port. If a legacy mask file cannot
    be read, the helper returns an all-one fallback with the MATLAB detector
    dimensions so workflows can still run and report the fallback explicitly.
    """

    detector = str(detector_type)
    if detector == "Eiger4M":
        # (2070, 2167): rows x cols as returned by `load_h5_frame`, which
        # transposes the raw H5 dataset (matching MATLAB `loadPGIXSH5Frame`).
        # A real Eiger4M frame confirmed this empirically; the other
        # detectors below are square so this row/col mixup never surfaced.
        return _mask_from_file_or_default("EigerMask.mat", (2070, 2167), 0.075, mask_dir)
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
    # Default to the copies bundled with the package. Before these were
    # shipped, `mask_dir=None` meant a bare relative path resolved against the
    # process CWD, which never matched -- so every detector silently ran on
    # the all-ones fallback and the real bad-pixel mask (6.2% of an Eiger4M,
    # 21.6% of an SX165) was never applied at all, unlike MATLAB.
    path = Path(mask_dir) / file_name if mask_dir is not None else _DATA_DIR / file_name

    try:
        mask = _orient_mask_to(_read_mat_mask(path), fallback_size)
    except Exception as exc:  # noqa: BLE001 - return MATLAB-compatible fallback plus diagnostic.
        mask = np.ones(fallback_size, dtype=np.float32)
        report.fallback_used = True
        report.message = str(exc)

    return DetectorMask(mask_zero_to_nan(mask), pixel_size_mm, report)


def _orient_mask_to(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    """Match the stored mask to the orientation the loaders produce.

    The mask files are saved in MATLAB's own orientation, which for the H5
    detectors is the transpose of what `load_h5_frame` returns (it transposes
    the raw dataset). Non-H5 loaders don't transpose, and those masks are
    square, so this only ever flips the Eiger4M one."""

    array = np.asarray(mask)
    if array.shape == tuple(expected_shape):
        return array
    if array.T.shape == tuple(expected_shape):
        return np.ascontiguousarray(array.T)
    raise ValueError(f"Mask shape {array.shape} matches neither {tuple(expected_shape)} nor its transpose.")


def _read_mat_mask(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to read legacy MATLAB mask files.") from exc

    data: dict[str, Any] = loadmat(path)
    if "Mask" not in data:
        raise KeyError(f"MAT mask file does not contain 'Mask': {path}")
    return np.asarray(data["Mask"], dtype=np.float32)
