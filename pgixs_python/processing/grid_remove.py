"""Eiger4M grid-line correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridRemoveReport:
    applied: bool = False
    message: str = ""
    median_ratio: float = float("nan")
    correction_factor: float = float("nan")


def grid_remove(data: np.ndarray, detector_type: str = "Eiger4M") -> tuple[np.ndarray, GridRemoveReport]:
    """Correct Eiger grid-line intensity artifacts conservatively.

    Unsupported detector types, small images, and images without valid
    reference pixels are returned unchanged with a diagnostic report.
    """

    report = GridRemoveReport()
    out = np.asarray(data, dtype=np.float32).copy()

    if detector_type and str(detector_type) != "Eiger4M":
        report.message = "Grid remover is available only for Eiger4M detector data."
        return out, report

    if out.ndim != 2:
        report.message = "Grid remover expects a 2D numeric detector image."
        return out, report

    hordata = _idx([255, 260, 513, 518, 771, 776, 1295, 1300, 1553, 1558, 1811, 1816])
    horgap = _idx([256, 259, 514, 517, 772, 775, 1296, 1299, 1554, 1557, 1812, 1815])
    verdata = _idx([255, 260, 806, 811, 1357, 1362, 1908, 1913])
    vergap = _idx([256, 259, 807, 810, 1358, 1361, 1909, 1912])

    row_groups = [_range1(256, 259), _range1(807, 810), _range1(1358, 1361), _range1(1909, 1912)]
    col_groups = [_range1(256, 259), _range1(514, 517), _range1(772, 775), _range1(1296, 1299), _range1(1554, 1557), _range1(1812, 1815)]

    required_rows = max(max(verdata), max(vergap), *(int(group.max()) for group in row_groups)) + 1
    required_cols = max(max(hordata), max(horgap), *(int(group.max()) for group in col_groups)) + 1
    if out.shape[0] < required_rows or out.shape[1] < required_cols:
        report.message = f"Detector image size {out.shape[0]}x{out.shape[1]} is too small for Eiger grid removal."
        return out, report

    original = np.concatenate([out[:, hordata].ravel(), out[verdata, :].ravel()]).astype(np.float64)
    grid = np.concatenate([out[:, horgap].ravel(), out[vergap, :].ravel()]).astype(np.float64)

    valid = np.isfinite(original) & np.isfinite(grid) & (original != 0) & (grid != 0)
    if not np.any(valid):
        report.message = "No valid nonzero reference pixels for Eiger grid removal."
        return out, report

    ratio = grid[valid] / original[valid]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        report.message = "No valid positive grid/reference ratio for Eiger grid removal."
        return out, report

    median_ratio = float(np.nanmedian(ratio))
    if not np.isfinite(median_ratio) or median_ratio <= 0:
        report.message = "Invalid Eiger grid correction ratio."
        return out, report

    correction_factor = float(np.clip(1.0 / median_ratio, 0.25, 4.0))
    grid_mask = np.zeros(out.shape, dtype=bool)
    for rows in row_groups:
        grid_mask[rows, :] = True
    for cols in col_groups:
        grid_mask[:, cols] = True

    out[grid_mask] *= np.float32(correction_factor)
    report.applied = True
    report.median_ratio = median_ratio
    report.correction_factor = correction_factor
    report.message = f"Applied Eiger grid correction factor {correction_factor:.6g}."
    return out, report


def _idx(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64) - 1


def _range1(start: int, stop: int) -> np.ndarray:
    return np.arange(start - 1, stop, dtype=np.int64)
