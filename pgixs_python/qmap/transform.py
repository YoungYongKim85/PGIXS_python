"""Pixel-space to reciprocal-space (Q) image regridding.

Ports MATLAB's ``ImageTransformQ_gi.m`` / ``ImageTransformQ_tr.m``: a bilinear
forward-splat of detector pixel intensities onto a uniform Q grid. Target
cells that receive too little contributing weight (sparse coverage near the
beam or at the detector edge) are left as NaN rather than producing a noisy
partial average -- this matches the MATLAB coverage-based normalization so
downstream 1D averaging automatically ignores those gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_COVERAGE_WEIGHT = 0.25


@dataclass
class QSpaceImage:
    horizontal_axis: np.ndarray
    vertical_axis: np.ndarray
    image: np.ndarray


def _forward_map_to_q_grid(
    data: np.ndarray,
    horizontal_q_map: np.ndarray,
    vertical_q_map: np.ndarray,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    wavelength_a: float,
    sdd_mm: float,
) -> QSpaceImage:
    rows, cols = data.shape
    d_q = np.float32((2.0 * np.pi * pixel_mm) / (wavelength_a * sdd_mm))

    horizontal_axis = d_q * (np.arange(1, cols + 1, dtype=np.float32) - beam_pos[0])
    vertical_axis = d_q * (beam_pos[1] - np.arange(1, rows + 1, dtype=np.float32))

    col_exact = horizontal_q_map / d_q + beam_pos[0]
    row_exact = beam_pos[1] - vertical_q_map / d_q

    eps = np.finfo(np.float32).eps
    valid = (
        (col_exact >= 1)
        & (col_exact <= cols - eps)
        & (row_exact >= 1)
        & (row_exact <= rows - eps)
        & np.isfinite(data)
    )

    if not np.any(valid):
        return QSpaceImage(horizontal_axis, vertical_axis, np.full((rows, cols), np.nan, dtype=np.float32))

    c = col_exact[valid].astype(np.float64)
    r = row_exact[valid].astype(np.float64)
    v = data[valid].astype(np.float64)

    c1 = np.floor(c).astype(np.int64)
    c2 = c1 + 1
    r1 = np.floor(r).astype(np.int64)
    r2 = r1 + 1

    wc1 = c2 - c
    wc2 = c - c1
    wr1 = r2 - r
    wr2 = r - r1

    c1 = np.clip(c1, 1, cols)
    c2 = np.clip(c2, 1, cols)
    r1 = np.clip(r1, 1, rows)
    r2 = np.clip(r2, 1, rows)

    isum = np.zeros((rows, cols), dtype=np.float64)
    wsum = np.zeros((rows, cols), dtype=np.float64)

    def _splat(rr: np.ndarray, cc: np.ndarray, weight: np.ndarray) -> None:
        np.add.at(isum, (rr - 1, cc - 1), v * weight)
        np.add.at(wsum, (rr - 1, cc - 1), weight)

    _splat(r1, c1, wr1 * wc1)
    _splat(r1, c2, wr1 * wc2)
    _splat(r2, c1, wr2 * wc1)
    _splat(r2, c2, wr2 * wc2)

    itrans = np.full((rows, cols), np.nan, dtype=np.float32)
    covered = wsum >= _MIN_COVERAGE_WEIGHT
    itrans[covered] = (isum[covered] / wsum[covered]).astype(np.float32)
    itrans[~np.isfinite(itrans)] = np.nan

    return QSpaceImage(horizontal_axis, vertical_axis, itrans)


def image_transform_q_gi(
    data: np.ndarray,
    horizontal_q_map: np.ndarray,
    qz_map: np.ndarray,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    wavelength_a: float,
    sdd_mm: float,
    mask_beamstop: bool = True,
) -> QSpaceImage:
    """Forward-map a GIXS detector image onto a uniform Q grid.

    ``horizontal_q_map`` is the per-pixel in-plane Q component to grid
    against: pass ``qr`` for the Qxy/Qz view or ``qy`` for the Qy/Qz view
    (matching ``GIXS_gui.m``'s two Map format options).
    """

    result = _forward_map_to_q_grid(data, horizontal_q_map, qz_map, beam_pos, pixel_mm, wavelength_a, sdd_mm)
    if mask_beamstop:
        cols = data.shape[1]
        beam_col = int(round(beam_pos[0]))
        if 1 <= beam_col <= cols:
            result.image[:, beam_col - 1] = np.nan
    return result


def image_transform_q_tr(
    data: np.ndarray,
    qy_map: np.ndarray,
    qz_map: np.ndarray,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    wavelength_a: float,
    sdd_mm: float,
) -> QSpaceImage:
    """Forward-map a TrXS detector image onto a uniform (qy, qz) grid."""

    return _forward_map_to_q_grid(data, qy_map, qz_map, beam_pos, pixel_mm, wavelength_a, sdd_mm)
