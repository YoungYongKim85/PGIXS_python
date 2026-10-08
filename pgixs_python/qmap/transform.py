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


def q_space_point(
    col: float,
    row: float,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    wavelength_a: float,
    sdd_mm: float,
) -> tuple[float, float]:
    """Convert one (col, row) pixel-space point into the same uniform
    Q-space (qx, qy) coordinate `image_transform_q_gi`/`image_transform_q_tr`'s
    own output grid axes use -- `d_q * (index - beam_pos)`, a single
    scale factor applied identically to both axes (see
    `_forward_map_to_q_grid`'s own `horizontal_axis`/`vertical_axis`).

    Used to draw a line/accumulation/cake cut's overlay directly as a
    straight line in Q-space, instead of sampling the true (nonlinear)
    per-pixel Q maps point by point -- that curves what is, in this
    uniform grid, actually a straight cut (the console's own Qxy/Qz
    line/cake cut now runs against this exact grid, not the true Q maps;
    see console_session.py's `_q_space_cut_inputs`). Reported: "Qxy/Qz
    맵에서 90도 line cut을... 해보면 일직선 라인이 아니라 휘어져 나와...
    그냥 일직선 라인이 되면 좋을 것 같아.\""""

    d_q = (2.0 * np.pi * pixel_mm) / (wavelength_a * sdd_mm)
    return d_q * (col - beam_pos[0]), d_q * (beam_pos[1] - row)


def pixel_point_from_q_space(
    qx: float,
    qy: float,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    wavelength_a: float,
    sdd_mm: float,
) -> tuple[float, float]:
    """Inverse of `q_space_point` -- a Q-space (qx, qy) coordinate (e.g. a
    click on the Qxy/Qz or Qy/Qz display) back to the (col, row) pixel
    point that grid cell was regridded from. Exact (this uniform grid's
    two axes are each linear in pixel position), unlike inverting the
    true per-pixel Q maps, which needs a nearest-neighbor search since
    they aren't invertible in closed form."""

    d_q = (2.0 * np.pi * pixel_mm) / (wavelength_a * sdd_mm)
    return qx / d_q + beam_pos[0], beam_pos[1] - qy / d_q


def polar_transform_q_grid(
    data: np.ndarray,
    horizontal_q_map: np.ndarray,
    vertical_q_map: np.ndarray,
    n_angle_bins: int = 360,
    n_r_bins: int = 300,
    angle_range: tuple[float, float] = (0.0, 360.0),
) -> QSpaceImage:
    """Remap a Cartesian (horizontal, vertical) Q image onto a polar grid
    centered at the origin (q=0, the beam position in Q-space) -- lets the
    Qxy/Qz or Qy/Qz Cartesian plot be viewed as a polar (angle vs |Q|) plot.

    Unlike `_forward_map_to_q_grid`'s bilinear splat (which only works
    because its target axes are *linear* functions of pixel position),
    polar (r, theta) are nonlinear functions of (horizontal, vertical), so
    this bins by nearest-cell mean instead: r = hypot(h, v), theta =
    atan2(v, -h) in [0, 360). This matches this codebase's one and only
    angle convention everywhere else (`pixel_angle_map`, the click-to-angle
    formula, Cakecut, the Azimuthal Process): 0deg at 9 o'clock (-horizontal,
    left), 90deg at 12 o'clock (+vertical, up), 180deg at 3 o'clock
    (+horizontal, right), increasing clockwise -- not the plain-math
    atan2(v, h) convention (0deg at 3 o'clock, counterclockwise).

    `angle_range` restricts the swept angles (and bins only that span, not
    the full circle) -- GIXS geometry only ever has meaningful scattering
    above the horizon, matching `angular_profile`'s same (0, 180) default
    for the Azimuthal Process; TrXS uses the full ring, (0, 360).

    The mapping is switched back to Cartesian simply by re-requesting
    `image_transform_q_gi`/`image_transform_q_tr`'s ordinary grid -- this
    function does not mutate or replace anything, so "recovering" the
    original view is just re-rendering it.
    """

    if not float(angle_range[0]) < float(angle_range[1]):
        # Sorting a wrap-around range like (350, 10) swept the other 340 deg
        # instead, and equal ends divided by a zero bin width.
        raise ValueError(f"angle_range must be increasing and non-empty, got {tuple(angle_range)}.")
    angle_min, angle_max = float(angle_range[0]), float(angle_range[1])
    angle_bin_width = (angle_max - angle_min) / n_angle_bins
    angle_axis = (angle_min + (np.arange(n_angle_bins, dtype=np.float64) + 0.5) * angle_bin_width).astype(np.float32)

    horizontal = np.asarray(horizontal_q_map, dtype=np.float64)
    vertical = np.asarray(vertical_q_map, dtype=np.float64)
    values = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(horizontal) & np.isfinite(vertical)

    if np.any(finite):
        theta_all = np.degrees(np.arctan2(vertical[finite], -horizontal[finite])) % 360.0
        finite_idx = np.flatnonzero(finite)
        in_range = finite_idx[(theta_all >= angle_min) & (theta_all <= angle_max)]
    else:
        in_range = np.array([], dtype=np.int64)

    if in_range.size == 0:
        r_axis = np.arange(n_r_bins, dtype=np.float32)
        return QSpaceImage(angle_axis, r_axis, np.full((n_r_bins, n_angle_bins), np.nan, dtype=np.float32))

    h = horizontal.ravel()[in_range]
    v = vertical.ravel()[in_range]
    val = values.ravel()[in_range]

    r = np.hypot(h, v)
    theta = np.degrees(np.arctan2(v, -h)) % 360.0

    r_max = float(r.max())
    r_bin_width = r_max / n_r_bins if r_max > 0 else 1.0

    r_idx = np.clip(np.floor(r / r_bin_width).astype(np.int64), 0, n_r_bins - 1)
    angle_idx = np.clip(np.floor((theta - angle_min) / angle_bin_width).astype(np.int64), 0, n_angle_bins - 1)
    flat_idx = r_idx * n_angle_bins + angle_idx
    n_total = n_r_bins * n_angle_bins

    sums = np.bincount(flat_idx, weights=val, minlength=n_total)
    counts = np.bincount(flat_idx, minlength=n_total)

    image = np.full(n_total, np.nan, dtype=np.float64)
    covered = counts > 0
    image[covered] = sums[covered] / counts[covered]
    image = image.reshape(n_r_bins, n_angle_bins).astype(np.float32)

    r_axis = ((np.arange(n_r_bins, dtype=np.float64) + 0.5) * r_bin_width).astype(np.float32)
    return QSpaceImage(angle_axis, r_axis, image)
