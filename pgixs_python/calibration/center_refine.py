"""Beam-center refinement for SDD calibration's "Smart Auto Run".

Ports MATLAB `SDDcal_gui.m`'s two local nested functions `refineCenter`
and `fineTuneCenter` -- they only ever existed inside `SDDcal_gui.m`'s
`AutoRun_Callback`, not as separate files.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pgixs_python.averaging.pixel_radial import compute_radial_profile
from pgixs_python.calibration.peak_fit import fit_gaussian_peak


def refine_center_from_ring(data: np.ndarray, init_center: tuple[float, float]) -> tuple[float, float]:
    """A first, coarse beam-center refinement: find the strongest ring in
    a radial profile around `init_center`, then minimize the variance of
    radii of that ring's brightest pixels (a circle fit) to find a better
    center. Falls back to `init_center` unchanged whenever there isn't
    enough signal to refine confidently -- matches `refineCenter`'s own
    early-return guards exactly."""

    init_y, init_z = float(init_center[0]), float(init_center[1])
    profile = compute_radial_profile(data, center=(init_y, init_z))
    if profile.r_axis.size == 0:
        return init_y, init_z

    valid = profile.r_axis > 80.0
    if int(np.count_nonzero(valid)) < 10:
        valid = profile.r_axis > 30.0
    if not np.any(valid):
        return init_y, init_z

    candidate_idx = np.flatnonzero(valid)
    local_intensity = profile.intensity[valid]
    if not np.any(np.isfinite(local_intensity)):
        return init_y, init_z
    local_max = int(np.nanargmax(local_intensity))
    approx_radius = float(profile.r_axis[candidate_idx[local_max]])

    n_rows, n_cols = np.asarray(data).shape
    cols_grid, rows_grid = np.meshgrid(
        np.arange(1, n_cols + 1, dtype=np.float64), np.arange(1, n_rows + 1, dtype=np.float64)
    )
    r_map = np.hypot(cols_grid - init_y, rows_grid - init_z)
    ring_mask = (r_map > approx_radius - 10.0) & (r_map < approx_radius + 10.0)

    values = np.asarray(data, dtype=np.float64)[ring_mask]
    xs = cols_grid[ring_mask]
    ys = rows_grid[ring_mask]
    finite = np.isfinite(values)
    values, xs, ys = values[finite], xs[finite], ys[finite]
    if values.size < 20:
        return init_y, init_z

    threshold = float(np.quantile(values, 0.80))
    if not np.isfinite(threshold):
        return init_y, init_z
    bright = values > threshold
    xs, ys = xs[bright], ys[bright]
    if xs.size < 10:
        return init_y, init_z

    def _cost(center: np.ndarray) -> float:
        return float(np.std(np.hypot(xs - center[0], ys - center[1])))

    result = minimize(_cost, x0=np.array([init_y, init_z]), method="Nelder-Mead", options={"xatol": 0.1})
    return float(result.x[0]), float(result.x[1])


def fine_tune_center(
    data: np.ndarray,
    init_center: tuple[float, float],
    target_radius: float,
    max_iterations: int = 10,
) -> tuple[float, float]:
    """Iteratively nudge the beam center so the ring at `target_radius`
    lands at the same pixel offset on all four sides (left/right/up/down)
    -- a 1D Gaussian fit per side each iteration, damped by 0.8. Matches
    `fineTuneCenter` exactly, including its per-side try/fail-to-NaN
    handling (a side simply doesn't contribute a shift that iteration if
    its fit fails)."""

    array = np.asarray(data, dtype=np.float64)
    n_rows, n_cols = array.shape
    cur_y, cur_z = float(init_center[0]), float(init_center[1])

    search_range = np.round(float(target_radius)) + np.arange(-20, 21, dtype=np.float64)
    search_range = search_range[search_range > 0]
    if search_range.size == 0:
        return cur_y, cur_z

    def _side_fit(offsets: np.ndarray, values: np.ndarray) -> float:
        if offsets.size == 0:
            return float("nan")
        fit = fit_gaussian_peak(offsets, values)
        return fit.center if fit.success else float("nan")

    for _ in range(max_iterations):
        i_col = int(round(cur_y))
        i_row = int(round(cur_z))
        if i_col < 1 or i_col > n_cols or i_row < 1 or i_row > n_rows:
            break

        idx_r = i_col + search_range
        keep_r = idx_r <= n_cols
        p_right = _side_fit(search_range[keep_r], array[i_row - 1, (idx_r[keep_r] - 1).astype(np.int64)])

        idx_l = i_col - search_range
        keep_l = idx_l > 0
        p_left = _side_fit(search_range[keep_l], array[i_row - 1, (idx_l[keep_l] - 1).astype(np.int64)])

        idx_d = i_row + search_range
        keep_d = idx_d <= n_rows
        p_down = _side_fit(search_range[keep_d], array[(idx_d[keep_d] - 1).astype(np.int64), i_col - 1])

        idx_u = i_row - search_range
        keep_u = idx_u > 0
        p_up = _side_fit(search_range[keep_u], array[(idx_u[keep_u] - 1).astype(np.int64), i_col - 1])

        shift_y = 0.0
        shift_z = 0.0
        moved = False
        if np.isfinite(p_right) and np.isfinite(p_left):
            shift_y = (p_right - p_left) / 2.0
            moved = True
        if np.isfinite(p_down) and np.isfinite(p_up):
            shift_z = (p_down - p_up) / 2.0
            moved = True
        if not moved:
            break

        cur_y += shift_y * 0.8
        cur_z += shift_z * 0.8
        if abs(shift_y) < 0.05 and abs(shift_z) < 0.05:
            break

    return cur_y, cur_z
