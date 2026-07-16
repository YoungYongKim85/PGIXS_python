"""GIXS-style cake cuts with relative-angle masking and coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgixs_python.averaging.angular import find_coverage_drop_bins
from pgixs_python.averaging.radial import RadiusAverageResult, build_radius_bin_map, radius_average


@dataclass
class CakeAverageResult:
    reference_angle: float
    angle_range: tuple[float, float]
    q_axis: np.ndarray
    r_axis: np.ndarray
    intensity: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    geom_counts: np.ndarray
    coverage: np.ndarray
    low_coverage_bins: np.ndarray
    coverage_drop_bins: np.ndarray
    selected_pixel_count: int
    is_valid: bool


def normalize_angle180(angle: np.ndarray | float) -> np.ndarray:
    """Normalize angles to [-180, 180), matching MATLAB helper behavior."""

    return np.mod(np.asarray(angle, dtype=np.float64) + 180.0, 360.0) - 180.0


def angle_in_relative_sector(
    angle_diff: np.ndarray,
    start_angle: float,
    finish_angle: float,
) -> np.ndarray:
    """Select relative angles, including +/-180 wrap."""

    diff = normalize_angle180(angle_diff)
    if not np.isfinite(start_angle) or not np.isfinite(finish_angle):
        return np.zeros(diff.shape, dtype=bool)
    if abs(float(finish_angle) - float(start_angle)) >= 360:
        return np.isfinite(diff)

    start = float(normalize_angle180(start_angle))
    finish = float(normalize_angle180(finish_angle))
    finite = np.isfinite(diff)
    if start <= finish:
        return finite & (diff >= start) & (diff <= finish)
    return finite & ((diff >= start) | (diff <= finish))


def cake_average(
    data: np.ndarray,
    q_map: np.ndarray,
    angle_map: np.ndarray,
    center: tuple[float, float],
    reference_angle: float,
    angle_range: tuple[float, float],
    use_median: bool = False,
    calc_std: bool = False,
) -> CakeAverageResult:
    """Run a cake cut using precomputed Q and angle maps.

    The map convention mirrors MATLAB `CakeCut_gi_prep`: select pixels by
    relative angle around `reference_angle`, then radius-bin by detector pixel
    distance and attach the mean Q per radius bin.
    """

    image = np.asarray(data, dtype=np.float32)
    q = np.asarray(q_map, dtype=np.float32)
    angles = np.asarray(angle_map, dtype=np.float32)
    if image.ndim != 2 or q.shape != image.shape or angles.shape != image.shape:
        raise ValueError("data, q_map, and angle_map must be 2D arrays with matching shape.")

    angle_diff = normalize_angle180(angles - float(reference_angle))
    selected = (
        np.isfinite(q)
        & (q > 0)
        & np.isfinite(angles)
        & angle_in_relative_sector(angle_diff, float(angle_range[0]), float(angle_range[1]))
    )
    radius_map = build_radius_bin_map(q, center, selected)
    if not radius_map.is_valid:
        empty_f = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int64)
        return CakeAverageResult(
            reference_angle=float(reference_angle),
            angle_range=(float(angle_range[0]), float(angle_range[1])),
            q_axis=empty_f,
            r_axis=empty_f,
            intensity=empty_f,
            std=empty_f,
            counts=empty_i,
            geom_counts=empty_i,
            coverage=empty_f,
            low_coverage_bins=empty_i,
            coverage_drop_bins=empty_i,
            selected_pixel_count=int(np.count_nonzero(selected)),
            is_valid=False,
        )

    radial: RadiusAverageResult = radius_average(image, radius_map, use_median=use_median, calc_std=calc_std)
    return CakeAverageResult(
        reference_angle=float(reference_angle),
        angle_range=(float(angle_range[0]), float(angle_range[1])),
        q_axis=radial.q_axis,
        r_axis=radial.r_axis,
        intensity=radial.intensity,
        std=radial.std,
        counts=radial.counts,
        geom_counts=radial.geom_counts,
        coverage=radial.coverage,
        low_coverage_bins=np.flatnonzero(radial.coverage < 0.15).astype(np.int64),
        coverage_drop_bins=find_coverage_drop_bins(radial.coverage),
        selected_pixel_count=int(np.count_nonzero(selected)),
        is_valid=True,
    )


def pixel_angle_map(image_shape: tuple[int, int], center: tuple[float, float]) -> np.ndarray:
    """Build MATLAB Pixel-mode cake angle map."""

    n_rows, n_cols = (int(image_shape[0]), int(image_shape[1]))
    cols, rows = np.meshgrid(
        np.arange(1, n_cols + 1, dtype=np.float64),
        np.arange(1, n_rows + 1, dtype=np.float64),
    )
    return np.degrees(np.arctan2(-(rows - float(center[1])), -(cols - float(center[0])))).astype(np.float32)
