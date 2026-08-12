"""Angular/sector averages with detector-gap coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgixs_python.averaging.qbin import qbin_average
from pgixs_python.averaging.radial import RadiusAverageResult, build_radius_bin_map, radius_average


@dataclass
class AngularProfileResult:
    radius_range: tuple[float, float]
    center: tuple[float, float]
    angle_axis: np.ndarray
    intensity: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    geom_counts: np.ndarray
    coverage: np.ndarray


def angular_profile(
    data: np.ndarray,
    center: tuple[float, float],
    radius_range: tuple[float, float],
    n_angle_bins: int = 180,
    use_median: bool = False,
    calc_std: bool = False,
) -> AngularProfileResult:
    """Ring intensity profile versus angle at a fixed pixel-radius range.

    Mirrors the MATLAB `DrawPlot_Callback` polar sampling: for each angle
    bin, average the pixels whose radius from `center` falls inside
    `radius_range`. Angle convention matches `make_sector_mask`.
    """

    image = np.asarray(data, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("data must be a 2D array.")

    n_rows, n_cols = image.shape
    cols, rows = np.meshgrid(
        np.arange(1, n_cols + 1, dtype=np.float64),
        np.arange(1, n_rows + 1, dtype=np.float64),
    )
    angle = np.degrees(np.arctan2(-(rows - float(center[1])), -(cols - float(center[0]))))
    angle = np.mod(angle, 360.0)
    radius = np.hypot(cols - float(center[0]), rows - float(center[1]))

    r_min, r_max = sorted((float(radius_range[0]), float(radius_range[1])))
    ring_mask = (radius >= r_min) & (radius <= r_max)

    n_bins = max(1, int(n_angle_bins))
    bin_width = 360.0 / n_bins
    bin_index = np.minimum(np.floor(angle / bin_width).astype(np.int64), n_bins - 1)

    binned = qbin_average(image, bin_index, n_bins, use_median=use_median, calc_std=calc_std, geom_mask=ring_mask)
    angle_axis = ((np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width).astype(np.float32)

    return AngularProfileResult(
        radius_range=(r_min, r_max),
        center=(float(center[0]), float(center[1])),
        angle_axis=angle_axis,
        intensity=binned.intensity,
        std=binned.std,
        counts=binned.counts,
        geom_counts=binned.geom_counts,
        coverage=binned.coverage,
    )


@dataclass
class SectorAverageResult:
    angle_range: tuple[float, float]
    center: tuple[float, float]
    q_axis: np.ndarray
    r_axis: np.ndarray
    intensity: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    sector_geom_counts: np.ndarray
    full_ring_counts: np.ndarray
    coverage: np.ndarray
    low_coverage_bins: np.ndarray
    coverage_drop_bins: np.ndarray
    sector_pixel_count: int
    full_pixel_count: int
    is_valid: bool


def make_sector_mask(
    image_shape: tuple[int, int],
    center: tuple[float, float],
    angle_range: tuple[float, float],
) -> np.ndarray:
    """Build a MATLAB-compatible angular sector mask.

    Angles follow `diagnosePGIXSSectorAverage.m`:
    `atan2d(-(row-center_z), -(col-center_y))`, wrapped to 0..360.
    Equal start/end means full circle.
    """

    n_rows, n_cols = (int(image_shape[0]), int(image_shape[1]))
    cols, rows = np.meshgrid(
        np.arange(1, n_cols + 1, dtype=np.float64),
        np.arange(1, n_rows + 1, dtype=np.float64),
    )
    angle = np.degrees(np.arctan2(-(rows - float(center[1])), -(cols - float(center[0]))))
    angle = np.mod(angle, 360.0)
    start_raw, end_raw = float(angle_range[0]), float(angle_range[1])
    start = start_raw % 360.0
    end = end_raw % 360.0
    if start_raw == end_raw:
        return np.ones((n_rows, n_cols), dtype=bool)
    if start < end:
        return (angle >= start) & (angle <= end)
    return (angle >= start) | (angle <= end)


def sector_average(
    data: np.ndarray,
    q_map: np.ndarray,
    center: tuple[float, float],
    angle_range: tuple[float, float],
    use_median: bool = False,
    calc_std: bool = False,
) -> SectorAverageResult:
    """Radius-bin average inside an angular sector with coverage sidecars."""

    image = np.asarray(data, dtype=np.float32)
    q = np.asarray(q_map, dtype=np.float32)
    if image.ndim != 2 or q.shape != image.shape:
        raise ValueError("data and q_map must be 2D arrays with matching shape.")

    finite_q = np.isfinite(q) & (q > 0)
    sector_mask = make_sector_mask(image.shape, center, angle_range)
    sector_geom_mask = finite_q & sector_mask

    sector_bins = build_radius_bin_map(q, center, sector_geom_mask)
    full_bins = build_radius_bin_map(q, center, finite_q)
    if sector_bins.is_valid:
        radial: RadiusAverageResult = radius_average(image, sector_bins, use_median=use_median, calc_std=calc_std)
        full_ring_counts = _estimate_full_counts_for_sector_bins(center, finite_q, radial.r_axis)
        low_coverage = np.flatnonzero(radial.coverage < 0.15)
        drop_bins = find_coverage_drop_bins(radial.coverage)
        return SectorAverageResult(
            angle_range=(float(angle_range[0]), float(angle_range[1])),
            center=(float(center[0]), float(center[1])),
            q_axis=radial.q_axis,
            r_axis=radial.r_axis,
            intensity=radial.intensity,
            std=radial.std,
            counts=radial.counts,
            sector_geom_counts=radial.geom_counts,
            full_ring_counts=full_ring_counts,
            coverage=radial.coverage,
            low_coverage_bins=low_coverage.astype(np.int64),
            coverage_drop_bins=drop_bins,
            sector_pixel_count=int(np.count_nonzero(sector_geom_mask)),
            full_pixel_count=int(np.count_nonzero(finite_q)),
            is_valid=True,
        )

    empty_f = np.array([], dtype=np.float32)
    empty_i = np.array([], dtype=np.int64)
    return SectorAverageResult(
        angle_range=(float(angle_range[0]), float(angle_range[1])),
        center=(float(center[0]), float(center[1])),
        q_axis=empty_f,
        r_axis=empty_f,
        intensity=empty_f,
        std=empty_f,
        counts=empty_i,
        sector_geom_counts=empty_i,
        full_ring_counts=empty_i,
        coverage=empty_f,
        low_coverage_bins=empty_i,
        coverage_drop_bins=empty_i,
        sector_pixel_count=int(np.count_nonzero(sector_geom_mask)),
        full_pixel_count=int(np.count_nonzero(finite_q)),
        is_valid=False,
    )


def find_coverage_drop_bins(coverage: np.ndarray) -> np.ndarray:
    """Find abrupt coverage changes using the MATLAB diagnostic heuristic."""

    cov = np.asarray(coverage, dtype=np.float64).reshape(-1)
    finite = np.isfinite(cov)
    if int(np.count_nonzero(finite)) < 5:
        return np.array([], dtype=np.int64)
    delta = np.r_[np.nan, np.abs(np.diff(cov))]
    finite_delta = delta[np.isfinite(delta)]
    med_drop = float(np.median(finite_delta)) if finite_delta.size else 0.05
    if not np.isfinite(med_drop) or med_drop <= 0:
        med_drop = 0.05
    return np.flatnonzero(delta > max(0.10, 5.0 * med_drop)).astype(np.int64)


def _estimate_full_counts_for_sector_bins(
    center: tuple[float, float],
    finite_q: np.ndarray,
    sector_radius_axis: np.ndarray,
) -> np.ndarray:
    rows, cols = np.nonzero(finite_q)
    if rows.size == 0 or sector_radius_axis.size == 0:
        return np.zeros(sector_radius_axis.size, dtype=np.int64)
    radius = np.hypot(cols.astype(np.float64) + 1.0 - float(center[0]), rows.astype(np.float64) + 1.0 - float(center[1]))
    bin_idx = np.floor(radius).astype(np.int64)
    full_counts = np.bincount(bin_idx, minlength=int(bin_idx.max()) + 1).astype(np.int64)
    sector_idx = np.floor(np.asarray(sector_radius_axis, dtype=np.float64)).astype(np.int64)
    counts = np.zeros(sector_idx.size, dtype=np.int64)
    valid = (sector_idx >= 0) & (sector_idx < full_counts.size)
    counts[valid] = full_counts[sector_idx[valid]]
    return counts
