"""GIXS-style cake cuts with relative-angle masking and coverage diagnostics."""

from __future__ import annotations

import warnings
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


def accumulation_average(
    data: np.ndarray,
    q_map: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
    perp_offset: float,
    band_minus: float,
    band_plus: float,
    use_median: bool = False,
) -> CakeAverageResult:
    """Perpendicular pixel-band mean along an axis-aligned cut direction.

    Mirrors MATLAB `Accumpix_gi`: an axis-aligned band (angle 0/90/180) is
    already perfectly aligned with the pixel grid, so this is a direct
    row-slice (0/180) or column-slice (90) mean rather than `cake_average`'s
    fuzzy floor-radius binning -- no interpolation needed. `perp_offset`
    shifts the band's fixed coordinate using this codebase's HD/VD
    convention (VD = beam_z - row, HD = col - beam_y), matching how
    Y-Range/Z-Range and the Center fields already treat pixel offsets
    elsewhere, rather than replicating MATLAB's internal row/col variable
    naming (which differs by sign for asymmetric band_minus/band_plus but
    is identical for MATLAB's own symmetric -1/+1 defaults).
    """

    angle = float(angle_deg)
    if angle not in (0.0, 90.0, 180.0):
        raise ValueError("Accumulation cut only supports angle 0, 90, or 180 degrees.")

    image = np.asarray(data, dtype=np.float32)
    q = np.asarray(q_map, dtype=np.float32)
    if image.ndim != 2 or q.shape != image.shape:
        raise ValueError("data and q_map must be 2D arrays with matching shape.")

    n_rows, n_cols = image.shape
    beam_y, beam_z = float(center[0]), float(center[1])
    reducer = np.nanmedian if use_median else np.nanmean

    if angle in (0.0, 180.0):
        row_center = beam_z - float(perp_offset)
        row_start = max(0, min(n_rows - 1, int(np.floor(row_center + band_minus))))
        row_end = max(row_start, min(n_rows - 1, int(np.ceil(row_center + band_plus))))
        band = image[row_start : row_end + 1, :]
        band_q = q[row_start : row_end + 1, :]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            # A detector column whose entire band is gap/masked pixels is
            # an expected, harmless case near detector edges -- nanmean/
            # nanmedian correctly return NaN for it (kept as NaN downstream
            # via the `keep` mask below), but warn on every occurrence via
            # Python's warnings module, which np.errstate does not cover.
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            intensity = reducer(band, axis=0)
            mean_q = np.nanmean(band_q, axis=0)
        finite_counts = np.count_nonzero(np.isfinite(band), axis=0)
        band_width = band.shape[0]
        col_offset = np.arange(1, n_cols + 1, dtype=np.float64) - beam_y
        keep = (col_offset <= 0) if angle == 0.0 else (col_offset >= 0)
        r_axis = np.abs(col_offset)
    else:  # angle == 90.0
        col_center = beam_y + float(perp_offset)
        col_start = max(0, min(n_cols - 1, int(np.floor(col_center + band_minus))))
        col_end = max(col_start, min(n_cols - 1, int(np.ceil(col_center + band_plus))))
        band = image[:, col_start : col_end + 1]
        band_q = q[:, col_start : col_end + 1]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            intensity = reducer(band, axis=1)
            mean_q = np.nanmean(band_q, axis=1)
        finite_counts = np.count_nonzero(np.isfinite(band), axis=1)
        band_width = band.shape[1]
        row_offset = -(np.arange(1, n_rows + 1, dtype=np.float64) - beam_z)
        keep = row_offset >= 0
        r_axis = np.abs(row_offset)

    coverage = finite_counts.astype(np.float32) / max(1, band_width)
    keep = keep & np.isfinite(mean_q) & (mean_q > 0) & np.isfinite(intensity)

    if not np.any(keep):
        empty_f = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int64)
        return CakeAverageResult(
            reference_angle=angle,
            angle_range=(float(band_minus), float(band_plus)),
            q_axis=empty_f,
            r_axis=empty_f,
            intensity=empty_f,
            std=empty_f,
            counts=empty_i,
            geom_counts=empty_i,
            coverage=empty_f,
            low_coverage_bins=empty_i,
            coverage_drop_bins=empty_i,
            selected_pixel_count=0,
            is_valid=False,
        )

    idx = np.flatnonzero(keep)[np.argsort(r_axis[keep])]
    coverage_out = coverage[idx]
    counts_out = finite_counts[idx].astype(np.int64)

    return CakeAverageResult(
        reference_angle=angle,
        angle_range=(float(band_minus), float(band_plus)),
        q_axis=mean_q[idx].astype(np.float32),
        r_axis=r_axis[idx].astype(np.float32),
        intensity=intensity[idx].astype(np.float32),
        std=np.full(idx.shape, np.nan, dtype=np.float32),
        counts=counts_out,
        geom_counts=counts_out,
        coverage=coverage_out,
        low_coverage_bins=np.flatnonzero(coverage_out < 0.15).astype(np.int64),
        coverage_drop_bins=find_coverage_drop_bins(coverage_out),
        selected_pixel_count=int(np.sum(counts_out)),
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
