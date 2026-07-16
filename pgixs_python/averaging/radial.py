"""Pixel-radius 1D averaging with Q-axis and coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgixs_python.averaging.qbin import QBinAverageResult, qbin_average


@dataclass
class RadiusBinMap:
    is_valid: bool
    bin_index: np.ndarray
    n_bins: int
    q_axis: np.ndarray
    r_axis: np.ndarray
    geom_counts: np.ndarray
    geom_mask: np.ndarray


@dataclass
class RadiusAverageResult:
    q_axis: np.ndarray
    r_axis: np.ndarray
    intensity: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    geom_counts: np.ndarray
    coverage: np.ndarray


def build_radius_bin_map(
    q_map: np.ndarray,
    center: tuple[float, float],
    geom_mask: np.ndarray | None = None,
) -> RadiusBinMap:
    """Precompute compact pixel-radius bins with mean Q per bin."""

    q = np.asarray(q_map, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("q_map must be a 2D array.")
    if geom_mask is None:
        geom = np.ones(q.shape, dtype=bool)
    else:
        geom = np.asarray(geom_mask, dtype=bool)
        if geom.shape != q.shape:
            raise ValueError("geom_mask must match q_map shape.")

    geom = geom & np.isfinite(q) & (q > 0)
    bin_index = np.full(q.shape, -1, dtype=np.int64)
    if not np.any(geom):
        return RadiusBinMap(False, bin_index, 0, np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int64), geom)

    rows, cols = np.nonzero(geom)
    radius = np.hypot(cols.astype(np.float64) + 1.0 - float(center[0]), rows.astype(np.float64) + 1.0 - float(center[1]))
    raw_bins = np.floor(radius).astype(np.int64)
    raw_n_bins = int(raw_bins.max()) + 1
    geom_counts_raw = np.bincount(raw_bins, minlength=raw_n_bins).astype(np.int64)
    q_sum = np.bincount(raw_bins, weights=q[rows, cols].astype(np.float64), minlength=raw_n_bins)
    r_sum = np.bincount(raw_bins, weights=radius, minlength=raw_n_bins)

    keep = geom_counts_raw > 0
    q_axis_raw = np.full(raw_n_bins, np.nan, dtype=np.float64)
    r_axis_raw = np.full(raw_n_bins, np.nan, dtype=np.float64)
    q_axis_raw[keep] = q_sum[keep] / geom_counts_raw[keep]
    r_axis_raw[keep] = r_sum[keep] / geom_counts_raw[keep]
    keep = keep & np.isfinite(q_axis_raw) & np.isfinite(r_axis_raw)

    compact = np.full(raw_n_bins, -1, dtype=np.int64)
    compact[keep] = np.arange(int(np.count_nonzero(keep)), dtype=np.int64)
    bin_index[rows, cols] = compact[raw_bins]

    return RadiusBinMap(
        is_valid=int(np.count_nonzero(keep)) > 0,
        bin_index=bin_index,
        n_bins=int(np.count_nonzero(keep)),
        q_axis=q_axis_raw[keep].astype(np.float32),
        r_axis=r_axis_raw[keep].astype(np.float32),
        geom_counts=geom_counts_raw[keep],
        geom_mask=geom,
    )


def radius_average(
    data: np.ndarray,
    radius_map: RadiusBinMap,
    use_median: bool = False,
    calc_std: bool = False,
) -> RadiusAverageResult:
    """Average a detector image using a precomputed radius bin map."""

    image = np.asarray(data, dtype=np.float32)
    if image.shape != radius_map.bin_index.shape:
        raise ValueError("data shape must match the radius bin map.")
    if not radius_map.is_valid:
        empty_i = np.array([], dtype=np.float32)
        empty_n = np.array([], dtype=np.int64)
        return RadiusAverageResult(empty_i, empty_i, empty_i, empty_i, empty_n, empty_n, empty_i)

    qbin: QBinAverageResult = qbin_average(
        image,
        radius_map.bin_index,
        radius_map.n_bins,
        use_median=use_median,
        calc_std=calc_std,
        geom_mask=radius_map.geom_mask,
    )
    return RadiusAverageResult(
        q_axis=radius_map.q_axis,
        r_axis=radius_map.r_axis,
        intensity=qbin.intensity,
        std=qbin.std,
        counts=qbin.counts,
        geom_counts=radius_map.geom_counts,
        coverage=qbin.coverage,
    )
