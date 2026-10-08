"""Pixel-radius (no Q-map needed) radial intensity profile, for SDD
calibration -- the sample-to-detector distance is the very thing being
calibrated, so there is no Q-map yet to bin against.

Ports MATLAB `computePGIXSRadialProfile.m`: bin every pixel by its raw pixel
distance from a candidate beam center, average per bin, and (by default)
apply a light Gaussian smoothing pass before returning -- ring-peak search
and fitting downstream (SDDcal's "Rad.Ave (Fit)"/Smart Auto Run) both work
off the smoothed curve in MATLAB, so this keeps that same shape.

Deliberately built on `averaging.qbin.qbin_average` rather than
`averaging.radial.build_radius_bin_map`: that function gates pixels on
`q > 0`, a real Q-specific constraint that would wrongly drop the r=0 bin
here (radius 0 is a legitimate, meaningful bin for a pixel-radius profile).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgixs_python.averaging.qbin import QBinAverageResult, qbin_average


@dataclass
class PixelRadialProfile:
    r_axis: np.ndarray
    intensity: np.ndarray
    raw_intensity: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    geom_counts: np.ndarray
    coverage: np.ndarray


def build_pixel_radius_bin_map(
    shape: tuple[int, int],
    center: tuple[float, float],
    geom_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Bin every pixel in `shape` by `floor(radius from center)`.

    Returns `(bin_index, n_bins, r_axis, geom)`: `bin_index` is a 2D array
    of zero-based bin ids (-1 outside the geometry), `r_axis[i] = i + 0.5`
    (bin-center convention, matching `edges(1:end-1) + 0.5` in
    `computePGIXSRadialProfile.m`), and `geom` is the boolean geometry mask
    actually used (all finite-radius pixels, further restricted by the
    caller's `geom_mask` if given -- e.g. a detector active-area mask).
    """

    n_rows, n_cols = int(shape[0]), int(shape[1])
    cols_grid, rows_grid = np.meshgrid(
        np.arange(1, n_cols + 1, dtype=np.float64),
        np.arange(1, n_rows + 1, dtype=np.float64),
    )
    radius = np.hypot(cols_grid - float(center[0]), rows_grid - float(center[1]))

    geom = np.isfinite(radius)
    if geom_mask is not None:
        mask = np.asarray(geom_mask, dtype=bool)
        if mask.shape != radius.shape:
            raise ValueError("geom_mask must match shape.")
        geom = geom & mask

    bin_index = np.full(radius.shape, -1, dtype=np.int64)
    if not np.any(geom):
        return bin_index, 0, np.array([], dtype=np.float64), geom

    # floor + 1, not ceil: a pixel at an exactly integer max radius has
    # floor(r) == ceil(r) and fell outside the last bin.
    n_bins = int(np.floor(np.max(radius[geom]))) + 1
    if n_bins <= 0:
        return bin_index, 0, np.array([], dtype=np.float64), geom

    raw_bin = np.floor(radius).astype(np.int64)
    valid = geom & (raw_bin >= 0) & (raw_bin < n_bins)
    bin_index[valid] = raw_bin[valid]
    r_axis = np.arange(n_bins, dtype=np.float64) + 0.5
    return bin_index, n_bins, r_axis, geom


def _smooth_profile(intensity: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware Gaussian smoothing, approximating MATLAB's
    `smoothdata(intensity, 'gaussian', window, 'omitnan')` via normalized
    convolution (convolve values-with-NaN-zeroed and a finite-weight mask
    with the same kernel, then divide) -- exact numerical parity with
    MATLAB's internal window-to-sigma mapping isn't the point here, this is
    a peak-finding preprocessing aid, not a physics calculation."""

    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(intensity, dtype=np.float64)
    finite = np.isfinite(values)
    if window <= 1 or not np.any(finite):
        return values.copy()

    sigma = max(float(window) / 5.0, 1e-6)
    weights = finite.astype(np.float64)
    filled = np.where(finite, values, 0.0)

    smoothed_sum = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    smoothed_weight = gaussian_filter1d(weights, sigma=sigma, mode="nearest")

    out = np.full_like(values, np.nan)
    ok = smoothed_weight > 1e-9
    out[ok] = smoothed_sum[ok] / smoothed_weight[ok]
    out[~finite & ~ok] = np.nan
    return out


def compute_radial_profile(
    data: np.ndarray,
    center: tuple[float, float],
    geom_mask: np.ndarray | None = None,
    use_median: bool = False,
    calc_std: bool = False,
    smooth: bool = True,
    smooth_window: int = 5,
) -> PixelRadialProfile:
    """Radius-binned detector profile with coverage, matching
    `computePGIXSRadialProfile.m`. `data` should already have detector gaps
    marked NaN (e.g. via `processing.masking.apply_detector_mask`) --
    unlike MATLAB's own zero-multiplied mask, NaN gap pixels are correctly
    excluded from both the average and (when `geom_mask` is also passed)
    the coverage denominator."""

    image = np.asarray(data, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("data must be a 2D array.")

    bin_index, n_bins, r_axis, geom = build_pixel_radius_bin_map(image.shape, center, geom_mask)
    if n_bins == 0:
        empty_f = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int64)
        return PixelRadialProfile(empty_f, empty_f, empty_f, empty_f, empty_i, empty_i, empty_f)

    binned: QBinAverageResult = qbin_average(image, bin_index, n_bins, use_median=use_median, calc_std=calc_std, geom_mask=geom)

    raw_intensity = binned.intensity.copy()
    intensity = raw_intensity
    if smooth:
        intensity = _smooth_profile(raw_intensity, smooth_window).astype(np.float32)

    return PixelRadialProfile(
        r_axis=r_axis.astype(np.float32),
        intensity=intensity,
        raw_intensity=raw_intensity,
        std=binned.std,
        counts=binned.counts,
        geom_counts=binned.geom_counts,
        coverage=binned.coverage,
    )
