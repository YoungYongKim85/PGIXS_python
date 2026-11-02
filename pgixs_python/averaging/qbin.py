"""Q-bin averaging with NaN omission and coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QBinAverageResult:
    intensity: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    geom_counts: np.ndarray
    coverage: np.ndarray


def summarize_coverage(coverage: np.ndarray, low_threshold: float = 0.15) -> dict[str, object]:
    """Return compact detector-gap diagnostics for a binned 1D curve."""

    cov = np.asarray(coverage, dtype=np.float64).reshape(-1)
    finite = np.isfinite(cov)
    if not np.any(finite):
        return {
            "coverage_min": float("nan"),
            "coverage_median": float("nan"),
            "coverage_low_bins": np.array([], dtype=np.int64),
            "coverage_drop_bins": np.array([], dtype=np.int64),
        }

    return {
        "coverage_min": float(np.nanmin(cov)),
        "coverage_median": float(np.nanmedian(cov)),
        "coverage_low_bins": np.flatnonzero(finite & (cov < float(low_threshold))).astype(np.int64),
        "coverage_drop_bins": _coverage_drop_bins(cov),
    }


def mean_by_bin(data: np.ndarray, bin_index: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return NaN-omitting mean and valid counts for integer bin indices."""

    values = np.asarray(data, dtype=np.float64).ravel()
    bins = np.asarray(bin_index).ravel()
    out = np.full(int(n_bins), np.nan, dtype=np.float64)
    counts = np.zeros(int(n_bins), dtype=np.int64)
    for idx in range(int(n_bins)):
        mask = (bins == idx) & np.isfinite(values)
        counts[idx] = int(np.count_nonzero(mask))
        if counts[idx] > 0:
            out[idx] = float(np.mean(values[mask]))
    return out.astype(np.float32), counts


def _coverage_drop_bins(coverage: np.ndarray) -> np.ndarray:
    cov = np.asarray(coverage, dtype=np.float64).reshape(-1)
    finite = np.isfinite(cov)
    if np.count_nonzero(finite) < 5:
        return np.array([], dtype=np.int64)
    delta = np.r_[np.nan, np.abs(np.diff(cov))]
    finite_delta = delta[np.isfinite(delta)]
    med_drop = float(np.median(finite_delta)) if finite_delta.size else 0.05
    if not np.isfinite(med_drop) or med_drop <= 0:
        med_drop = 0.05
    return np.flatnonzero(delta > max(0.10, 5.0 * med_drop)).astype(np.int64)


def qbin_average(
    data: np.ndarray,
    bin_index: np.ndarray,
    n_bins: int,
    use_median: bool = False,
    calc_std: bool = False,
    geom_mask: np.ndarray | None = None,
) -> QBinAverageResult:
    """NaN-omitting q-bin mean (or median) with detector-gap diagnostics.

    The mean is the plain arithmetic mean, as in MATLAB's circular average.
    A robust (Huber) mean used to be the default here, but it treats signal
    that fills only part of a ring -- Bragg peaks of a textured film, a
    single-crystal spot -- as outliers and removes most of it.

    `bin_index` is zero-based. Negative bin ids are ignored. Coverage is
    `finite data count / geometry pixel count`, which makes detector-gap
    induced dips visible instead of silently flattening the intensity.
    """

    values = np.asarray(data, dtype=np.float64).ravel()
    bins = np.asarray(bin_index, dtype=np.int64).ravel()
    if values.size != bins.size:
        raise ValueError("data and bin_index must have the same flattened size.")
    n_bins_int = int(n_bins)
    if n_bins_int <= 0:
        raise ValueError("n_bins must be positive.")

    if geom_mask is None:
        geom = bins >= 0
    else:
        geom = np.asarray(geom_mask, dtype=bool).ravel()
        if geom.size != values.size:
            raise ValueError("geom_mask must have the same flattened size as data.")
        geom = geom & (bins >= 0)

    valid_geom = geom & (bins < n_bins_int)
    finite = valid_geom & np.isfinite(values)
    geom_counts = np.bincount(bins[valid_geom], minlength=n_bins_int)[:n_bins_int].astype(np.int64)
    counts = np.bincount(bins[finite], minlength=n_bins_int)[:n_bins_int].astype(np.int64)

    if use_median:
        intensity = _median_by_bin(values[finite], bins[finite], n_bins_int)
        std = _mad_std_by_bin(values[finite], bins[finite], intensity, n_bins_int)
    else:
        intensity, std = _mean_std_by_bin(values[finite], bins[finite], counts)

    if not calc_std:
        std = np.full(n_bins_int, np.nan, dtype=np.float32)

    coverage = np.full(n_bins_int, np.nan, dtype=np.float32)
    ok = geom_counts > 0
    coverage[ok] = (counts[ok] / geom_counts[ok]).astype(np.float32)

    intensity[counts < 1] = np.nan
    std[counts < 1] = np.nan
    return QBinAverageResult(
        intensity=intensity.astype(np.float32),
        std=std.astype(np.float32),
        counts=counts,
        geom_counts=geom_counts,
        coverage=coverage,
    )


def _median_by_bin(values: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(n_bins, np.nan, dtype=np.float64)
    if values.size == 0:
        return out
    # Group by bin only. A median doesn't care how its inputs are ordered, so
    # sorting the values within each bin as well (`np.lexsort((values, bins))`)
    # bought nothing and cost ~5x as much -- on a full Eiger4M frame that was
    # ~1.0 s per call versus ~0.2 s, paid twice per median/MAD reduction, on
    # every frame of a series. Results are bit-identical either way.
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1]
    ends = np.r_[starts[1:], sorted_bins.size]
    for start, end in zip(starts, ends):
        bin_id = int(sorted_bins[start])
        if 0 <= bin_id < n_bins:
            out[bin_id] = np.median(sorted_values[start:end])
    return out


def _mad_std_by_bin(values: np.ndarray, bins: np.ndarray, center: np.ndarray, n_bins: int) -> np.ndarray:
    finite_bin = (bins >= 0) & (bins < n_bins) & np.isfinite(center[bins])
    if not np.any(finite_bin):
        return np.full(n_bins, np.nan, dtype=np.float64)
    abs_dev = np.abs(values[finite_bin] - center[bins[finite_bin]])
    return 1.4826 * _median_by_bin(abs_dev, bins[finite_bin], n_bins)


def _mean_std_by_bin(values: np.ndarray, bins: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic mean and sample standard deviation (N-1, NaN below two
    pixels) per bin, as MATLAB `processPGIXSRadiusBin1D`'s `mean`/`std`."""

    n_bins = counts.size
    mean = np.full(n_bins, np.nan, dtype=np.float64)
    std = np.full(n_bins, np.nan, dtype=np.float64)
    ok = counts > 0
    mean[ok] = np.bincount(bins, weights=values, minlength=n_bins)[:n_bins][ok] / counts[ok]
    # Two-pass: squared deviations from the bin mean, not E[x^2] - E[x]^2,
    # which cancels badly on bright, flat bins.
    sq_dev = np.bincount(bins, weights=(values - mean[bins]) ** 2, minlength=n_bins)[:n_bins]
    many = counts > 1
    std[many] = np.sqrt(sq_dev[many] / (counts[many] - 1))
    return mean, std
