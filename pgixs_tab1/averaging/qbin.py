"""Q-bin averaging with NaN omission."""

from __future__ import annotations

import numpy as np


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

