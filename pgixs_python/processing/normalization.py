"""Normalization helpers."""

from __future__ import annotations

import numpy as np


def normalize_by_count(data: np.ndarray, count: float | None) -> np.ndarray:
    """Normalize by a scalar count while preserving NaNs."""

    arr = np.asarray(data, dtype=np.float32)
    if count is None or not np.isfinite(count) or count == 0:
        return arr.copy()
    return arr / np.float32(count)

