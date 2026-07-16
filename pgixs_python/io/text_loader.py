"""Numeric text detector image loading for PGIXS Python."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_numeric_text_image(file_path: str | Path, delimiter: str | None = None) -> np.ndarray:
    """Load a CSV/DAT/TXT numeric matrix as float32."""

    path = Path(file_path)
    data = np.loadtxt(path, delimiter=delimiter)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] <= 1 or data.shape[1] <= 1:
        raise ValueError(f"No numeric 2D matrix found in text file: {path}")
    return data
