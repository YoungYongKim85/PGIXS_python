"""TIFF detector image loading for PGIXS Python."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_tif_image(file_path: str | Path) -> np.ndarray:
    """Load the first TIFF image as a float32 detector matrix."""

    tifffile = _import_tifffile()
    path = Path(file_path)
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"No TIFF pages found: {path}")
        data = tif.pages[0].asarray()
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[-1] in {3, 4}:
        data = data[..., :3].mean(axis=-1)
    if data.ndim != 2:
        raise ValueError(f"TIFF image must reduce to one 2D detector matrix, got {data.shape}: {path}")
    return np.asarray(data, dtype=np.float32)


def _import_tifffile() -> Any:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required for TIFF detector image loading.") from exc
    return tifffile
