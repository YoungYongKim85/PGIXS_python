"""Detector image loading entry points.

This module will mirror MATLAB `loadPGIXSImageFile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.io.h5_loader import load_h5_frame
from pgixs_python.io.h5_metadata import read_h5_metadata


@dataclass
class ImageMetadata:
    file_path: Path
    file_format: str
    data_size: tuple[int, int]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageLoadResult:
    data: np.ndarray
    counts: np.ndarray
    metadata: ImageMetadata


def load_image_file(file_path: str | Path, file_format: str | None = None) -> ImageLoadResult:
    """Load one detector image.

    The implementation is intentionally minimal in the scaffold. The next step
    is to port MATLAB format-specific behavior into this dispatcher.
    """

    path = Path(file_path)
    suffix = (file_format or path.suffix).lower()
    if suffix == ".h5":
        data = load_h5_frame(path)
        h5_meta = read_h5_metadata(path)
        metadata = ImageMetadata(
            file_path=path,
            file_format=suffix,
            data_size=tuple(int(v) for v in data.shape),
            extra={
                "timestamp": h5_meta.timestamp,
                "bc_counter": h5_meta.bc_counter,
                "temperature": h5_meta.temperature,
            },
        )
        counts = np.asarray([h5_meta.counter1, h5_meta.counter_bc], dtype=np.float32)
        return ImageLoadResult(data=data, counts=counts, metadata=metadata)
    raise NotImplementedError(f"Image loading for {suffix} is not implemented yet: {path}")
