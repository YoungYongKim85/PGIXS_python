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
from pgixs_python.io.mat_loader import load_mat_image
from pgixs_python.io.text_loader import load_numeric_text_image
from pgixs_python.io.tif_loader import load_tif_image


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
    suffix = _normalize_suffix(file_format or path.suffix)
    if suffix in {".h5", ".hdf5"}:
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
    if suffix == ".mat":
        data, mat_info = load_mat_image(path)
        return _result_without_counts(
            path,
            suffix,
            data,
            {
                "mat_variable_name": mat_info.variable_name,
                "mat_variable_shape": mat_info.variable_shape,
                "mat_variable_class": mat_info.variable_class,
            },
        )
    if suffix in {".tif", ".tiff"}:
        return _result_without_counts(path, suffix, load_tif_image(path))
    if suffix in {".txt", ".dat", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        return _result_without_counts(path, suffix, load_numeric_text_image(path, delimiter=delimiter))
    raise NotImplementedError(f"Image loading for {suffix} is not implemented yet: {path}")


def _result_without_counts(
    path: Path,
    suffix: str,
    data: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> ImageLoadResult:
    metadata = ImageMetadata(
        file_path=path,
        file_format=suffix,
        data_size=tuple(int(v) for v in data.shape),
        extra=extra or {},
    )
    return ImageLoadResult(data=data, counts=np.asarray([], dtype=np.float32), metadata=metadata)


def _normalize_suffix(value: str) -> str:
    suffix = str(value).lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix
