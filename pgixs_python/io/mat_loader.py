"""MAT detector image loading for PGIXS Python."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PRIORITY_IMAGE_NAMES = ("data", "image", "img", "frame", "i")
NUMERIC_KINDS = {"b", "u", "i", "f", "c"}


@dataclass
class MatImageInfo:
    variable_name: str
    variable_shape: tuple[int, ...]
    variable_class: str


def load_mat_image(file_path: str | Path) -> tuple[np.ndarray, MatImageInfo]:
    """Load the most image-like numeric variable from a MATLAB MAT file."""

    scipy_io = _import_scipy_io()
    path = Path(file_path)
    try:
        variables = scipy_io.whosmat(path)
    except NotImplementedError:
        return _load_hdf5_mat_image(path)

    candidates = [_mat_candidate(item) for item in variables]
    candidates = [item for item in candidates if item is not None]
    if not candidates:
        raise ValueError(f"No numeric 2D image-like variables found in MAT file: {path}")

    selected = max(candidates, key=_candidate_score)
    data_dict = scipy_io.loadmat(path, variable_names=[selected.variable_name], squeeze_me=False, struct_as_record=False)
    data = np.asarray(data_dict[selected.variable_name], dtype=np.float32)
    return _as_detector_matrix(data, path), selected


def _mat_candidate(item: tuple[str, tuple[int, ...], str]) -> MatImageInfo | None:
    name, shape, class_name = item
    if name.startswith("__") or len(shape) < 2 or shape[0] <= 1 or shape[1] <= 1:
        return None
    if class_name in {"cell", "struct", "char", "object"}:
        return None
    return MatImageInfo(variable_name=name, variable_shape=tuple(int(v) for v in shape), variable_class=class_name)


def _candidate_score(info: MatImageInfo) -> float:
    score = float(np.prod(info.variable_shape[:2]))
    if info.variable_name.lower() in PRIORITY_IMAGE_NAMES:
        score += 1e12
    return score


def _as_detector_matrix(data: np.ndarray, path: Path) -> np.ndarray:
    if data.ndim < 2 or data.shape[0] <= 1 or data.shape[1] <= 1:
        raise ValueError(f"Selected MAT variable is not a 2D image-like matrix: {path}")
    if data.ndim > 2:
        trailing = tuple(int(v) for v in data.shape[2:])
        if int(np.prod(trailing)) != 1:
            raise ValueError(f"Selected MAT variable has unsupported image dimensions {data.shape}: {path}")
        data = np.reshape(data, data.shape[:2])
    return np.asarray(data, dtype=np.float32)


def _load_hdf5_mat_image(path: Path) -> tuple[np.ndarray, MatImageInfo]:
    h5py = _import_h5py()
    candidates: list[tuple[str, tuple[int, ...], Any]] = []
    with h5py.File(path, "r") as h5:
        h5.visititems(lambda name, obj: _collect_hdf5_dataset(name, obj, candidates))
        if not candidates:
            raise ValueError(f"No numeric 2D image-like datasets found in HDF5 MAT file: {path}")
        name, shape, dataset = max(
            candidates,
            key=lambda item: (1e12 if Path(item[0]).name.lower() in PRIORITY_IMAGE_NAMES else 0) + np.prod(item[1][:2]),
        )
        variable_class = dataset.attrs.get("MATLAB_class", dataset.dtype.name)
        if isinstance(variable_class, bytes):
            variable_class = variable_class.decode("ascii")
        # MATLAB writes v7.3 arrays column-major, so h5py sees every axis
        # reversed; transposing back gives the same orientation scipy reads
        # from a v7 file of the same image.
        data = np.asarray(dataset[()], dtype=np.float32).T

    info = MatImageInfo(variable_name=name, variable_shape=shape, variable_class=str(variable_class))
    return _as_detector_matrix(data, path), info


def _collect_hdf5_dataset(name: str, obj: Any, candidates: list[tuple[str, tuple[int, ...], Any]]) -> None:
    if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
        return
    shape = tuple(int(v) for v in reversed(obj.shape))  # MATLAB's axis order
    if len(shape) >= 2 and shape[0] > 1 and shape[1] > 1 and obj.dtype.kind in NUMERIC_KINDS:
        candidates.append((name, shape, obj))


def _import_scipy_io() -> Any:
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("scipy is required for MATLAB MAT image loading.") from exc
    return scipy.io


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5/v7.3 MATLAB MAT image loading.") from exc
    return h5py
