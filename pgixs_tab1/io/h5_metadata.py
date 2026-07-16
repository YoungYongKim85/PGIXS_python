"""H5 metadata readers for PGIXS detector files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PARAM_BASE = "/entry/instrument/user/params"


@dataclass
class H5Metadata:
    counter1: float
    counter_bc: float
    bc_counter: str
    timestamp: float
    temperature: np.ndarray


def read_h5_metadata(file_path: str | Path, mode: str = "", reference_timestamp: float | None = None) -> H5Metadata:
    """Read counters, timestamp, and optional scan values from a PGIXS H5 file."""

    h5py = _import_h5py()
    path = Path(file_path)
    with h5py.File(path, "r") as h5:
        counter1 = _read_scalar(h5, f"{PARAM_BASE}/Counter1")
        counter3 = _read_scalar(h5, f"{PARAM_BASE}/Counter3")
        if np.isnan(counter3):
            counter_bc = _read_scalar(h5, f"{PARAM_BASE}/Counter2")
            bc_counter = "Counter2"
        else:
            counter_bc = counter3
            bc_counter = "Counter3"

        timestamp = _read_scalar(h5, f"{PARAM_BASE}/TimeStamp")
        if reference_timestamp is not None and np.isfinite(timestamp) and np.isfinite(reference_timestamp):
            timestamp = timestamp - float(reference_timestamp)

        temperature = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        if mode == "Temperature":
            temperature[0] = _read_first_available(h5, ["Temp1", "Temperature1"])
            temperature[1] = _read_first_available(h5, ["Temp2", "Temperature2"])
            temperature[2] = _read_first_available(h5, ["Temp3", "Temperature3"])
        elif mode == "LowTemp.":
            temperature[0] = _read_first_available(h5, ["LowTemp1"])
            temperature[1] = _read_first_available(h5, ["LowTemp2"])
        elif mode == "Angle":
            temperature[0] = _read_first_available(h5, ["IncAngle"])

    return H5Metadata(
        counter1=float(counter1),
        counter_bc=float(counter_bc),
        bc_counter=bc_counter,
        timestamp=float(timestamp),
        temperature=temperature,
    )


def _read_first_available(h5: Any, names: list[str]) -> float:
    for name in names:
        value = _read_scalar(h5, f"{PARAM_BASE}/{name}")
        if not np.isnan(value):
            return value
    return float("nan")


def _read_scalar(h5: Any, dataset_path: str) -> float:
    try:
        raw = h5[dataset_path][()]
    except Exception:
        return float("nan")
    arr = np.asarray(raw)
    if arr.size == 0:
        return float("nan")
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return float("nan")


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for PGIXS H5 metadata reading.") from exc
    return h5py

