"""H5-native Tab1 2D-info structures.

Python Tab1 uses H5 as its native internal output format because it can store
large arrays, metadata, diagnostics, and provenance in one extensible file.
MATLAB `.mat` export should be treated as a compatibility layer, not the
primary Python storage contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "python-tab1-h5-0.1"


@dataclass
class TwoDInfo:
    measurement_mode: str
    data2d_pix: np.ndarray
    data2d_q: np.ndarray
    q_map: np.ndarray
    qz_map: np.ndarray
    qz_axis: np.ndarray
    beam_position: np.ndarray
    wavelength: float
    sdd: float
    pixel: float
    qx_map: np.ndarray | None = None
    qy_map: np.ndarray | None = None
    qy_axis: np.ndarray | None = None
    qr_map: np.ndarray | None = None
    qr_axis: np.ndarray | None = None
    energy_kev: float | None = None
    detector: str = ""
    incident_angle: float | None = None
    source_files: list[str] = field(default_factory=list)
    retry_failed_frames: np.ndarray | None = None
    metadata_failed_frames: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def save_two_d_info_h5(file_path: str | Path, info: TwoDInfo) -> None:
    """Save Python-native 2D-info H5 output."""

    h5py = _import_h5py()
    path = Path(file_path)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["measurement_mode"] = info.measurement_mode

        data_group = h5.create_group("data")
        _write_dataset(data_group, "data2d_pix", info.data2d_pix)
        _write_dataset(data_group, "data2d_q", info.data2d_q)

        qmap_group = h5.create_group("qmap")
        _write_dataset(qmap_group, "q", info.q_map)
        _write_dataset(qmap_group, "qz", info.qz_map)
        _write_dataset(qmap_group, "qz_axis", info.qz_axis)
        _write_optional_dataset(qmap_group, "qx", info.qx_map)
        _write_optional_dataset(qmap_group, "qy", info.qy_map)
        _write_optional_dataset(qmap_group, "qy_axis", info.qy_axis)
        _write_optional_dataset(qmap_group, "qr", info.qr_map)
        _write_optional_dataset(qmap_group, "qr_axis", info.qr_axis)

        calibration_group = h5.create_group("calibration")
        _write_dataset(calibration_group, "beam_center", np.asarray(info.beam_position, dtype=np.float32))
        calibration_group.attrs["wavelength"] = float(info.wavelength)
        calibration_group.attrs["sdd"] = float(info.sdd)
        calibration_group.attrs["pixel"] = float(info.pixel)
        calibration_group.attrs["detector"] = info.detector
        if info.energy_kev is not None:
            calibration_group.attrs["energy_kev"] = float(info.energy_kev)
        if info.incident_angle is not None:
            calibration_group.attrs["incident_angle"] = float(info.incident_angle)

        metadata_group = h5.create_group("metadata")
        metadata_group.attrs["source_files"] = np.array(info.source_files, dtype=h5py.string_dtype("utf-8"))

        processing_group = h5.create_group("processing")
        _write_optional_dataset(processing_group, "retry_failed_frames", info.retry_failed_frames)
        _write_optional_dataset(processing_group, "metadata_failed_frames", info.metadata_failed_frames)

        diagnostics_group = h5.create_group("diagnostics")
        _write_diagnostics(diagnostics_group, info.diagnostics)


def load_two_d_info_h5(file_path: str | Path) -> TwoDInfo:
    """Load Python-native 2D-info H5 output."""

    h5py = _import_h5py()
    path = Path(file_path)
    with h5py.File(path, "r") as h5:
        calibration = h5["calibration"]
        metadata = h5["metadata"] if "metadata" in h5 else None
        processing = h5["processing"] if "processing" in h5 else None
        qmap = h5["qmap"]
        data = h5["data"]
        return TwoDInfo(
            measurement_mode=_read_str_attr(h5.attrs.get("measurement_mode", "")),
            data2d_pix=np.asarray(data["data2d_pix"][()]),
            data2d_q=np.asarray(data["data2d_q"][()]),
            q_map=np.asarray(qmap["q"][()]),
            qz_map=np.asarray(qmap["qz"][()]),
            qz_axis=np.asarray(qmap["qz_axis"][()]),
            beam_position=np.asarray(calibration["beam_center"][()]),
            wavelength=float(calibration.attrs["wavelength"]),
            sdd=float(calibration.attrs["sdd"]),
            pixel=float(calibration.attrs["pixel"]),
            qx_map=_read_optional_dataset(qmap, "qx"),
            qy_map=_read_optional_dataset(qmap, "qy"),
            qy_axis=_read_optional_dataset(qmap, "qy_axis"),
            qr_map=_read_optional_dataset(qmap, "qr"),
            qr_axis=_read_optional_dataset(qmap, "qr_axis"),
            energy_kev=_read_optional_float_attr(calibration.attrs, "energy_kev"),
            detector=_read_str_attr(calibration.attrs.get("detector", "")),
            incident_angle=_read_optional_float_attr(calibration.attrs, "incident_angle"),
            source_files=_read_source_files(metadata),
            retry_failed_frames=_read_optional_dataset(processing, "retry_failed_frames"),
            metadata_failed_frames=_read_optional_dataset(processing, "metadata_failed_frames"),
            diagnostics=_read_diagnostics(h5["diagnostics"] if "diagnostics" in h5 else None),
        )


def export_two_d_info_mat(file_path: str | Path, info: TwoDInfo) -> None:
    """Export a MATLAB-readable compatibility MAT file."""

    scipy_io = _import_scipy_io()
    path = Path(file_path)
    data = {
        "data2dPix": info.data2d_pix,
        "data2dQ": info.data2d_q,
        "QMap": info.q_map,
        "QzMap": info.qz_map,
        "qzaxis": info.qz_axis,
        "Beampos": info.beam_position,
        "Wavelength": info.wavelength,
        "SDD": info.sdd,
        "pixel": info.pixel,
        "MeasurementMode": info.measurement_mode,
        "SchemaVersion": SCHEMA_VERSION,
    }
    if info.qx_map is not None:
        data["QxMap"] = info.qx_map
    if info.qy_map is not None:
        data["QyMap"] = info.qy_map
    if info.qy_axis is not None:
        data["qyaxis"] = info.qy_axis
    if info.qr_map is not None:
        data["QrMap"] = info.qr_map
    if info.qr_axis is not None:
        data["qraxis"] = info.qr_axis
    scipy_io.savemat(path, data, do_compression=True)


def _write_dataset(group: Any, name: str, value: Any) -> None:
    group.create_dataset(name, data=np.asarray(value), compression="gzip")


def _write_optional_dataset(group: Any, name: str, value: Any) -> None:
    if value is not None:
        _write_dataset(group, name, value)


def _write_diagnostics(group: Any, diagnostics: dict[str, Any]) -> None:
    for key, value in diagnostics.items():
        if value is None:
            continue
        if np.isscalar(value):
            group.attrs[key] = value
        else:
            _write_dataset(group, key, value)


def _read_optional_dataset(group: Any, name: str) -> np.ndarray | None:
    if group is None or name not in group:
        return None
    return np.asarray(group[name][()])


def _read_optional_float_attr(attrs: Any, name: str) -> float | None:
    if name not in attrs:
        return None
    return float(attrs[name])


def _read_str_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_source_files(metadata_group: Any) -> list[str]:
    if metadata_group is None or "source_files" not in metadata_group.attrs:
        return []
    value = metadata_group.attrs["source_files"]
    if np.isscalar(value):
        return [_read_str_attr(value)]
    return [_read_str_attr(item) for item in value]


def _read_diagnostics(group: Any) -> dict[str, Any]:
    if group is None:
        return {}
    diagnostics: dict[str, Any] = {key: value for key, value in group.attrs.items()}
    for key in group.keys():
        diagnostics[key] = np.asarray(group[key][()])
    return diagnostics


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for Python-native PGIXS H5 2D-info files.") from exc
    return h5py


def _import_scipy_io() -> Any:
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("scipy is required only for MATLAB compatibility MAT export.") from exc
    return scipy.io

