"""Read and write PGIXS SDD standard calibration files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CalibrationLoadReport:
    ok: bool
    path: Path
    message: str = ""
    missing_required: list[str] = field(default_factory=list)
    detector_mismatch: bool = False


@dataclass
class StandardCalibration:
    beam_center: np.ndarray
    wavelength: float
    pixel_size: float
    sdd: float
    detector: str = ""
    energy_kev: float | None = None
    standard_peak: np.ndarray | None = None
    standard_q_value: float | None = None
    detector_index: str = ""
    image_shape: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_standard_calibration(
    file_path: str | Path,
    expected_detector: str = "",
) -> tuple[StandardCalibration | None, CalibrationLoadReport]:
    """Load a MATLAB `.mat` or Python-friendly `.json` SDD calibration file."""

    path = Path(file_path)
    report = CalibrationLoadReport(ok=False, path=path)
    if not path.is_file():
        report.message = f"File not found: {path}"
        return None, report

    try:
        raw = _load_raw_calibration(path)
    except Exception as exc:  # noqa: BLE001 - keep the original load reason.
        report.message = str(exc)
        return None, report

    required = ["pc", "wavelength", "pixelsize", "sdd"]
    missing = [name for name in required if name not in raw]
    if missing:
        report.missing_required = missing
        report.message = f"Missing required calibration field(s): {', '.join(missing)}"
        return None, report

    detector = _to_string(raw.get("dttypename", ""))
    if expected_detector and detector and expected_detector != detector:
        report.detector_mismatch = True
        report.message = f"Detector mismatch: expected {expected_detector}, file contains {detector}."
        return None, report

    wavelength = _to_float(raw["wavelength"])
    calibration = StandardCalibration(
        beam_center=_as_1d_float_array(raw["pc"]),
        wavelength=wavelength,
        pixel_size=_to_float(raw["pixelsize"]),
        sdd=_to_float(raw["sdd"]),
        detector=detector,
        energy_kev=wavelength_to_energy_kev(wavelength),
        standard_peak=_optional_array(raw, "STpeak"),
        standard_q_value=_optional_float(raw, "STqvalue"),
        detector_index=_to_string(raw.get("index_dt", "")),
        image_shape=_optional_array(raw, "imgdim"),
        diagnostics=_to_plain_dict(raw.get("SDDDiagnostics", {})),
        raw=raw,
    )
    report.ok = True
    return calibration, report


def save_standard_calibration_json(file_path: str | Path, calibration: StandardCalibration) -> None:
    """Save a Python-friendly JSON copy of an SDD calibration."""

    path = Path(file_path)
    payload = {
        "pc": _array_to_json(calibration.beam_center),
        "wavelength": calibration.wavelength,
        "pixelsize": calibration.pixel_size,
        "sdd": calibration.sdd,
        "dttypename": calibration.detector,
        "Energy": calibration.energy_kev,
        "STpeak": _array_to_json(calibration.standard_peak),
        "STqvalue": calibration.standard_q_value,
        "index_dt": calibration.detector_index,
        "imgdim": _array_to_json(calibration.image_shape),
        "SDDDiagnostics": _json_safe(calibration.diagnostics),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def wavelength_to_energy_kev(wavelength: float) -> float:
    """Return photon energy in keV from wavelength in Angstrom."""

    wl = float(wavelength)
    if not np.isfinite(wl) or wl <= 0:
        return float("nan")
    return 12.398419843320026 / wl


def _load_raw_calibration(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported calibration file format: {path.suffix}")
    scipy_io = _import_scipy_io()
    loaded = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {key: _mat_to_python(value) for key, value in loaded.items() if not key.startswith("__")}


def _mat_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return {name: _mat_to_python(value[name]) for name in value.dtype.names}
        if value.size == 1:
            return _mat_to_python(value.reshape(-1)[0])
        return value
    if hasattr(value, "_fieldnames"):
        return {name: _mat_to_python(getattr(value, name)) for name in value._fieldnames}
    return value


def _as_1d_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _optional_array(raw: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in raw or raw[name] is None:
        return None
    return _as_1d_float_array(raw[name])


def _optional_float(raw: dict[str, Any], name: str) -> float | None:
    if name not in raw or raw[name] is None:
        return None
    return _to_float(raw[name])


def _to_float(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    arr = np.asarray(value)
    if arr.size == 1:
        value = arr.reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _json_safe(item) for key, item in value.items()}


def _array_to_json(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).tolist()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_safe(value.__dict__)
    return str(value)


def _import_scipy_io() -> Any:
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("scipy is required for MATLAB MAT calibration loading.") from exc
    return scipy.io
