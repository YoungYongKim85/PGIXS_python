"""Read and write PGIXS SDD standard calibration files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
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
    """Load a MATLAB `.mat` or Python-friendly `.json` SDD calibration file.

    `pc` is taken verbatim as a 1-based beam center, which is what this app's
    pixel grids expect and what both MATLAB and this app write to `.mat`/
    `.std`/`.json`. MATLAB's `SDDcal_gui.m` used to subtract 1 from the center
    in its `.json` on the assumption that Python indexes from 0; that was
    fixed there, so only `.json` files written by a MATLAB build older than
    that still read one pixel low on both axes."""

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
        "SDDDiagnostics": calibration.diagnostics,
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_standard_calibration_mat(file_path: str | Path, calibration: StandardCalibration) -> None:
    """Save a MATLAB-compatible `.mat` copy of an SDD calibration, matching
    `SDDcal_gui.m`'s `Savepara_Callback` field names exactly (round-trips
    through `load_standard_calibration` unchanged, same as a real MATLAB
    `.mat` calibration file already does)."""

    scipy_io = _import_scipy_io()
    path = Path(file_path)
    payload: dict[str, Any] = {
        "pc": np.asarray(calibration.beam_center, dtype=np.float64),
        "wavelength": float(calibration.wavelength),
        "pixelsize": float(calibration.pixel_size),
        "sdd": float(calibration.sdd),
        "dttypename": calibration.detector,
        "index_dt": calibration.detector_index,
    }
    if calibration.standard_peak is not None:
        payload["STpeak"] = np.asarray(calibration.standard_peak, dtype=np.float64)
    if calibration.standard_q_value is not None:
        payload["STqvalue"] = float(calibration.standard_q_value)
    if calibration.image_shape is not None:
        payload["imgdim"] = np.asarray(calibration.image_shape, dtype=np.float64)
    # SDDDiagnostics (Phase 2's ring-candidate table) contains a list of
    # per-ring dicts -- scipy.io.savemat can't represent that shape
    # directly (no heterogeneous struct-array support without extra
    # wrangling), so it's left out of the .mat copy for now; .json already
    # carries it in full via _json_safe.
    scipy_io.savemat(path, payload)


def save_standard_calibration_std(file_path: str | Path, calibration: StandardCalibration) -> None:
    """Save the plain-ASCII `.std` calibration format (MATLAB `stdsave.m`),
    extended with a photon energy field alongside wavelength in section 3
    -- this port always has both readily available, unlike the original
    MATLAB tool. `calibration.beam_center` is written as-is: the 1-based
    center this app uses everywhere, the same value `sddcal_session.sddcal_save`
    writes to `.json` and `.mat` (MATLAB is 1-based too)."""

    path = Path(file_path)
    peak = calibration.standard_peak
    radius = float(peak[0]) if peak is not None and peak.size else float("nan")
    q_value = calibration.standard_q_value if calibration.standard_q_value is not None else float("nan")
    energy_kev = calibration.energy_kev if calibration.energy_kev is not None else float("nan")
    center = np.asarray(calibration.beam_center, dtype=np.float64).reshape(-1)
    lines = [
        "%1 center coordinate(yc,zc)",
        f"  {center[0]:.6E} ,   {center[1]:.6E}",
        "%2. Distances from center to peak  and q value ( h1st, q1st)",
        f"  {radius:.6E} ,   {q_value:.6E}",
        "%3. Wave length[A], Energy[keV], and Pixel size[mm]",
        f"  {calibration.wavelength:.6E}  ,  {energy_kev:.6E}  ,   {calibration.pixel_size:.6E}",
        "%4. Sample to detector distance[mm]",
        f"  {calibration.sdd:.6E}",
        "%5. detector type and name",
        f"{calibration.detector_index},{calibration.detector}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def wavelength_to_energy_kev(wavelength: float) -> float:
    """Return photon energy in keV from wavelength in Angstrom."""

    wl = float(wavelength)
    if not np.isfinite(wl) or wl <= 0:
        return float("nan")
    return 12.398419843320026 / wl


def _load_raw_calibration(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".std":
        return _parse_std_calibration(path)
    if suffix != ".mat":
        raise ValueError(f"Unsupported calibration file format: {path.suffix}")
    scipy_io = _import_scipy_io()
    loaded = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {key: _mat_to_python(value) for key, value in loaded.items() if not key.startswith("__")}


def _parse_std_calibration(path: Path) -> dict[str, Any]:
    """Parse `stdsave.m`'s plain-ASCII format into the same raw-dict shape
    the `.mat`/`.json` branches produce. Handles both the original MATLAB
    layout (section 3: wavelength, pixel size) and this app's own extended
    layout (section 3: wavelength, energy, pixel size) -- see
    `save_standard_calibration_std`."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value_lines = [line for line in lines if not line.startswith("%")]
    if len(value_lines) < 5:
        raise ValueError(f"Malformed .std calibration file (expected 5 value lines): {path}")

    def _values(line: str) -> list[float]:
        return [float(part.strip()) for part in line.split(",")]

    yc, zc = _values(value_lines[0])
    radius, q_value = _values(value_lines[1])
    section3 = _values(value_lines[2])
    if len(section3) == 3:
        wavelength, energy_kev, pixel_size = section3
    else:
        wavelength, pixel_size = section3
        energy_kev = None
    (sdd,) = _values(value_lines[3])
    detector_parts = value_lines[4].split(",")
    index_dt = detector_parts[0].strip() if detector_parts else ""
    dttypename = detector_parts[1].strip() if len(detector_parts) > 1 else ""

    raw: dict[str, Any] = {
        "pc": [yc, zc],
        "wavelength": wavelength,
        "pixelsize": pixel_size,
        "sdd": sdd,
        "dttypename": dttypename,
        "index_dt": index_dt,
        "STpeak": radius,
        "STqvalue": q_value,
    }
    if energy_kev is not None:
        raw["Energy"] = energy_kev
    return raw


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
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        # Bare NaN/Infinity tokens are not JSON; strict parsers (PGIXS_web's
        # JavaScript JSON.parse) reject the whole file. Loads back as None/NaN.
        return None
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
