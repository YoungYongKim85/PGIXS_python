"""In-process interactive session backing the GIXS/TrXS console web UI.

This mirrors the real MATLAB `GIXS_gui.m` / `TrXS_gui.m` interaction model:
load one detector image, keep it (plus calibration/qmap state) in memory,
and recompute previews/curves on demand as the user edits parameters,
instead of running a full batch workflow to completion and writing files
at every step. This is a local, single-user desktop tool, so one
module-level session is enough -- it matches the MATLAB single-window
model and avoids multi-session/auth complexity that would add nothing
here. The session's `mode` ("gixs" or "trxs") is fixed for the life of a
loaded dataset, exactly like MATLAB where GIXS and TrXS are separate
console windows, never the same window switching geometry mid-session.

File loading mirrors MATLAB's `Directory` + `LOAD DATA` split exactly
(`Openimage_Callback` in `GIXS_gui.m`/`TrXS_gui.m`): `Directory` names a
folder; `LOAD DATA` picks one file from it; `Multishot`, when checked,
takes the picked file's name, strips a trailing `_NNNN` series index
(`normalizePGIXSSeriesPrefix.m`), and averages every same-prefix
same-extension file in that folder (`listPGIXSSeriesFiles.m` /
`loadPGIXSMultishotAverage.m`) -- not "every file in the folder."

Deliberate simplifications versus MATLAB, noted in
`docs/console_ui_design.md`:
- "Line" 1D processing is implemented as a cake-cut (angular wedge) around
  the reference angle. For GIXS this stands in for the Accum-Pixel /
  multi-function dispatch; for TrXS it matches `Anglecut1d` directly
  (TrXS only ever used one line-cut function).
- Detector correction applied during q-Map Cal. is solid-angle x
  polarization only (no kapton/silicon attenuation tables).
- TrXS `PointFind_Callback`'s live click-and-drag ROI updates and the
  optional `roipoly` mask in `CircularAve_Callback`/`DrawPlot_Callback`
  are not implemented; positions are set by a single click, and circular
  average always uses the full unmasked geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.averaging.angular import angular_profile, sector_average
from pgixs_python.averaging.cake import cake_average, pixel_angle_map
from pgixs_python.averaging.qbin import summarize_coverage
from pgixs_python.calibration.calibration_io import load_standard_calibration
from pgixs_python.calibration.sdd import wavelength_from_energy_kev
from pgixs_python.export.data_export import OneDExport, save_1d_data
from pgixs_python.io.image_loader import load_image_file
from pgixs_python.io.two_d_info import TwoDInfo, save_two_d_info_h5
from pgixs_python.processing.corrections import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    detector_correction_factor,
)
from pgixs_python.processing.grid_remove import grid_remove
from pgixs_python.processing.masking import apply_detector_mask, detector_mask_for_type
from pgixs_python.processing.normalization import process_normalization
from pgixs_python.qmap.gixs import qmap_gixs
from pgixs_python.qmap.transform import image_transform_q_gi, image_transform_q_tr
from pgixs_python.qmap.trxs import qmap_trxs
from pgixs_python.ui_backend.preview import downsample_1d_for_json, downsample_2d_for_json


_MODES = {"gixs", "trxs"}


_IMAGE_SUFFIXES = {".h5", ".hdf5", ".mat", ".tif", ".tiff", ".txt", ".dat", ".csv"}


@dataclass
class ConsoleSession:
    mode: str = "gixs"
    detector: str = "Eiger4M"
    data: np.ndarray | None = None
    corrected_data: np.ndarray | None = None
    mask: np.ndarray | None = None
    counts: np.ndarray | None = None
    background: np.ndarray | None = None
    background_counts: np.ndarray | None = None
    source_path: str = ""
    calibration: dict[str, Any] = field(
        default_factory=lambda: {
            "energy_kev": 9.5,
            "sdd_mm": 3000.0,
            "beam_pos": (1000.0, 1800.0),
            "pixel_mm": 0.075,
            "incident_angle_deg": 0.1,
            "footprint_mm": 1.0,
        }
    )
    qmap: dict[str, np.ndarray] | None = None
    angle_map: np.ndarray | None = None
    normalized: bool = False
    grid_removed: bool = False
    last_curve: dict[str, Any] | None = None
    load_progress: dict[str, Any] | None = None


_SESSION = ConsoleSession()


def get_session() -> ConsoleSession:
    """Return the module-level session (mainly for tests)."""

    return _SESSION


def set_session(session: ConsoleSession) -> None:
    """Install `session` as the active module-level session.

    All `console_*` functions operate on whatever session is currently
    installed here. The web console only ever ran one console at a time, so
    the single-session-forever default was never a problem; the desktop app
    keeps a GIXS tab and a TrXS tab alive simultaneously, each owning its
    own `ConsoleSession`, and swaps the active one in immediately before
    dispatching to this module -- otherwise the two tabs would silently
    clobber each other's loaded data.
    """

    global _SESSION
    _SESSION = session


def reset_session() -> None:
    """Reset the module-level session to a fresh state (mainly for tests)."""

    global _SESSION
    _SESSION = ConsoleSession()


def console_state() -> dict[str, Any]:
    loaded = _SESSION.data is not None
    return {
        "ok": True,
        "mode": _SESSION.mode,
        "detector": _SESSION.detector,
        "loaded": loaded,
        "shape": list(_SESSION.data.shape) if loaded else None,
        "source_path": _SESSION.source_path,
        "calibration": _json_calibration(),
        "has_qmap": _SESSION.qmap is not None,
        "normalized": _SESSION.normalized,
        "grid_removed": _SESSION.grid_removed,
    }


def console_load_progress() -> dict[str, Any]:
    """Poll target for Multishot progress, matching MATLAB's `uiprogressdlg`
    updates during `loadPGIXSMultishotAverage`. Returns `active: false` once
    the in-flight `console_load`/`console_background_load` call finishes
    (or if nothing is loading)."""

    progress = _SESSION.load_progress
    if progress is None:
        return {"ok": True, "active": False}
    return {"ok": True, "active": True, **progress}


def console_list_files(payload: dict[str, Any]) -> dict[str, Any]:
    """List loadable files in a directory, for the LOAD DATA file picker.

    Mirrors what MATLAB's `uigetfile(..., pathway)` shows when it opens in
    the `Directory` folder -- a plain listing, not the series grouping
    (that only happens once a specific file is picked and Multishot is on).
    """

    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Directory is required."}
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return {"ok": False, "message": f"Directory not found: {directory}"}

    suffixes_raw = payload.get("suffixes")
    if suffixes_raw:
        allowed_suffixes = {f".{str(item).strip().lstrip('.').lower()}" for item in suffixes_raw if str(item).strip()}
    else:
        allowed_suffixes = _IMAGE_SUFFIXES

    files = sorted(
        item.name
        for item in dir_path.iterdir()
        if item.is_file() and item.suffix.lower() in allowed_suffixes
    )
    return {"ok": True, "directory": directory, "files": files}


def console_load(payload: dict[str, Any]) -> dict[str, Any]:
    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    detector = str(payload.get("detector", "Eiger4M")).strip() or "Eiger4M"
    multishot = bool(payload.get("multishot", False))
    mode = str(payload.get("mode", "gixs")).strip().lower()
    if mode not in _MODES:
        mode = "gixs"
    if not directory or not filename:
        return {"ok": False, "message": "Choose a directory and a file to load."}

    file_path = Path(directory) / filename
    _SESSION.load_progress = None
    try:
        if multishot:
            data, counts, series_files = _load_series_average(file_path)
        else:
            result = load_image_file(file_path)
            data, counts, series_files = result.data, result.counts, [file_path]
        detector_mask = detector_mask_for_type(detector)
        masked = apply_detector_mask(data, detector_mask.mask)
        # Display/detector-axis convention fix: the raw H5 frame (after
        # `load_h5_frame`'s own transpose) still comes out rotated 90deg
        # relative to the real y/z detector axes for this beamline's data.
        # Transposing once here, before qmap/averaging ever see the array,
        # keeps every downstream consumer (q-map, click-to-pixel, canvas
        # render) consistent -- confirmed against a real Eiger4M frame.
        masked = np.ascontiguousarray(masked.T)
    except Exception as exc:  # noqa: BLE001 - console endpoint must return a visible failure.
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    _SESSION.mode = mode
    _SESSION.detector = detector
    _SESSION.data = masked
    _SESSION.mask = detector_mask.mask
    _SESSION.counts = np.asarray(counts, dtype=np.float32)
    _SESSION.source_path = str(file_path)
    _SESSION.calibration["pixel_mm"] = detector_mask.pixel_size_mm
    _reset_for_new_data()

    return {
        "ok": True,
        "path": str(file_path),
        "mode": mode,
        "detector": detector,
        "shape": list(masked.shape),
        "ic": _counter_value(_SESSION.counts, 0),
        "bc": _counter_value(_SESSION.counts, 1),
        "series_file_count": len(series_files) if multishot else None,
        "mask_report": {
            "fallback_used": detector_mask.report.fallback_used,
            "message": detector_mask.report.message,
        },
        "calibration": _json_calibration(),
        "preview": downsample_2d_for_json(masked, max_points=1024),
    }


def console_load_info(payload: dict[str, Any]) -> dict[str, Any]:
    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    if not directory or not filename:
        return {"ok": False, "message": "Choose a directory and a calibration file to load."}
    path = str(Path(directory) / filename)

    calibration, report = load_standard_calibration(path, expected_detector=_SESSION.detector)
    if calibration is None:
        return {"ok": False, "message": report.message}

    beam = np.asarray(calibration.beam_center, dtype=np.float64).reshape(-1)
    if beam.size >= 2:
        _SESSION.calibration["beam_pos"] = (float(beam[0]), float(beam[1]))
    if calibration.energy_kev is not None and np.isfinite(calibration.energy_kev):
        _SESSION.calibration["energy_kev"] = float(calibration.energy_kev)
    _SESSION.calibration["sdd_mm"] = float(calibration.sdd)
    _SESSION.calibration["pixel_mm"] = float(calibration.pixel_size)

    return {"ok": True, "path": path, "calibration": _json_calibration()}


def console_grid_remove(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001 - payload kept for endpoint symmetry.
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    # `grid_remove`'s Eiger4M pixel indices are calibrated against the raw
    # (pre-display-transpose) orientation, matching the batch workflow's
    # untransposed data. Un-transpose, correct, then re-transpose so the
    # console keeps its display orientation without touching those indices
    # (which would break the batch pathway that never transposes).
    corrected, report = grid_remove(_SESSION.data.T, _SESSION.detector)
    if not report.applied:
        return {"ok": False, "message": report.message or "Grid remove is not applicable."}
    corrected = np.ascontiguousarray(corrected.T)

    _SESSION.data = corrected
    _SESSION.grid_removed = True
    return {
        "ok": True,
        "message": report.message,
        "preview": downsample_2d_for_json(corrected, max_points=1024),
    }


def console_background_load(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    multishot = bool(payload.get("multishot", False))
    if not directory or not filename:
        return {"ok": False, "message": "Choose a directory and a file for the background."}

    file_path = Path(directory) / filename
    _SESSION.load_progress = None
    try:
        if multishot:
            data, counts, _series_files = _load_series_average(file_path)
        else:
            result = load_image_file(file_path)
            data, counts = result.data, result.counts
        masked = apply_detector_mask(data, _SESSION.mask)
        masked = np.ascontiguousarray(masked.T)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    if masked.shape != _SESSION.data.shape:
        return {
            "ok": False,
            "message": f"Background shape {masked.shape} does not match data shape {_SESSION.data.shape}.",
        }

    _SESSION.background = masked
    _SESSION.background_counts = np.asarray(counts, dtype=np.float32)
    return {"ok": True, "path": str(file_path)}


def console_normalize(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}
    if _SESSION.normalized:
        return {"ok": False, "message": "This data is already background-subtracted/normalized."}

    method = str(payload.get("method", "None"))
    plus_data = float(payload.get("plus_data", 1.0))
    confirm = bool(payload.get("confirm", False))

    preview, report = process_normalization(
        _SESSION.data,
        background=_SESSION.background,
        counts=_SESSION.counts,
        background_counts=_SESSION.background_counts,
        method=method,
        plus_data=plus_data,
    )
    if not report.success:
        return {"ok": False, "message": report.message}

    if not confirm:
        return {
            "ok": True,
            "requires_confirm": True,
            "message": "Preview only. Resend with confirm=true to keep this result.",
            "preview": downsample_2d_for_json(preview, max_points=1024),
        }

    _SESSION.data = preview
    _SESSION.normalized = True
    return {
        "ok": True,
        "message": f"Applied {report.method} normalization.",
        "preview": downsample_2d_for_json(preview, max_points=1024),
    }


def console_qmap(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    try:
        energy_kev = float(payload.get("energy_kev", _SESSION.calibration["energy_kev"]))
        sdd_mm = float(payload.get("sdd_mm", _SESSION.calibration["sdd_mm"]))
        beam_y = float(payload.get("beam_y", _SESSION.calibration["beam_pos"][0]))
        beam_z = float(payload.get("beam_z", _SESSION.calibration["beam_pos"][1]))
        incident_angle_deg = float(payload.get("incident_angle_deg", _SESSION.calibration["incident_angle_deg"]))
        footprint_mm = float(payload.get("footprint_mm", _SESSION.calibration.get("footprint_mm", 1.0)))
        apply_correction = bool(payload.get("apply_correction", True))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid calibration input: {exc}"}

    pixel_mm = float(_SESSION.calibration["pixel_mm"])
    wavelength = wavelength_from_energy_kev(energy_kev)
    if not np.isfinite(wavelength) or wavelength <= 0:
        return {"ok": False, "message": "Invalid energy for wavelength conversion."}
    k0 = float(2.0 * np.pi / wavelength)

    n_rows, n_cols = _SESSION.data.shape
    try:
        if _SESSION.mode == "trxs":
            qx, qy, qz, q = qmap_trxs(
                det_y=n_cols,
                det_z=n_rows,
                beam_pos=(beam_y, beam_z),
                pixel_mm=pixel_mm,
                sdd_mm=sdd_mm,
                k0=k0,
            )
            qmap = {"qx": qx, "qy": qy, "qz": qz, "q": q}
        else:
            qy, qr, qz, q, tth2d, af2d = qmap_gixs(
                det_y=n_cols,
                det_z=n_rows,
                beam_pos=(beam_y, beam_z),
                pixel_mm=pixel_mm,
                sdd_mm=sdd_mm,
                k0=k0,
                incident_angle_deg=incident_angle_deg,
            )
            qmap = {"qy": qy, "qr": qr, "qz": qz, "q": q, "tth2d": tth2d, "af2d": af2d}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    _SESSION.calibration.update(
        {
            "energy_kev": energy_kev,
            "sdd_mm": sdd_mm,
            "beam_pos": (beam_y, beam_z),
            "incident_angle_deg": incident_angle_deg,
            "footprint_mm": footprint_mm,
        }
    )
    _SESSION.qmap = qmap
    _SESSION.angle_map = pixel_angle_map(_SESSION.data.shape, center=(beam_y, beam_z))

    corrected_data = _SESSION.data
    if apply_correction:
        try:
            factor = detector_correction_factor(
                DetectorCorrectionInputs(
                    data_size=_SESSION.data.shape,
                    beam_position=(beam_y, beam_z),
                    pixel_size_mm=pixel_mm,
                    sdd_mm=sdd_mm,
                    q_map=q,
                    energy_kev=energy_kev,
                    detector_type=_SESSION.detector,
                ),
                k0=k0,
            )
            corrected_data = apply_detector_correction(_SESSION.data, factor)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"Q-map computed, but detector correction failed: {exc}"}
    _SESSION.corrected_data = corrected_data

    return {
        "ok": True,
        "calibration": _json_calibration(),
        "q_range": _finite_range(qmap["q"]),
        "qz_range": _finite_range(qmap["qz"]),
        "qr_range": _finite_range(qmap["qr"]) if "qr" in qmap else _finite_range(qmap["qx"]),
        "preview": downsample_2d_for_json(corrected_data, max_points=1024),
    }


def console_qspace_image(payload: dict[str, Any]) -> dict[str, Any]:
    """Forward-map the current image onto a uniform Q grid for display.

    Backs the console's "Map format" selector (`Qxy/Qz` / `Qy/Qz`) --
    ports `ImageTransformQ_gi.m` / `ImageTransformQ_tr.m`'s bilinear splat,
    unlike `console_qmap`, which only labels pixel-space data with q-range
    text and never actually regrids the displayed image.
    """

    if _SESSION.data is None or _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    mapping = str(payload.get("mapping", "Qxy/Qz")).strip()
    data_for_transform = _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    cal = _SESSION.calibration
    beam_pos = (float(cal["beam_pos"][0]), float(cal["beam_pos"][1]))
    pixel_mm = float(cal["pixel_mm"])
    wavelength_a = wavelength_from_energy_kev(float(cal["energy_kev"]))
    sdd_mm = float(cal["sdd_mm"])
    qmap = _SESSION.qmap

    if _SESSION.mode == "trxs":
        if "qy" not in qmap or "qz" not in qmap:
            return {"ok": False, "message": "Qy/Qz map is not available."}
        result = image_transform_q_tr(data_for_transform, qmap["qy"], qmap["qz"], beam_pos, pixel_mm, wavelength_a, sdd_mm)
    else:
        horizontal_map = qmap.get("qy") if mapping == "Qy/Qz" else qmap.get("qr")
        if horizontal_map is None or "qz" not in qmap:
            return {"ok": False, "message": f"{mapping} map is not available."}
        result = image_transform_q_gi(data_for_transform, horizontal_map, qmap["qz"], beam_pos, pixel_mm, wavelength_a, sdd_mm)

    return {
        "ok": True,
        "mapping": mapping,
        "horizontal_range": _finite_range(result.horizontal_axis),
        "vertical_range": _finite_range(result.vertical_axis),
        "preview": downsample_2d_for_json(result.image, max_points=1024),
    }


def console_line_1d(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.qmap is None or _SESSION.data is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    try:
        angle_deg = float(payload.get("angle_deg", 90.0))
        cakecut = bool(payload.get("cakecut", False))
        range_minus = float(payload.get("range_minus", -1.0))
        range_plus = float(payload.get("range_plus", 1.0))
        use_median = bool(payload.get("use_median", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid line parameters: {exc}"}

    angle_range = (range_minus, range_plus) if cakecut else (-1.0, 1.0)
    data_for_1d = _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    beam_pos = _SESSION.calibration["beam_pos"]
    angle_map = _SESSION.angle_map
    if angle_map is None or angle_map.shape != data_for_1d.shape:
        angle_map = pixel_angle_map(data_for_1d.shape, center=beam_pos)
        _SESSION.angle_map = angle_map

    result = cake_average(
        data_for_1d,
        _SESSION.qmap["q"],
        angle_map,
        center=beam_pos,
        reference_angle=angle_deg,
        angle_range=angle_range,
        use_median=use_median,
    )
    if not result.is_valid:
        return {"ok": False, "message": "No pixels selected for this line/cake-cut geometry."}

    _SESSION.last_curve = {
        "kind": "line",
        "q": result.q_axis,
        "intensity": result.intensity,
        "coverage": result.coverage,
    }
    return {
        "ok": True,
        "angle_deg": angle_deg,
        "angle_range": list(result.angle_range),
        "coverage_diagnostics": _coverage_summary(result.coverage),
        "preview": {
            "q": downsample_1d_for_json(result.q_axis, max_points=600),
            "intensity": downsample_1d_for_json(result.intensity, max_points=600),
            "coverage": downsample_1d_for_json(result.coverage, max_points=600),
        },
    }


def console_azimuthal_1d(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.qmap is None or _SESSION.data is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    try:
        pixel = float(payload.get("pixel", 202.0))
        range_minus = float(payload.get("range_minus", -1.0))
        range_plus = float(payload.get("range_plus", 1.0))
        use_median = str(payload.get("mean_or_median", "Mean")).strip().lower() == "median"
        n_angle_bins = int(payload.get("n_angle_bins", 180))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid azimuthal parameters: {exc}"}

    data_for_1d = _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    beam_pos = _SESSION.calibration["beam_pos"]
    radius_range = (pixel + range_minus, pixel + range_plus)

    result = angular_profile(
        data_for_1d,
        center=beam_pos,
        radius_range=radius_range,
        n_angle_bins=n_angle_bins,
        use_median=use_median,
    )
    _SESSION.last_curve = {
        "kind": "azimuthal",
        "angle": result.angle_axis,
        "intensity": result.intensity,
        "coverage": result.coverage,
    }
    return {
        "ok": True,
        "radius_range": list(result.radius_range),
        "coverage_diagnostics": _coverage_summary(result.coverage),
        "preview": {
            "angle": downsample_1d_for_json(result.angle_axis, max_points=600),
            "intensity": downsample_1d_for_json(result.intensity, max_points=600),
            "coverage": downsample_1d_for_json(result.coverage, max_points=600),
        },
    }


def console_circular_average(payload: dict[str, Any]) -> dict[str, Any]:
    """TrXS `CircularAve_Callback`: q-binned average over an angle range.

    Reuses `averaging.angular.sector_average` (a full 0..360 range collapses
    to the same full-circle mask `make_sector_mask` already returns for
    matching start/end angles), then nulls out bins below `min_coverage`,
    matching MATLAB's coverage-threshold behavior.
    """

    if _SESSION.qmap is None or _SESSION.data is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    try:
        angle_min = float(payload.get("angle_min", 0.0))
        angle_max = float(payload.get("angle_max", 360.0))
        min_coverage = float(payload.get("min_coverage", 0.80))
        use_median = bool(payload.get("use_median", False))
        calc_std = bool(payload.get("calc_std", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid circular average parameters: {exc}"}

    data_for_1d = _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    beam_pos = _SESSION.calibration["beam_pos"]

    result = sector_average(
        data_for_1d,
        _SESSION.qmap["q"],
        center=beam_pos,
        angle_range=(angle_min, angle_max),
        use_median=use_median,
        calc_std=calc_std,
    )
    if not result.is_valid:
        return {"ok": False, "message": "No pixels selected for this angular range."}

    intensity = np.asarray(result.intensity, dtype=np.float32).copy()
    coverage = np.asarray(result.coverage, dtype=np.float32)
    below = np.isfinite(coverage) & (coverage < min_coverage)
    intensity[below] = np.nan

    _SESSION.last_curve = {
        "kind": "line",
        "q": result.q_axis,
        "intensity": intensity,
        "coverage": coverage,
    }
    return {
        "ok": True,
        "angle_range": list(result.angle_range),
        "min_coverage": min_coverage,
        "coverage_diagnostics": _coverage_summary(coverage),
        "preview": {
            "q": downsample_1d_for_json(result.q_axis, max_points=600),
            "intensity": downsample_1d_for_json(intensity, max_points=600),
            "coverage": downsample_1d_for_json(coverage, max_points=600),
        },
    }


def console_save(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}
    if _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Map Cal. before saving."}

    out_dir = str(payload.get("out_dir", "")).strip()
    stem = _output_stem(payload)
    target_dir = Path(out_dir) if out_dir else (Path(_SESSION.source_path).parent if _SESSION.source_path else Path("console_outputs"))
    target_dir.mkdir(parents=True, exist_ok=True)
    h5_path = target_dir / f"{stem}_2Dinfo.h5"

    cal = _SESSION.calibration
    is_trxs = _SESSION.mode == "trxs"
    info = TwoDInfo(
        measurement_mode="TrXS" if is_trxs else "GIXS",
        data2d_pix=_SESSION.data,
        data2d_q=_SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data,
        q_map=_SESSION.qmap["q"],
        qz_map=_SESSION.qmap["qz"],
        qz_axis=np.array([], dtype=np.float32),
        beam_position=np.asarray(cal["beam_pos"], dtype=np.float32),
        wavelength=wavelength_from_energy_kev(cal["energy_kev"]),
        sdd=float(cal["sdd_mm"]),
        pixel=float(cal["pixel_mm"]),
        qx_map=_SESSION.qmap.get("qx"),
        qy_map=_SESSION.qmap.get("qy"),
        qr_map=_SESSION.qmap.get("qr"),
        energy_kev=float(cal["energy_kev"]),
        detector=_SESSION.detector,
        incident_angle=None if is_trxs else float(cal["incident_angle_deg"]),
        source_files=[_SESSION.source_path] if _SESSION.source_path else [],
    )
    try:
        save_two_d_info_h5(h5_path, info)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    curve_path: Path | None = None
    if _SESSION.last_curve is not None:
        curve = _SESSION.last_curve
        x_axis = curve["q"] if curve["kind"] == "line" else curve["angle"]
        export = OneDExport(
            x=x_axis,
            intensity=curve["intensity"],
            coverage=curve.get("coverage"),
            x_label="q" if curve["kind"] == "line" else "angle_deg",
        )
        curve_path = target_dir / f"{stem}_1D.csv"
        try:
            save_1d_data(curve_path, export, include_diagnostics=True)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"2D-info saved, but 1D export failed: {exc}", "h5_path": str(h5_path)}

    return {
        "ok": True,
        "h5_path": str(h5_path),
        "curve_path": str(curve_path) if curve_path is not None else None,
    }


def _reset_for_new_data() -> None:
    _SESSION.qmap = None
    _SESSION.corrected_data = None
    _SESSION.normalized = False
    _SESSION.grid_removed = False
    _SESSION.angle_map = None
    _SESSION.last_curve = None
    _SESSION.background = None
    _SESSION.background_counts = None


_SERIES_SUFFIX_RE = re.compile(r"^(.*)_\d{4,}$")


def _normalize_series_prefix(filename: str) -> str:
    """Port of MATLAB `normalizePGIXSSeriesPrefix.m`.

    Strips a trailing `_NNNN` (4+ digit) series index off the file stem,
    e.g. `HDH_155_0001.h5` -> `HDH_155`. Files with no such suffix use
    their own stem as the "prefix" (a single-file series of one).
    """

    stem = Path(filename).stem
    match = _SERIES_SUFFIX_RE.match(stem)
    return match.group(1) if match else stem


def _list_series_files(directory: Path, filename: str) -> list[Path]:
    """Port of MATLAB `listPGIXSSeriesFiles.m`: same prefix, same extension,
    non-recursive, sorted by name."""

    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    prefix = _normalize_series_prefix(filename)
    suffix = Path(filename).suffix.lower()
    matches = [
        item
        for item in directory.iterdir()
        if item.is_file() and suffix == item.suffix.lower() and prefix in item.stem
    ]
    return sorted(matches, key=lambda item: item.name)


def _load_series_average(file_path: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Multishot load: average every file in `file_path`'s series (same
    directory, same `_NNNN`-stripped prefix, same extension as `file_path`),
    matching MATLAB `loadPGIXSMultishotAverage.m`. Updates
    `_SESSION.load_progress` per file, matching MATLAB's `uiprogressdlg`
    callback (`@(ii, n) set(d, 'Value', ii / n)`)."""

    files = _list_series_files(file_path.parent, file_path.name)
    if not files:
        raise ValueError(f'No files found for prefix "{_normalize_series_prefix(file_path.name)}" next to {file_path}.')

    stack = []
    counts_list = []
    total = len(files)
    for index, series_path in enumerate(files, start=1):
        _SESSION.load_progress = {"current": index, "total": total, "label": series_path.name}
        result = load_image_file(series_path)
        stack.append(result.data)
        if result.counts.size:
            counts_list.append(result.counts)

    averaged = np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)
    counts = (
        np.nanmean(np.stack(counts_list, axis=0), axis=0).astype(np.float32)
        if counts_list
        else np.asarray([], dtype=np.float32)
    )
    return averaged, counts, files


def _output_stem(payload: dict[str, Any]) -> str:
    hint = str(payload.get("file_name", "")).strip()
    if hint:
        return Path(hint).stem
    if _SESSION.source_path:
        return Path(_SESSION.source_path).stem
    return "console"


def _counter_value(counts: np.ndarray | None, index: int) -> float | None:
    if counts is None or counts.size <= index:
        return None
    value = float(counts[index])
    return value if np.isfinite(value) else None


def _json_calibration() -> dict[str, Any]:
    cal = _SESSION.calibration
    beam = cal.get("beam_pos", (0.0, 0.0))
    return {
        "energy_kev": float(cal.get("energy_kev", 0.0)),
        "sdd_mm": float(cal.get("sdd_mm", 0.0)),
        "beam_pos": [float(beam[0]), float(beam[1])],
        "pixel_mm": float(cal.get("pixel_mm", 0.0)),
        "incident_angle_deg": float(cal.get("incident_angle_deg", 0.0)),
        "footprint_mm": float(cal.get("footprint_mm", 0.0)),
    }


def _finite_range(values: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"min": None, "max": None}
    return {"min": float(np.nanmin(values[finite])), "max": float(np.nanmax(values[finite]))}


def _coverage_summary(coverage: np.ndarray) -> dict[str, Any]:
    summary = summarize_coverage(np.asarray(coverage, dtype=np.float32), low_threshold=0.15)
    return {
        "coverage_min": _json_float(summary["coverage_min"]),
        "coverage_median": _json_float(summary["coverage_median"]),
        "low_bin_count": int(np.asarray(summary["coverage_low_bins"]).size),
        "drop_bin_count": int(np.asarray(summary["coverage_drop_bins"]).size),
    }


def _json_float(value: object) -> float | None:
    number = float(value)
    return None if not np.isfinite(number) else number


__all__ = [
    "ConsoleSession",
    "console_azimuthal_1d",
    "console_background_load",
    "console_circular_average",
    "console_grid_remove",
    "console_line_1d",
    "console_list_files",
    "console_load",
    "console_load_info",
    "console_load_progress",
    "console_normalize",
    "console_qmap",
    "console_qspace_image",
    "console_save",
    "console_state",
    "get_session",
    "reset_session",
    "set_session",
]
