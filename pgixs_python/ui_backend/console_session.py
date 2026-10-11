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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.averaging.angular import angular_profile, sector_average
from pgixs_python.averaging.cake import accumulation_average, cake_average, pixel_angle_map
from pgixs_python.averaging.qbin import summarize_coverage
from pgixs_python.calibration.calibration_io import load_standard_calibration
from pgixs_python.calibration.sdd import wavelength_from_energy_kev
from pgixs_python.export.data_export import OneDExport, save_1d_data
from pgixs_python.io.h5_series import natural_sort_key
from pgixs_python.io.image_loader import load_image_file
from pgixs_python.io.two_d_info import TwoDInfo, save_two_d_info_h5
from pgixs_python.processing.corrections import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    detector_correction_factor,
)
from pgixs_python.processing.grid_remove import grid_remove
from pgixs_python.processing.masking import apply_detector_mask, detector_mask_for_type, dilate_gap_mask
from pgixs_python.processing.normalization import process_normalization
from pgixs_python.qmap.gixs import qmap_gixs
from pgixs_python.qmap.transform import (
    image_transform_q_gi,
    image_transform_q_tr,
    pixel_point_from_q_space,
    polar_transform_q_grid,
    q_space_point,
)
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
    # Detector gap/dead pixels grown by 3 px (`dilate_gap_mask`). Pixels
    # bordering a module gap can carry biased counts, so every 1D reduction
    # drops that shoulder -- the 2D image keeps them, matching MATLAB, which
    # applies this only inside its averaging paths.
    gap_buffer: np.ndarray | None = None
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
    # Full-resolution BKG sub./Norm. preview (console_normalize's own
    # `confirm=False` call) -- the JSON `preview` field in that response
    # is downsampled for the web UI's transport, which the desktop widget
    # never needed for any *other* render (it always re-reads full data
    # from this session instead); stashing it here lets the widget show
    # the real result on the 2D Image Plot Viewer *before* the keep/
    # discard confirmation dialog, instead of the dialog's text alone
    # (reported: "메시지만 나오고 subtracted 또는 normalized image가
    # 어떤지 나오지 않아"). Cleared once confirmed (the result moves into
    # `data` itself) or discarded.
    pending_normalize_preview: np.ndarray | None = None
    normalized: bool = False
    grid_removed: bool = False
    # Set True only by a *successful* console_load_info -- console_qmap
    # refuses to run without it, since `calibration` above always holds
    # plausible-looking placeholder values even before Load Info ever runs,
    # and silently q-Cal'ing against those produces a wrong-but-not-
    # obviously-wrong q-map (reported: "Load Info에 파일이 들어가 있지
    # 않으면, q-cal이 되지 않도록 해줘").
    calibration_loaded: bool = False
    last_curve: dict[str, Any] | None = None
    load_progress: dict[str, Any] | None = None
    # User-drawn ROI mask for Circular Process (True = keep, False =
    # exclude) -- distinct from `mask` above, which is the fixed detector
    # gap/hot-pixel mask. Mirrors TrXS_gui.m's CircularAve_Callback
    # `selectROIMask`/`BW`.
    circular_mask: np.ndarray | None = None


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
    # Transposed to match `data`/`stack`, which the loaders store in the
    # mask's transpose (display orientation); the shapes differ (2070x2167
    # vs 2167x2070 on an Eiger4M), so a mismatch would silently no-op.
    _SESSION.gap_buffer = (
        np.ascontiguousarray(dilate_gap_mask(detector_mask.mask, 3).T)
        if detector_mask.mask is not None
        else None
    )
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
    _SESSION.calibration_loaded = True
    message = ""
    if _SESSION.qmap is not None:
        # Cuts read the new beam center against the old q-map otherwise.
        _SESSION.qmap = None
        _SESSION.angle_map = None
        _SESSION.corrected_data = None
        _SESSION.last_curve = None
        message = "New calibration loaded: run q-Map Cal. again."

    return {"ok": True, "path": path, "calibration": _json_calibration(), "message": message}


def console_grid_remove(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001 - payload kept for endpoint symmetry.
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    # `grid_remove` takes the `load_h5_frame` orientation (what the batch
    # workflow passes), so undo the console's display transpose, correct,
    # then re-transpose.
    corrected, report = grid_remove(_SESSION.data.T, _SESSION.detector)
    if not report.applied:
        return {"ok": False, "message": report.message or "Grid remove is not applicable."}
    corrected = np.ascontiguousarray(corrected.T)

    _SESSION.data = corrected
    _SESSION.grid_removed = True
    # Same staleness trap BKG sub. & Norm. hit (see console_normalize): every
    # render and every 1D cut prefers `corrected_data` when it is set, so
    # leaving the pre-grid-removal copy in place would make Grid Remove run
    # after q-Map Cal. a silent no-op -- unchanged image, unchanged curves.
    # MATLAB has no such second array (its correction overwrites `param.data`
    # in place), so this only needs guarding here. Drop the stale copy and let
    # the widget flag q-Map Cal. as needing a re-run.
    _SESSION.corrected_data = None
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
        _SESSION.pending_normalize_preview = preview
        return {
            "ok": True,
            "requires_confirm": True,
            "message": "Preview only. Resend with confirm=true to keep this result.",
            "preview": downsample_2d_for_json(preview, max_points=1024),
        }

    _SESSION.data = preview
    _SESSION.normalized = True
    _SESSION.pending_normalize_preview = None
    # `corrected_data` (if q-Map Cal. already ran) was built from the
    # *pre*-normalization `data` -- the widget's own render always prefers
    # `corrected_data` over `data` when it's set, so leaving it as-is here
    # made BKG sub. & Norm's confirmation dialog appear to do nothing at
    # all (reported: "BKGsub. & Norm을 누르면 바뀐 이미지를 유지하겠냐고
    # 나오는데, 사실 이미지가 바뀌지 않고 메시지 박스만 나와"). The
    # detector-correction factor itself only depends on geometry, not
    # data, so it's still valid -- but this port never re-caches it
    # outside q-Map Cal. to reapply here, so the simplest correct fix is
    # to drop the stale corrected copy (the render then falls back to the
    # now-current `data`) and let the qmap_button's own "needs re-run"
    # indicator (see console_widget.py's _on_subtract_clicked) prompt the
    # user to redo q-Map Cal. for a freshly-corrected image.
    _SESSION.corrected_data = None
    return {
        "ok": True,
        "message": f"Applied {report.method} normalization.",
        "preview": downsample_2d_for_json(preview, max_points=1024),
    }


def console_qmap(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}
    if not _SESSION.calibration_loaded:
        return {"ok": False, "message": "Run Load Info first."}

    try:
        energy_kev = float(payload.get("energy_kev", _SESSION.calibration["energy_kev"]))
        sdd_mm = float(payload.get("sdd_mm", _SESSION.calibration["sdd_mm"]))
        beam_y = float(payload.get("beam_y", _SESSION.calibration["beam_pos"][0]))
        beam_z = float(payload.get("beam_z", _SESSION.calibration["beam_pos"][1]))
        incident_angle_deg = float(payload.get("incident_angle_deg", _SESSION.calibration["incident_angle_deg"]))
        footprint_mm = float(payload.get("footprint_mm", _SESSION.calibration.get("footprint_mm", 1.0)))
        q_uncertainty = bool(payload.get("q_uncertainty", False))
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
            if q_uncertainty and footprint_mm > 0:
                # Port of GIXS_gui.m's calculateGIXSQUncertainty: a finite
                # illuminated footprint on the sample means photons
                # scattered from different points along it travel
                # different effective path lengths to the detector, so
                # it's treated as a +/-FP/2 spread in SDD -- recompute the
                # Q maps at each SDD extreme and derive an approximate
                # 1-sigma q uncertainty from the resulting range. `q`
                # above is already hypot(qr, qz) -- exactly MATLAB's
                # "Qxy/Qz" case -- so no separate combination step is
                # needed to match it.
                #
                # Divisor: MATLAB's own convention divides by 4 (the
                # min<->max range treated as ~4 sigma, i.e. a roughly
                # Gaussian illumination profile). For typical GIXS
                # conditions -- sample thickness of order hundreds of nm
                # or less, a ~60um vertical beam -- the illuminated
                # footprint is effectively uniform across its length
                # regardless of incidence angle (confirmed: "샘플 두께는
                # 약 수백 나노 또는 그이하 사이즈이고, 빔사이즈는 vertical로만
                # 60um정도 이기때문에... 거의 uniform하게 들어오지"), so a
                # uniform distribution's actual std (range / sqrt(12)) is
                # used here instead of MATLAB's Gaussian-shaped /4 --
                # about 13% larger than MATLAB's own number for the same
                # geometry.
                sdd_lo = max(sdd_mm - footprint_mm / 2.0, 1e-3)
                sdd_hi = sdd_mm + footprint_mm / 2.0
                qy_lo, qr_lo, qz_lo, q_lo, _, _ = qmap_gixs(
                    det_y=n_cols, det_z=n_rows, beam_pos=(beam_y, beam_z),
                    pixel_mm=pixel_mm, sdd_mm=sdd_lo, k0=k0, incident_angle_deg=incident_angle_deg,
                )
                qy_hi, qr_hi, qz_hi, q_hi, _, _ = qmap_gixs(
                    det_y=n_cols, det_z=n_rows, beam_pos=(beam_y, beam_z),
                    pixel_mm=pixel_mm, sdd_mm=sdd_hi, k0=k0, incident_angle_deg=incident_angle_deg,
                )
                uniform_divisor = np.sqrt(12.0)
                qmap["q_std"] = np.abs(q_hi - q_lo) / uniform_divisor
                qmap["qr_std"] = np.abs(qr_hi - qr_lo) / uniform_divisor
                qmap["qz_std"] = np.abs(qz_hi - qz_lo) / uniform_divisor
                qmap["qy_std"] = np.abs(qy_hi - qy_lo) / uniform_divisor
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


def _q_space_scale_inputs() -> tuple[float, float, float] | None:
    """(pixel_mm, wavelength_a, sdd_mm) for `q_space_point`/
    `pixel_point_from_q_space`, or None if the calibration can't produce
    a valid wavelength/scale."""

    cal = _SESSION.calibration
    wavelength_a = wavelength_from_energy_kev(float(cal["energy_kev"]))
    sdd_mm = float(cal["sdd_mm"])
    if not np.isfinite(wavelength_a) or wavelength_a <= 0 or sdd_mm <= 0:
        return None
    return float(cal["pixel_mm"]), wavelength_a, sdd_mm


def console_q_space_point(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert one pixel (col, row) point into the uniform Q-space (qx,
    qy) coordinate the console's Qxy/Qz or Qy/Qz display and cut geometry
    actually uses (see `q_space_point`'s own docstring) -- used to draw a
    cut's overlay as a straight line in Q-space directly, instead of
    sampling the true per-pixel Q maps point by point, which curves what
    is, in this uniform grid, actually a straight cut. Reported: "Qxy/Qz
    맵에서 90도 line cut을... 해보면 일직선 라인이 아니라 휘어져 나와...
    그냥 일직선 라인이 되면 좋을 것 같아.\""""

    if _SESSION.data is None or _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}
    scale_inputs = _q_space_scale_inputs()
    if scale_inputs is None:
        return {"ok": False, "message": "Invalid energy/SDD for Q-space scale."}
    try:
        col = float(payload["col"])
        row = float(payload["row"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid pixel point: {exc}"}
    beam_pos = (float(_SESSION.calibration["beam_pos"][0]), float(_SESSION.calibration["beam_pos"][1]))
    qx, qy = q_space_point(col, row, beam_pos, *scale_inputs)
    return {"ok": True, "qx": qx, "qy": qy}


def console_pixel_from_q_space(payload: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `console_q_space_point` -- a Q-space (qx, qy) click back
    to the (col, row) pixel point it was regridded from, exactly (this
    uniform grid's axes are each linear in pixel position, unlike the
    true per-pixel Q maps a click used to search)."""

    if _SESSION.data is None or _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}
    scale_inputs = _q_space_scale_inputs()
    if scale_inputs is None:
        return {"ok": False, "message": "Invalid energy/SDD for Q-space scale."}
    try:
        qx = float(payload["qx"])
        qy = float(payload["qy"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid Q-space point: {exc}"}
    beam_pos = (float(_SESSION.calibration["beam_pos"][0]), float(_SESSION.calibration["beam_pos"][1]))
    col, row = pixel_point_from_q_space(qx, qy, beam_pos, *scale_inputs)
    return {"ok": True, "col": col, "row": row}


def console_polar_image(payload: dict[str, Any]) -> dict[str, Any]:
    """Polar (angle vs |Q|) remap of the current Cartesian Q image,
    centered at q=0 (the beam) -- the "Polar" toggle next to Map format.
    Switching back to a plain Qxy/Qz or Qy/Qz view (console_qspace_image)
    is how the Cartesian view is "recovered"; this function never mutates
    session state, so nothing needs undoing."""

    if _SESSION.data is None or _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    mapping = str(payload.get("mapping", "Qxy/Qz")).strip()
    data_for_transform = _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    qmap = _SESSION.qmap
    horizontal_map = qmap.get("qy") if mapping == "Qy/Qz" else qmap.get("qr")
    if horizontal_map is None or "qz" not in qmap:
        return {"ok": False, "message": f"{mapping} map is not available."}

    # Same reasoning as console_azimuthal_1d: GIXS's grazing-incidence
    # geometry only ever scatters above the horizon, so restrict the polar
    # sweep to a half-circle; TrXS keeps the full ring.
    angle_range = (0.0, 180.0) if _SESSION.mode == "gixs" else (0.0, 360.0)
    result = polar_transform_q_grid(data_for_transform, horizontal_map, qmap["qz"], angle_range=angle_range)

    return {
        "ok": True,
        "mapping": mapping,
        "horizontal_range": _finite_range(result.horizontal_axis),
        "vertical_range": _finite_range(result.vertical_axis),
        "preview": downsample_2d_for_json(result.image, max_points=1024),
    }


def _q_space_cut_inputs(mapping: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the (regridded intensity, Q-magnitude, horizontal component,
    vertical component) tuple `console_line_1d` swaps in for `data`/`q_map`
    when Map format isn't "Pixel". The two component maps label an *offset*
    axis-aligned cut, which runs at a fixed nonzero Qz (or Qxy) and so is
    labelled by the coordinate that actually varies along it, not by |Q|
    (`Anglecut1d_gi.m`'s `calcQ_Horizontal`/`calcQ_Vertical`) -- port of
    GIXS_gui.m's `Anglecut1d_gi.m` Mapping branch (`data2d = param.Itrans`
    for Qxy/Qz and Qy/Qz), so a line/accumulation/cake cut done in
    Qxy/Qz (or Qy/Qz) mode runs against the actual Q-remapped image
    instead of silently reusing the pixel-space cut (reported: "1D
    process해서 pixel map에서 cutting하는 것이 Qxy/Qz cutting에
    반영되서 진행되는데, 그렇게 하지말고 독립적으로 되면 좋겠어").

    `image_transform_q_gi`/`image_transform_q_tr`'s own output grid is
    index-aligned 1:1 with the pixel grid -- its axes are `d_q * (index -
    beam_pos)`, a single uniform scale factor applied identically to both
    the horizontal and vertical axis, not a per-pixel nonlinear remap.
    That means `console_line_1d`'s existing `beam_pos` and (angle-
    invariant under a uniform scale) `angle_map` are *already* correct
    for this regridded image unchanged -- only the array of values being
    cut (Itrans instead of raw counts) and the Q label attached to each
    cut sample (this grid's own true Q magnitude, computed here in closed
    form, instead of the true per-pixel Q) actually differ. This is what
    makes a "Qxy/Qz cut" a genuinely independent cut through the
    Q-remapped image rather than the pixel-space cut just relabeled, and
    it lets `cake_average`/`accumulation_average` be reused completely
    generically -- no need to port MATLAB's own separate per-direction
    `calcQ_Horizontal`/`calcQ_Vertical`/`calcQ_Diagonal` Q-labeling logic,
    since this grid's Q value at any index is already known exactly.

    The gap shoulder is blanked *before* regridding, as Pixel-mode cuts and
    GIXS-series' "Reflect missing wedge" do: once on the Q grid the
    detector-pixel mask no longer maps onto it, and bilinear interpolation
    would otherwise spread the shoulder's edge bias into the cut. MATLAB
    does the same since v2.2.0 (`buildPGIXSCutImage` -> `ItransCut`);
    before that it stamped the detector-pixel mask straight onto Itrans."""

    data_for_transform = _values_for_reduction(
        _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    )
    cal = _SESSION.calibration
    beam_pos = (float(cal["beam_pos"][0]), float(cal["beam_pos"][1]))
    pixel_mm = float(cal["pixel_mm"])
    wavelength_a = wavelength_from_energy_kev(float(cal["energy_kev"]))
    sdd_mm = float(cal["sdd_mm"])
    qmap = _SESSION.qmap

    if _SESSION.mode == "trxs":
        if "qy" not in qmap or "qz" not in qmap:
            raise ValueError("Qy/Qz map is not available.")
        result = image_transform_q_tr(data_for_transform, qmap["qy"], qmap["qz"], beam_pos, pixel_mm, wavelength_a, sdd_mm)
        # `result.horizontal_axis`/`vertical_axis` are `image_transform_q_tr`'s
        # own small-angle-linear (`d_q * pixel_index`) approximation --
        # accurate near the beam but off by ~8% at a 200mm SDD's detector
        # corner (confirmed against qmap_trxs's real geometry). TrXS's
        # flat-detector, normal-incidence transmission makes the true |Q|
        # exactly a function of radial *pixel* distance from the beam
        # alone (2theta = atan(R_px*pixel_mm/sdd_mm), q = 2*k0*sin(theta)
        # -- confirmed azimuthally symmetric to <0.5%), so correcting just
        # this *label* -- not the regridded image, or anything cutting/
        # clicking/drawing against it -- makes every cut sample's reported
        # Q exact everywhere, including the corners, with nothing else
        # changed (reported: "실제 Qmap이... 광각으로 갈수록 벌어져...
        # 코너 쪽만 정밀했으면 좋겠어, 전체적으로는 지금도 충분해").
        d_q = (2.0 * np.pi * pixel_mm) / (wavelength_a * sdd_mm)
        radius_px = np.hypot(result.horizontal_axis[np.newaxis, :], result.vertical_axis[:, np.newaxis]) / d_q
        two_theta = np.arctan(radius_px * pixel_mm / sdd_mm)
        k0 = 2.0 * np.pi / wavelength_a
        q_magnitude = (2.0 * k0 * np.sin(two_theta / 2.0)).astype(np.float32)
        return result.image, q_magnitude, *_axis_component_maps(result)

    horizontal_map = qmap.get("qy") if mapping == "Qy/Qz" else qmap.get("qr")
    if horizontal_map is None or "qz" not in qmap:
        raise ValueError(f"{mapping} map is not available.")
    result = image_transform_q_gi(data_for_transform, horizontal_map, qmap["qz"], beam_pos, pixel_mm, wavelength_a, sdd_mm)
    q_magnitude = np.hypot(result.horizontal_axis[np.newaxis, :], result.vertical_axis[:, np.newaxis]).astype(np.float32)
    return result.image, q_magnitude, *_axis_component_maps(result)


def _axis_component_maps(result) -> tuple[np.ndarray, np.ndarray]:
    """The regridded image's own horizontal/vertical Q axes broadcast to full
    2D maps, matching MATLAB's `meshgrid(param.qr, param.qz)` in the offset
    branches of `Anglecut1d_gi.m`."""

    shape = result.image.shape
    horizontal = np.broadcast_to(result.horizontal_axis[np.newaxis, :], shape).astype(np.float32)
    vertical = np.broadcast_to(result.vertical_axis[:, np.newaxis], shape).astype(np.float32)
    return horizontal, vertical


def _values_for_reduction(values: np.ndarray) -> np.ndarray:
    """Blank the gap shoulder before a 1D reduction.

    The pixels themselves stay in the geometry, so they still count against
    `coverage` -- a bin sitting mostly on a module edge should read as poorly
    sampled, not as fully sampled from fewer pixels."""

    buffer_mask = _SESSION.gap_buffer
    array = np.asarray(values)
    if buffer_mask is None or buffer_mask.shape != array.shape:
        return array
    out = array.astype(np.float32, copy=True)
    out[buffer_mask] = np.nan
    return out


def console_line_1d(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.qmap is None or _SESSION.data is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    try:
        angle_deg = float(payload.get("angle_deg", 90.0))
        cakecut = bool(payload.get("cakecut", False))
        range_minus = float(payload.get("range_minus", -1.0))
        range_plus = float(payload.get("range_plus", 1.0))
        use_median = bool(payload.get("use_median", False))
        y_pos_move = float(payload.get("y_pos_move", 0.0))
        z_pos_move = float(payload.get("z_pos_move", 0.0))
        accumulation = bool(payload.get("accumulation", False))
        accum_minus = float(payload.get("accum_minus", -1.0))
        accum_plus = float(payload.get("accum_plus", 1.0))
        mapping = str(payload.get("mapping", "Pixel")).strip()
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid line parameters: {exc}"}

    if accumulation and cakecut:
        return {"ok": False, "message": "Turn off Cakecut before using Accumulation."}

    if mapping == "Pixel":
        data_for_1d = _values_for_reduction(
            _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
        )
        q_for_1d = _SESSION.qmap["q"]
        # GIXS keeps the signed in-plane component as "qr"; TrXS names its
        # equivalent "qy". Either way these label an offset cut (below).
        horizontal_component = _SESSION.qmap.get("qr")
        if horizontal_component is None:
            horizontal_component = _SESSION.qmap.get("qy")
        vertical_component = _SESSION.qmap.get("qz")
    else:
        try:
            data_for_1d, q_for_1d, horizontal_component, vertical_component = _q_space_cut_inputs(mapping)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}
    beam_pos = _SESSION.calibration["beam_pos"]
    angle_map = _SESSION.angle_map
    if angle_map is None or angle_map.shape != data_for_1d.shape:
        angle_map = pixel_angle_map(data_for_1d.shape, center=beam_pos)
        _SESSION.angle_map = angle_map

    # Three cut modes, matching GIXS_gui.m's calculateLinePositions: a plain
    # angle cut (default -- narrow wedge from the beam center, or an
    # axis-aligned offset ray once y/z-pos move is nonzero), an angular
    # Cakecut wedge (never affected by y/z-pos move, matching MATLAB), and
    # a perpendicular-band Accumulation cut (axis-aligned only, offset by
    # y/z-pos move). Factored into a closure over the resolved geometry so
    # the exact same cut can be replayed against a *second* array below
    # (the per-pixel q-uncertainty map, if q-Map Cal. computed one) without
    # duplicating the branch logic -- same geometry, different "data".
    axis_aligned = angle_deg in (0.0, 90.0, 180.0)

    # An offset axis-aligned cut sits at a fixed nonzero Qz (horizontal cut)
    # or Qxy (vertical cut), so |Q| along it is inflated by that constant
    # perpendicular component and is not the coordinate the cut scans.
    # MATLAB switches the axis to the varying component in exactly this case
    # (`Anglecut1d_gi.m`'s `calcQ_Horizontal`/`calcQ_Vertical`: `Qdata = Qr`
    # when `zposmv ~= 0`, `Qdata = Qz` when `yposmv ~= 0`); use its magnitude,
    # since the component is signed and every downstream binning step here
    # requires q > 0. Accumulation is deliberately excluded -- MATLAB's
    # `Accumpix_gi.m` labels its band with |Q| regardless of the offset.
    offset_component = horizontal_component if angle_deg in (0.0, 180.0) else vertical_component
    offset_cut_q_map = q_for_1d if offset_component is None else np.abs(offset_component)
    # Whether the cut that actually runs below is the offset one, i.e. whether
    # the plotted axis is a single Q component rather than |Q|. The
    # q-uncertainty map sampled further down has to match that choice.
    uses_offset_axis = (
        not accumulation
        and not cakecut
        and axis_aligned
        and (y_pos_move != 0.0 or z_pos_move != 0.0)
        and offset_component is not None
    )

    def _run_cut(values: np.ndarray):
        if accumulation:
            if not axis_aligned:
                raise ValueError("Accumulation cut only supports angle 0, 90, or 180 degrees.")
            perp_offset = z_pos_move if angle_deg in (0.0, 180.0) else y_pos_move
            return accumulation_average(
                values,
                q_for_1d,
                center=beam_pos,
                angle_deg=angle_deg,
                perp_offset=perp_offset,
                band_minus=accum_minus,
                band_plus=accum_plus,
                use_median=use_median,
            )
        if cakecut:
            return cake_average(
                values,
                q_for_1d,
                angle_map,
                center=beam_pos,
                reference_angle=angle_deg,
                angle_range=(range_minus, range_plus),
                use_median=use_median,
            )
        if axis_aligned and (y_pos_move != 0.0 or z_pos_move != 0.0):
            perp_offset = z_pos_move if angle_deg in (0.0, 180.0) else y_pos_move
            return accumulation_average(
                values,
                offset_cut_q_map,
                center=beam_pos,
                angle_deg=angle_deg,
                perp_offset=perp_offset,
                band_minus=0.0,
                band_plus=0.0,
                use_median=use_median,
            )
        return cake_average(
            values,
            q_for_1d,
            angle_map,
            center=beam_pos,
            reference_angle=angle_deg,
            angle_range=(-1.0, 1.0),
            use_median=use_median,
        )

    try:
        result = _run_cut(data_for_1d)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    if not result.is_valid:
        return {"ok": False, "message": "No pixels selected for this line/cake-cut geometry."}

    # GIXS's "q uncertainty" checkbox (see console_qmap) -- only meaningful
    # against the true per-pixel map, so Q-space (Qxy/Qz-independent) cuts
    # don't carry it. Port of GIXS_gui.m's Anglecut1d_giplus: the same cut
    # geometry is replayed against the per-pixel q_std map to get a q_std
    # value per output bin, right alongside the intensity value.
    #
    # Cakecut is excluded outright (not just warned about, unlike MATLAB's
    # own "Cakecut not working for Q uncertainty!"): a wedge spans a real
    # spread of q directions, and simply averaging per-pixel q_std across
    # it the same way intensity is averaged isn't real error propagation
    # (variances would need combining in quadrature, not averaged
    # directly) -- confirmed not physically meaningful for this app's use
    # (reported: "cake_cut에서는 당연히 반영하면 안돼").
    q_std_curve = None
    q_uncertainty_warning = None
    # The error bar has to describe the coordinate actually plotted: an offset
    # cut is labelled by one Q component, so it carries that component's own
    # uncertainty (`qr_std`/`qz_std`, computed alongside `q_std` in
    # console_qmap), not |Q|'s. `q_std` gates on availability either way,
    # since all four are computed together or not at all.
    q_std_key = "q_std"
    if uses_offset_axis:
        q_std_key = "qr_std" if angle_deg in (0.0, 180.0) else "qz_std"
        if _SESSION.qmap.get(q_std_key) is None:
            q_std_key = "qy_std"  # TrXS names its in-plane component qy
    q_std_available = mapping == "Pixel" and _SESSION.qmap.get("q_std") is not None
    q_std_map = _SESSION.qmap.get(q_std_key) if q_std_available else None
    if q_std_map is not None and cakecut:
        q_uncertainty_warning = "Q uncertainty not computed for Cakecut."
    elif q_std_map is not None:
        try:
            # accumulation_average's own output can be *shorter* than
            # data_for_1d's own cut: which output bins survive depends
            # partly on whether that specific call's own reduced value
            # ends up finite (see its `keep` mask), not on geometry
            # alone. q_std_map is purely geometric and almost never NaN,
            # unlike real detector data (masked/gap/grid-removed
            # pixels), so run unmasked it silently kept extra bins the
            # real intensity cut dropped -- a shape mismatch that crashed
            # errorbar() outright (reported live, with the traceback:
            # "xerr (shape: (1700,)) must be... whose shape matches x
            # (shape: (1690,))"). Matching data_for_1d's own NaN pattern
            # onto q_std_map first makes both calls drop exactly the same
            # positions; the shape check after is a defensive fallback in
            # case some other divergence still slips through.
            masked_q_std_map = np.where(np.isfinite(data_for_1d), q_std_map, np.nan).astype(np.float32)
            std_result = _run_cut(masked_q_std_map)
            if std_result.intensity.shape == result.intensity.shape:
                q_std_curve = std_result.intensity
        except ValueError:
            q_std_curve = None

    _SESSION.last_curve = {
        "kind": "line",
        "q": result.q_axis,
        "intensity": result.intensity,
        "coverage": result.coverage,
        "angle_deg": angle_deg,
        "mapping": mapping,
        "q_std": q_std_curve,
    }
    return {
        "ok": True,
        "angle_deg": angle_deg,
        "angle_range": list(result.angle_range),
        "mapping": mapping,
        "coverage_diagnostics": _coverage_summary(result.coverage),
        "warning": q_uncertainty_warning,
        "preview": {
            "q": downsample_1d_for_json(result.q_axis, max_points=600),
            "intensity": downsample_1d_for_json(result.intensity, max_points=600),
            "coverage": downsample_1d_for_json(result.coverage, max_points=600),
            "q_std": downsample_1d_for_json(q_std_curve, max_points=600) if q_std_curve is not None else None,
        },
    }


def console_azimuthal_1d(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.qmap is None or _SESSION.data is None:
        return {"ok": False, "message": "Run q-Map Cal. first."}

    # GIXS's grazing-incidence geometry only ever scatters above the
    # horizon -- the lower half-circle is symmetric-empty, so restrict the
    # sweep the same way MATLAB GIXS_gui.m does (`theta = 0:0.2:180`).
    # TrXS's transmission geometry uses the full ring, matching TrXS_gui.m.
    angle_range = (0.0, 180.0) if _SESSION.mode == "gixs" else (0.0, 360.0)
    # Default bin count gives a 1deg/bin resolution regardless of sweep
    # span -- a single literal default shared between the 180deg (GIXS) and
    # 360deg (TrXS) sweeps left TrXS at a coarser 2deg/bin (180 bins over
    # 360deg) even though GIXS's own 180-over-180 was already 1deg/bin
    # (reported: "azimuthal에서 angle간에 2도간격으로 되어 있는데, 1도간격으로
    # 만들어줘").
    default_n_angle_bins = int(angle_range[1] - angle_range[0])

    try:
        pixel = float(payload.get("pixel", 202.0))
        range_minus = float(payload.get("range_minus", -1.0))
        range_plus = float(payload.get("range_plus", 1.0))
        use_median = str(payload.get("mean_or_median", "Mean")).strip().lower() == "median"
        n_angle_bins = int(payload.get("n_angle_bins", default_n_angle_bins))
        mapping = str(payload.get("mapping", "Pixel")).strip()
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid azimuthal parameters: {exc}"}

    if mapping == "Pixel":
        data_for_1d = _values_for_reduction(
            _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
        )
    else:
        # Same independence already given to the line/accumulation/cake
        # cuts: cut the actual Qxy/Qz-regridded (Itrans) image, not the
        # raw pixel data relabeled -- Itrans's own masked beamstop column
        # and low-coverage bilinear-splat gaps fan out into the "missing
        # wedge" visible in the Qxy/Qz 2D display at large Qz, which a
        # pixel-space cut would never see (reported: "range를 선택하여
        # azimuthal을 했는데... 90도 부근에 missing wedge가 있음에도
        # 연결되서 그렸어" -- because it was still silently cutting the
        # raw pixel image regardless of Map format).
        try:
            data_for_1d, _, _, _ = _q_space_cut_inputs(mapping)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}
    beam_pos = _SESSION.calibration["beam_pos"]
    # `pixel`/Range(-/+) are always real pixel-index radii regardless of
    # Map format (see console_widget.py's own _pixel_radius_to_q_radius --
    # only the *display* circle needs a Q-space radius) -- Itrans is
    # index-aligned 1:1 with the pixel grid, so this radius_range already
    # addresses the same ring of array cells in either data source.
    radius_range = (pixel + range_minus, pixel + range_plus)

    result = angular_profile(
        data_for_1d,
        center=beam_pos,
        radius_range=radius_range,
        n_angle_bins=n_angle_bins,
        use_median=use_median,
        angle_range=angle_range,
    )
    _SESSION.last_curve = {
        "kind": "azimuthal",
        "angle": result.angle_axis,
        "intensity": result.intensity,
        "coverage": result.coverage,
        "mapping": mapping,
    }
    return {
        "ok": True,
        "mapping": mapping,
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

    data_for_1d = _values_for_reduction(
        _SESSION.corrected_data if _SESSION.corrected_data is not None else _SESSION.data
    )
    mask_active = False
    if _SESSION.circular_mask is not None and _SESSION.circular_mask.shape == data_for_1d.shape:
        # ROI mask (see console_set_circular_mask): pixels outside the
        # user-drawn polygon become NaN, matching TrXS_gui.m's
        # `I2d(BW <= 0) = NaN`. np.where copies -- session data is never
        # mutated in place.
        data_for_1d = np.where(_SESSION.circular_mask, data_for_1d, np.nan).astype(np.float32)
        mask_active = True
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

    # `sector_average` already computes this whenever calc_std=True (see
    # its own calc_std param) -- it just never made it past this function
    # before, so the widget's Std checkbox had nothing to actually plot
    # (reported: "Std가 켜고 프로세스 했을때, 1D Result에서 Y축 errorbar
    # plot이 되면 좋을 것 같아"). Same low-coverage mask as intensity, so
    # a bin nulled out above doesn't still carry a stray error bar.
    std = None
    if calc_std:
        std = np.asarray(result.std, dtype=np.float32).copy()
        std[below] = np.nan

    _SESSION.last_curve = {
        "kind": "circular",
        "q": result.q_axis,
        "intensity": intensity,
        "coverage": coverage,
        "std": std,
    }
    return {
        "ok": True,
        "angle_range": list(result.angle_range),
        "min_coverage": min_coverage,
        "mask_active": mask_active,
        "coverage_diagnostics": _coverage_summary(coverage),
        "preview": {
            "q": downsample_1d_for_json(result.q_axis, max_points=600),
            "intensity": downsample_1d_for_json(intensity, max_points=600),
            "coverage": downsample_1d_for_json(coverage, max_points=600),
            "std": downsample_1d_for_json(std, max_points=600) if std is not None else None,
        },
    }


def console_set_circular_mask(payload: dict[str, Any]) -> dict[str, Any]:
    """Set or clear the Circular Process ROI mask (True = keep, False =
    excluded from Circular Average) -- port of TrXS_gui.m's
    `selectROIMask`, but as a persistent Draw/Clear pair on the console
    instead of a per-run modal popup. `console_circular_average` applies
    whatever mask is currently set, regardless of angle range (MATLAB only
    offered it for a full 0-360 sweep; that restriction was about when the
    popup asked, not a real requirement of the masking itself)."""

    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    if payload.get("clear"):
        _SESSION.circular_mask = None
        return {"ok": True, "cleared": True}

    mask = payload.get("mask")
    if mask is None:
        return {"ok": False, "message": "No mask provided."}
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != _SESSION.data.shape:
        return {"ok": False, "message": "Mask shape does not match the loaded image."}

    _SESSION.circular_mask = mask_arr
    return {"ok": True, "pixel_count": int(np.count_nonzero(mask_arr)), "total_pixels": int(mask_arr.size)}


def console_save(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}
    if _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Map Cal. before saving."}

    stem = _output_stem(payload)
    target_dir = _resolve_output_dir(payload, stem)
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
        try:
            curve_path = _export_last_curve(target_dir, stem)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"2D-info saved, but 1D export failed: {exc}", "h5_path": str(h5_path)}

    return {
        "ok": True,
        "h5_path": str(h5_path),
        "curve_path": str(curve_path) if curve_path is not None else None,
    }


def console_save_curve(payload: dict[str, Any]) -> dict[str, Any]:
    """Save just the current 1D Result curve (Line/Azimuthal/Circular) as a
    CSV -- unlike console_save, doesn't require q-Map Cal. to have been run
    (no 2D-info bundle), just a curve to have been plotted."""

    if _SESSION.last_curve is None:
        return {"ok": False, "message": "Run a 1D plot first."}

    stem = _output_stem(payload)
    target_dir = _resolve_output_dir(payload, stem)

    try:
        curve_path = _export_last_curve(target_dir, stem)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    return {"ok": True, "curve_path": str(curve_path)}


def _resolve_output_dir(payload: dict[str, Any], stem: str) -> Path:
    """Every save (console_save / console_save_curve) lands in its own
    <stem>/ subfolder rather than dropping files directly into the source
    directory -- repeated saves for different frames (or repeated saves of
    the same frame with different settings) stay grouped instead of piling
    up loose files side by side."""

    out_dir = str(payload.get("out_dir", "")).strip()
    base = Path(out_dir) if out_dir else (Path(_SESSION.source_path).parent if _SESSION.source_path else Path("console_outputs"))
    target_dir = base / stem
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _export_last_curve(target_dir: Path, stem: str) -> Path:
    curve = _SESSION.last_curve
    kind = curve["kind"]
    is_q_axis = kind in ("line", "circular")
    x_axis = curve["q"] if is_q_axis else curve["angle"]
    export = OneDExport(
        x=x_axis,
        intensity=curve["intensity"],
        coverage=curve.get("coverage"),
        x_label="q" if is_q_axis else "angle_deg",
        # Only "line" curves ever carry this (GIXS's q uncertainty,
        # see console_line_1d) -- absent (None) for every other kind.
        x_std=curve.get("q_std"),
    )
    curve_path = target_dir / f"{stem}_1D{_curve_filename_suffix(curve)}.csv"
    save_1d_data(curve_path, export, include_diagnostics=True)
    return curve_path


def _curve_filename_suffix(curve: dict[str, Any]) -> str:
    """A q-axis "line" export on its own can't be told apart from another
    line cut at a different angle, or from a q-axis "circular" average --
    stamp the kind (and, for a line cut, the reference angle it was taken
    at) into the filename so a folder of saved 1D profiles stays legible
    without having to open each one."""

    kind = curve["kind"]
    if kind == "line":
        angle_deg = curve.get("angle_deg")
        return f"_line_ang{angle_deg:g}deg" if angle_deg is not None else "_line"
    if kind == "azimuthal":
        return "_azimuthal"
    if kind == "circular":
        return "_circular"
    return ""


def _reset_for_new_data() -> None:
    _SESSION.qmap = None
    _SESSION.corrected_data = None
    _SESSION.normalized = False
    _SESSION.grid_removed = False
    _SESSION.angle_map = None
    _SESSION.last_curve = None
    _SESSION.background = None
    _SESSION.background_counts = None
    _SESSION.circular_mask = None


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
    """List same-series files: same normalized prefix, same extension,
    non-recursive, sorted by name.

    MATLAB `listPGIXSSeriesFiles.m` matches via `contains(allNames, prefix)`,
    a plain substring check -- so a series like "PTCDIC13_0001.h5" also
    sweeps up unrelated series such as "PTCDIC13_pPS_0001.h5" whose stem
    merely contains "PTCDIC13" as a substring. We instead require the
    candidate's own `_NNNN`-stripped prefix to match exactly, so only true
    same-series files are grouped.
    """

    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    prefix = _normalize_series_prefix(filename)
    suffix = Path(filename).suffix.lower()
    matches = [
        item
        for item in directory.iterdir()
        if item.is_file() and suffix == item.suffix.lower() and _normalize_series_prefix(item.name) == prefix
    ]
    return sorted(matches, key=natural_sort_key)


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

    with warnings.catch_warnings():
        # A detector gap pixel that is NaN in every frame of the series is
        # an expected, harmless case -- nanmean correctly returns NaN for
        # it, but warns on every occurrence via Python's warnings module.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
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
    "console_polar_image",
    "console_qmap",
    "console_qspace_image",
    "console_save",
    "console_save_curve",
    "console_set_circular_mask",
    "console_state",
    "get_session",
    "reset_session",
    "set_session",
]
