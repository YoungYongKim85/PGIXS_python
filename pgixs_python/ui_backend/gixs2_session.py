"""Backend for the GIXS2 tab: a port of MATLAB `GIXS2_gui.m`'s core
series-analysis workflow, including "Change ai" angle-scanning q-Cal
(`GUI.hChange`) -- Frame Playback, Q/Azimuthal Heatmap viewers, whole-stack
save, and custom colormaps remain deferred (see the implementation plan).

Unlike the single-frame `console_session.py`, GIXS2 holds a whole *stack*
of frames (a time/temperature/angle series) and produces per-frame curve
*matrices* (one row per frame) rather than a single curve -- the
per-frame geometry math itself (`cake_average`/`accumulation_average`/
`angular_profile`/`qmap_gixs`/detector corrections) is unchanged from the
single-frame console; `series_processing.py` is the new orchestration
layer that loops those across the stack.

Mirrors `console_session.py`/`sddcal_session.py`'s established pattern:
one module-level session, `gixs2_*` functions each returning
`{"ok": bool, ...}`.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.averaging.cake import pixel_angle_map
from pgixs_python.calibration.calibration_io import load_standard_calibration
from pgixs_python.calibration.sdd import wavelength_from_energy_kev
from pgixs_python.export.data_export import save_series_data
from pgixs_python.io.h5_series import load_h5_series
from pgixs_python.io.image_loader import load_image_file
from pgixs_python.processing.corrections import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    detector_correction_factor,
    kapton_efficiency,
    load_attenuation_table,
    polarization,
    silicon_efficiency,
    solid_angle,
)
from pgixs_python.processing.grid_remove import grid_remove
from pgixs_python.processing.masking import apply_detector_mask, detector_mask_for_type
from pgixs_python.processing.normalization import normalize_by_count
from pgixs_python.qmap.gixs import qmap_gixs
from pgixs_python.qmap.transform import image_transform_q_gi
from pgixs_python.ui_backend.preview import downsample_1d_for_json, downsample_2d_for_json
from pgixs_python.ui_backend.series_processing import (
    compute_experiment_x_axis,
    process_azimuthal_stack,
    process_linecut_stack,
    process_roi_stack,
    sort_x_axis,
)

_INSITU_MODES = ("Time", "Temperature", "Angle", "LowTemp.")


@dataclass
class GIXS2Session:
    detector: str = "Eiger4M"
    data: np.ndarray | None = None  # mean-of-stack display image
    stack: np.ndarray | None = None  # (n_frames, rows, cols)
    corrected_stack: np.ndarray | None = None
    mask: np.ndarray | None = None
    counts: np.ndarray | None = None  # (n_frames, 2) IC/BC
    timestamps: np.ndarray | None = None  # (n_frames,), baseline-relative
    temperatures: np.ndarray | None = None  # (n_frames, 3)
    file_paths: list[Path] = field(default_factory=list)
    source_path: str = ""
    # The raw text last entered in the Filename field when Set/Table
    # actually found files with it (gixs2_list_series_files) -- Auto
    # Refresh reuses this rather than re-deriving a prefix from
    # file_paths[0]'s own name, which only recovers the right value for a
    # plain "PREFIX_0001.h5..." series. A hand-typed prefix that's only a
    # *substring* of each file's name (e.g. "pvk_om_05" matching
    # "pvk_om_05_002_30_0001.h5", an angle-scan series where each file's
    # own middle segment varies) can't be recovered that way at all.
    series_filename_prefix: str = ""
    insitu_mode: str = "Time"
    # Raw (pre-baseline) timestamp of the series' own first frame -- kept
    # separately from `timestamps` (which is stored already baselined to
    # 0.0) so Auto Refresh can put newly-appended frames on the same
    # baseline instead of restarting from their own first new timestamp.
    timestamp_reference: float | None = None
    calibration: dict[str, Any] = field(
        default_factory=lambda: {
            "energy_kev": 9.5,
            "sdd_mm": 3000.0,
            "beam_pos": (1000.0, 1800.0),
            "pixel_mm": 0.075,
            "incident_angle_deg": 0.1,
        }
    )
    qmap: dict[str, np.ndarray] | None = None
    angle_map: np.ndarray | None = None
    # "Change ai" angle-scanning q-Cal (GIXS2_gui.m's `GUI.hChange`): when
    # active, `qmap` above is unused (there is no single shared Q-map) --
    # `aivar[i]` is frame i's own incident angle (`ai + i*ai_step`) and
    # `q_stack[i]` is that frame's own |Q| magnitude map. Only |Q| is kept
    # per frame (not Qr/Qz separately, unlike MATLAB's `Qr_stack`/
    # `Qz_stack`) since `cake_average`/`accumulation_average` only ever
    # consume the scalar q_map argument.
    change_ai: bool = False
    ai_step: float = 0.01
    aivar: np.ndarray | None = None
    q_stack: np.ndarray | None = None
    grid_removed: bool = False
    corrected: bool = False
    # Set True only by a *successful* gixs2_load_info -- gixs2_qcal refuses
    # to run without it, since `calibration` above always holds plausible-
    # looking placeholder values even before Load Info ever runs, and
    # silently q-Cal'ing against those produces a wrong-but-not-obviously-
    # wrong q-map (reported: "Load Info에 파일이 들어가 있지 않으면, q-cal이
    # 되지 않도록 해줘").
    calibration_loaded: bool = False
    roi_list: list[tuple[int, int, int, int]] = field(default_factory=list)
    last_1d: dict[str, Any] | None = None
    last_azimuthal: dict[str, Any] | None = None
    last_roi: np.ndarray | None = None
    x_axis_values: np.ndarray | None = None
    x_axis_label: str = ""
    # Which result the Series Integration panel currently shows, and how
    # many frames were loaded as of the last full load/Auto-Refresh --
    # matches MATLAB's `Auto.Met`/`Auto.Nold`.
    auto_met: str = "linecut"
    auto_n_old: int = 0
    # Set per-file during gixs2_load_series/gixs2_auto_refresh, polled from
    # the main thread via gixs2_load_progress() while the background
    # QThread runs -- matches console_session.py's own load_progress.
    load_progress: dict[str, Any] | None = None


_SESSION = GIXS2Session()


def get_session() -> GIXS2Session:
    return _SESSION


def set_session(session: GIXS2Session) -> None:
    global _SESSION
    _SESSION = session


def reset_session() -> None:
    global _SESSION
    _SESSION = GIXS2Session()


_SERIES_SUFFIX_RE = re.compile(r"^(.*)_\d{4,}$")


def _normalize_series_prefix(filename: str) -> str:
    """Port of `normalizePGIXSSeriesPrefix.m` (duplicated from
    `console_session.py` rather than imported -- this module is meant to
    stand alone, same reasoning as `sddcal_session.py`)."""

    stem = Path(filename).stem
    match = _SERIES_SUFFIX_RE.match(stem)
    return match.group(1) if match else stem


def _list_series_files(directory: Path, filename: str, extension: str = ".h5") -> list[Path]:
    """Port of `listPGIXSSeriesFiles.m`: same normalized prefix, same
    extension, non-recursive, sorted by name.

    `extension` is a separate parameter (default ".h5", matching MATLAB's
    own hardcoded `listPGIXSSeriesFiles(param.Path, param.FileName, '.h5')`
    call) rather than derived from `filename` -- GIXS2's Filename field
    (`GX2.hFilename`/`self.filename_edit`) only ever holds the bare series
    prefix (no extension), so deriving the extension from it would always
    come up empty and match nothing.

    The match itself is `prefix in item.name` (substring containment),
    matching MATLAB's own `contains(allNames, prefix)` -- NOT equality
    against each candidate file's own separately-normalized prefix (an
    earlier version of this port did that, and it silently found nothing
    for an angle-scan series like "pvk_om_05_002_30_0001.h5", "...
    _004_30_0001.h5", ...: each file's *own* normalized prefix is
    "pvk_om_05_002_30"/"pvk_om_05_004_30"/... -- never equal to each
    other, let alone to a hand-typed "pvk_om_05" meant to catch all of
    them). A plain substring check against the typed prefix is what
    actually lets a shorter, hand-edited Filename widen the match.
    """

    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    prefix = _normalize_series_prefix(filename)
    suffix = extension.lower()
    matches = [
        item
        for item in directory.iterdir()
        if item.is_file() and suffix == item.suffix.lower() and prefix in item.name
    ]
    return sorted(matches, key=lambda item: item.name)


def _set_load_progress(completed: int, total: int, label: str) -> None:
    _SESSION.load_progress = {"current": completed, "total": total, "label": label}


def gixs2_load_progress() -> dict[str, Any]:
    """Poll target for series-load progress while `gixs2_load_series`/
    `gixs2_auto_refresh` runs on a background thread, matching MATLAB's
    `uiprogressdlg` updates in `Loadingdata()`. Returns `active: false`
    once the in-flight call finishes (or if nothing is loading)."""

    progress = _SESSION.load_progress
    if progress is None:
        return {"ok": True, "active": False}
    return {"ok": True, "active": True, **progress}


def _metadata_mode_for_insitu(insitu_mode: str) -> str:
    """`read_h5_metadata`'s `mode` param uses the same strings as MATLAB's
    `hinsitumode` items except "Time" (no per-file scan value to read)."""

    return "" if insitu_mode == "Time" else insitu_mode


def _mean_ignoring_all_nan(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmean(stack, axis=0)


def _reset_for_new_series() -> None:
    _SESSION.qmap = None
    _SESSION.angle_map = None
    _SESSION.change_ai = False
    _SESSION.aivar = None
    _SESSION.q_stack = None
    _SESSION.corrected_stack = None
    _SESSION.corrected = False
    _SESSION.grid_removed = False
    _SESSION.roi_list = []
    _SESSION.last_1d = None
    _SESSION.last_azimuthal = None
    _SESSION.last_roi = None
    _SESSION.x_axis_values = None
    _SESSION.x_axis_label = ""


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def gixs2_load(payload: dict[str, Any]) -> dict[str, Any]:
    """LOAD DATA: pick one frame, derive the series filename prefix,
    matching `Openimage2_Callback`."""

    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    detector = str(payload.get("detector", "Eiger4M")).strip() or "Eiger4M"
    if not directory or not filename:
        return {"ok": False, "message": "Choose a file to load."}

    file_path = Path(directory) / filename
    try:
        result = load_image_file(file_path)
        detector_mask = detector_mask_for_type(detector)
        masked = apply_detector_mask(result.data, detector_mask.mask)
        masked = np.ascontiguousarray(masked.T)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    _SESSION.detector = detector
    # Kept in `detector_mask.mask`'s own native orientation, *not*
    # transposed to match `_SESSION.data`'s display orientation -- this is
    # only ever fed back into `load_h5_series` below (Set/Table -> Import,
    # Auto Refresh), which calls `load_h5_frame` internally; that function
    # transposes each raw H5 frame once before masking, so the mask it
    # needs is in the same single-transposed orientation `apply_detector_mask`
    # already validated two lines up (`result.data.shape` there is that
    # same single-transposed shape). Transposing this mask an extra time
    # (matching `_SESSION.data`'s own *second* transpose) silently made
    # every frame's masking multiply mismatched shapes, which
    # `load_h5_frame_with_retry` swallowed into a same-shaped-every-time
    # NaN fallback -- no error, just an entirely blank averaged image.
    _SESSION.mask = detector_mask.mask
    _SESSION.calibration["pixel_mm"] = detector_mask.pixel_size_mm
    _SESSION.source_path = str(file_path)
    _SESSION.stack = masked[np.newaxis, ...]
    _SESSION.data = masked
    _SESSION.counts = None
    _SESSION.timestamps = None
    _SESSION.temperatures = None
    _SESSION.timestamp_reference = None
    _SESSION.file_paths = [file_path]
    _reset_for_new_series()
    _SESSION.auto_n_old = 0

    return {
        "ok": True,
        "path": str(file_path),
        "shape": list(masked.shape),
        "prefix": _normalize_series_prefix(file_path.name),
        "preview": downsample_2d_for_json(masked, max_points=1024),
    }


def gixs2_list_series_files(payload: dict[str, Any]) -> dict[str, Any]:
    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    if not directory or not filename:
        return {"ok": False, "message": "Load one frame first to set the series prefix."}
    try:
        files = _list_series_files(Path(directory), filename)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    if not files:
        return {"ok": False, "message": "No files matched this series prefix."}
    # Remembered for Auto Refresh -- see series_filename_prefix's own
    # docstring on why file_paths[0]'s name alone can't always recover it.
    _SESSION.series_filename_prefix = filename
    return {"ok": True, "files": [f.name for f in files]}


def gixs2_load_series(payload: dict[str, Any]) -> dict[str, Any]:
    """Set/Table -> Import: load the checked series files into an
    in-memory stack, matching `Loadingdata()`. `parallel=True` (matches
    MATLAB's `hParCom`/`parfor` checkbox) loads files concurrently."""

    directory = str(payload.get("directory", "")).strip()
    filenames = list(payload.get("filenames", []))
    insitu_mode = str(payload.get("insitu_mode", "Time"))
    parallel = bool(payload.get("parallel", False))
    if insitu_mode not in _INSITU_MODES:
        insitu_mode = "Time"
    if not directory or not filenames:
        return {"ok": False, "message": "Choose a directory and at least one series file."}

    paths = [Path(directory) / name for name in filenames]
    _SESSION.load_progress = None
    try:
        result = load_h5_series(
            paths, mask=_SESSION.mask, metadata_mode=_metadata_mode_for_insitu(insitu_mode),
            progress_callback=_set_load_progress, parallel=parallel,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    # Baseline the series to its own first frame here (rather than passing
    # `load_h5_series`'s own `reference_timestamp` param, which would need
    # a first pass just to learn that value) -- matches MATLAB computing
    # `timestamps - timestamps(1)` once, at X-axis build time.
    timestamps = result.timestamps.astype(np.float64)
    _SESSION.timestamp_reference = float(timestamps[0]) if timestamps.size and np.isfinite(timestamps[0]) else None
    if _SESSION.timestamp_reference is not None:
        timestamps = timestamps - _SESSION.timestamp_reference

    stack = np.ascontiguousarray(np.transpose(result.stack, (0, 2, 1)))
    _SESSION.stack = stack
    _SESSION.data = _mean_ignoring_all_nan(stack)
    _SESSION.counts = result.counts.astype(np.float32)
    _SESSION.timestamps = timestamps
    _SESSION.temperatures = result.temperatures.astype(np.float64)
    _SESSION.file_paths = paths
    _SESSION.insitu_mode = insitu_mode
    _reset_for_new_series()
    _SESSION.auto_n_old = len(paths)

    n_load_failed = int(result.retry_failed_frames.size)
    n_meta_failed = int(result.metadata_failed_frames.size)
    warnings_list = []
    if n_load_failed:
        warnings_list.append(f"{n_load_failed} frame(s) failed to load after retry: {result.retry_failed_frames.tolist()}")
    if n_meta_failed:
        warnings_list.append(f"{n_meta_failed} frame(s) failed to read metadata: {result.metadata_failed_frames.tolist()}")

    return {
        "ok": True,
        "n_files": len(paths),
        "shape": list(stack.shape),
        "warnings": warnings_list,
        "preview": downsample_2d_for_json(_SESSION.data, max_points=1024),
    }


def gixs2_load_info(payload: dict[str, Any]) -> dict[str, Any]:
    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    if not directory or not filename:
        return {"ok": False, "message": "Choose a calibration file to load."}
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

    return {"ok": True, "path": path, "calibration": dict(_SESSION.calibration)}


def gixs2_grid_remove(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}

    n_frames = _SESSION.stack.shape[0]
    corrected_frames = []
    applied_message = ""
    for i in range(n_frames):
        corrected, report = grid_remove(_SESSION.stack[i].T, _SESSION.detector)
        if not report.applied:
            return {"ok": False, "message": report.message or "Grid remove is not applicable."}
        applied_message = report.message
        corrected_frames.append(np.ascontiguousarray(corrected.T))

    _SESSION.stack = np.stack(corrected_frames, axis=0)
    _SESSION.data = _mean_ignoring_all_nan(_SESSION.stack)
    _SESSION.grid_removed = True
    return {"ok": True, "message": applied_message, "preview": downsample_2d_for_json(_SESSION.data, max_points=1024)}


# ----------------------------------------------------------------------
# Incidence angle calibration
# ----------------------------------------------------------------------
def gixs2_z_plot(payload: dict[str, Any]) -> dict[str, Any]:
    """Z-Plot: a Z-profile column through `y_pos`, matching
    `aiZplot_Callback` -- the caller clicks a peak on the returned curve
    and applies it via `gixs2_ai_from_pixel`."""

    if _SESSION.data is None:
        return {"ok": False, "message": "Load data first."}
    beam_pos = _SESSION.calibration.get("beam_pos")
    if beam_pos is None:
        return {"ok": False, "message": "Run Load Info first to get the Beam Center."}

    try:
        y_pos = float(payload.get("y_pos", 0.0))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid y-pos: {exc}"}

    n_rows, n_cols = _SESSION.data.shape
    target_col = int(round(beam_pos[0] + y_pos))
    target_col = max(1, min(n_cols, target_col)) - 1

    z_axis = -(np.arange(1, n_rows + 1, dtype=np.float64) - beam_pos[1])
    intensity = _SESSION.data[:, target_col].astype(np.float64)

    return {
        "ok": True,
        "x": downsample_1d_for_json(z_axis, max_points=600),
        "y": downsample_1d_for_json(intensity, max_points=600),
    }


def gixs2_ai_from_pixel(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `CalIncidence2_Callback`: `ai_deg = atand(pix*pixel/SDD)/2`."""

    if "SDD" not in payload and _SESSION.calibration.get("sdd_mm") is None:
        return {"ok": False, "message": "Load Info first to get the SDD."}
    try:
        pix = abs(float(payload["pix"]))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid pixel value: {exc}"}

    sdd_mm = float(_SESSION.calibration["sdd_mm"])
    pixel_mm = float(_SESSION.calibration["pixel_mm"])
    ai_deg = np.degrees(np.arctan((pix * pixel_mm) / sdd_mm)) / 2.0
    _SESSION.calibration["incident_angle_deg"] = float(ai_deg)
    return {"ok": True, "ai_deg": float(ai_deg)}


def gixs2_ai_from_degree(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `CalIncidencePixel2_Callback`: `pix = round(abs(SDD*tand(2*ai)/pixel))`."""

    try:
        ai_deg = float(payload["ai_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid angle: {exc}"}

    sdd_mm = float(_SESSION.calibration["sdd_mm"])
    pixel_mm = float(_SESSION.calibration["pixel_mm"])
    pix = round(abs(sdd_mm * np.tan(np.radians(2.0 * ai_deg)) / pixel_mm))
    _SESSION.calibration["incident_angle_deg"] = float(ai_deg)
    return {"ok": True, "pix": int(pix)}


# ----------------------------------------------------------------------
# q-Cal -- static incidence angle (one shared q-map), or "Change ai"
# angle-scanning (a per-frame q-map, port of `GUI.hChange`)
# ----------------------------------------------------------------------
def _change_ai_base_correction(shape: tuple[int, int], beam_pos: tuple[float, float], pixel_mm: float, sdd_mm: float) -> np.ndarray:
    """`SolAng .* Pol` -- computed once, reused for every frame, matching
    `BaseCorrection` in `Qcalc_Callback`'s Change-ai branch."""

    return solid_angle(shape, beam_pos, pixel_mm, sdd_mm) * polarization(shape, beam_pos, pixel_mm, sdd_mm)


def _change_ai_attenuation_tables() -> tuple[np.ndarray, np.ndarray | None]:
    kapton_table = load_attenuation_table("kapton")
    si_table = load_attenuation_table("silicon") if _SESSION.detector == "Eiger4M" else None
    return kapton_table, si_table


def _change_ai_frame_factor(
    base_correction: np.ndarray,
    q_frame: np.ndarray,
    k0: float,
    energy_kev: float,
    shape: tuple[int, int],
    beam_pos: tuple[float, float],
    pixel_mm: float,
    kapton_table: np.ndarray,
    si_table: np.ndarray | None,
) -> np.ndarray:
    factor = base_correction * kapton_efficiency(q_frame, kapton_table, k0, energy_kev, shape, beam_pos, pixel_mm)
    if si_table is not None:
        factor = factor * silicon_efficiency(q_frame, k0, si_table, energy_kev)
    return factor


def gixs2_qcal(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}
    if not _SESSION.calibration_loaded:
        return {"ok": False, "message": "Run Load Info first."}

    try:
        energy_kev = float(payload.get("energy_kev", _SESSION.calibration["energy_kev"]))
        sdd_mm = float(payload.get("sdd_mm", _SESSION.calibration["sdd_mm"]))
        beam_y = float(payload.get("beam_y", _SESSION.calibration["beam_pos"][0]))
        beam_z = float(payload.get("beam_z", _SESSION.calibration["beam_pos"][1]))
        incident_angle_deg = float(payload.get("incident_angle_deg", _SESSION.calibration["incident_angle_deg"]))
        change_ai = bool(payload.get("change_ai", False))
        ai_step = float(payload.get("ai_step", _SESSION.ai_step))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid calibration input: {exc}"}

    pixel_mm = float(_SESSION.calibration["pixel_mm"])
    wavelength = wavelength_from_energy_kev(energy_kev)
    if not np.isfinite(wavelength) or wavelength <= 0:
        return {"ok": False, "message": "Invalid energy for wavelength conversion."}
    k0 = float(2.0 * np.pi / wavelength)
    beam_pos = (beam_y, beam_z)

    n_frames, n_rows, n_cols = _SESSION.stack.shape
    shape = (n_rows, n_cols)

    if not change_ai:
        try:
            qy, qr, qz, q, tth2d, af2d = qmap_gixs(
                det_y=n_cols, det_z=n_rows, beam_pos=beam_pos,
                pixel_mm=pixel_mm, sdd_mm=sdd_mm, k0=k0, incident_angle_deg=incident_angle_deg,
            )
            factor = detector_correction_factor(
                DetectorCorrectionInputs(
                    data_size=shape, beam_position=beam_pos, pixel_size_mm=pixel_mm,
                    sdd_mm=sdd_mm, q_map=q, energy_kev=energy_kev, detector_type=_SESSION.detector,
                ),
                k0=k0,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": str(exc)}

        corrected = np.stack([apply_detector_correction(_SESSION.stack[i], factor) for i in range(n_frames)], axis=0)
        _SESSION.qmap = {"qy": qy, "qr": qr, "qz": qz, "q": q, "tth2d": tth2d, "af2d": af2d}
        _SESSION.aivar = None
        _SESSION.q_stack = None
    else:
        try:
            base_correction = _change_ai_base_correction(shape, beam_pos, pixel_mm, sdd_mm)
            kapton_table, si_table = _change_ai_attenuation_tables()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": str(exc)}

        aivar = incident_angle_deg + np.arange(n_frames, dtype=np.float64) * ai_step
        q_stack = np.empty((n_frames, n_rows, n_cols), dtype=np.float32)
        corrected_frames: list[np.ndarray] = []
        # "Change ai" recomputes a whole new q-map per frame (unlike the
        # plain branch above, one shared map for the series) -- report
        # progress through the same session field Exec Cut/Exec Azi/series
        # loading already poll, so the widget can show a progress dialog
        # for what can otherwise be a long, silent freeze.
        _SESSION.load_progress = None
        try:
            for i in range(n_frames):
                _, _, _, q_i, _, _ = qmap_gixs(
                    det_y=n_cols, det_z=n_rows, beam_pos=beam_pos,
                    pixel_mm=pixel_mm, sdd_mm=sdd_mm, k0=k0, incident_angle_deg=float(aivar[i]),
                )
                factor = _change_ai_frame_factor(base_correction, q_i, k0, energy_kev, shape, beam_pos, pixel_mm, kapton_table, si_table)
                corrected_frames.append(apply_detector_correction(_SESSION.stack[i], factor))
                q_stack[i] = q_i
                _set_load_progress(i + 1, n_frames, "q-Cal (Change ai)")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": str(exc)}
        finally:
            _SESSION.load_progress = None

        corrected = np.stack(corrected_frames, axis=0)
        _SESSION.qmap = None
        _SESSION.aivar = aivar
        _SESSION.q_stack = q_stack

    _SESSION.calibration.update(
        {"energy_kev": energy_kev, "sdd_mm": sdd_mm, "beam_pos": beam_pos, "incident_angle_deg": incident_angle_deg}
    )
    _SESSION.change_ai = change_ai
    _SESSION.ai_step = ai_step
    _SESSION.angle_map = pixel_angle_map(shape, center=beam_pos)
    _SESSION.corrected_stack = corrected
    _SESSION.corrected = True
    _SESSION.data = _mean_ignoring_all_nan(corrected)

    return {
        "ok": True,
        "calibration": dict(_SESSION.calibration),
        "change_ai": change_ai,
        "preview": downsample_2d_for_json(_SESSION.data, max_points=1024),
    }


def _data_for_processing() -> np.ndarray:
    return _SESSION.corrected_stack if _SESSION.corrected_stack is not None else _SESSION.stack


def _q_remap_stack(stack: np.ndarray, progress_label: str) -> np.ndarray | None:
    """Regrid every frame in `stack` onto the same uniform Qxy/Qz grid the
    single-frame console's own Q-space cut uses (`image_transform_q_gi`;
    see console_session.py's `_q_space_cut_inputs`), so a plain radius/
    angle cut against the *result* genuinely reflects the display's real
    Q-space coverage gaps -- including the "missing wedge" near Qxy=0.

    A plain pixel-space cut can never show this gap no matter which
    pixels get excluded: every real detector pixel holds a real count, so
    there is nothing "missing" in pixel space to begin with. The gap is
    purely an artifact of interpolating that real pixel data onto a
    uniform Q grid finer, near the pole, than the pixel sampling can
    actually support -- an earlier attempt at a cheap geometry-only
    exclusion mask confirmed this: it only ever excluded the exact
    beamstop column, never anything resembling the wedge actually visible
    on screen (reported, with the fix shown live and still not reflected:
    "그림과 같이 딱히 반영된것 같지 않은데?"). Only genuinely regridding
    each frame -- not merely excluding some of its pixels -- can produce
    the same holes the display shows.

    Costs one `image_transform_q_gi` bilinear splat per frame -- run only
    when Azimuthal Process's own "Reflect missing wedge" checkbox opts in
    (reported: "미싱웻지 반영 모드와 반영되지 않은 모드 2가지가 있으면
    좋을 것 같아"), using the same background-thread + progress-dialog
    Exec Azi already runs its normal per-frame loop under, so this
    doesn't freeze the UI, just takes longer.

    Only available after a plain q-Cal, where `qmap` holds one shared
    qr/qz for the whole series -- "Change ai" recomputes qr/qz per frame
    and only keeps the scalar |Q| magnitude (`q_stack`), not qr/qz
    separately, so there is nothing to regrid against there; returns None
    in that case."""

    if _SESSION.qmap is None:
        return None
    cal = _SESSION.calibration
    beam_pos = (float(cal["beam_pos"][0]), float(cal["beam_pos"][1]))
    pixel_mm = float(cal["pixel_mm"])
    wavelength_a = wavelength_from_energy_kev(float(cal["energy_kev"]))
    sdd_mm = float(cal["sdd_mm"])
    qr = _SESSION.qmap["qr"]
    qz = _SESSION.qmap["qz"]

    n_frames = stack.shape[0]
    remapped = np.empty_like(stack)
    for i in range(n_frames):
        remapped[i] = image_transform_q_gi(stack[i], qr, qz, beam_pos, pixel_mm, wavelength_a, sdd_mm).image
        _set_load_progress(i + 1, n_frames, progress_label)
    return remapped


# ----------------------------------------------------------------------
# 1D / Azimuthal / ROI series exec
# ----------------------------------------------------------------------
def gixs2_linecut_exec(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.stack is None or (_SESSION.qmap is None and _SESSION.q_stack is None):
        return {"ok": False, "message": "Run q-Cal first."}

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
        parallel = bool(payload.get("parallel", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid line parameters: {exc}"}

    beam_pos = _SESSION.calibration["beam_pos"]
    kwargs = dict(
        angle_deg=angle_deg, cakecut=cakecut, range_minus=range_minus, range_plus=range_plus, use_median=use_median,
        y_pos_move=y_pos_move, z_pos_move=z_pos_move, accumulation=accumulation,
        accum_minus=accum_minus, accum_plus=accum_plus, parallel=parallel,
    )
    def _report(completed: int, total: int) -> None:
        _set_load_progress(completed, total, "1D cut")

    _SESSION.load_progress = None
    try:
        if _SESSION.change_ai and _SESSION.q_stack is not None:
            result = process_linecut_stack(
                _data_for_processing(), None, _SESSION.angle_map, center=beam_pos,
                q_map_stack=_SESSION.q_stack, progress_callback=_report, **kwargs
            )
        else:
            result = process_linecut_stack(
                _data_for_processing(), _SESSION.qmap["q"], _SESSION.angle_map, center=beam_pos,
                progress_callback=_report, **kwargs
            )
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    # `kwargs` is replayed against just the newly-appended frames by
    # `_extend_active_curve` on the next Auto Refresh -- must stay in sync
    # with whatever geometry actually produced this curve.
    _SESSION.last_1d = {"x_axis": result.x_axis, "intensity": result.intensity, "coverage": result.coverage, "params": kwargs}
    _SESSION.auto_met = "linecut"
    return {
        "ok": True,
        "n_frames": int(result.intensity.shape[0]),
        "n_points": int(result.x_axis.shape[0]),
        "x": downsample_1d_for_json(result.x_axis, max_points=600),
    }


def gixs2_azimuthal_exec(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}

    try:
        pixel = float(payload.get("pixel", 202.0))
        range_minus = float(payload.get("range_minus", -1.0))
        range_plus = float(payload.get("range_plus", 1.0))
        use_median = str(payload.get("mean_or_median", "Mean")).strip().lower() == "median"
        n_angle_bins = int(payload.get("n_angle_bins", 180))
        reflect_missing_wedge = bool(payload.get("reflect_missing_wedge", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid azimuthal parameters: {exc}"}

    beam_pos = _SESSION.calibration["beam_pos"]
    radius_range = (pixel + range_minus, pixel + range_plus)
    kwargs = dict(radius_range=radius_range, n_angle_bins=n_angle_bins, use_median=use_median, angle_range=(0.0, 180.0))

    def _report(completed: int, total: int) -> None:
        _set_load_progress(completed, total, "Azimuthal")

    _SESSION.load_progress = None
    try:
        stack = _data_for_processing()
        if reflect_missing_wedge:
            stack = _q_remap_stack(stack, "Azimuthal (Qxy/Qz remap)")
            if stack is None:
                return {"ok": False, "message": 'Reflect missing wedge needs a plain q-Cal (not "Change ai").'}
        result = process_azimuthal_stack(stack, center=beam_pos, progress_callback=_report, **kwargs)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    # `reflect_missing_wedge` is kept separate from `kwargs` -- `kwargs`
    # is splatted straight into `process_azimuthal_stack` by both this
    # call and `_extend_active_curve`'s Auto Refresh replay, and that
    # function has no such parameter of its own.
    _SESSION.last_azimuthal = {
        "x_axis": result.x_axis, "intensity": result.intensity, "coverage": result.coverage,
        "params": kwargs, "reflect_missing_wedge": reflect_missing_wedge,
    }
    _SESSION.auto_met = "azimuthal"
    return {
        "ok": True,
        "n_frames": int(result.intensity.shape[0]),
        "n_points": int(result.x_axis.shape[0]),
        "x": downsample_1d_for_json(result.x_axis, max_points=600),
    }


def gixs2_roi_exec(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}

    rois_raw = payload.get("rois", _SESSION.roi_list)
    try:
        rois = [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rois_raw]
    except (TypeError, ValueError, IndexError) as exc:
        return {"ok": False, "message": f"Invalid ROI list: {exc}"}
    if not rois:
        return {"ok": False, "message": "Draw at least one ROI first."}

    sums = process_roi_stack(_data_for_processing(), rois)
    _SESSION.roi_list = rois
    _SESSION.last_roi = sums
    _SESSION.auto_met = "roi"
    return {"ok": True, "n_frames": int(sums.shape[0]), "n_roi": int(sums.shape[1]), "sums": sums.tolist()}


# ----------------------------------------------------------------------
# X-axis
# ----------------------------------------------------------------------
def gixs2_set_x_axis(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.timestamps is None:
        return {"ok": False, "message": "Load a series first."}

    sub_mode = str(payload.get("sub_mode", ""))
    should_sort = bool(payload.get("order", False))
    # None (the default -- not simply absent-from-payload) tells
    # compute_experiment_x_axis to auto-derive the exposure-time offset
    # from this series' own average frame-to-frame timestamp interval;
    # only an explicit payload value overrides that.
    raw_offset = payload.get("time_offset_seconds")
    time_offset_seconds = None if raw_offset is None else float(raw_offset)
    time_unit = str(payload.get("time_unit", "second"))

    # "Change ai" q-Cal computes the *real* per-frame incident angle each
    # frame's own q-map was actually built with (aivar = ai + i*ai_step) --
    # prefer that over compute_experiment_x_axis's own "Angle" branch (the
    # per-file *logged* Angle metadata column, an unrelated value with no
    # connection to the ai/ai_step ramp q-Cal applied) whenever it's
    # available, matching MATLAB's own "Angle" axis, which always reads
    # param.aivar. Reported: "angle 모드에서 1D plot에 incidence angle
    # 밸류들이 제대로 반영이 되지 않아 color가 이상하게 나왔어." sub_mode
    # == "Time" is still honored as an explicit opt-out back to the time
    # axis (a valid choice within Angle insitu mode too).
    if _SESSION.insitu_mode == "Angle" and sub_mode != "Time" and _SESSION.change_ai and _SESSION.aivar is not None:
        if _SESSION.aivar.shape[0] != _SESSION.timestamps.shape[0]:
            return {"ok": False, "message": "Incidence-angle ramp doesn't match the loaded series length."}
        values, label = _SESSION.aivar, "Incidence Angle, deg"
    else:
        try:
            values, label = compute_experiment_x_axis(
                _SESSION.insitu_mode, _SESSION.timestamps, _SESSION.temperatures,
                sub_mode=sub_mode, time_offset_seconds=time_offset_seconds, time_unit=time_unit,
            )
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}

    order = sort_x_axis(values, should_sort)
    _SESSION.x_axis_values = values[order]
    _SESSION.x_axis_label = label
    return {"ok": True, "x_axis": _SESSION.x_axis_values.tolist(), "label": label, "order": order.tolist()}


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
def gixs2_normalized_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    """Apply the None/IC/BC per-frame normalization used throughout GIXS2
    -- shared by Save 1D/Azi (below) and by the live 1D/Azimuthal Plot
    Viewer renders in gixs2_widget.py, so the on-screen curves and the
    saved file always agree."""

    if method == "None" or _SESSION.counts is None:
        return matrix
    column = 0 if method == "IC" else 1
    out = matrix.copy()
    for i in range(matrix.shape[0]):
        out[i] = normalize_by_count(matrix[i], _SESSION.counts[i, column])
    return out


def _column_labels() -> list[str]:
    if _SESSION.x_axis_values is not None:
        return [f"{v:g}" for v in _SESSION.x_axis_values]
    return [f"frame{i + 1}" for i in range(_SESSION.stack.shape[0])]


def _save_path(directory: str, suffix: str) -> Path:
    """`{directory}/{prefix}/{prefix}{suffix}` -- every Save button gets
    its own subfolder named after the series (the same prefix shown in the
    Filename field), rather than dropping same-named exports from
    different series/detectors loosely into one shared directory.

    `source_path` (set by the single-frame LOAD DATA step) is the usual
    source for the prefix; `file_paths[0]` (set by Set/Table Import too)
    is the fallback for a session that went straight to Import with a
    hand-typed Filename and never ran LOAD DATA."""

    name_source = _SESSION.source_path or (str(_SESSION.file_paths[0]) if _SESSION.file_paths else "")
    prefix = _normalize_series_prefix(Path(name_source).name) if name_source else "series"
    return Path(directory) / prefix / f"{prefix}{suffix}"


def gixs2_save_1d(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.last_1d is None:
        return {"ok": False, "message": "Run Exec Cut first."}
    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    method = str(payload.get("normalize", "None"))
    matrix = gixs2_normalized_matrix(_SESSION.last_1d["intensity"], method)
    path = _save_path(directory, "_1D.csv")
    try:
        save_series_data(path, _SESSION.last_1d["x_axis"], "Q_1/A", matrix, _column_labels())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


def gixs2_save_azimuthal(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.last_azimuthal is None:
        return {"ok": False, "message": "Run Exec Azi first."}
    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    method = str(payload.get("normalize", "None"))
    matrix = gixs2_normalized_matrix(_SESSION.last_azimuthal["intensity"], method)
    path = _save_path(directory, "_Azi.csv")
    try:
        save_series_data(path, _SESSION.last_azimuthal["x_axis"], "Angle_deg", matrix, _column_labels())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


def gixs2_save_integration(payload: dict[str, Any]) -> dict[str, Any]:
    """Save Integ.: whichever of 1D/Azimuthal/ROI is currently active
    (`Auto.Met`), summed over its X-axis, or the ROI sums directly."""

    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}
    if _SESSION.x_axis_values is None:
        return {"ok": False, "message": "Set the series X-axis first."}

    if _SESSION.auto_met == "roi":
        if _SESSION.last_roi is None:
            return {"ok": False, "message": "Run Exec ROI first."}
        columns = [f"ROI{i + 1}" for i in range(_SESSION.last_roi.shape[1])]
        method = str(payload.get("normalize", "None"))
        matrix = gixs2_normalized_matrix(_SESSION.last_roi, method).T
    elif _SESSION.auto_met == "azimuthal":
        if _SESSION.last_azimuthal is None:
            return {"ok": False, "message": "Run Exec Azi first."}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            matrix = np.nansum(_SESSION.last_azimuthal["intensity"], axis=1, keepdims=True).T
        columns = ["Integrated_Azimuthal"]
    else:
        if _SESSION.last_1d is None:
            return {"ok": False, "message": "Run Exec Cut first."}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            matrix = np.nansum(_SESSION.last_1d["intensity"], axis=1, keepdims=True).T
        columns = ["Integrated_1D"]

    path = _save_path(directory, "_Integ.csv")
    try:
        save_series_data(path, _SESSION.x_axis_values, _SESSION.x_axis_label, matrix, columns)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


def gixs2_save_x_axis(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.x_axis_values is None:
        return {"ok": False, "message": "Set the series X-axis first."}
    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    path = _save_path(directory, "_XAxis.csv")
    try:
        save_series_data(path, _SESSION.x_axis_values, _SESSION.x_axis_label, _SESSION.x_axis_values[np.newaxis, :], ["value"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


# ----------------------------------------------------------------------
# Auto Refresh (Find New)
# ----------------------------------------------------------------------
def gixs2_auto_refresh(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `Autoplot_Callback`: poll-on-click (not a timer), rescan the
    directory for the same series prefix, load only the files beyond
    `auto_n_old`, append them to the stack/counts/timestamps/temperatures,
    and extend only the *currently active* curve type (`auto_met`) --
    never a full series recompute."""

    if _SESSION.stack is None or not _SESSION.file_paths:
        return {"ok": False, "message": "Load an initial series first."}

    directory = str(payload.get("directory", str(_SESSION.file_paths[0].parent)))
    # Prefer the prefix actually typed into the Filename field over
    # re-deriving one from file_paths[0]'s own name -- see
    # series_filename_prefix's own docstring. Empty when the session
    # loaded straight through gixs2_load_series without ever going through
    # Set/Table (e.g. a test, or a future non-UI caller); file_paths[0]'s
    # name is still the right fallback there.
    match_name = _SESSION.series_filename_prefix or _SESSION.file_paths[0].name
    try:
        all_files = _list_series_files(Path(directory), match_name, _SESSION.file_paths[0].suffix)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    n_total = len(all_files)
    if n_total <= _SESSION.auto_n_old:
        return {"ok": True, "n_new": 0, "message": "No new files."}

    new_files = all_files[_SESSION.auto_n_old : n_total]
    _SESSION.load_progress = None
    try:
        result = load_h5_series(
            new_files, mask=_SESSION.mask, metadata_mode=_metadata_mode_for_insitu(_SESSION.insitu_mode),
            progress_callback=_set_load_progress, parallel=bool(payload.get("parallel", False)),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    new_stack = np.ascontiguousarray(np.transpose(result.stack, (0, 2, 1)))
    new_timestamps = result.timestamps.astype(np.float64)
    if _SESSION.timestamp_reference is not None:
        new_timestamps = new_timestamps - _SESSION.timestamp_reference

    if _SESSION.corrected_stack is not None and _SESSION.change_ai and _SESSION.q_stack is not None and _SESSION.aivar is not None:
        pixel_mm = float(_SESSION.calibration["pixel_mm"])
        beam_pos = _SESSION.calibration["beam_pos"]
        sdd_mm = float(_SESSION.calibration["sdd_mm"])
        energy_kev = float(_SESSION.calibration["energy_kev"])
        wavelength = wavelength_from_energy_kev(energy_kev)
        k0 = float(2.0 * np.pi / wavelength)
        n_new, n_rows, n_cols = new_stack.shape
        shape = (n_rows, n_cols)
        last_ai = float(_SESSION.aivar[-1])
        # Port of `Autoplot_Callback`'s `aivar_candidates = last_ai +
        # aistep .* (1:Nnew)'` -- continues the same linear ramp the
        # original series' q-Cal established, not a fresh one starting
        # over from the incidence-angle field's current value.
        new_aivar = last_ai + _SESSION.ai_step * np.arange(1, n_new + 1, dtype=np.float64)
        try:
            base_correction = _change_ai_base_correction(shape, beam_pos, pixel_mm, sdd_mm)
            kapton_table, si_table = _change_ai_attenuation_tables()
            new_q_stack = np.empty((n_new, n_rows, n_cols), dtype=np.float32)
            new_corrected_frames: list[np.ndarray] = []
            for i in range(n_new):
                _, _, _, q_i, _, _ = qmap_gixs(
                    det_y=n_cols, det_z=n_rows, beam_pos=beam_pos,
                    pixel_mm=pixel_mm, sdd_mm=sdd_mm, k0=k0, incident_angle_deg=float(new_aivar[i]),
                )
                factor = _change_ai_frame_factor(base_correction, q_i, k0, energy_kev, shape, beam_pos, pixel_mm, kapton_table, si_table)
                new_corrected_frames.append(apply_detector_correction(new_stack[i], factor))
                new_q_stack[i] = q_i
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": str(exc)}
        new_corrected = np.stack(new_corrected_frames, axis=0)
        _SESSION.corrected_stack = np.concatenate([_SESSION.corrected_stack, new_corrected], axis=0)
        _SESSION.q_stack = np.concatenate([_SESSION.q_stack, new_q_stack], axis=0)
        _SESSION.aivar = np.concatenate([_SESSION.aivar, new_aivar])
    elif _SESSION.corrected_stack is not None and _SESSION.qmap is not None:
        pixel_mm = float(_SESSION.calibration["pixel_mm"])
        wavelength = wavelength_from_energy_kev(float(_SESSION.calibration["energy_kev"]))
        k0 = float(2.0 * np.pi / wavelength)
        factor = detector_correction_factor(
            DetectorCorrectionInputs(
                data_size=new_stack.shape[1:], beam_position=_SESSION.calibration["beam_pos"], pixel_size_mm=pixel_mm,
                sdd_mm=float(_SESSION.calibration["sdd_mm"]), q_map=_SESSION.qmap["q"],
                energy_kev=float(_SESSION.calibration["energy_kev"]), detector_type=_SESSION.detector,
            ),
            k0=k0,
        )
        new_corrected = np.stack([apply_detector_correction(new_stack[i], factor) for i in range(new_stack.shape[0])], axis=0)
        _SESSION.corrected_stack = np.concatenate([_SESSION.corrected_stack, new_corrected], axis=0)

    _SESSION.stack = np.concatenate([_SESSION.stack, new_stack], axis=0)
    _SESSION.data = _mean_ignoring_all_nan(_data_for_processing())
    _SESSION.counts = np.concatenate([_SESSION.counts, result.counts.astype(np.float32)], axis=0) if _SESSION.counts is not None else result.counts
    _SESSION.timestamps = np.concatenate([_SESSION.timestamps, new_timestamps]) if _SESSION.timestamps is not None else new_timestamps
    _SESSION.temperatures = (
        np.concatenate([_SESSION.temperatures, result.temperatures.astype(np.float64)], axis=0)
        if _SESSION.temperatures is not None
        else result.temperatures
    )
    _SESSION.file_paths = _SESSION.file_paths + new_files

    n_appended = new_stack.shape[0]
    new_q_stack_for_curve = (
        _SESSION.q_stack[-n_appended:] if _SESSION.change_ai and _SESSION.q_stack is not None else None
    )
    extended = _extend_active_curve(
        new_stack if _SESSION.corrected_stack is None else _SESSION.corrected_stack[-n_appended:], new_q_stack_for_curve
    )
    _SESSION.auto_n_old = n_total

    return {
        "ok": True,
        "n_new": len(new_files),
        "n_total": n_total,
        "curve_extended": extended,
        "preview": downsample_2d_for_json(_SESSION.data, max_points=1024),
    }


def _extend_active_curve(new_frames: np.ndarray, new_q_stack: np.ndarray | None = None) -> bool:
    """Append the new frames' contribution onto whichever curve type is
    active -- silently no-ops (returns False) if the new block's shape
    doesn't match the existing curve's, matching MATLAB's own
    append-only-if-shape-matches guard in `autoUpdateLineCut`/
    `autoUpdateAziCut`/`autoUpdateROI`.

    `new_q_stack`, the newly-appended frames' own per-frame Q-maps
    (Change-ai mode only), is aligned onto the *existing* `last_1d`
    reference axis (`ref_axis=`) rather than a fresh one derived from just
    this call's own first frame -- see `process_linecut_stack`'s own
    docstring for why that distinction matters here specifically."""

    beam_pos = _SESSION.calibration.get("beam_pos")
    try:
        if _SESSION.auto_met == "linecut" and _SESSION.last_1d is not None and (_SESSION.qmap is not None or new_q_stack is not None):
            if _SESSION.change_ai and new_q_stack is not None:
                extra = process_linecut_stack(
                    new_frames, None, _SESSION.angle_map, center=beam_pos,
                    q_map_stack=new_q_stack, ref_axis=_SESSION.last_1d["x_axis"], **_SESSION.last_1d["params"]
                )
            else:
                extra = process_linecut_stack(
                    new_frames, _SESSION.qmap["q"], _SESSION.angle_map, center=beam_pos, **_SESSION.last_1d["params"]
                )
            if extra.x_axis.shape != _SESSION.last_1d["x_axis"].shape:
                return False
            _SESSION.last_1d["intensity"] = np.vstack([_SESSION.last_1d["intensity"], extra.intensity])
            _SESSION.last_1d["coverage"] = np.vstack([_SESSION.last_1d["coverage"], extra.coverage])
            return True
        if _SESSION.auto_met == "azimuthal" and _SESSION.last_azimuthal is not None:
            frames = new_frames
            if _SESSION.last_azimuthal.get("reflect_missing_wedge"):
                frames = _q_remap_stack(frames, "Azimuthal (Qxy/Qz remap)")
                if frames is None:
                    return False
            extra = process_azimuthal_stack(frames, center=beam_pos, **_SESSION.last_azimuthal["params"])
            if extra.x_axis.shape != _SESSION.last_azimuthal["x_axis"].shape:
                return False
            _SESSION.last_azimuthal["intensity"] = np.vstack([_SESSION.last_azimuthal["intensity"], extra.intensity])
            _SESSION.last_azimuthal["coverage"] = np.vstack([_SESSION.last_azimuthal["coverage"], extra.coverage])
            return True
        if _SESSION.auto_met == "roi" and _SESSION.last_roi is not None and _SESSION.roi_list:
            extra = process_roi_stack(new_frames, _SESSION.roi_list)
            _SESSION.last_roi = np.vstack([_SESSION.last_roi, extra])
            return True
    except ValueError:
        return False
    return False


__all__ = [
    "GIXS2Session",
    "get_session",
    "gixs2_ai_from_degree",
    "gixs2_ai_from_pixel",
    "gixs2_auto_refresh",
    "gixs2_grid_remove",
    "gixs2_linecut_exec",
    "gixs2_azimuthal_exec",
    "gixs2_list_series_files",
    "gixs2_load",
    "gixs2_load_info",
    "gixs2_load_progress",
    "gixs2_load_series",
    "gixs2_qcal",
    "gixs2_roi_exec",
    "gixs2_save_1d",
    "gixs2_save_azimuthal",
    "gixs2_save_integration",
    "gixs2_save_x_axis",
    "gixs2_set_x_axis",
    "gixs2_z_plot",
    "reset_session",
    "set_session",
]
