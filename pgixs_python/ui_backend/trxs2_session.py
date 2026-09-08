"""Backend for the TrXS2 tab: a port of MATLAB `TrXS2_gui.m`'s
series-analysis workflow -- Circular Average (the primary 1D mode),
1D Line Ang, Azimuthal Process, ROI Process, X-Axis/Auto Refresh, and a
live Background Subtraction pipeline. Structured like `gixs2_session.py`
but simpler: TrXS is normal-incidence transmission, so there is no
incidence-angle calibration and no "Change ai" angle-scanning branch
anywhere in this module.

Unlike GIXS2, TrXS2 has one genuinely new architectural piece: Background
Subtraction. A background series gets averaged into one 2D frame once
(`trxs2_background_load`, mirroring how `gixs2_load_series`/MATLAB's own
`Bkgload2_Callback` average a whole matched file series into one frame --
not a literal copy of `console_session.console_background_load`, which
only ever averages via the older Multishot single-frame path), corrected
by the same detector-correction factor as the main stack, cut once
(circular/line/azimuthal, whichever mode is active) into a single
background curve, and a live Subtraction toggle (plus the Normalize
combo, plus Auto Refresh) triggers a cheap column-wise re-normalize-and-
subtract over the already-computed stack curve matrix -- never a re-run
of the per-frame averaging loop. ROI never subtracts background (matches
MATLAB's own `Saveget4_Callback`/`ROIdef2_Callback`, which have no
background-related code at all).

Mirrors `console_session.py`/`gixs2_session.py`'s established pattern:
one module-level session, `trxs2_*` functions each returning
`{"ok": bool, ...}`.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.averaging.angular import angular_profile
from pgixs_python.averaging.cake import pixel_angle_map
from pgixs_python.averaging.radial import RadiusBinMap, radius_average
from pgixs_python.calibration.calibration_io import load_standard_calibration
from pgixs_python.calibration.sdd import wavelength_from_energy_kev
from pgixs_python.export.data_export import save_series_data
from pgixs_python.io.h5_series import load_h5_series
from pgixs_python.io.image_loader import load_image_file
from pgixs_python.processing.corrections import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    detector_correction_factor,
)
from pgixs_python.processing.grid_remove import grid_remove
from pgixs_python.processing.masking import apply_detector_mask, detector_mask_for_type, dilate_gap_mask
from pgixs_python.processing.normalization import normalize_by_count
from pgixs_python.qmap.trxs import qmap_trxs
from pgixs_python.ui_backend.preview import downsample_1d_for_json, downsample_2d_for_json
from pgixs_python.ui_backend.series_processing import (
    apply_correction_to,
    build_circular_radius_map,
    compute_experiment_x_axis,
    process_azimuthal_stack,
    process_circular_stack,
    process_linecut_stack,
    process_roi_stack,
    sort_x_axis,
)

_INSITU_MODES = ("Time", "Temperature")


@dataclass
class TrXS2Session:
    detector: str = "Eiger4M"
    data: np.ndarray | None = None  # mean-of-stack display image
    stack: np.ndarray | None = None  # (n_frames, rows, cols)
    # TrXS2 has no per-frame-factor mode, so q-Cal keeps just the factor
    # (one detector-sized array) and applies it per frame at reduction
    # time instead of holding a second full copy of the stack.
    corrected_stack: np.ndarray | None = None
    correction_factor: np.ndarray | None = None
    mask: np.ndarray | None = None
    # Detector gap/dead pixels grown by 3 px -- blanked in the 1D
    # reductions (not in the 2D image, and not in ROI sums), matching where
    # MATLAB applies `dilatePGIXSGapMask`.
    gap_buffer: np.ndarray | None = None
    counts: np.ndarray | None = None  # (n_frames, 2) IC/BC
    timestamps: np.ndarray | None = None  # (n_frames,), baseline-relative
    temperatures: np.ndarray | None = None  # (n_frames, 3)
    file_paths: list[Path] = field(default_factory=list)
    source_path: str = ""
    # See gixs2_session.GIXS2Session.series_filename_prefix's own
    # docstring -- Auto Refresh reuses this rather than re-deriving a
    # prefix from file_paths[0]'s own name.
    series_filename_prefix: str = ""
    insitu_mode: str = "Time"
    timestamp_reference: float | None = None
    calibration: dict[str, Any] = field(
        default_factory=lambda: {
            "energy_kev": 9.5,
            "sdd_mm": 3000.0,
            "beam_pos": (1000.0, 1800.0),
            "pixel_mm": 0.075,
        }
    )
    qmap: dict[str, np.ndarray] | None = None
    angle_map: np.ndarray | None = None
    grid_removed: bool = False
    corrected: bool = False
    # Set True only by a *successful* trxs2_load_info -- trxs2_qcal refuses
    # to run without it, since `calibration` above always holds plausible-
    # looking placeholder values even before Load Info ever runs, and
    # silently q-Cal'ing against those produces a wrong-but-not-obviously-
    # wrong q-map (reported: "Load Info에 파일이 들어가 있지 않으면, q-cal이
    # 되지 않도록 해줘").
    calibration_loaded: bool = False
    # Background series, averaged into one 2D frame by
    # trxs2_background_load and corrected by the same detector-correction
    # factor as the main stack -- mirrors console_session.ConsoleSession's
    # background/background_counts fields.
    background: np.ndarray | None = None
    background_counts: np.ndarray | None = None
    # User-drawn ROI mask for Circular Process (True = keep, False =
    # exclude) -- mirrors console_session.py's own circular_mask.
    circular_mask: np.ndarray | None = None
    # The once-built RadiusBinMap from the last Circular Exec, reused
    # unchanged (never rebuilt) by both the background's own cut and Auto
    # Refresh's extension of newly-appended frames -- matches MATLAB's own
    # `param.qbin`/"radiusPrep" reuse.
    circular_radius_map: RadiusBinMap | None = None
    # The background's own once-computed cut (circular or line -- whichever
    # is active) / azimuthal cut, in raw (un-normalized) units. Subtraction
    # re-derives a per-frame-normalized version of these from scratch on
    # every render rather than caching a normalized copy.
    bkg_1d: np.ndarray | None = None
    bkg_azi: np.ndarray | None = None
    # Live "Subtraction" checkbox state -- read by trxs2_render_1d/
    # trxs2_render_azimuthal.
    subtract_background: bool = False
    roi_list: list[tuple[int, int, int, int]] = field(default_factory=list)
    last_1d: dict[str, Any] | None = None
    last_azimuthal: dict[str, Any] | None = None
    last_roi: np.ndarray | None = None
    x_axis_values: np.ndarray | None = None
    # Frame permutation that produced `x_axis_values` -- see `order_frames`.
    x_axis_order: np.ndarray | None = None
    x_axis_label: str = ""
    # Which result the Series Integration panel currently shows, and how
    # many frames were loaded as of the last full load/Auto-Refresh --
    # matches MATLAB's `Auto.Met`/`Auto.Nold`. "circular" is TrXS2's
    # primary 1D mode (vs. GIXS2's "linecut"), matching MATLAB's own
    # `Auto.Met = AUTO_CIRCULAR` default.
    auto_met: str = "circular"
    auto_n_old: int = 0
    load_progress: dict[str, Any] | None = None


_SESSION = TrXS2Session()


def get_session() -> TrXS2Session:
    return _SESSION


def set_session(session: TrXS2Session) -> None:
    global _SESSION
    _SESSION = session


def reset_session() -> None:
    global _SESSION
    _SESSION = TrXS2Session()


_SERIES_SUFFIX_RE = re.compile(r"^(.*)_\d{4,}$")


def _normalize_series_prefix(filename: str) -> str:
    """Port of `normalizePGIXSSeriesPrefix.m` (duplicated from
    `gixs2_session.py` rather than imported -- this module is meant to
    stand alone, same reasoning as `sddcal_session.py`/`gixs2_session.py`)."""

    stem = Path(filename).stem
    match = _SERIES_SUFFIX_RE.match(stem)
    return match.group(1) if match else stem


def _list_series_files(directory: Path, filename: str, extension: str = ".h5") -> list[Path]:
    """Port of `listPGIXSSeriesFiles.m` -- see `gixs2_session._list_series_files`'s
    own docstring for why this is a plain substring match against the
    normalized prefix (`prefix in item.name`), not equality against each
    candidate file's own separately-normalized prefix."""

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


def trxs2_load_progress() -> dict[str, Any]:
    progress = _SESSION.load_progress
    if progress is None:
        return {"ok": True, "active": False}
    return {"ok": True, "active": True, **progress}


def _metadata_mode_for_insitu(insitu_mode: str) -> str:
    return "" if insitu_mode == "Time" else insitu_mode


def _mean_ignoring_all_nan(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmean(stack, axis=0)


def _reset_for_new_series() -> None:
    """Reset per-series derived state on a fresh Load/Set-Table Import --
    deliberately leaves `background`/`background_counts`/
    `subtract_background` untouched (confirmed against `TrXS2_gui.m`'s own
    `Loadingdata`, which never clears `param.bkg`: a background stays
    loaded across a new series until the user explicitly reloads it via
    Bkgload2_Callback again)."""

    _SESSION.qmap = None
    _SESSION.angle_map = None
    _SESSION.corrected_stack = None
    _SESSION.correction_factor = None
    _SESSION.corrected = False
    _SESSION.grid_removed = False
    _SESSION.roi_list = []
    _SESSION.last_1d = None
    _SESSION.last_azimuthal = None
    _SESSION.last_roi = None
    _SESSION.x_axis_values = None
    _SESSION.x_axis_order = None
    _SESSION.x_axis_label = ""
    _SESSION.circular_mask = None
    _SESSION.circular_radius_map = None
    _SESSION.bkg_1d = None
    _SESSION.bkg_azi = None


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def trxs2_load(payload: dict[str, Any]) -> dict[str, Any]:
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
        # Transposed to match `stack`, stored in the mask's transpose.
        _SESSION.gap_buffer = (
            np.ascontiguousarray(dilate_gap_mask(detector_mask.mask, 3).T)
            if detector_mask.mask is not None
            else None
        )
        masked = apply_detector_mask(result.data, detector_mask.mask)
        masked = np.ascontiguousarray(masked.T)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    _SESSION.detector = detector
    # Kept in `detector_mask.mask`'s own native (pre-display-transpose)
    # orientation -- see gixs2_session.gixs2_load's own long comment on why
    # this must NOT be transposed a second time here.
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


def trxs2_list_series_files(payload: dict[str, Any]) -> dict[str, Any]:
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
    _SESSION.series_filename_prefix = filename
    return {"ok": True, "files": [f.name for f in files]}


def trxs2_load_series(payload: dict[str, Any]) -> dict[str, Any]:
    """Set/Table -> Import: load the checked series files into an
    in-memory stack, matching `Loadingdata()`. `parallel=True` loads files
    concurrently."""

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


def trxs2_load_info(payload: dict[str, Any]) -> dict[str, Any]:
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


def trxs2_grid_remove(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    """Applies `grid_remove` to every frame in `session.stack` individually
    (each frame's own reference-pixel ratio, not one correction reused
    across the series). Must run before q-Cal, not after: q-Cal caches its
    own `corrected_stack` derived from `stack` at that moment, and unlike
    MATLAB's single in-place-mutated array, this port never re-derives
    `corrected_stack` afterward -- running Grid Remove post-q-Cal would
    silently leave every downstream Circular/Line/Azimuthal Exec cutting
    the stale, not-grid-removed corrected data despite `session.data`'s own
    preview (and `session.stack`) showing the correction applied. Guarded
    the same way `trxs2_qcal`'s own one-shot lifecycle guard is."""

    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}
    if _SESSION.corrected:
        return {"ok": False, "message": "Run Grid Remove before q-Cal (reload the series to redo Grid Remove after q-Cal)."}

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
# Background
# ----------------------------------------------------------------------
def trxs2_background_load(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `Bkgload2_Callback`: pick one background file, then average
    every same-series file next to it (matched the same way
    `_list_series_files` matches a main-series prefix) into one 2D frame.

    Correctness fix over the literal MATLAB port: if q-Cal already ran
    (`_SESSION.corrected`), apply the same detector-correction factor to
    this newly-loaded background immediately, recomputed from the current
    calibration/q-map rather than requiring a separately-cached factor
    (same non-cached-recompute pattern `gixs2_auto_refresh` already uses
    for the main stack) -- MATLAB's own `Bkgload2_Callback` has no such
    fix: a background loaded *after* `Qcalc_Callback` has already run
    stays silently uncorrected there.
    """

    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}
    directory = str(payload.get("directory", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    if not directory or not filename:
        return {"ok": False, "message": "Choose a directory and a file for the background."}

    file_path = Path(directory) / filename
    extension = file_path.suffix or ".h5"
    try:
        files = _list_series_files(Path(directory), filename, extension)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    if not files:
        files = [file_path]

    target_shape = _SESSION.stack.shape[1:]
    running_sum = np.zeros(target_shape, dtype=np.float64)
    counts_sum = np.zeros(2, dtype=np.float64)
    loaded = 0
    meta_loaded = 0
    total = len(files)
    _SESSION.load_progress = None
    try:
        for idx, series_path in enumerate(files, start=1):
            _set_load_progress(idx, total, series_path.name)
            try:
                result = load_image_file(series_path)
                masked = apply_detector_mask(result.data, _SESSION.mask)
                masked = np.ascontiguousarray(masked.T)
            except Exception:  # noqa: BLE001
                continue
            if masked.shape != target_shape:
                continue
            running_sum += masked
            loaded += 1
            counts = np.asarray(result.counts, dtype=np.float64).reshape(-1)
            if counts.size >= 2 and np.all(np.isfinite(counts[:2])):
                counts_sum += counts[:2]
                meta_loaded += 1
    finally:
        _SESSION.load_progress = None

    if loaded == 0:
        return {"ok": False, "message": "No background frames could be loaded."}

    background = (running_sum / loaded).astype(np.float32)
    background[background == 0] = np.nan
    background_counts = (
        (counts_sum / meta_loaded).astype(np.float32) if meta_loaded else np.full(2, np.nan, dtype=np.float32)
    )

    _SESSION.background = background
    _SESSION.background_counts = background_counts

    warning = None
    if _SESSION.corrected and _SESSION.qmap is not None and _SESSION.background.shape == target_shape:
        try:
            factor = detector_correction_factor(
                DetectorCorrectionInputs(
                    data_size=target_shape, beam_position=_SESSION.calibration["beam_pos"],
                    pixel_size_mm=float(_SESSION.calibration["pixel_mm"]), sdd_mm=float(_SESSION.calibration["sdd_mm"]),
                    q_map=_SESSION.qmap["q"], energy_kev=float(_SESSION.calibration["energy_kev"]),
                    detector_type=_SESSION.detector,
                ),
                k0=float(2.0 * np.pi / wavelength_from_energy_kev(float(_SESSION.calibration["energy_kev"]))),
            )
            _SESSION.background = apply_detector_correction(_SESSION.background, factor)
        except Exception as exc:  # noqa: BLE001
            warning = f"Background loaded but correction failed: {exc}"

    result_dict = {"ok": True, "path": str(file_path), "n_files": loaded}
    if warning is not None:
        result_dict["warning"] = warning
    return result_dict


def trxs2_set_circular_mask(payload: dict[str, Any]) -> dict[str, Any]:
    """Set or clear the Circular Process ROI mask -- port of
    `console_set_circular_mask`, same True=keep convention."""

    if _SESSION.data is None:
        return {"ok": False, "message": "Load data first."}

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


# ----------------------------------------------------------------------
# q-Cal -- static only (TrXS is normal-incidence transmission; no
# incidence-angle/"Change ai" branch at all)
# ----------------------------------------------------------------------
def trxs2_qcal(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `Qcalc_Callback`. Mirrors MATLAB's own one-shot guard
    (`if param.correctionI == 1, alert 'Already corrected!'; return`)
    rather than being freely re-runnable like `gixs2_qcal` -- this keeps
    the background-correction bookkeeping below correct without a second
    cached-factor field: since this recomputes `corrected_stack` fresh
    from the untouched raw `stack` on every call (unlike MATLAB's in-place
    multiply), a freely-repeatable q-Cal would double-apply the correction
    factor to `session.background` on a second call. A new series load
    (`_reset_for_new_series`) resets `corrected` back to False, allowing
    exactly one more q-Cal run for that series -- same lifecycle as
    MATLAB's `correctionI`.
    """

    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}
    if not _SESSION.calibration_loaded:
        return {"ok": False, "message": "Run Load Info first."}
    if _SESSION.corrected:
        return {"ok": False, "message": "Already corrected!"}

    try:
        energy_kev = float(payload.get("energy_kev", _SESSION.calibration["energy_kev"]))
        sdd_mm = float(payload.get("sdd_mm", _SESSION.calibration["sdd_mm"]))
        beam_y = float(payload.get("beam_y", _SESSION.calibration["beam_pos"][0]))
        beam_z = float(payload.get("beam_z", _SESSION.calibration["beam_pos"][1]))
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

    try:
        qx, qy, qz, q = qmap_trxs(det_y=n_cols, det_z=n_rows, beam_pos=beam_pos, pixel_mm=pixel_mm, sdd_mm=sdd_mm, k0=k0)
        factor = detector_correction_factor(
            DetectorCorrectionInputs(
                data_size=shape, beam_position=beam_pos, pixel_size_mm=pixel_mm,
                sdd_mm=sdd_mm, q_map=q, energy_kev=energy_kev, detector_type=_SESSION.detector,
            ),
            k0=k0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    _SESSION.calibration.update({"energy_kev": energy_kev, "sdd_mm": sdd_mm, "beam_pos": beam_pos})
    _SESSION.qmap = {"qx": qx, "qy": qy, "qz": qz, "q": q}
    _SESSION.angle_map = pixel_angle_map(shape, center=beam_pos)
    _SESSION.correction_factor = factor
    _SESSION.corrected = True
    # mean_i(x_i / f) == mean_i(x_i) / f for a frame-independent f.
    _SESSION.data = apply_detector_correction(_mean_ignoring_all_nan(_SESSION.stack), factor)

    if _SESSION.background is not None and _SESSION.background.shape == shape:
        _SESSION.background = apply_detector_correction(_SESSION.background, factor)

    return {
        "ok": True,
        "calibration": dict(_SESSION.calibration),
        "preview": downsample_2d_for_json(_SESSION.data, max_points=1024),
    }


def _data_for_processing() -> np.ndarray:
    return _SESSION.corrected_stack if _SESSION.corrected_stack is not None else _SESSION.stack


def _processing_correction() -> np.ndarray | None:
    """Correction factor still to be applied to `_data_for_processing()`."""

    if _SESSION.corrected_stack is not None:
        return None
    return _SESSION.correction_factor


def corrected_frame(index: int) -> np.ndarray:
    """Frame `index` with the detector correction applied, whichever way this
    session happens to be holding it -- a materialized `corrected_stack`
    or the raw stack plus a shared `correction_factor`."""

    return apply_correction_to(_data_for_processing()[index], _processing_correction())


# ----------------------------------------------------------------------
# Circular / Line / Azimuthal / ROI series exec
# ----------------------------------------------------------------------
def trxs2_circular_exec(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `CircularAve2_Callback`: build the sector+mask geometry
    once, run Circular Average across the whole stack, and (if a
    background is loaded) cut the background through that exact same
    `RadiusBinMap` -- trivially aligned since the bin map is built purely
    from pixel/Q geometry, independent of any one frame's own data."""

    if _SESSION.stack is None or _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Cal first."}

    try:
        angle_min = float(payload.get("angle_min", 0.0))
        angle_max = float(payload.get("angle_max", 360.0))
        min_coverage = float(payload.get("min_coverage", 0.80))
        use_median = bool(payload.get("use_median", False))
        calc_std = bool(payload.get("calc_std", False))
        parallel = bool(payload.get("parallel", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid circular average parameters: {exc}"}

    beam_pos = _SESSION.calibration["beam_pos"]
    try:
        radius_map = build_circular_radius_map(
            _SESSION.qmap["q"], beam_pos, angle_range=(angle_min, angle_max), mask=_SESSION.circular_mask,
        )
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    if not radius_map.is_valid:
        return {"ok": False, "message": "No valid pixels in geometry mask."}

    def _report(completed: int, total: int) -> None:
        _set_load_progress(completed, total, "Circular average")

    _SESSION.load_progress = None
    try:
        result = process_circular_stack(
            _data_for_processing(), radius_map, use_median=use_median, calc_std=calc_std,
            correction_factor=_processing_correction(), gap_buffer=_SESSION.gap_buffer,
            min_coverage=min_coverage, progress_callback=_report, parallel=parallel,
        )
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    _SESSION.circular_radius_map = radius_map
    _SESSION.last_1d = {
        "x_axis": result.x_axis, "intensity": result.intensity, "coverage": result.coverage, "std": result.std,
        "params": {"use_median": use_median, "calc_std": calc_std, "min_coverage": min_coverage},
    }
    if _SESSION.background is not None:
        # The background frame is already corrected (trxs2_qcal /
        # trxs2_background_load), so only the gap shoulder is left to blank --
        # it has to be blanked on the same pixels as the stack, or Subtraction
        # takes a background that averaged gap-edge pixels the sample did not.
        bkg_result = radius_average(
            apply_correction_to(_SESSION.background, None, _SESSION.gap_buffer),
            radius_map, use_median=use_median, calc_std=False)
        _SESSION.bkg_1d = bkg_result.intensity
    else:
        _SESSION.bkg_1d = None
    _SESSION.auto_met = "circular"

    render = trxs2_render_1d({})
    return {
        "ok": True,
        "n_frames": int(result.intensity.shape[0]),
        "n_points": int(result.x_axis.shape[0]),
        "x": downsample_1d_for_json(result.x_axis, max_points=600),
        "render": render,
    }


def trxs2_line_exec(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `Linecut2_Callback`/"1D Line Ang": zero new averaging code,
    reuses `process_linecut_stack` with `cakecut=False, accumulation=False`
    (a plain angle cut -- TrXS never used Cakecut/Accumulation), plus one
    more call on the background frame for `bkg_1d`."""

    if _SESSION.stack is None or _SESSION.qmap is None:
        return {"ok": False, "message": "Run q-Cal first."}

    try:
        angle_deg = float(payload.get("angle_deg", 0.0)) % 360.0
        use_median = bool(payload.get("use_median", False))
        parallel = bool(payload.get("parallel", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid line parameters: {exc}"}

    beam_pos = _SESSION.calibration["beam_pos"]
    if _SESSION.angle_map is None or _SESSION.angle_map.shape != _SESSION.stack.shape[1:]:
        _SESSION.angle_map = pixel_angle_map(_SESSION.stack.shape[1:], center=beam_pos)

    kwargs = dict(angle_deg=angle_deg, cakecut=False, accumulation=False, use_median=use_median, parallel=parallel)

    def _report(completed: int, total: int) -> None:
        _set_load_progress(completed, total, "1D Line")

    _SESSION.load_progress = None
    try:
        result = process_linecut_stack(
            _data_for_processing(), _SESSION.qmap["q"], _SESSION.angle_map, center=beam_pos,
            correction_factor=_processing_correction(), gap_buffer=_SESSION.gap_buffer,
            progress_callback=_report, **kwargs,
        )
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    _SESSION.last_1d = {
        "x_axis": result.x_axis, "intensity": result.intensity, "coverage": result.coverage, "std": None,
        "params": kwargs,
    }
    if _SESSION.background is not None:
        try:
            bkg_result = process_linecut_stack(
                _SESSION.background[np.newaxis, ...], _SESSION.qmap["q"], _SESSION.angle_map, center=beam_pos,
                ref_axis=result.x_axis, gap_buffer=_SESSION.gap_buffer, **kwargs,
            )
            _SESSION.bkg_1d = bkg_result.intensity[0]
        except ValueError:
            # The background's own valid-pixel geometry didn't produce the
            # same-length axis as the main stack (see this module's own
            # test coverage) -- Subtraction just has nothing to subtract,
            # rather than failing the whole Exec.
            _SESSION.bkg_1d = None
    else:
        _SESSION.bkg_1d = None
    _SESSION.auto_met = "line1d"

    render = trxs2_render_1d({})
    return {
        "ok": True,
        "n_frames": int(result.intensity.shape[0]),
        "n_points": int(result.x_axis.shape[0]),
        "x": downsample_1d_for_json(result.x_axis, max_points=600),
        "render": render,
    }


def trxs2_azimuthal_exec(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `Azilinecut2_Callback`: calls existing `process_azimuthal_stack`
    with `angle_range=(0.0, 360.0)` (TrXS's full-ring transmission geometry,
    vs. GIXS2's half-circle), plus one `angular_profile` call on the
    background for `bkg_azi`."""

    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}

    try:
        pixel = float(payload.get("pixel", 202.0))
        range_minus = float(payload.get("range_minus", -1.0))
        range_plus = float(payload.get("range_plus", 1.0))
        use_median = str(payload.get("mean_or_median", "Mean")).strip().lower() == "median"
        # 360 (not GIXS2's 180) gives a 1deg/bin resolution over TrXS's own
        # full 360deg sweep -- a shared 180-bin default across both modes
        # left TrXS at a coarser 2deg/bin (reported: "azimuthal에서
        # angle간에 2도간격으로 되어 있는데, 1도간격으로 만들어줘").
        n_angle_bins = int(payload.get("n_angle_bins", 360))
        parallel = bool(payload.get("parallel", False))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid azimuthal parameters: {exc}"}

    beam_pos = _SESSION.calibration["beam_pos"]
    radius_range = (pixel + range_minus, pixel + range_plus)
    # `kwargs` is also reused below for the single-frame background cut via
    # `angular_profile`, which has no `parallel`/`max_workers` parameters --
    # keep `parallel` out of it and pass it only to the stack call.
    kwargs = dict(radius_range=radius_range, n_angle_bins=n_angle_bins, use_median=use_median, angle_range=(0.0, 360.0))

    def _report(completed: int, total: int) -> None:
        _set_load_progress(completed, total, "Azimuthal")

    _SESSION.load_progress = None
    try:
        result = process_azimuthal_stack(
            _data_for_processing(), center=beam_pos, progress_callback=_report, parallel=parallel,
            correction_factor=_processing_correction(), gap_buffer=_SESSION.gap_buffer, **kwargs
        )
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    finally:
        _SESSION.load_progress = None

    _SESSION.last_azimuthal = {
        "x_axis": result.x_axis, "intensity": result.intensity, "coverage": result.coverage, "params": kwargs,
    }
    if _SESSION.background is not None:
        bkg_result = angular_profile(
            apply_correction_to(_SESSION.background, None, _SESSION.gap_buffer),
            center=beam_pos, **kwargs)
        _SESSION.bkg_azi = bkg_result.intensity
    else:
        _SESSION.bkg_azi = None
    _SESSION.auto_met = "azimuthal"

    render = trxs2_render_azimuthal({})
    return {
        "ok": True,
        "n_frames": int(result.intensity.shape[0]),
        "n_points": int(result.x_axis.shape[0]),
        "x": downsample_1d_for_json(result.x_axis, max_points=600),
        "render": render,
    }


def trxs2_roi_exec(payload: dict[str, Any]) -> dict[str, Any]:
    """Direct copy of `gixs2_roi_exec` (`process_roi_stack` unchanged).
    No background subtraction -- confirmed in MATLAB (`ROIdef2_Callback`/
    `Saveget4_Callback` have no background-related code at all)."""

    if _SESSION.stack is None:
        return {"ok": False, "message": "Load a series first."}

    rois_raw = payload.get("rois", _SESSION.roi_list)
    try:
        rois = [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rois_raw]
    except (TypeError, ValueError, IndexError) as exc:
        return {"ok": False, "message": f"Invalid ROI list: {exc}"}
    if not rois:
        return {"ok": False, "message": "Draw at least one ROI first."}

    sums = process_roi_stack(_data_for_processing(), rois, correction_factor=_processing_correction())
    _SESSION.roi_list = rois
    _SESSION.last_roi = sums
    _SESSION.auto_met = "roi"
    return {"ok": True, "n_frames": int(sums.shape[0]), "n_roi": int(sums.shape[1]), "sums": sums.tolist()}


# ----------------------------------------------------------------------
# Render / Subtraction -- the core new runtime behavior. A cheap
# column-wise (per-frame) re-normalize-and-subtract over the already-
# computed stack curve matrix; never re-runs the per-frame averaging loop.
# ----------------------------------------------------------------------
def _normalized_series_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    """Per-frame IC/BC divide (or leave raw for "None") -- same loop shape
    as `gixs2_session.gixs2_normalized_matrix`, shared by both the live
    render below and Save Integ.'s ROI branch."""

    if method == "None" or _SESSION.counts is None:
        return np.asarray(matrix, dtype=np.float32).copy()
    column = 0 if method == "IC" else 1
    out = np.asarray(matrix, dtype=np.float32).copy()
    for i in range(matrix.shape[0]):
        out[i] = normalize_by_count(matrix[i], _SESSION.counts[i, column])
    return out


def _background_count_value(method: str) -> float | None:
    if method == "None" or _SESSION.background_counts is None:
        return None
    idx = 0 if method == "IC" else 1
    counts = _SESSION.background_counts
    if counts.size <= idx:
        return None
    value = float(counts[idx])
    return value if np.isfinite(value) else None


def trxs2_render_1d(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-normalize (None/IC/BC) + conditionally subtract the background
    over `session.last_1d["intensity"]` (the raw per-frame matrix Circular/
    Line Exec computed), storing the result as
    `session.last_1d["rendered"]` -- what gets plotted/saved. O(n_frames x
    n_bins); never touches the stack or re-runs averaging. Call on: Exec
    finishing, Normalize combo change, Subtraction toggle (pass
    `"subtract"` in `payload` to update `session.subtract_background` in
    the same call), Auto Refresh finishing."""

    if _SESSION.last_1d is None:
        return {"ok": False, "message": "Run Circular/Line Exec first."}

    if "subtract" in payload:
        _SESSION.subtract_background = bool(payload["subtract"])
    method = str(payload.get("normalize", _SESSION.last_1d.get("normalize", "None")))
    _SESSION.last_1d["normalize"] = method

    rendered = _normalized_series_matrix(_SESSION.last_1d["intensity"], method)
    if _SESSION.subtract_background and _SESSION.bkg_1d is not None:
        bkg_curve = normalize_by_count(_SESSION.bkg_1d, _background_count_value(method))
        rendered = rendered - bkg_curve[np.newaxis, :]

    _SESSION.last_1d["rendered"] = rendered.astype(np.float32)
    return {"ok": True, "x": downsample_1d_for_json(_SESSION.last_1d["x_axis"], max_points=600)}


def trxs2_render_azimuthal(payload: dict[str, Any]) -> dict[str, Any]:
    """Same shape as `trxs2_render_1d`, against `last_azimuthal`/`bkg_azi`."""

    if _SESSION.last_azimuthal is None:
        return {"ok": False, "message": "Run Azimuthal Exec first."}

    if "subtract" in payload:
        _SESSION.subtract_background = bool(payload["subtract"])
    method = str(payload.get("normalize", _SESSION.last_azimuthal.get("normalize", "None")))
    _SESSION.last_azimuthal["normalize"] = method

    rendered = _normalized_series_matrix(_SESSION.last_azimuthal["intensity"], method)
    if _SESSION.subtract_background and _SESSION.bkg_azi is not None:
        bkg_curve = normalize_by_count(_SESSION.bkg_azi, _background_count_value(method))
        rendered = rendered - bkg_curve[np.newaxis, :]

    _SESSION.last_azimuthal["rendered"] = rendered.astype(np.float32)
    return {"ok": True, "x": downsample_1d_for_json(_SESSION.last_azimuthal["x_axis"], max_points=600)}


# ----------------------------------------------------------------------
# X-axis
# ----------------------------------------------------------------------
def trxs2_set_x_axis(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.timestamps is None:
        return {"ok": False, "message": "Load a series first."}

    sub_mode = str(payload.get("sub_mode", ""))
    should_sort = bool(payload.get("order", False))
    raw_offset = payload.get("time_offset_seconds")
    time_offset_seconds = None if raw_offset is None else float(raw_offset)
    time_unit = str(payload.get("time_unit", "second"))

    try:
        values, label = compute_experiment_x_axis(
            _SESSION.insitu_mode, _SESSION.timestamps, _SESSION.temperatures,
            sub_mode=sub_mode, time_offset_seconds=time_offset_seconds, time_unit=time_unit,
        )
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    order = sort_x_axis(values, should_sort)
    _SESSION.x_axis_values = values[order]
    _SESSION.x_axis_order = order
    _SESSION.x_axis_label = label
    return {"ok": True, "x_axis": _SESSION.x_axis_values.tolist(), "label": label, "order": order.tolist()}


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
def order_frames(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Reorder a per-frame array to match `x_axis_values` (Temp.Sort permutes
    that axis out of file order, so anything paired with it must follow) --
    same contract as `gixs2_session.order_frames`."""

    order = _SESSION.x_axis_order
    array = np.asarray(values)
    if order is None or array.ndim <= axis or array.shape[axis] != order.shape[0]:
        return array
    return np.take(array, order, axis=axis)


def _column_labels() -> list[str]:
    if _SESSION.x_axis_values is not None:
        return [f"{v:g}" for v in _SESSION.x_axis_values]
    return [f"frame{i + 1}" for i in range(_SESSION.stack.shape[0])]


def _save_path(directory: str, suffix: str) -> Path:
    """See `gixs2_session._save_path`'s own docstring -- same
    `{directory}/{prefix}/{prefix}{suffix}` convention."""

    name_source = _SESSION.source_path or (str(_SESSION.file_paths[0]) if _SESSION.file_paths else "")
    prefix = _normalize_series_prefix(Path(name_source).name) if name_source else "series"
    return Path(directory) / prefix / f"{prefix}{suffix}"


def trxs2_save_1d(payload: dict[str, Any]) -> dict[str, Any]:
    """Saves `session.last_1d["rendered"]` directly (already Normalize/
    Subtraction-applied at render time) -- unlike `gixs2_save_1d`, which
    re-derives its own normalization fresh at save time, matching MATLAB's
    own `Saveget3_Callback`, which saves whatever `param.data1dstackN`
    already holds rather than recomputing it.

    When Circular Exec's Std checkbox populated `session.last_1d["std"]`
    (only Circular Average ever does -- Line cut has no std column, same
    as MATLAB's own `std1dstack`), each frame's std column is interleaved
    right after its own data column -- Q, frame1, frame1_STD, frame2,
    frame2_STD, ... -- in this one file, rather than a separate Save 1D
    STD file (reported: "Save 1D STD를 따로 세이브 하기보다는... Save 1D에서
    Column을 q, 1st data, 1st std, 2nd data, 2nd std 식으로 세이브 되면
    좋을 것 같아")."""

    if _SESSION.last_1d is None or _SESSION.last_1d.get("rendered") is None:
        return {"ok": False, "message": "Run Circular/Line Exec first."}
    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    rendered = order_frames(_SESSION.last_1d["rendered"])
    std = _SESSION.last_1d.get("std")
    if std is not None:
        std = order_frames(std)
    labels = _column_labels()
    if std is not None:
        n_frames = rendered.shape[0]
        matrix = np.empty((2 * n_frames, rendered.shape[1]), dtype=np.float32)
        matrix[0::2] = rendered
        matrix[1::2] = std
        column_labels = [item for label in labels for item in (label, f"{label}_STD")]
    else:
        matrix = rendered
        column_labels = labels

    path = _save_path(directory, "_1D.csv")
    try:
        save_series_data(path, _SESSION.last_1d["x_axis"], "Q_1/A", matrix, column_labels)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


def trxs2_save_azimuthal(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.last_azimuthal is None or _SESSION.last_azimuthal.get("rendered") is None:
        return {"ok": False, "message": "Run Azimuthal Exec first."}
    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    path = _save_path(directory, "_Azi.csv")
    try:
        save_series_data(
            path, _SESSION.last_azimuthal["x_axis"], "Angle_deg",
            order_frames(_SESSION.last_azimuthal["rendered"]), _column_labels(),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


def trxs2_save_integration(payload: dict[str, Any]) -> dict[str, Any]:
    """Save Integ.: whichever of Circular/Line/Azimuthal/ROI is currently
    active (`auto_met`), summed over its X-axis (rendered matrix for
    1D/Azimuthal, freshly-normalized-at-save-time for ROI, matching
    `gixs2_save_integration`'s own ROI branch)."""

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
        matrix = order_frames(_normalized_series_matrix(_SESSION.last_roi, method)).T
    elif _SESSION.auto_met == "azimuthal":
        if _SESSION.last_azimuthal is None or _SESSION.last_azimuthal.get("rendered") is None:
            return {"ok": False, "message": "Run Azimuthal Exec first."}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            matrix = order_frames(np.nansum(_SESSION.last_azimuthal["rendered"], axis=1, keepdims=True)).T
        columns = ["Integrated_Azimuthal"]
    else:
        if _SESSION.last_1d is None or _SESSION.last_1d.get("rendered") is None:
            return {"ok": False, "message": "Run Circular/Line Exec first."}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            matrix = order_frames(np.nansum(_SESSION.last_1d["rendered"], axis=1, keepdims=True)).T
        columns = ["Integrated_1D"]

    path = _save_path(directory, "_Integ.csv")
    try:
        save_series_data(path, _SESSION.x_axis_values, _SESSION.x_axis_label, matrix, columns)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "path": str(path)}


def trxs2_save_x_axis(payload: dict[str, Any]) -> dict[str, Any]:
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
def trxs2_auto_refresh(payload: dict[str, Any]) -> dict[str, Any]:
    """Port of `Autoplot_Callback`: rescan the directory for the same
    series prefix, load only the files beyond `auto_n_old`, append them to
    the stack/counts/timestamps/temperatures, and extend only the
    currently-active curve type (`auto_met`) -- then re-render so
    Subtraction/Normalize apply to the newly-appended columns too."""

    if _SESSION.stack is None or not _SESSION.file_paths:
        return {"ok": False, "message": "Load an initial series first."}

    directory = str(payload.get("directory", str(_SESSION.file_paths[0].parent)))
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

    _SESSION.stack = np.concatenate([_SESSION.stack, new_stack], axis=0)
    # q-Cal stores only the (geometry-only) correction factor, which already
    # covers the appended frames -- nothing to correct up front here, just
    # apply it to the refreshed mean image.
    _SESSION.data = apply_correction_to(
        _mean_ignoring_all_nan(_data_for_processing()), _processing_correction()
    )
    _SESSION.counts = np.concatenate([_SESSION.counts, result.counts.astype(np.float32)], axis=0) if _SESSION.counts is not None else result.counts
    _SESSION.timestamps = np.concatenate([_SESSION.timestamps, new_timestamps]) if _SESSION.timestamps is not None else new_timestamps
    _SESSION.temperatures = (
        np.concatenate([_SESSION.temperatures, result.temperatures.astype(np.float64)], axis=0)
        if _SESSION.temperatures is not None
        else result.temperatures
    )
    _SESSION.file_paths = _SESSION.file_paths + new_files

    n_appended = new_stack.shape[0]
    new_frames = new_stack if _SESSION.corrected_stack is None else _SESSION.corrected_stack[-n_appended:]
    extended = _extend_active_curve(new_frames)
    _SESSION.auto_n_old = n_total

    return {
        "ok": True,
        "n_new": len(new_files),
        "n_total": n_total,
        "curve_extended": extended,
        "preview": downsample_2d_for_json(_SESSION.data, max_points=1024),
    }


def _extend_active_curve(new_frames: np.ndarray) -> bool:
    """Append the new frames' contribution onto whichever curve type is
    active, then re-render (so Subtraction/Normalize apply to the newly-
    appended columns too) -- silently no-ops (returns False) if the new
    block's shape doesn't match the existing curve's, matching MATLAB's
    own append-only-if-shape-matches guard.

    "circular" reuses `session.circular_radius_map` unchanged (never
    rebuilt) -- matches `updateCircularAve`'s own `radiusPrep = p.qbin`
    reuse.
    """

    beam_pos = _SESSION.calibration.get("beam_pos")
    factor = _processing_correction()
    try:
        if _SESSION.auto_met == "circular" and _SESSION.last_1d is not None and _SESSION.circular_radius_map is not None:
            params = _SESSION.last_1d["params"]
            extra = process_circular_stack(
                new_frames, _SESSION.circular_radius_map, correction_factor=factor, gap_buffer=_SESSION.gap_buffer, **params
            )
            if extra.x_axis.shape != _SESSION.last_1d["x_axis"].shape:
                return False
            _SESSION.last_1d["intensity"] = np.vstack([_SESSION.last_1d["intensity"], extra.intensity])
            _SESSION.last_1d["coverage"] = np.vstack([_SESSION.last_1d["coverage"], extra.coverage])
            if _SESSION.last_1d.get("std") is not None and extra.std is not None:
                _SESSION.last_1d["std"] = np.vstack([_SESSION.last_1d["std"], extra.std])
            trxs2_render_1d({})
            return True
        if _SESSION.auto_met == "line1d" and _SESSION.last_1d is not None and _SESSION.qmap is not None:
            extra = process_linecut_stack(
                new_frames, _SESSION.qmap["q"], _SESSION.angle_map, center=beam_pos,
                ref_axis=_SESSION.last_1d["x_axis"], correction_factor=factor, gap_buffer=_SESSION.gap_buffer, **_SESSION.last_1d["params"],
            )
            if extra.x_axis.shape != _SESSION.last_1d["x_axis"].shape:
                return False
            _SESSION.last_1d["intensity"] = np.vstack([_SESSION.last_1d["intensity"], extra.intensity])
            _SESSION.last_1d["coverage"] = np.vstack([_SESSION.last_1d["coverage"], extra.coverage])
            trxs2_render_1d({})
            return True
        if _SESSION.auto_met == "azimuthal" and _SESSION.last_azimuthal is not None:
            extra = process_azimuthal_stack(
                new_frames, center=beam_pos, correction_factor=factor, gap_buffer=_SESSION.gap_buffer, **_SESSION.last_azimuthal["params"]
            )
            if extra.x_axis.shape != _SESSION.last_azimuthal["x_axis"].shape:
                return False
            _SESSION.last_azimuthal["intensity"] = np.vstack([_SESSION.last_azimuthal["intensity"], extra.intensity])
            _SESSION.last_azimuthal["coverage"] = np.vstack([_SESSION.last_azimuthal["coverage"], extra.coverage])
            trxs2_render_azimuthal({})
            return True
        if _SESSION.auto_met == "roi" and _SESSION.last_roi is not None and _SESSION.roi_list:
            extra = process_roi_stack(new_frames, _SESSION.roi_list, correction_factor=factor)
            _SESSION.last_roi = np.vstack([_SESSION.last_roi, extra])
            return True
    except ValueError:
        return False
    return False


__all__ = [
    "TrXS2Session",
    "get_session",
    "reset_session",
    "set_session",
    "trxs2_auto_refresh",
    "trxs2_azimuthal_exec",
    "trxs2_background_load",
    "trxs2_circular_exec",
    "trxs2_grid_remove",
    "trxs2_line_exec",
    "trxs2_list_series_files",
    "trxs2_load",
    "trxs2_load_info",
    "trxs2_load_progress",
    "trxs2_load_series",
    "trxs2_qcal",
    "trxs2_render_1d",
    "trxs2_render_azimuthal",
    "trxs2_roi_exec",
    "trxs2_save_1d",
    "trxs2_save_azimuthal",
    "trxs2_save_integration",
    "trxs2_save_x_axis",
    "trxs2_set_circular_mask",
    "trxs2_set_x_axis",
]
