"""In-process session backing the SDD Calibration tab.

Mirrors `console_session.py`'s architecture (one module-level session,
`console_*`-style functions that each return `{"ok": bool, ...}`), but for
MATLAB's standalone `SDDcal_gui.m` instead of `GIXS_gui.m`/`TrXS_gui.m`:
load one calibration-ring image (e.g. Silver behenate), mark/refine the beam
center, compute a radial or line-cut profile, Gaussian-fit a ring peak, and
compute + save the sample-to-detector distance (SDD) calibration that
GIXS/TrXS's own "Load Info" already reads back (`.json`/`.mat`).

`sddcal_calc_sdd` is the manual path (pixel radius + standard sample +
order + energy -> SDD); `sddcal_auto_run` is "Smart Auto Run" -- automatic
beam-center refinement (`calibration.center_refine`) + multi-ring
candidate search and order assignment (`calibration.ring_candidates`),
matching `SDDcal_gui.m`'s `AutoRun_Callback` step for step. Both funnel
into the same `_SESSION.sdd_result`/`sddcal_save`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from pgixs_python.averaging.pixel_radial import compute_radial_profile
from pgixs_python.calibration.calibration_io import (
    StandardCalibration,
    save_standard_calibration_json,
    save_standard_calibration_mat,
    save_standard_calibration_std,
)
from pgixs_python.calibration.center_refine import fine_tune_center, refine_center_from_ring
from pgixs_python.calibration.peak_fit import fit_gaussian_peak
from pgixs_python.calibration.ring_candidates import SDDCandidateReport, analyze_sdd_candidates, candidates_to_dicts
from pgixs_python.calibration.sdd import compute_sdd_from_ring
from pgixs_python.calibration.standards import standard_q_value
from pgixs_python.io.image_loader import load_image_file
from pgixs_python.processing.masking import apply_detector_mask, detector_mask_for_type
from pgixs_python.ui_backend.preview import downsample_1d_for_json, downsample_2d_for_json

_DETECTOR_INDEX = {
    "Eiger4M": "eiger4m",
    "MarCCD": "marccd",
    "SX165": "sx165",
    "MX170HS": "mx170hs",
    "PI-CCD": "piccd",
    "PI_CCD": "piccd",
    "MAR345": "mar345",
}


@dataclass
class SDDcalSession:
    detector: str = "Eiger4M"
    data: np.ndarray | None = None
    mask: np.ndarray | None = None
    pixel_size_mm: float = 0.075
    source_path: str = ""
    center: tuple[float, float] = (0.0, 0.0)  # (y, z), 1-based pixel coordinates
    last_curve: dict[str, Any] | None = None
    fit_result: dict[str, Any] | None = None
    sdd_result: dict[str, Any] | None = None
    candidate_report: SDDCandidateReport | None = field(default=None, repr=False)


_SESSION = SDDcalSession()


def get_session() -> SDDcalSession:
    """Return the module-level session (mainly for tests)."""

    return _SESSION


def set_session(session: SDDcalSession) -> None:
    """Install `session` as the active module-level session."""

    global _SESSION
    _SESSION = session


def reset_session() -> None:
    """Reset the module-level session to a fresh state (mainly for tests)."""

    global _SESSION
    _SESSION = SDDcalSession()


def sddcal_load(payload: dict[str, Any]) -> dict[str, Any]:
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
        # Same raw-frame axis-transpose fix console_load applies (this
        # beamline's H5 frames come out rotated 90deg relative to the real
        # y/z detector axes) -- keeps SDDcal's pixel convention consistent
        # with GIXS/TrXS, since they share calibration files.
        masked = np.ascontiguousarray(masked.T)
    except Exception as exc:  # noqa: BLE001 - surface as a visible failure.
        return {"ok": False, "message": str(exc)}

    # Openimage_Callback's saturated-pixel clip.
    masked = masked.astype(np.float32)
    masked[masked >= 1.0e9] = np.nan

    # Unlike console_session.py (which only ever re-applies `_SESSION.mask`
    # to a fresh, untransposed frame), sddcal_radial_profile below uses the
    # stored mask directly against `_SESSION.data`, which is already
    # transposed -- so the mask has to be transposed the same way to match.
    mask_t = np.ascontiguousarray(detector_mask.mask.T) if detector_mask.mask is not None else None

    _SESSION.detector = detector
    _SESSION.data = masked
    _SESSION.mask = mask_t
    _SESSION.pixel_size_mm = detector_mask.pixel_size_mm
    _SESSION.source_path = str(file_path)
    _SESSION.last_curve = None
    _SESSION.fit_result = None
    _SESSION.sdd_result = None
    _SESSION.candidate_report = None

    rows, cols = masked.shape
    _SESSION.center = (cols / 2.0, rows / 2.0)

    return {
        "ok": True,
        "path": str(file_path),
        "shape": list(masked.shape),
        "center": list(_SESSION.center),
        "pixel_size_mm": _SESSION.pixel_size_mm,
        "mask_report": {
            "fallback_used": detector_mask.report.fallback_used,
            "message": detector_mask.report.message,
        },
        "preview": downsample_2d_for_json(masked, max_points=1024),
    }


def sddcal_set_center(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    try:
        y = float(payload["y"])
        z = float(payload["z"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid center: {exc}"}

    _SESSION.center = (y, z)
    return {"ok": True, "center": [y, z]}


def sddcal_line_cut(payload: dict[str, Any]) -> dict[str, Any]:
    """Y_Plot ('y') or Z_Plot ('z'): a beam-centered line cut through the
    current center, matching `Yplot_Callback`/`Zplot_Callback`."""

    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    direction = str(payload.get("direction", "y")).strip().lower()
    if direction not in ("y", "z"):
        return {"ok": False, "message": "direction must be 'y' or 'z'."}

    rows, cols = _SESSION.data.shape
    beam_y, beam_z = _SESSION.center

    if direction == "y":
        target_row = max(1, min(rows, int(round(beam_z))))
        values = _SESSION.data[target_row - 1, :]
        x_axis = np.arange(1, cols + 1, dtype=np.float64) - beam_y
    else:
        target_col = max(1, min(cols, int(round(beam_y))))
        values = _SESSION.data[:, target_col - 1]
        x_axis = np.arange(1, rows + 1, dtype=np.float64) - beam_z

    _SESSION.last_curve = {"kind": f"{direction}plot", "x": x_axis, "y": np.asarray(values, dtype=np.float64)}
    _SESSION.fit_result = None

    return {
        "ok": True,
        "direction": direction,
        "preview": {
            "x": downsample_1d_for_json(x_axis, max_points=600),
            "y": downsample_1d_for_json(values, max_points=600),
        },
    }


def sddcal_radial_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Rad.Ave (Fit): a pixel-radius profile around the current center,
    matching `CircularAve_Callback`."""

    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    use_median = bool(payload.get("use_median", False))
    geom_mask = _SESSION.mask.astype(bool) if _SESSION.mask is not None else None

    profile = compute_radial_profile(_SESSION.data, center=_SESSION.center, geom_mask=geom_mask, use_median=use_median)
    if profile.r_axis.size == 0:
        return {"ok": False, "message": "No valid pixels for a radial profile at this center."}

    _SESSION.last_curve = {"kind": "radial", "x": profile.r_axis.astype(np.float64), "y": profile.intensity.astype(np.float64)}
    _SESSION.fit_result = None

    return {
        "ok": True,
        "preview": {
            "x": downsample_1d_for_json(profile.r_axis, max_points=600),
            "y": downsample_1d_for_json(profile.intensity, max_points=600),
            "coverage": downsample_1d_for_json(profile.coverage, max_points=600),
        },
    }


def sddcal_fit_peak(payload: dict[str, Any]) -> dict[str, Any]:
    """Gaussian-fit the last curve (line cut or radial profile) over a
    selected range. `mode="single"` needs `c1`/`c2`; `mode="symmetric"`
    (line cuts only, matches `OneDplotpixel_gui.m`'s dual-peak mode) needs
    `cl1`/`cl2`/`cr1`/`cr2` -- the true center offset is the midpoint of the
    two fitted peak centers, and `radius` is half their separation."""

    if _SESSION.last_curve is None:
        return {"ok": False, "message": "Select a curve (Y_Plot/Z_Plot/Rad.Ave) first."}

    x = np.asarray(_SESSION.last_curve["x"], dtype=np.float64)
    y = np.asarray(_SESSION.last_curve["y"], dtype=np.float64)
    mode = str(payload.get("mode", "single")).strip().lower()

    if mode == "single":
        try:
            c1 = float(payload["c1"])
            c2 = float(payload["c2"])
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "message": f"Invalid range: {exc}"}
        sel = (x >= min(c1, c2)) & (x <= max(c1, c2))
        result = fit_gaussian_peak(x[sel], y[sel])
        if not result.success:
            return {"ok": False, "message": result.message}
        _SESSION.fit_result = {"mode": "single", "center": result.center, "radius": None}
        return {
            "ok": True,
            "mode": "single",
            "center": result.center,
            "sigma": result.sigma,
            "amplitude": result.amplitude,
            "offset": result.offset,
            "r_square": result.r_square,
            "fit_x": downsample_1d_for_json(result.fit_x, max_points=600),
            "fit_y": downsample_1d_for_json(result.fit_y, max_points=600),
        }

    if mode == "symmetric":
        if _SESSION.last_curve["kind"] == "radial":
            return {"ok": False, "message": "Symmetric fit mode only applies to Y_Plot/Z_Plot line cuts."}
        try:
            cl1 = float(payload["cl1"])
            cl2 = float(payload["cl2"])
            cr1 = float(payload["cr1"])
            cr2 = float(payload["cr2"])
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "message": f"Invalid range: {exc}"}
        sel_l = (x >= min(cl1, cl2)) & (x <= max(cl1, cl2))
        sel_r = (x >= min(cr1, cr2)) & (x <= max(cr1, cr2))
        left = fit_gaussian_peak(x[sel_l], y[sel_l])
        if not left.success:
            return {"ok": False, "message": f"Left peak fit failed: {left.message}"}
        right = fit_gaussian_peak(x[sel_r], y[sel_r])
        if not right.success:
            return {"ok": False, "message": f"Right peak fit failed: {right.message}"}

        center_offset = (left.center + right.center) / 2.0
        ring_radius = abs(right.center - left.center) / 2.0
        _SESSION.fit_result = {"mode": "symmetric", "center": center_offset, "radius": ring_radius}
        return {
            "ok": True,
            "mode": "symmetric",
            "center": center_offset,
            "radius": ring_radius,
            "left_fit_x": downsample_1d_for_json(left.fit_x, max_points=600),
            "left_fit_y": downsample_1d_for_json(left.fit_y, max_points=600),
            "right_fit_x": downsample_1d_for_json(right.fit_x, max_points=600),
            "right_fit_y": downsample_1d_for_json(right.fit_y, max_points=600),
        }

    return {"ok": False, "message": f"Unknown fit mode: {mode}"}


def sddcal_apply_fit_to_center(payload: dict[str, Any]) -> dict[str, Any]:
    """"Send to Main": apply the last fit's offset onto the beam center
    along whichever axis the last curve was taken (matches
    `OneDplotpixel_gui.m`'s `Sending_Callback`)."""

    if _SESSION.fit_result is None or _SESSION.last_curve is None:
        return {"ok": False, "message": "Fit a peak first."}

    kind = _SESSION.last_curve["kind"]
    offset = _SESSION.fit_result["center"]
    beam_y, beam_z = _SESSION.center
    if kind == "yplot":
        beam_y = beam_y + offset
    elif kind == "zplot":
        beam_z = beam_z + offset
    else:
        return {"ok": False, "message": "Apply to Center only applies to a Y_Plot/Z_Plot fit."}

    _SESSION.center = (round(beam_y, 2), round(beam_z, 2))
    return {"ok": True, "center": list(_SESSION.center)}


def sddcal_calc_sdd(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    try:
        pixel_radius = float(payload["pixel_radius"])
        order = int(payload.get("order", 1))
        energy_kev = float(payload["energy_kev"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid Calc SDD parameters: {exc}"}
    standard_sample = str(payload.get("standard_sample", "Silver behenate"))

    try:
        q_value = standard_q_value(standard_sample, order)
        result = compute_sdd_from_ring(pixel_radius, _SESSION.pixel_size_mm, energy_kev, q_value)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    _SESSION.sdd_result = {
        "sdd_mm": result.sdd,
        "wavelength": result.wavelength,
        "two_theta_deg": result.two_theta_deg,
        "pixel_radius": pixel_radius,
        "q_value": q_value,
        "order": order,
        "standard_sample": standard_sample,
        "energy_kev": energy_kev,
    }
    return {"ok": True, **_SESSION.sdd_result}


def sddcal_auto_run(payload: dict[str, Any]) -> dict[str, Any]:
    """"Smart Auto Run": automatic beam-center refinement + multi-ring SDD
    candidate search, matching `SDDcal_gui.m`'s `AutoRun_Callback` step for
    step. Starts from whatever center is already set (`_SESSION.center`)
    -- like MATLAB, this is a *local* refinement, not a global search, so
    it needs a reasonably close starting point (e.g. a rough click), not a
    blind image-center guess."""

    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}

    standard_sample = str(payload.get("standard_sample", "Silver behenate"))
    try:
        energy_kev = float(payload.get("energy_kev", 9.5))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid energy: {exc}"}

    # Both refinement steps fall back like AutoRun_Callback's own try/catch,
    # but say so: the center they leave behind fixes every q axis downstream.
    warnings: list[str] = []

    # Step 1: coarse center refinement (circle fit to the strongest ring).
    approx_y, approx_z = _SESSION.center
    try:
        refined_y, refined_z = refine_center_from_ring(_SESSION.data, (approx_y, approx_z))
    except Exception as exc:  # noqa: BLE001
        refined_y, refined_z = approx_y, approx_z
        warnings.append(f"Center refinement failed ({exc}); keeping the center you set. Please verify it.")

    # Step 2: fine-tune using the single strongest ring above r=80 with
    # good coverage -- AutoRun_Callback's own separate, simpler peak
    # search, not step 3's multi-candidate one.
    try:
        profile = compute_radial_profile(_SESSION.data, center=(refined_y, refined_z))
        if profile.r_axis.size:
            valid = (profile.r_axis > 80.0) & (profile.coverage >= 0.80)
            finite_full = profile.intensity[np.isfinite(profile.intensity)]
            if np.any(valid) and finite_full.size:
                prominence = float(np.max(finite_full)) * 0.05
                local_intensity = profile.intensity[valid]
                local_peaks, properties = find_peaks(local_intensity, prominence=prominence)
                if local_peaks.size:
                    strongest_first = np.argsort(properties["prominences"])[::-1]
                    best_local = local_peaks[strongest_first[0]]
                    valid_idx = np.flatnonzero(valid)
                    target_radius = float(profile.r_axis[valid_idx[best_local]])
                    refined_y, refined_z = fine_tune_center(_SESSION.data, (refined_y, refined_z), target_radius)
    except Exception as exc:  # noqa: BLE001 - best-effort; keep the step-1 center on failure.
        warnings.append(f"Center fine-tuning failed ({exc}); keeping the coarse center. Please verify it.")

    _SESSION.center = (round(refined_y, 2), round(refined_z, 2))

    # Step 3: multi-ring candidate search + order assignment at the refined center.
    profile = compute_radial_profile(_SESSION.data, center=_SESSION.center)
    if profile.r_axis.size == 0:
        return {"ok": False, "message": "No valid pixels for a radial profile at the refined center.", "center": list(_SESSION.center)}

    report = analyze_sdd_candidates(
        profile.r_axis,
        profile.intensity,
        profile.coverage,
        standard_sample,
        energy_kev,
        _SESSION.pixel_size_mm,
        min_coverage=0.80,
    )
    _SESSION.candidate_report = report
    accepted = [c for c in report.candidates if c.accepted]
    if not accepted:
        message = f"No reliable rings found with coverage >= 0.80. {report.message}".strip()
        return {"ok": False, "message": message, "center": list(_SESSION.center), "candidates": candidates_to_dicts(report)}

    primary = accepted[0]
    fit = fit_gaussian_peak(profile.r_axis, profile.intensity, window_center=primary.rough_radius, half_width=30.0)
    if not fit.success:
        return {
            "ok": False,
            "message": f"Gaussian fitting failed: {fit.message}",
            "center": list(_SESSION.center),
            "candidates": candidates_to_dicts(report),
        }

    _SESSION.last_curve = {"kind": "radial", "x": profile.r_axis.astype(np.float64), "y": profile.intensity.astype(np.float64)}
    _SESSION.fit_result = {"mode": "single", "center": fit.center, "radius": None}

    # Only used to derive wavelength/two_theta for display -- the actual
    # SDD kept is report.final_sdd (median across all accepted rings, more
    # robust than any single ring), matching AutoRun_Callback discarding
    # this call's own SDD output (`[~, sddInfo] = computePGIXSSDDFromRing(...)`).
    ring_info = compute_sdd_from_ring(fit.center, _SESSION.pixel_size_mm, energy_kev, primary.q_value)
    _SESSION.sdd_result = {
        "sdd_mm": report.final_sdd,
        "wavelength": ring_info.wavelength,
        "two_theta_deg": ring_info.two_theta_deg,
        "pixel_radius": fit.center,
        "q_value": primary.q_value,
        "order": primary.order,
        "standard_sample": standard_sample,
        "energy_kev": energy_kev,
    }

    if np.isfinite(fit.r_square) and fit.r_square < 0.85:
        warnings.append(f"Gaussian fit quality is low (R^2={fit.r_square:.3f}). Please verify manually.")
    if report.accepted_count > 1 and np.isfinite(report.sdd_std) and report.sdd_std > 5.0:
        warnings.append(f"Multi-ring SDD spread is {report.sdd_std:.2f} mm. Please verify manually.")

    return {
        "ok": True,
        "center": list(_SESSION.center),
        "pixel_radius": fit.center,
        "order": primary.order,
        "q_value": primary.q_value,
        "sdd_mm": report.final_sdd,
        "sdd_std": report.sdd_std,
        "accepted_count": report.accepted_count,
        "r_square": fit.r_square,
        "candidates": candidates_to_dicts(report),
        "warnings": warnings,
        "profile": {
            "x": downsample_1d_for_json(profile.r_axis, max_points=600),
            "y": downsample_1d_for_json(profile.intensity, max_points=600),
        },
        "fit_x": downsample_1d_for_json(fit.fit_x, max_points=600),
        "fit_y": downsample_1d_for_json(fit.fit_y, max_points=600),
    }


def sddcal_save(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.data is None:
        return {"ok": False, "message": "Load a detector image first."}
    if _SESSION.sdd_result is None:
        return {"ok": False, "message": "Run Calc SDD first."}
    if not _SESSION.source_path:
        return {"ok": False, "message": "No source file to derive the save path from."}

    source = Path(_SESSION.source_path)
    out_dir = Path(payload["out_dir"]) if payload.get("out_dir") else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source.stem}_standard"

    # `_SESSION.center` is 1-based, the same convention MATLAB uses and the
    # one every pixel grid here is built on (`np.arange(1, n+1)` in
    # `pixel_radial`/`center_refine`/`qmap`). MATLAB's `.mat`/`.std` are
    # 1-based too (`SDDcal_gui.m` writes the raw GUI values), so every file
    # below gets the session center verbatim -- an earlier +1 shift on
    # `.mat`/`.std` here was compensating for a 0-based convention this app
    # does not actually use, and put those files one pixel off on both axes.
    # (MATLAB's own `.json` is separately 0-based; see the note in
    # `load_standard_calibration` about reading MATLAB-written JSON.)
    calibration_json = StandardCalibration(
        beam_center=np.asarray(_SESSION.center, dtype=np.float64),
        wavelength=_SESSION.sdd_result["wavelength"],
        pixel_size=_SESSION.pixel_size_mm,
        sdd=_SESSION.sdd_result["sdd_mm"],
        detector=_SESSION.detector,
        energy_kev=_SESSION.sdd_result["energy_kev"],
        standard_peak=np.asarray([_SESSION.sdd_result["pixel_radius"]], dtype=np.float64),
        standard_q_value=_SESSION.sdd_result["q_value"],
        detector_index=_DETECTOR_INDEX.get(_SESSION.detector, "unknown"),
        image_shape=np.asarray(_SESSION.data.shape, dtype=np.float64),
    )

    json_path = out_dir / f"{stem}.json"
    save_standard_calibration_json(json_path, calibration_json)

    mat_path = out_dir / f"{stem}.mat"
    mat_message = ""
    try:
        save_standard_calibration_mat(mat_path, calibration_json)
        mat_saved = True
    except Exception as exc:  # noqa: BLE001 - .mat/.std are best-effort, .json already saved.
        mat_saved = False
        mat_message = str(exc)

    std_path = out_dir / f"{stem}.std"
    std_message = ""
    try:
        save_standard_calibration_std(std_path, calibration_json)
        std_saved = True
    except Exception as exc:  # noqa: BLE001
        std_saved = False
        std_message = str(exc)

    return {
        "ok": True,
        "json_path": str(json_path),
        "mat_path": str(mat_path) if mat_saved else None,
        "mat_message": mat_message,
        "std_path": str(std_path) if std_saved else None,
        "std_message": std_message,
    }


__all__ = [
    "SDDcalSession",
    "get_session",
    "reset_session",
    "sddcal_apply_fit_to_center",
    "sddcal_auto_run",
    "sddcal_calc_sdd",
    "sddcal_fit_peak",
    "sddcal_line_cut",
    "sddcal_load",
    "sddcal_radial_profile",
    "sddcal_save",
    "sddcal_set_center",
    "set_session",
]
