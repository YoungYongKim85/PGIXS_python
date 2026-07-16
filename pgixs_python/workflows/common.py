"""Shared command-line helpers for PGIXS Tab1 workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pgixs_python.calibration.calibration_io import StandardCalibration, load_standard_calibration
from pgixs_python.io.image_loader import load_image_file
from pgixs_python.io.h5_loader import DEFAULT_DETECTOR_DATASET
from pgixs_python.io.h5_series import H5SeriesLoadResult, load_h5_series
from pgixs_python.io.two_d_info import TwoDInfo, save_two_d_info_h5
from pgixs_python.processing import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    detector_correction_factor,
    detector_mask_for_type,
    grid_remove,
    process_normalization,
)
from pgixs_python.qmap.gixs import qmap_gixs
from pgixs_python.qmap.trxs import qmap_trxs


@dataclass
class WorkflowQMap:
    calibration: StandardCalibration
    q_map: np.ndarray
    qz_map: np.ndarray
    qx_map: np.ndarray | None = None
    qy_map: np.ndarray | None = None
    qr_map: np.ndarray | None = None
    tth2d: np.ndarray | None = None
    af2d: np.ndarray | None = None


@dataclass
class WorkflowProcessingResult:
    stack_result: H5SeriesLoadResult
    diagnostics: dict[str, np.ndarray | str | int | float]


def add_h5_input_arguments(parser) -> None:
    parser.add_argument("--input", required=False, help="Input H5 detector file or folder.")
    parser.add_argument("--dataset", default=DEFAULT_DETECTOR_DATASET, help="H5 detector dataset path.")
    parser.add_argument("--metadata-mode", default="", help="Metadata scan mode: Temperature, LowTemp., Angle, or blank.")
    parser.add_argument("--reference-timestamp", type=float, default=None, help="Optional timestamp zero-point.")
    parser.add_argument("--retry-count", type=int, default=1, help="Frame load retry count before NaN fallback.")


def add_calibration_arguments(parser, include_incident_angle: bool = False) -> None:
    parser.add_argument("--standard", required=False, help="Standard/calibration MAT/JSON file.")
    parser.add_argument("--detector", default="", help="Expected detector name for calibration validation.")
    if include_incident_angle:
        parser.add_argument("--incident-angle", type=float, default=0.0, help="GIXS incident angle in degrees.")


def add_output_arguments(parser) -> None:
    parser.add_argument("--out", required=False, help="Output H5 file or directory for Python-native 2D-info.")


def add_processing_arguments(parser) -> None:
    parser.add_argument(
        "--normalization",
        default="None",
        choices=["None", "IC", "BC"],
        help="Per-frame normalization mode before averaging.",
    )
    parser.add_argument("--background", required=False, help="Optional background image file.")
    parser.add_argument("--background-ic", type=float, default=None, help="Optional background IC counter.")
    parser.add_argument("--background-bc", type=float, default=None, help="Optional background BC counter.")
    parser.add_argument(
        "--no-clamp-negative",
        action="store_true",
        help="Keep negative values after background subtraction instead of clamping them to zero.",
    )
    parser.add_argument("--mask-detector", action="store_true", help="Apply detector default mask while loading H5 frames.")
    parser.add_argument("--grid-remove", action="store_true", help="Apply Eiger4M grid-line correction per frame.")
    parser.add_argument(
        "--detector-correction",
        action="store_true",
        help="Apply solid-angle/polarization detector correction after normalization.",
    )


def collect_h5_inputs(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Only H5/HDF5 inputs are wired into the workflow scaffold: {path}")
        return [path]
    if path.is_dir():
        files = sorted([*path.glob("*.h5"), *path.glob("*.hdf5")])
        if not files:
            raise FileNotFoundError(f"No .h5/.hdf5 files found in input folder: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def load_h5_input_stack(args) -> H5SeriesLoadResult:
    files = collect_h5_inputs(args.input)
    mask = None
    if getattr(args, "mask_detector", False):
        mask = detector_mask_for_type(getattr(args, "detector", "")).mask
    return load_h5_series(
        files,
        dataset_path=args.dataset,
        mask=mask,
        retry_count=args.retry_count,
        metadata_mode=args.metadata_mode,
        reference_timestamp=args.reference_timestamp,
    )


def format_h5_series_summary(label: str, result: H5SeriesLoadResult) -> str:
    return (
        f"{label} H5 input loaded\n"
        f"  frames: {result.stack.shape[0]}\n"
        f"  image shape: {result.stack.shape[1]} x {result.stack.shape[2]}\n"
        f"  retry failed frames: {result.retry_failed_frames.tolist()}\n"
        f"  metadata failed frames: {result.metadata_failed_frames.tolist()}\n"
    )


def load_workflow_qmap(
    args,
    image_shape: tuple[int, int],
    mode: str,
) -> WorkflowQMap | None:
    if not getattr(args, "standard", None):
        return None

    calibration, report = load_standard_calibration(args.standard, expected_detector=getattr(args, "detector", ""))
    if not report.ok or calibration is None:
        raise RuntimeError(f"Calibration load failed: {report.message}")

    det_z, det_y = (int(image_shape[0]), int(image_shape[1]))
    k0 = 2.0 * np.pi / float(calibration.wavelength)
    beam_pos = (float(calibration.beam_center[0]), float(calibration.beam_center[1]))
    if mode == "trxs":
        qx, qy, qz, q = qmap_trxs(det_y, det_z, beam_pos, calibration.pixel_size, calibration.sdd, k0)
        return WorkflowQMap(calibration=calibration, q_map=q, qz_map=qz, qx_map=qx, qy_map=qy)
    if mode == "gixs":
        qy, qr, qz, q, tth2d, af2d = qmap_gixs(
            det_y,
            det_z,
            beam_pos,
            calibration.pixel_size,
            calibration.sdd,
            k0,
            float(getattr(args, "incident_angle", 0.0)),
        )
        return WorkflowQMap(calibration=calibration, q_map=q, qz_map=qz, qy_map=qy, qr_map=qr, tth2d=tth2d, af2d=af2d)
    raise ValueError(f"Unknown workflow qmap mode: {mode}")


def format_qmap_summary(result: WorkflowQMap | None) -> str:
    if result is None:
        return "  qmap: skipped; pass --standard to build Q maps\n"
    cal = result.calibration
    finite_q = result.q_map[np.isfinite(result.q_map)]
    finite_qz = result.qz_map[np.isfinite(result.qz_map)]
    q_min = float(np.min(finite_q)) if finite_q.size else float("nan")
    q_max = float(np.max(finite_q)) if finite_q.size else float("nan")
    qz_min = float(np.min(finite_qz)) if finite_qz.size else float("nan")
    qz_max = float(np.max(finite_qz)) if finite_qz.size else float("nan")
    return (
        "  qmap: built from standard calibration\n"
        f"  detector: {cal.detector or '(unspecified)'}\n"
        f"  beam center: {cal.beam_center.tolist()}\n"
        f"  wavelength: {cal.wavelength:g} A\n"
        f"  sdd: {cal.sdd:g} mm\n"
        f"  pixel size: {cal.pixel_size:g} mm\n"
        f"  q range: {q_min:g} to {q_max:g}\n"
        f"  qz range: {qz_min:g} to {qz_max:g}\n"
    )


def process_workflow_stack(
    args,
    stack_result: H5SeriesLoadResult,
    qmap_result: WorkflowQMap | None = None,
) -> WorkflowProcessingResult:
    """Apply optional Tab1 processing steps to each loaded frame."""

    stack = np.asarray(stack_result.stack, dtype=np.float32).copy()
    frame_count = stack.shape[0]
    background, background_counts = _load_background_for_processing(args)
    norm_success = np.ones(frame_count, dtype=np.int8)
    grid_applied = np.zeros(frame_count, dtype=np.int8)
    correction_applied = np.zeros(frame_count, dtype=np.int8)

    correction_factor = None
    if getattr(args, "detector_correction", False):
        if qmap_result is None:
            raise RuntimeError("--detector-correction requires --standard because Q maps are required.")
        cal = qmap_result.calibration
        k0 = 2.0 * np.pi / float(cal.wavelength)
        correction_factor = detector_correction_factor(
            DetectorCorrectionInputs(
                data_size=tuple(int(v) for v in stack.shape[1:3]),
                beam_position=(float(cal.beam_center[0]), float(cal.beam_center[1])),
                pixel_size_mm=float(cal.pixel_size),
                sdd_mm=float(cal.sdd),
                q_map=qmap_result.q_map,
                energy_kev=float(cal.energy_kev) if cal.energy_kev is not None else float("nan"),
                detector_type=cal.detector or getattr(args, "detector", ""),
            ),
            k0=k0,
        )

    for idx in range(frame_count):
        frame, norm_report = process_normalization(
            stack[idx],
            background=background,
            counts=stack_result.counts[idx, :] if stack_result.counts.size else None,
            background_counts=background_counts,
            method=getattr(args, "normalization", "None"),
            clamp_negative=not getattr(args, "no_clamp_negative", False),
        )
        norm_success[idx] = 1 if norm_report.success else 0

        if getattr(args, "grid_remove", False):
            frame, grid_report = grid_remove(frame, detector_type=getattr(args, "detector", ""))
            grid_applied[idx] = 1 if grid_report.applied else 0

        if correction_factor is not None:
            frame = apply_detector_correction(frame, correction_factor)
            correction_applied[idx] = 1

        stack[idx, :, :] = frame

    processed = replace(stack_result, stack=stack)
    diagnostics: dict[str, np.ndarray | str | int | float] = {
        "processing_normalization": str(getattr(args, "normalization", "None")),
        "processing_norm_success": norm_success,
        "processing_grid_applied": grid_applied,
        "processing_detector_correction_applied": correction_applied,
        "processing_background_used": int(background is not None),
    }
    return WorkflowProcessingResult(stack_result=processed, diagnostics=diagnostics)


def format_processing_summary(result: WorkflowProcessingResult) -> str:
    diagnostics = result.diagnostics
    norm_success = np.asarray(diagnostics["processing_norm_success"])
    grid_applied = np.asarray(diagnostics["processing_grid_applied"])
    correction_applied = np.asarray(diagnostics["processing_detector_correction_applied"])
    return (
        "  processing: applied\n"
        f"  normalization: {diagnostics['processing_normalization']} "
        f"({int(np.count_nonzero(norm_success))}/{norm_success.size} successful)\n"
        f"  background used: {bool(diagnostics['processing_background_used'])}\n"
        f"  grid-remove frames: {int(np.count_nonzero(grid_applied))}\n"
        f"  detector-correction frames: {int(np.count_nonzero(correction_applied))}\n"
    )


def build_two_d_info(
    measurement_mode: str,
    stack_result: H5SeriesLoadResult,
    qmap_result: WorkflowQMap,
    processing_diagnostics: dict[str, np.ndarray | str | int | float] | None = None,
    incident_angle: float | None = None,
) -> TwoDInfo:
    """Build Python-native 2D-info from a loaded stack and workflow Q-map."""

    data2d = _nanmean_no_warning(stack_result.stack, axis=0).astype(np.float32)
    cal = qmap_result.calibration
    return TwoDInfo(
        measurement_mode=measurement_mode,
        data2d_pix=data2d,
        data2d_q=data2d.copy(),
        q_map=qmap_result.q_map,
        qz_map=qmap_result.qz_map,
        qz_axis=_axis_from_map(qmap_result.qz_map, axis=0),
        beam_position=cal.beam_center.astype(np.float32, copy=False),
        wavelength=float(cal.wavelength),
        sdd=float(cal.sdd),
        pixel=float(cal.pixel_size),
        qx_map=qmap_result.qx_map,
        qy_map=qmap_result.qy_map,
        qy_axis=_axis_from_map(qmap_result.qy_map, axis=1),
        qr_map=qmap_result.qr_map,
        qr_axis=_axis_from_map(qmap_result.qr_map, axis=1),
        energy_kev=cal.energy_kev,
        detector=cal.detector,
        incident_angle=incident_angle,
        source_files=[str(record.file_path) for record in stack_result.records],
        retry_failed_frames=stack_result.retry_failed_frames,
        metadata_failed_frames=stack_result.metadata_failed_frames,
        diagnostics={
            "frame_count": int(stack_result.stack.shape[0]),
            "input_stack_shape": np.asarray(stack_result.stack.shape, dtype=np.int32),
            "counts": stack_result.counts,
            "timestamps": stack_result.timestamps,
            "temperatures": stack_result.temperatures,
            **(processing_diagnostics or {}),
        },
    )


def save_workflow_two_d_info(
    args,
    measurement_mode: str,
    stack_result: H5SeriesLoadResult,
    qmap_result: WorkflowQMap | None,
    processing_diagnostics: dict[str, np.ndarray | str | int | float] | None = None,
) -> Path | None:
    """Save workflow 2D-info when `--out` and a qmap are available."""

    out_arg = getattr(args, "out", None)
    if not out_arg:
        return None
    if qmap_result is None:
        raise RuntimeError("Cannot save 2D-info without --standard because Q maps are required.")

    out_path = _resolve_two_d_info_output_path(out_arg, measurement_mode, stack_result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = build_two_d_info(
        measurement_mode=measurement_mode,
        stack_result=stack_result,
        qmap_result=qmap_result,
        processing_diagnostics=processing_diagnostics,
        incident_angle=getattr(args, "incident_angle", None),
    )
    save_two_d_info_h5(out_path, info)
    return out_path


def format_output_summary(path: Path | None) -> str:
    if path is None:
        return "  output: skipped; pass --out with --standard to save *_2Dinfo.h5\n"
    return f"  output: saved {path}\n"


def _axis_from_map(qmap: np.ndarray | None, axis: int) -> np.ndarray | None:
    if qmap is None:
        return None
    if axis == 0:
        column = qmap[:, qmap.shape[1] // 2]
        return np.asarray(column, dtype=np.float32)
    row = qmap[qmap.shape[0] // 2, :]
    return np.asarray(row, dtype=np.float32)


def _resolve_two_d_info_output_path(
    out_arg: str | Path,
    measurement_mode: str,
    stack_result: H5SeriesLoadResult,
) -> Path:
    out_path = Path(out_arg)
    if out_path.suffix.lower() in {".h5", ".hdf5"}:
        return out_path
    first_file = Path(stack_result.records[0].file_path)
    stem = first_file.stem or measurement_mode.lower()
    return out_path / f"{stem}_{measurement_mode}_2Dinfo.h5"


def _nanmean_no_warning(values: np.ndarray, axis: int) -> np.ndarray:
    """NaN-omitting mean that leaves all-NaN pixels as NaN without warnings."""

    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, arr, 0), axis=axis)
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _load_background_for_processing(args) -> tuple[np.ndarray | None, np.ndarray | None]:
    background_path = getattr(args, "background", None)
    if not background_path:
        return None, None

    image = load_image_file(background_path)
    background = np.asarray(image.data, dtype=np.float32)
    counts = image.counts if image.counts.size >= 2 else None
    bg_ic = getattr(args, "background_ic", None)
    bg_bc = getattr(args, "background_bc", None)
    if bg_ic is not None or bg_bc is not None:
        counts = np.asarray(
            [
                np.nan if bg_ic is None else float(bg_ic),
                np.nan if bg_bc is None else float(bg_bc),
            ],
            dtype=np.float32,
        )
    return background, counts
