"""Shared command-line helpers for PGIXS Tab1 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pgixs_python.calibration.calibration_io import StandardCalibration, load_standard_calibration
from pgixs_python.io.h5_loader import DEFAULT_DETECTOR_DATASET
from pgixs_python.io.h5_series import H5SeriesLoadResult, load_h5_series
from pgixs_python.io.two_d_info import TwoDInfo, save_two_d_info_h5
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
    return load_h5_series(
        files,
        dataset_path=args.dataset,
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


def build_two_d_info(
    measurement_mode: str,
    stack_result: H5SeriesLoadResult,
    qmap_result: WorkflowQMap,
    incident_angle: float | None = None,
) -> TwoDInfo:
    """Build Python-native 2D-info from a loaded stack and workflow Q-map."""

    data2d = np.nanmean(stack_result.stack, axis=0).astype(np.float32)
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
        },
    )


def save_workflow_two_d_info(
    args,
    measurement_mode: str,
    stack_result: H5SeriesLoadResult,
    qmap_result: WorkflowQMap | None,
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
