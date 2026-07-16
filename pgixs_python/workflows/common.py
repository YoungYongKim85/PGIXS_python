"""Shared command-line helpers for PGIXS Tab1 workflows."""

from __future__ import annotations

import json
from numbers import Real
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pgixs_python.calibration.calibration_io import StandardCalibration, load_standard_calibration
from pgixs_python.averaging import qbin_average
from pgixs_python.export import OneDExport, WorkflowReport, save_1d_data, save_1d_figure, save_workflow_report_json
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


@dataclass
class WorkflowExportResult:
    data_path: Path | None = None
    figure_path: Path | None = None
    message: str = ""


@dataclass
class WorkflowReportResult:
    report_path: Path | None = None
    message: str = ""


def add_h5_input_arguments(parser) -> None:
    parser.add_argument("--config", required=False, help="Optional JSON workflow configuration file.")
    parser.add_argument("--input", required=False, help="Input H5 detector file or folder.")
    parser.add_argument("--dataset", default=DEFAULT_DETECTOR_DATASET, help="H5 detector dataset path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to read from each H5 detector dataset.")
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
    parser.add_argument("--export-1d", action="store_true", help="Export q-binned 1D data from the processed 2D image.")
    parser.add_argument("--export-format", default="csv", choices=["csv", "dat"], help="1D export file format.")
    parser.add_argument("--q-bins", type=int, default=600, help="Number of q bins for 1D export.")
    parser.add_argument("--q-min", type=float, default=None, help="Optional minimum q for 1D export.")
    parser.add_argument("--q-max", type=float, default=None, help="Optional maximum q for 1D export.")
    parser.add_argument("--q-scale", default="linear", choices=["linear", "log"], help="1D q-bin spacing.")
    parser.add_argument("--export-std", action="store_true", help="Include standard deviation in 1D export.")
    parser.add_argument("--export-diagnostics", action="store_true", help="Include valid-count and coverage columns.")
    parser.add_argument("--export-figure", action="store_true", help="Also save a PNG 1D figure when matplotlib is installed.")
    parser.add_argument("--report-json", action="store_true", help="Save a JSON report with workflow options and diagnostics.")


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


def parse_workflow_args(parser, argv: list[str] | None = None):
    """Parse workflow args, using optional JSON config values as defaults."""

    pre_parser = parser.__class__(add_help=False)
    pre_parser.add_argument("--config", required=False)
    pre_args, _ = pre_parser.parse_known_args(argv)
    if pre_args.config:
        config_values = load_workflow_config(pre_args.config)
        unknown = sorted(set(config_values) - parser_config_keys(parser))
        if unknown:
            parser.error(f"Unknown config option(s): {', '.join(unknown)}")
        try:
            validate_workflow_config_values(config_values)
        except ValueError as exc:
            parser.error(str(exc))
        parser.set_defaults(**config_values)
    return parser.parse_args(argv)


def load_workflow_config(file_path: str | Path) -> dict[str, object]:
    """Load workflow JSON config using argparse destination names."""

    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Workflow config JSON must contain an object at the top level.")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def parser_config_keys(parser) -> set[str]:
    """Return argparse destination names that are valid config keys."""

    return {
        action.dest
        for action in parser._actions  # noqa: SLF001 - argparse has no public iterator for actions.
        if action.dest and action.dest != "help"
    }


def validate_workflow_config_values(config: dict[str, object]) -> None:
    """Validate JSON config values that argparse would otherwise accept as defaults."""

    int_keys = {"q_bins": (1, None), "retry_count": (0, None), "frame_index": (0, None)}
    float_keys = {
        "q_min",
        "q_max",
        "incident_angle",
        "reference_timestamp",
        "background_ic",
        "background_bc",
    }
    bool_keys = {
        "export_1d",
        "export_std",
        "export_diagnostics",
        "export_figure",
        "report_json",
        "no_clamp_negative",
        "mask_detector",
        "grid_remove",
        "detector_correction",
    }
    choice_keys = {
        "export_format": {"csv", "dat"},
        "q_scale": {"linear", "log"},
        "normalization": {"None", "IC", "BC"},
    }
    string_keys = {
        "config",
        "input",
        "dataset",
        "metadata_mode",
        "standard",
        "detector",
        "out",
        "background",
    }

    for key, value in config.items():
        if key in int_keys and not _is_int_like(value):
            raise ValueError(f"Config option '{key}' must be an integer.")
        if key in int_keys:
            min_value, max_value = int_keys[key]
            numeric_value = int(value)
            if min_value is not None and numeric_value < min_value:
                raise ValueError(f"Config option '{key}' must be >= {min_value}.")
            if max_value is not None and numeric_value > max_value:
                raise ValueError(f"Config option '{key}' must be <= {max_value}.")
        if key in float_keys and value is not None and not _is_real_number(value):
            raise ValueError(f"Config option '{key}' must be a number or null.")
        if key in bool_keys and not isinstance(value, bool):
            raise ValueError(f"Config option '{key}' must be true or false.")
        if key in choice_keys and value not in choice_keys[key]:
            choices = ", ".join(sorted(choice_keys[key]))
            raise ValueError(f"Config option '{key}' must be one of: {choices}.")
        if key in string_keys and value is not None and not isinstance(value, str):
            raise ValueError(f"Config option '{key}' must be a string or null.")

    q_min = config.get("q_min")
    q_max = config.get("q_max")
    if q_min is not None and q_max is not None and float(q_max) <= float(q_min):
        raise ValueError("Config option 'q_max' must be greater than 'q_min'.")


def _is_int_like(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


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
        frame_index=args.frame_index,
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


def export_workflow_1d(
    args,
    measurement_mode: str,
    stack_result: H5SeriesLoadResult,
    qmap_result: WorkflowQMap | None,
) -> WorkflowExportResult:
    """Export a q-binned 1D curve when requested."""

    if not getattr(args, "export_1d", False):
        return WorkflowExportResult(message="  1D export: skipped\n")
    if qmap_result is None:
        raise RuntimeError("Cannot export 1D data without --standard because Q maps are required.")

    out_arg = getattr(args, "out", None) or "outputs"
    out_dir = _resolve_export_output_dir(out_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(stack_result.records[0].file_path).stem or measurement_mode.lower()

    data2d = _nanmean_no_warning(stack_result.stack, axis=0)
    q = np.asarray(qmap_result.q_map, dtype=np.float32)
    centers, bin_index, geom_mask = _build_q_bins(
        q,
        n_bins=int(getattr(args, "q_bins", 600)),
        q_min=getattr(args, "q_min", None),
        q_max=getattr(args, "q_max", None),
        q_scale=getattr(args, "q_scale", "linear"),
    )
    average = qbin_average(
        data2d,
        bin_index,
        centers.size,
        calc_std=bool(getattr(args, "export_std", False)),
        geom_mask=geom_mask,
    )

    std = average.std if getattr(args, "export_std", False) else None
    export = OneDExport(
        x=centers,
        intensity=average.intensity,
        std=std,
        counts=average.counts.astype(np.float32),
        coverage=average.coverage,
        x_label="q",
        intensity_label="intensity",
    )
    suffix = str(getattr(args, "export_format", "csv")).lower()
    data_path = save_1d_data(
        out_dir / f"{stem}_{measurement_mode}_1D.{suffix}",
        export,
        include_diagnostics=bool(getattr(args, "export_diagnostics", False)),
    )

    figure_path = None
    if getattr(args, "export_figure", False):
        figure_path = save_1d_figure(out_dir / f"{stem}_{measurement_mode}_1D.png", export)

    msg = f"  1D export: saved {data_path}"
    if figure_path is not None:
        msg += f"\n  1D figure: saved {figure_path}"
    return WorkflowExportResult(data_path=data_path, figure_path=figure_path, message=msg + "\n")


def format_1d_export_summary(result: WorkflowExportResult) -> str:
    return result.message


def save_workflow_report(
    args,
    measurement_mode: str,
    raw_stack: H5SeriesLoadResult,
    processed_stack: H5SeriesLoadResult,
    qmap_result: WorkflowQMap | None,
    processing_result: WorkflowProcessingResult,
    output_path: Path | None,
    export_result: WorkflowExportResult,
) -> WorkflowReportResult:
    """Save optional JSON provenance/diagnostics report."""

    if not getattr(args, "report_json", False):
        return WorkflowReportResult(message="  report: skipped\n")

    out_arg = getattr(args, "out", None) or "outputs"
    out_dir = _resolve_export_output_dir(out_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(raw_stack.records[0].file_path).stem or measurement_mode.lower()
    report_path = out_dir / f"{stem}_{measurement_mode}_report.json"

    report = WorkflowReport(
        measurement_mode=measurement_mode,
        input_files=[str(record.file_path) for record in raw_stack.records],
        output_files={
            "two_d_info": str(output_path) if output_path is not None else None,
            "one_d_data": str(export_result.data_path) if export_result.data_path is not None else None,
            "one_d_figure": str(export_result.figure_path) if export_result.figure_path is not None else None,
        },
        options=_workflow_options_for_report(args),
        calibration=_qmap_report(qmap_result),
        diagnostics={
            "frame_count": int(processed_stack.stack.shape[0]),
            "image_shape": [int(processed_stack.stack.shape[1]), int(processed_stack.stack.shape[2])],
            "retry_failed_frames": raw_stack.retry_failed_frames,
            "metadata_failed_frames": raw_stack.metadata_failed_frames,
            "counts": raw_stack.counts,
            "timestamps": raw_stack.timestamps,
            "temperatures": raw_stack.temperatures,
            **processing_result.diagnostics,
        },
    )
    save_workflow_report_json(report_path, report)
    return WorkflowReportResult(report_path=report_path, message=f"  report: saved {report_path}\n")


def format_report_summary(result: WorkflowReportResult) -> str:
    return result.message


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


def _resolve_export_output_dir(out_arg: str | Path) -> Path:
    out_path = Path(out_arg)
    if out_path.suffix.lower() in {".h5", ".hdf5"}:
        return out_path.parent if str(out_path.parent) else Path(".")
    return out_path


def _nanmean_no_warning(values: np.ndarray, axis: int) -> np.ndarray:
    """NaN-omitting mean that leaves all-NaN pixels as NaN without warnings."""

    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, arr, 0), axis=axis)
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _build_q_bins(
    q_map: np.ndarray,
    n_bins: int,
    q_min: float | None,
    q_max: float | None,
    q_scale: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_map, dtype=np.float32)
    finite = np.isfinite(q)
    finite_values = q[finite]
    if finite_values.size == 0:
        raise ValueError("q_map has no finite values.")

    lo = float(np.nanmin(finite_values)) if q_min is None else float(q_min)
    hi = float(np.nanmax(finite_values)) if q_max is None else float(q_max)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid q export range.")

    if str(q_scale).lower() == "log":
        positive = finite_values[finite_values > 0]
        if positive.size == 0:
            raise ValueError("Log q export requires positive finite q values.")
        lo = max(lo, float(np.nanmin(positive)))
        edges = np.geomspace(lo, hi, int(n_bins) + 1, dtype=np.float32)
        centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    else:
        edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float32)
        centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)

    bin_index = np.full(q.shape, -1, dtype=np.int64)
    in_range = finite & (q >= edges[0]) & (q <= edges[-1])
    bins = np.searchsorted(edges, q[in_range], side="right") - 1
    bins = np.clip(bins, 0, centers.size - 1)
    bin_index[in_range] = bins
    return centers, bin_index, in_range


def _workflow_options_for_report(args) -> dict[str, object]:
    names = [
        "dataset",
        "frame_index",
        "detector",
        "incident_angle",
        "metadata_mode",
        "reference_timestamp",
        "retry_count",
        "normalization",
        "background",
        "background_ic",
        "background_bc",
        "no_clamp_negative",
        "mask_detector",
        "grid_remove",
        "detector_correction",
        "export_1d",
        "export_format",
        "q_bins",
        "q_min",
        "q_max",
        "q_scale",
        "export_std",
        "export_diagnostics",
        "export_figure",
    ]
    return {name: getattr(args, name) for name in names if hasattr(args, name)}


def _qmap_report(qmap_result: WorkflowQMap | None) -> dict[str, object]:
    if qmap_result is None:
        return {"qmap_built": False}
    cal = qmap_result.calibration
    q = qmap_result.q_map[np.isfinite(qmap_result.q_map)]
    qz = qmap_result.qz_map[np.isfinite(qmap_result.qz_map)]
    return {
        "qmap_built": True,
        "detector": cal.detector,
        "beam_center": cal.beam_center,
        "wavelength": cal.wavelength,
        "energy_kev": cal.energy_kev,
        "sdd": cal.sdd,
        "pixel_size": cal.pixel_size,
        "q_min": float(np.min(q)) if q.size else float("nan"),
        "q_max": float(np.max(q)) if q.size else float("nan"),
        "qz_min": float(np.min(qz)) if qz.size else float("nan"),
        "qz_max": float(np.max(qz)) if qz.size else float("nan"),
    }


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
