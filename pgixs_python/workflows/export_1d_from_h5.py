"""Export 1D curves from an existing Python-native PGIXS 2D-info H5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pgixs_python.averaging import qbin_average, summarize_coverage
from pgixs_python.export import OneDExport, WorkflowReport, save_1d_data, save_1d_figure, save_workflow_report_json
from pgixs_python.io.two_d_info import TwoDInfo, load_two_d_info_h5
from pgixs_python.workflows.common import load_workflow_config, parse_workflow_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export q-binned 1D data from PGIXS *_2Dinfo.h5.")
    parser.add_argument("--config", required=False, help="Optional JSON export configuration file.")
    parser.add_argument("--input", required=False, help="Input Python-native *_2Dinfo.h5 file.")
    parser.add_argument("--out", required=False, help="Output file or directory for 1D export.")
    parser.add_argument(
        "--export-format", default=None, choices=["csv", "dat"],
        help="1D export file format (default: the --out file's extension, else csv).",
    )
    parser.add_argument("--q-bins", type=int, default=600, help="Number of q bins for 1D export.")
    parser.add_argument("--q-min", type=float, default=None, help="Optional minimum q for 1D export.")
    parser.add_argument("--q-max", type=float, default=None, help="Optional maximum q for 1D export.")
    parser.add_argument("--q-scale", default="linear", choices=["linear", "log"], help="1D q-bin spacing.")
    parser.add_argument("--export-std", action="store_true", help="Include standard deviation in 1D export.")
    parser.add_argument("--export-diagnostics", action="store_true", help="Include valid-count and coverage columns.")
    parser.add_argument("--export-figure", action="store_true", help="Also save a PNG 1D figure when matplotlib is installed.")
    parser.add_argument("--report-json", action="store_true", help="Save a JSON export report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parse_workflow_args(parser, argv)
    if not args.input:
        parser.exit(status=0, message="1D H5 export is ready; pass --input *_2Dinfo.h5.\n")

    info = load_two_d_info_h5(args.input)
    data_path, figure_path = export_1d_from_two_d_info(info, args.input, args)
    report_path = save_export_report(info, args.input, args, data_path, figure_path)

    message = f"1D export: saved {data_path}\n"
    if figure_path is not None:
        message += f"1D figure: saved {figure_path}\n"
    if report_path is not None:
        message += f"report: saved {report_path}\n"
    parser.exit(status=0, message=message)
    return 0


def export_1d_from_two_d_info(info: TwoDInfo, source_path: str | Path, args) -> tuple[Path, Path | None]:
    centers, bin_index, geom_mask = build_q_bins_from_map(
        info.q_map,
        n_bins=int(getattr(args, "q_bins", 600)),
        q_min=getattr(args, "q_min", None),
        q_max=getattr(args, "q_max", None),
        q_scale=getattr(args, "q_scale", "linear"),
    )
    average = qbin_average(
        info.data2d_q,
        bin_index,
        centers.size,
        calc_std=bool(getattr(args, "export_std", False)),
        geom_mask=geom_mask,
    )
    export = OneDExport(
        x=centers,
        intensity=average.intensity,
        std=average.std if getattr(args, "export_std", False) else None,
        counts=average.counts.astype(np.float32),
        coverage=average.coverage,
        x_label="q",
        intensity_label="intensity",
    )
    out_path = resolve_1d_output_path(getattr(args, "out", None), source_path, info.measurement_mode, getattr(args, "export_format", None))
    data_path = save_1d_data(out_path, export, include_diagnostics=bool(getattr(args, "export_diagnostics", False)))

    figure_path = None
    if getattr(args, "export_figure", False):
        figure_path = save_1d_figure(data_path.with_suffix(".png"), export)
    return data_path, figure_path


def save_export_report(info: TwoDInfo, source_path: str | Path, args, data_path: Path, figure_path: Path | None) -> Path | None:
    if not getattr(args, "report_json", False):
        return None

    report_path = data_path.with_name(f"{data_path.stem}_report.json")
    q = info.q_map[np.isfinite(info.q_map)]
    one_d_diagnostics = build_export_coverage_diagnostics(info, args)
    report = WorkflowReport(
        measurement_mode=info.measurement_mode,
        input_files=[str(source_path)],
        output_files={
            "one_d_data": str(data_path),
            "one_d_figure": str(figure_path) if figure_path is not None else None,
        },
        options={
            "q_bins": getattr(args, "q_bins", None),
            "q_min": getattr(args, "q_min", None),
            "q_max": getattr(args, "q_max", None),
            "q_scale": getattr(args, "q_scale", None),
            "export_format": data_path.suffix.lstrip("."),
            "export_std": getattr(args, "export_std", None),
            "export_diagnostics": getattr(args, "export_diagnostics", None),
            "export_figure": getattr(args, "export_figure", None),
        },
        calibration={
            "detector": info.detector,
            "beam_center": info.beam_position,
            "wavelength": info.wavelength,
            "energy_kev": info.energy_kev,
            "sdd": info.sdd,
            "pixel_size": info.pixel,
            "q_min": float(np.nanmin(q)) if q.size else float("nan"),
            "q_max": float(np.nanmax(q)) if q.size else float("nan"),
        },
        diagnostics={
            "source_files": info.source_files,
            "retry_failed_frames": info.retry_failed_frames,
            "metadata_failed_frames": info.metadata_failed_frames,
            **one_d_diagnostics,
        },
    )
    return save_workflow_report_json(report_path, report)


def build_export_coverage_diagnostics(info: TwoDInfo, args) -> dict[str, object]:
    centers, bin_index, geom_mask = build_q_bins_from_map(
        info.q_map,
        n_bins=int(getattr(args, "q_bins", 600)),
        q_min=getattr(args, "q_min", None),
        q_max=getattr(args, "q_max", None),
        q_scale=getattr(args, "q_scale", "linear"),
    )
    average = qbin_average(info.data2d_q, bin_index, centers.size, geom_mask=geom_mask)
    return {
        "one_d_q_min": float(np.nanmin(centers)) if centers.size else float("nan"),
        "one_d_q_max": float(np.nanmax(centers)) if centers.size else float("nan"),
        "one_d_bin_count": int(centers.size),
        "one_d_valid_bins": int(np.count_nonzero(average.counts > 0)),
        **{f"one_d_{key}": value for key, value in summarize_coverage(average.coverage).items()},
    }


def build_q_bins_from_map(
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


def resolve_1d_output_path(
    out_arg: str | Path | None, source_path: str | Path, measurement_mode: str, export_format: str | None
) -> Path:
    """The file's extension decides the delimiter (see save_1d_data), so an
    --out file name and --export-format must agree; whichever one is given
    decides, and csv is the default when neither is."""

    source = Path(source_path)
    fmt = str(export_format).lower() if export_format else None
    if out_arg:
        out = Path(out_arg)
        named = out.suffix.lower().lstrip(".")
        if named in {"csv", "dat"}:
            if fmt is not None and fmt != named:
                raise ValueError(f"--out {out.name} is a .{named} file but --export-format is {fmt}.")
            return out
        return out / f"{source.stem}_{measurement_mode}_1D.{fmt or 'csv'}"
    return source.with_name(f"{source.stem}_{measurement_mode}_1D.{fmt or 'csv'}")


__all__ = [
    "build_parser",
    "build_export_coverage_diagnostics",
    "build_q_bins_from_map",
    "export_1d_from_two_d_info",
    "load_workflow_config",
    "main",
    "resolve_1d_output_path",
    "save_export_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
