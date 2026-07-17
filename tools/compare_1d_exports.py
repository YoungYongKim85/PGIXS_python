"""Compare two exported 1D PGIXS curves for validation work."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Curve1D:
    q: np.ndarray
    intensity: np.ndarray
    std: np.ndarray | None = None


@dataclass
class CurveComparison:
    q: np.ndarray
    reference_intensity: np.ndarray
    candidate_intensity: np.ndarray
    absolute_diff: np.ndarray
    relative_diff: np.ndarray
    max_abs_diff: float
    rms_abs_diff: float
    max_rel_diff: float
    compared_points: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MATLAB and Python 1D export curves.")
    parser.add_argument("--reference", required=True, help="Reference 1D CSV/DAT file, usually MATLAB output.")
    parser.add_argument("--candidate", required=True, help="Candidate 1D CSV/DAT file, usually Python output.")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate candidate intensity onto reference q-grid.")
    parser.add_argument("--q-tolerance", type=float, default=1e-8, help="q-grid tolerance when interpolation is disabled.")
    parser.add_argument("--relative-eps", type=float, default=1e-12, help="Small denominator for relative differences.")
    parser.add_argument("--json-out", required=False, help="Optional JSON summary output path.")
    parser.add_argument("--csv-out", required=False, help="Optional pointwise comparison CSV output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = load_curve(args.reference)
    candidate = load_curve(args.candidate)
    comparison = compare_curves(
        reference,
        candidate,
        interpolate=bool(args.interpolate),
        q_tolerance=float(args.q_tolerance),
        relative_eps=float(args.relative_eps),
    )
    summary = comparison_summary(comparison, args.reference, args.candidate)
    print(format_summary(summary))
    if args.json_out:
        write_json_summary(args.json_out, summary)
    if args.csv_out:
        write_pointwise_csv(args.csv_out, comparison)
    return 0


def load_curve(file_path: str | Path) -> Curve1D:
    path = Path(file_path)
    delimiter = "," if path.suffix.lower() == ".csv" else None
    table = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=np.float64, encoding="utf-8")
    if table.size == 0:
        raise ValueError(f"1D curve file is empty: {path}")
    names = list(table.dtype.names or [])
    if len(names) < 2:
        raise ValueError(f"1D curve file must contain at least q and intensity columns: {path}")

    q = np.asarray(table[names[0]], dtype=np.float64).reshape(-1)
    intensity = np.asarray(table[names[1]], dtype=np.float64).reshape(-1)
    std = _optional_std_column(table, names)
    return Curve1D(q=q, intensity=intensity, std=std)


def compare_curves(
    reference: Curve1D,
    candidate: Curve1D,
    interpolate: bool = False,
    q_tolerance: float = 1e-8,
    relative_eps: float = 1e-12,
) -> CurveComparison:
    ref_q = _as_1d(reference.q, "reference q")
    ref_i = _as_1d(reference.intensity, "reference intensity")
    cand_q = _as_1d(candidate.q, "candidate q")
    cand_i = _as_1d(candidate.intensity, "candidate intensity")

    if ref_q.size != ref_i.size:
        raise ValueError("Reference q and intensity lengths do not match.")
    if cand_q.size != cand_i.size:
        raise ValueError("Candidate q and intensity lengths do not match.")

    if interpolate:
        candidate_on_ref = interpolate_candidate(ref_q, cand_q, cand_i)
        q = ref_q
    else:
        if ref_q.size != cand_q.size or not np.allclose(ref_q, cand_q, atol=q_tolerance, rtol=0.0, equal_nan=True):
            raise ValueError("q grids differ; rerun with --interpolate or export matching q bins.")
        candidate_on_ref = cand_i
        q = ref_q

    valid = np.isfinite(ref_i) & np.isfinite(candidate_on_ref) & np.isfinite(q)
    if not np.any(valid):
        raise ValueError("No finite points are available for comparison.")

    q_valid = q[valid]
    ref_valid = ref_i[valid]
    cand_valid = candidate_on_ref[valid]
    abs_diff = np.abs(cand_valid - ref_valid)
    rel_diff = abs_diff / np.maximum(np.abs(ref_valid), float(relative_eps))

    return CurveComparison(
        q=q_valid,
        reference_intensity=ref_valid,
        candidate_intensity=cand_valid,
        absolute_diff=abs_diff,
        relative_diff=rel_diff,
        max_abs_diff=float(np.max(abs_diff)),
        rms_abs_diff=float(np.sqrt(np.mean(abs_diff**2))),
        max_rel_diff=float(np.max(rel_diff)),
        compared_points=int(q_valid.size),
    )


def interpolate_candidate(reference_q: np.ndarray, candidate_q: np.ndarray, candidate_i: np.ndarray) -> np.ndarray:
    order = np.argsort(candidate_q)
    sorted_q = candidate_q[order]
    sorted_i = candidate_i[order]
    valid = np.isfinite(sorted_q) & np.isfinite(sorted_i)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Candidate curve needs at least two finite points for interpolation.")
    sorted_q = sorted_q[valid]
    sorted_i = sorted_i[valid]
    unique_q, unique_index = np.unique(sorted_q, return_index=True)
    unique_i = sorted_i[unique_index]
    return np.interp(reference_q, unique_q, unique_i, left=np.nan, right=np.nan)


def comparison_summary(comparison: CurveComparison, reference_path: str | Path, candidate_path: str | Path) -> dict[str, object]:
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "compared_points": comparison.compared_points,
        "q_min": float(np.min(comparison.q)),
        "q_max": float(np.max(comparison.q)),
        "max_abs_diff": comparison.max_abs_diff,
        "rms_abs_diff": comparison.rms_abs_diff,
        "max_rel_diff": comparison.max_rel_diff,
    }


def format_summary(summary: dict[str, object]) -> str:
    return (
        "PGIXS 1D comparison\n"
        f"  compared points: {summary['compared_points']}\n"
        f"  q range: {summary['q_min']:.10g} to {summary['q_max']:.10g}\n"
        f"  max abs diff: {summary['max_abs_diff']:.10g}\n"
        f"  rms abs diff: {summary['rms_abs_diff']:.10g}\n"
        f"  max rel diff: {summary['max_rel_diff']:.10g}\n"
    )


def write_json_summary(file_path: str | Path, summary: dict[str, object]) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_pointwise_csv(file_path: str | Path, comparison: CurveComparison) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        [
            comparison.q,
            comparison.reference_intensity,
            comparison.candidate_intensity,
            comparison.absolute_diff,
            comparison.relative_diff,
        ]
    )
    header = "q,reference_intensity,candidate_intensity,absolute_diff,relative_diff"
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.10g")
    return path


def _optional_std_column(table: np.ndarray, names: list[str]) -> np.ndarray | None:
    for name in names[2:]:
        normalized = name.lower().replace(" ", "_")
        if normalized in {"std", "sigma", "standard_deviation"}:
            return np.asarray(table[name], dtype=np.float64).reshape(-1)
    return None


def _as_1d(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


if __name__ == "__main__":
    raise SystemExit(main())

