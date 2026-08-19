"""Multi-ring SDD candidate search for SDD calibration's "Smart Auto Run".

Ports MATLAB `analyzePGIXSSDDCandidates.m` in full: rough-peak search over
a radial profile, a Gaussian fit per candidate ring, then `assignRingOrders`'s
brute-force search over (candidate-start x order-start) combinations,
scored by `scoreOrderAssignment`, to find the most self-consistent ring
order assignment (only meaningful for Silver behenate, which has multiple
observable orders; every other standard sample only ever tries order 1
starting at the first candidate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from pgixs_python.calibration.peak_fit import fit_gaussian_peak
from pgixs_python.calibration.sdd import compute_sdd_from_ring
from pgixs_python.calibration.standards import standard_q_value


@dataclass
class RingCandidate:
    rough_radius: float
    fitted_radius: float
    order: int | None = None
    q_value: float | None = None
    coverage: float = float("nan")
    r_square: float = float("nan")
    sigma: float = float("nan")
    sdd: float | None = None
    accepted: bool = False
    message: str = ""


@dataclass
class SDDCandidateReport:
    candidates: list[RingCandidate] = field(default_factory=list)
    final_sdd: float = float("nan")
    sdd_std: float = float("nan")
    accepted_count: int = 0
    message: str = ""


def find_rough_peaks(
    r_axis: np.ndarray,
    intensity: np.ndarray,
    coverage: np.ndarray,
    *,
    min_radius: float = 50.0,
    min_coverage: float = 0.80,
    prominence_fraction: float = 0.03,
    max_candidates: int = 7,
) -> np.ndarray:
    """Indices into `r_axis`/`intensity`/`coverage` of candidate ring
    peaks -- matches `analyzePGIXSSDDCandidates.m`'s `valid`/`findpeaks`
    block, including its quirk of peak-finding over the *filtered*
    subsequence (gaps between valid bins are invisible to the search, same
    as MATLAB's `findpeaks(intensity(valid))`)."""

    r_axis = np.asarray(r_axis, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)

    valid = (
        np.isfinite(r_axis)
        & np.isfinite(intensity)
        & np.isfinite(coverage)
        & (r_axis > min_radius)
        & (coverage >= min_coverage)
    )
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.array([], dtype=np.int64)

    # Prominence threshold comes from the *whole* profile's max (including
    # the excluded near-center region, typically far brighter than any
    # ring), not just the `valid` subset -- matches
    # `max(intensity, [], 'omitnan')` in the MATLAB source, computed
    # before the `valid` filter is applied to `findpeaks`'s own input.
    finite_full = intensity[np.isfinite(intensity)]
    if finite_full.size == 0:
        return np.array([], dtype=np.int64)
    prominence = float(np.max(finite_full)) * prominence_fraction

    finite_intensity = intensity[valid]  # already all-finite, `valid` requires it
    local_peaks, _ = find_peaks(finite_intensity, prominence=prominence)
    if local_peaks.size == 0:
        return np.array([], dtype=np.int64)

    locs = valid_idx[local_peaks]
    return locs[: min(locs.size, max_candidates)]


def analyze_sdd_candidates(
    r_axis: np.ndarray,
    intensity: np.ndarray,
    coverage: np.ndarray,
    sample_name: str,
    energy_kev: float,
    pixel_size_mm: float,
    *,
    min_radius: float = 50.0,
    min_coverage: float = 0.80,
    min_r_square: float = 0.85,
    max_candidates: int = 7,
    prominence_fraction: float = 0.03,
    fit_half_width: float = 30.0,
    max_order_start: int = 4,
    max_skip_candidates: int = 3,
) -> SDDCandidateReport:
    """Full port of `analyzePGIXSSDDCandidates.m`."""

    r_axis = np.asarray(r_axis, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)

    locs = find_rough_peaks(
        r_axis,
        intensity,
        coverage,
        min_radius=min_radius,
        min_coverage=min_coverage,
        prominence_fraction=prominence_fraction,
        max_candidates=max_candidates,
    )
    if locs.size == 0:
        return SDDCandidateReport(message="No peak candidates found.")

    candidates: list[RingCandidate] = []
    for idx in locs:
        rough_radius = float(r_axis[idx])
        fit = fit_gaussian_peak(r_axis, intensity, window_center=rough_radius, half_width=fit_half_width)
        candidates.append(
            RingCandidate(
                rough_radius=rough_radius,
                fitted_radius=fit.center if fit.success else float("nan"),
                coverage=float(coverage[idx]),
                r_square=fit.r_square if fit.success else float("nan"),
                sigma=fit.sigma if fit.success else float("nan"),
                message="" if fit.success else fit.message,
            )
        )

    _assign_ring_orders(
        candidates, sample_name, energy_kev, pixel_size_mm, min_r_square, min_coverage, max_order_start, max_skip_candidates
    )

    accepted_sdd = [c.sdd for c in candidates if c.accepted and c.sdd is not None]
    if not accepted_sdd:
        return SDDCandidateReport(candidates=candidates, message="No accepted SDD ring candidates.")

    final_sdd = float(np.median(accepted_sdd))
    spread = float(np.std(accepted_sdd)) if len(accepted_sdd) > 1 else 0.0
    return SDDCandidateReport(candidates=candidates, final_sdd=final_sdd, sdd_std=spread, accepted_count=len(accepted_sdd))


def _clone_candidates(candidates: list[RingCandidate]) -> list[RingCandidate]:
    return [
        RingCandidate(
            rough_radius=c.rough_radius,
            fitted_radius=c.fitted_radius,
            coverage=c.coverage,
            r_square=c.r_square,
            sigma=c.sigma,
            message=c.message,
        )
        for c in candidates
    ]


def _assign_ring_orders(
    candidates: list[RingCandidate],
    sample_name: str,
    energy_kev: float,
    pixel_size_mm: float,
    min_r_square: float,
    min_coverage: float,
    max_order_start: int,
    max_skip_candidates: int,
) -> None:
    n = len(candidates)
    if n == 0:
        return

    if str(sample_name).strip().lower() == "silver behenate":
        order_starts = range(1, max_order_start + 1)
        candidate_starts = range(1, min(max_skip_candidates, n) + 1)
    else:
        order_starts = [1]
        candidate_starts = [1]

    best_score = float("inf")
    best_trial: list[RingCandidate] | None = None

    for cand_start in candidate_starts:
        for order_start in order_starts:
            trial = _clone_candidates(candidates)
            for row in range(cand_start - 1, n):
                order = order_start + (row - (cand_start - 1))
                trial[row].order = order
                try:
                    q_value = standard_q_value(sample_name, order)
                    trial[row].q_value = q_value
                    trial[row].sdd = compute_sdd_from_ring(trial[row].fitted_radius, pixel_size_mm, energy_kev, q_value).sdd
                except (ValueError, ZeroDivisionError) as exc:
                    trial[row].message = str(exc)
                sdd = trial[row].sdd
                trial[row].accepted = (
                    sdd is not None
                    and np.isfinite(sdd)
                    and trial[row].r_square >= min_r_square
                    and trial[row].coverage >= min_coverage
                )

            score = _score_order_assignment(trial, cand_start)
            if score < best_score:
                best_score = score
                best_trial = trial

    if best_trial is None:
        return
    for i, c in enumerate(best_trial):
        candidates[i].order = c.order
        candidates[i].q_value = c.q_value
        candidates[i].sdd = c.sdd
        candidates[i].accepted = c.accepted
        candidates[i].message = "Skipped before selected ring series" if c.order is None else c.message


def _score_order_assignment(trial: list[RingCandidate], cand_start: int) -> float:
    accepted_sdd = [c.sdd for c in trial if c.accepted and c.sdd is not None]
    n_accepted = len(accepted_sdd)
    if n_accepted == 0:
        return float("inf")
    center = float(np.median(accepted_sdd))
    if not np.isfinite(center) or center <= 0:
        return float("inf")
    rel_spread = 0.25 if n_accepted == 1 else float(np.std(accepted_sdd)) / center
    return rel_spread - 0.05 * n_accepted + 0.02 * (cand_start - 1)


def candidates_to_dicts(report: SDDCandidateReport) -> list[dict[str, Any]]:
    """JSON/UI-friendly representation of a report's candidates, for the
    desktop widget's candidate table."""

    return [
        {
            "rough_radius": c.rough_radius,
            "fitted_radius": c.fitted_radius,
            "order": c.order,
            "q_value": c.q_value,
            "coverage": c.coverage,
            "r_square": c.r_square,
            "sigma": c.sigma,
            "sdd_mm": c.sdd,
            "accepted": c.accepted,
            "message": c.message,
        }
        for c in report.candidates
    ]
