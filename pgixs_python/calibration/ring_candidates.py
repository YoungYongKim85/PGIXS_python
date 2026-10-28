"""Multi-ring SDD candidate search for SDD calibration's "Smart Auto Run".

Same algorithm as MATLAB `analyzePGIXSSDDCandidates.m`: rough-peak search
over a radial profile, a Gaussian fit per candidate ring, then a search over
"candidate i is order k" hypotheses. Each hypothesis's SDD predicts every
order's radius; candidates that land on one join the series, the rest are
left out, and `_score_order_assignment` picks the hypothesis the most rings
agree on without unexplained gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from pgixs_python.calibration.peak_fit import fit_gaussian_peak
from pgixs_python.calibration.sdd import compute_sdd_from_ring, predicted_ring_radius
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

    # Prominence threshold comes from the `valid` analysis domain only, not
    # the whole profile -- matches `max(intensity(valid), [], 'omitnan')` in
    # `analyzePGIXSSDDCandidates.m`. The excluded near-center region holds the
    # beam stop / direct beam, orders of magnitude brighter than any standard
    # ring; scaling the threshold off that would suppress every real ring.
    finite_intensity = intensity[valid]  # already all-finite, `valid` requires it
    prominence = float(np.max(finite_intensity)) * prominence_fraction
    if not np.isfinite(prominence) or prominence <= 0:
        return np.array([], dtype=np.int64)
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
        candidates,
        sample_name,
        energy_kev,
        pixel_size_mm,
        min_r_square,
        min_coverage,
        max_order_start,
        r_axis,
        coverage,
        min_radius,
    )

    accepted_sdd = [c.sdd for c in candidates if c.accepted and c.sdd is not None]
    if not accepted_sdd:
        return SDDCandidateReport(candidates=candidates, message="No accepted SDD ring candidates.")

    final_sdd = float(np.median(accepted_sdd))
    # Sample std (N-1), as MATLAB analyzePGIXSSDDCandidates.m's `std`.
    spread = float(np.std(accepted_sdd, ddof=1)) if len(accepted_sdd) > 1 else 0.0
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


# A candidate belongs to an order when its fitted radius lies within this
# distance of the radius the hypothesis predicts for that order. Real AgBH
# and SEBS rings fit to well under 1% of the predicted radius; the pixel
# floor covers small rings whose centre is biased by the beam stop edge.
RING_MATCH_REL = 0.02
RING_MATCH_PX = 3.0
# Standards whose rings are a harmonic series (q, 2q, 3q, ...). Sucrose's
# other reflections are not multiples of its reference peak, which is its
# lowest-q ring: it is only ever the first candidate, as order 1.
HARMONIC_STANDARDS = {"silver behenate", "sebs", "mesoporous silica"}


def _assign_ring_orders(
    candidates: list[RingCandidate],
    sample_name: str,
    energy_kev: float,
    pixel_size_mm: float,
    min_r_square: float,
    min_coverage: float,
    max_order_start: int,
    r_axis: np.ndarray,
    coverage: np.ndarray,
    min_radius: float,
) -> None:
    """Pick the ring-order assignment the whole candidate set agrees on.

    Each hypothesis is "candidate i is order k". Its SDD predicts where every
    order falls; candidates within RING_MATCH_* of a predicted ring take that
    order. Only the longest run of consecutive orders is kept -- an order may
    be skipped only where it could not have been seen -- so a beam-stop edge,
    a detector seam or a far stray peak stays out of the series instead of
    shifting or stretching it. k > 1 (1st order hidden behind the beam stop)
    is tried for Silver behenate only, as it is the one standard measured at
    SDDs that short."""

    if not candidates:
        return

    sample_key = " ".join(str(sample_name).lower().split())
    max_order = None if sample_key in HARMONIC_STANDARDS else 1
    anchor_orders = range(1, (max_order_start if sample_key == "silver behenate" else 1) + 1)
    max_radius = float(np.nanmax(r_axis)) if r_axis.size else min_radius

    def q_of(order: int) -> float:
        return standard_q_value(sample_name, order)

    def radius_of(sdd: float, order: int) -> float:
        try:
            return predicted_ring_radius(sdd, pixel_size_mm, energy_kev, q_of(order))
        except (ValueError, ZeroDivisionError):
            return float("nan")

    def observable(radius: float) -> bool:
        # A ring here would have been a candidate: inside the analysed range
        # and on enough of the detector (same coverage rule as the peak search).
        if not (np.isfinite(radius) and min_radius <= radius <= max_radius):
            return False
        return bool(coverage[int(np.argmin(np.abs(r_axis - radius)))] >= min_coverage)

    radii = np.array([c.fitted_radius for c in candidates], dtype=np.float64)
    best_score = float("inf")
    best_trial: list[RingCandidate] | None = None

    anchors = radii[np.isfinite(radii)]
    for anchor_radius in anchors if max_order is None else anchors[:1]:
        for anchor_order in anchor_orders:
            try:
                sdd = compute_sdd_from_ring(anchor_radius, pixel_size_mm, energy_kev, q_of(anchor_order)).sdd
            except (ValueError, ZeroDivisionError):
                continue
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            predicted = {}
            order = 1
            while max_order is None or order <= max_order:
                radius = radius_of(sdd, order)
                if not np.isfinite(radius) or radius > 1.05 * max_radius:
                    break
                predicted[order] = radius
                order += 1
            if not predicted:
                continue

            matched: dict[int, int] = {}
            for row, radius in enumerate(radii):
                if not np.isfinite(radius):
                    continue
                order, expected = min(predicted.items(), key=lambda kv: abs(kv[1] - radius))
                if abs(radius - expected) <= max(RING_MATCH_REL * expected, RING_MATCH_PX):
                    matched[row] = order

            runs: list[list[int]] = []
            for order in sorted(set(matched.values())):
                if runs and not any(observable(predicted[o]) for o in range(runs[-1][-1] + 1, order)):
                    runs[-1].append(order)
                else:
                    runs.append([order])
            if not runs:
                continue

            trial = _clone_candidates(candidates)
            for row, order in matched.items():
                trial[row].order = order
                trial[row].q_value = q_of(order)
                try:
                    trial[row].sdd = compute_sdd_from_ring(radii[row], pixel_size_mm, energy_kev, trial[row].q_value).sdd
                except (ValueError, ZeroDivisionError) as exc:
                    trial[row].message = str(exc)
                trial[row].accepted = (
                    trial[row].sdd is not None
                    and np.isfinite(trial[row].sdd)
                    and trial[row].r_square >= min_r_square
                    and trial[row].coverage >= min_coverage
                )
            series = max(runs, key=lambda run: (len(run), sum(c.accepted for c in trial if c.order in run)))
            for c in trial:
                if c.order is not None and c.order not in series:
                    c.order, c.q_value, c.sdd, c.accepted = None, None, None, False

            score = _score_order_assignment(trial, series, radius_of, observable)
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
        candidates[i].message = "Not in the ring series" if c.order is None else c.message


def _score_order_assignment(trial: list[RingCandidate], series: list[int], radius_of, observable) -> float:
    """Lower is better. Each order in the series counts -1; an order the
    hypothesis needs below the series but nobody saw counts against it."""

    accepted_sdd = [c.sdd for c in trial if c.accepted and c.sdd is not None]
    if not accepted_sdd:
        return float("inf")
    center = float(np.median(accepted_sdd))
    if not np.isfinite(center) or center <= 0:
        return float("inf")
    rel_spread = float(np.std(accepted_sdd, ddof=1)) / center if len(accepted_sdd) > 1 else 0.0  # N-1, as MATLAB
    # Accepted rings add a little each, weighted by fit quality: between two
    # otherwise equal readings (a lone sharp ring vs a broad beam-stop edge)
    # the better-fitting ring wins.
    score = -len(series) - 0.01 * sum(c.r_square for c in trial if c.accepted) + rel_spread

    # Every ring of one standard must report the same SDD, and ring centres
    # fit to a fraction of a pixel, so a correct assignment agrees far better
    # than a percent. Disagreement past that means the assignment is wrong.
    spread_tolerance = 0.01
    if rel_spread > spread_tolerance:
        score += 20.0 * (rel_spread - spread_tolerance)

    lowest = series[0]
    for order in range(1, lowest):
        if observable(radius_of(center, order)):
            # A lower order that should have been on the detector. One that
            # falls inside the beam stop region is the normal reason a
            # pattern starts at order 2.
            score += 0.5

    if len(series) == 1 and lowest == 1 and observable(radius_of(center, 2)):
        # A lone ring read as order 1 claims order 2 was visible but missed.
        # AgBH orders 1 and 2 are comparably bright and order 2 sits further
        # from the beam stop, so this weighs as much as a missing lower order.
        score += 0.5

    # Ties: fewest rings assumed hidden below the series.
    return score + 0.001 * (lowest - 1)


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
