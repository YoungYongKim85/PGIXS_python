import numpy as np
import pytest

from pgixs_python.calibration.ring_candidates import analyze_sdd_candidates, candidates_to_dicts, find_rough_peaks
from pgixs_python.calibration.sdd import compute_sdd_from_ring
from pgixs_python.calibration.standards import standard_q_value


def _synthetic_multi_ring_profile(radii, energy_kev=9.5, pixel_size_mm=0.075, n_bins=500, ring_sigma=2.0):
    """A radial profile with real Gaussian rings at the given pixel radii,
    plus a strong near-center feature (mimicking direct/parasitic
    scattering) that must NOT be picked up as a candidate (r <= min_radius)."""

    r_axis = np.arange(n_bins, dtype=np.float64) + 0.5
    intensity = np.full(n_bins, 5.0)
    intensity += 5000.0 * np.exp(-((r_axis - 20.0) ** 2) / (2.0 * 5.0**2))  # near-center feature
    for radius in radii:
        intensity += 500.0 * np.exp(-((r_axis - radius) ** 2) / (2.0 * ring_sigma**2))
    coverage = np.ones(n_bins, dtype=np.float64)
    return r_axis, intensity.astype(np.float64), coverage


def test_find_rough_peaks_excludes_near_center_feature():
    r_axis, intensity, coverage = _synthetic_multi_ring_profile([120.0, 180.0])

    locs = find_rough_peaks(r_axis, intensity, coverage, min_radius=50.0)

    radii_found = r_axis[locs]
    assert not np.any(radii_found < 50.0)
    assert np.any(np.isclose(radii_found, 120.5, atol=1.0))
    assert np.any(np.isclose(radii_found, 180.5, atol=1.0))


def test_analyze_sdd_candidates_recovers_known_sdd_for_multiple_orders():
    """Build a synthetic radial profile with rings at exactly the radii a
    real SDD/energy/q combination would put Silver behenate's first few
    orders at, and confirm the recovered FinalSDD matches."""

    true_sdd = 300.0
    pixel_size_mm = 0.075
    energy_kev = 9.5

    radii = []
    for order in range(1, 5):
        q = standard_q_value("Silver behenate", order)
        wavelength = 12.398419843320026 / energy_kev
        two_theta = 2.0 * np.arcsin(q * wavelength / (4.0 * np.pi))
        radius = true_sdd * np.tan(two_theta) / pixel_size_mm
        radii.append(radius)

    r_axis, intensity, coverage = _synthetic_multi_ring_profile(radii, ring_sigma=1.5)

    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Silver behenate", energy_kev, pixel_size_mm)

    assert report.accepted_count >= 3
    assert report.final_sdd == pytest.approx(true_sdd, rel=0.02)
    for candidate in report.candidates:
        if candidate.accepted:
            assert candidate.q_value == pytest.approx(standard_q_value("Silver behenate", candidate.order))


def test_analyze_sdd_candidates_handles_no_peaks():
    r_axis = np.arange(200, dtype=np.float64) + 0.5
    intensity = np.full(200, 10.0)
    coverage = np.ones(200, dtype=np.float64)

    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Silver behenate", 9.5, 0.075)

    assert report.candidates == []
    assert not np.isfinite(report.final_sdd)
    assert report.accepted_count == 0
    assert report.message


def test_analyze_sdd_candidates_non_silver_behenate_only_tries_order_one():
    """Every non-SEBS/SB standard sample has no meaningful multi-order
    search -- only order 1 starting at the first candidate is tried."""

    q1 = standard_q_value("Sucrose", 1)
    r_axis, intensity, coverage = _synthetic_multi_ring_profile([150.0])

    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Sucrose", 9.5, 0.075)

    assert len(report.candidates) == 1
    assert report.candidates[0].order == 1
    assert report.candidates[0].q_value == pytest.approx(q1)


def test_multiple_visible_rings_with_first_order_hidden_are_assigned_orders_two_through_four():
    """The more typical real-data version of the beamstop-hidden-1st-order
    scenario: several higher orders (2nd-4th) are all clearly visible and
    mutually self-consistent, with only the 1st order missing (hidden
    behind the beamstop). The cross-ring rel_spread term already handles
    this correctly on its own (order_start=1 would misalign every ring by
    one order and blow up the spread) -- this just locks in that the new
    single-ring plausibility checks don't regress the already-working
    multi-ring path."""

    true_sdd = 150.0
    pixel_size_mm = 0.075
    energy_kev = 9.5
    radii = []
    for order in range(1, 5):
        q = standard_q_value("Silver behenate", order)
        wavelength = 12.398419843320026 / energy_kev
        two_theta = 2.0 * np.arcsin(q * wavelength / (4.0 * np.pi))
        radii.append(true_sdd * np.tan(two_theta) / pixel_size_mm)

    r_axis, intensity, coverage = _synthetic_multi_ring_profile(radii[1:], n_bins=800, ring_sigma=2.0)

    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Silver behenate", energy_kev, pixel_size_mm)

    assert [c.order for c in report.candidates] == [2, 3, 4]
    assert report.final_sdd == pytest.approx(true_sdd, rel=0.01)


def test_single_visible_ring_hidden_behind_beamstop_is_recognized_as_order_two():
    """Regression: "1st order위치를 잘못 인식하는거야... 1st peak은 빔스탑에
    가려져 있어" -- a short-SDD (WAXS-like) geometry where the 1st-order
    Silver behenate ring falls inside the excluded near-beamstop region
    and only the 2nd-order ring is ever detected as a candidate. The old
    scoring always defaulted a lone ring to order=1 (a tie it could never
    lose), silently ignoring that the implied 1st-order ring would sit
    well inside the excluded region -- exactly what a hidden ring behind
    the beamstop should look like."""

    pixel_size_mm = 0.075
    energy_kev = 9.5
    q1 = standard_q_value("Silver behenate", 1)
    q2 = standard_q_value("Silver behenate", 2)

    # Choose a true SDD by pinning the (only visible) 2nd-order ring to a
    # rough_radius comfortably past min_radius, then confirm the 1st-order
    # ring it implies is comfortably inside the excluded (< min_radius)
    # region -- i.e. genuinely hidden, not just barely.
    true_sdd = compute_sdd_from_ring(80.0, pixel_size_mm, energy_kev, q2).sdd
    order1_radius = 80.0 * (q1 / q2)  # small-angle approx, just for the assertion below
    assert order1_radius < 50.0  # sanity check: 1st order is inside the excluded region

    r_axis, intensity, coverage = _synthetic_multi_ring_profile([80.0], ring_sigma=2.0)

    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Silver behenate", energy_kev, pixel_size_mm)

    assert len(report.candidates) == 1
    assert report.candidates[0].order == 2
    assert report.candidates[0].sdd == pytest.approx(true_sdd, rel=0.01)


def test_single_visible_ring_far_from_beamstop_stays_order_one():
    """The counterpart to the hidden-1st-order case above: a single ring
    where the implied 2nd-order ring would fall *beyond* the analyzed
    radial range (not just unseen but genuinely out of view) must still be
    recognized as a real, unremarkable order=1 ring -- the "was a lower/
    next order plausibly hidden" checks must not overcorrect every lone
    ring into a higher order."""

    r_axis, intensity, coverage = _synthetic_multi_ring_profile([80.0], n_bins=110, ring_sigma=2.0)

    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Silver behenate", 9.5, 0.075)

    assert len(report.candidates) == 1
    assert report.candidates[0].order == 1


def test_candidates_to_dicts_round_trips_fields():
    r_axis, intensity, coverage = _synthetic_multi_ring_profile([150.0])
    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Sucrose", 9.5, 0.075)

    payload = candidates_to_dicts(report)

    assert len(payload) == 1
    assert set(payload[0]) == {
        "rough_radius", "fitted_radius", "order", "q_value", "coverage",
        "r_square", "sigma", "sdd_mm", "accepted", "message",
    }


def test_find_rough_peaks_threshold_ignores_the_beamstop_core():
    """Regression: the peak-prominence threshold must be scaled off the
    *analysis domain* (`r > min_radius`), not the whole profile, matching
    `analyzePGIXSSDDCandidates.m`'s `max(intensity(valid), [], 'omitnan')`
    and the reason its comment gives ("A saturated beam-stop/direct-beam
    feature at a radius rejected by valid must not suppress otherwise usable
    standard rings"). Scaling off the whole profile made the threshold track
    the beam-stop core, which on real data is orders of magnitude brighter
    than any calibrant ring, so Smart Auto Run silently found nothing."""

    n_bins = 500
    r_axis = np.arange(n_bins, dtype=np.float64) + 0.5
    intensity = np.full(n_bins, 5.0)
    # A direct-beam/beam-stop core 1000x the rings, entirely inside r <= 50.
    intensity += 5.0e5 * np.exp(-((r_axis - 15.0) ** 2) / (2.0 * 6.0**2))
    for radius in (120.0, 240.0):
        intensity += 500.0 * np.exp(-((r_axis - radius) ** 2) / (2.0 * 2.0**2))
    coverage = np.ones(n_bins, dtype=np.float64)

    peaks = find_rough_peaks(r_axis, intensity, coverage)

    found = np.sort(r_axis[peaks])
    assert found.size == 2, f"expected both rings, got radii {found}"
    np.testing.assert_allclose(found, [120.5, 240.5], atol=1.5)


def test_order_assignment_prefers_the_self_consistent_sdd_over_more_rings():
    """Regression: on a short-SDD AgBH pattern the visible rings are orders 2
    and 3 (radius ratio exactly 1.5), with order 1 behind the beam stop. The
    scorer used to read them as orders 1, 2, 3 -- one extra "explained" ring,
    but SDD values spread 6.7% instead of 0.04%. Raw rel_spread moved the
    score less than the 0.5 hidden-order penalty, so the inconsistent reading
    won; ring SDD agreement now dominates past a 1% tolerance."""

    r = np.arange(1, 221, dtype=np.float64)
    profile = (
        250.0 * np.exp(-((r - 60.0) ** 2) / (2 * 5.0**2))
        + 100.0 * np.exp(-((r - 140.0) ** 2) / (2 * 3.0**2))
        + 70.0 * np.exp(-((r - 210.0) ** 2) / (2 * 4.0**2))
        + 2.0
    )

    report = analyze_sdd_candidates(
        r, profile, np.ones_like(r), "Silver behenate", 9.5, 0.075, min_coverage=0.8
    )

    rows = candidates_to_dicts(report)
    assert not rows[0]["accepted"], "the 60 px feature is not an AgBH ring here"
    accepted = [c["order"] for c in rows if c["accepted"]]
    assert accepted == [2, 3]
    # 140 px and 210 px agree on the SDD to well under a percent.
    assert report.final_sdd == pytest.approx(239.98, abs=0.5)


def test_order_assignment_still_reads_a_genuine_first_order_pattern():
    """The tolerance must not push every pattern to higher orders: rings at a
    1:2 radius ratio are orders 1 and 2 and stay that way."""

    r = np.arange(1, 221, dtype=np.float64)
    profile = (
        100.0 * np.exp(-((r - 70.0) ** 2) / (2 * 3.0**2))
        + 70.0 * np.exp(-((r - 140.0) ** 2) / (2 * 4.0**2))
        + 2.0
    )

    report = analyze_sdd_candidates(
        r, profile, np.ones_like(r), "Silver behenate", 9.5, 0.075, min_coverage=0.8
    )

    assert [c["order"] for c in candidates_to_dicts(report) if c["accepted"]] == [1, 2]
