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


def test_candidates_to_dicts_round_trips_fields():
    r_axis, intensity, coverage = _synthetic_multi_ring_profile([150.0])
    report = analyze_sdd_candidates(r_axis, intensity, coverage, "Sucrose", 9.5, 0.075)

    payload = candidates_to_dicts(report)

    assert len(payload) == 1
    assert set(payload[0]) == {
        "rough_radius", "fitted_radius", "order", "q_value", "coverage",
        "r_square", "sigma", "sdd_mm", "accepted", "message",
    }
