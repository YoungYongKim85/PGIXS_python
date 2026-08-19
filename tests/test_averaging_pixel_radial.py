import numpy as np
import pytest

from pgixs_python.averaging.pixel_radial import build_pixel_radius_bin_map, compute_radial_profile


def _synthetic_ring(n=400, center=(200.5, 200.5), ring_radius=100.0, ring_sigma=3.0):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    data = 10.0 + 500.0 * np.exp(-((r - ring_radius) ** 2) / (2.0 * ring_sigma**2))
    return data.astype(np.float32)


def test_build_pixel_radius_bin_map_bins_by_floor_radius():
    bin_index, n_bins, r_axis, geom = build_pixel_radius_bin_map((5, 5), center=(3.0, 3.0))

    assert geom.all()
    assert n_bins > 0
    assert r_axis[0] == 0.5
    assert bin_index[2, 2] == 0  # the pixel at the center itself: radius 0


def test_build_pixel_radius_bin_map_respects_geom_mask():
    geom_mask = np.ones((5, 5), dtype=bool)
    geom_mask[0, 0] = False

    bin_index, n_bins, r_axis, geom = build_pixel_radius_bin_map((5, 5), center=(3.0, 3.0), geom_mask=geom_mask)

    assert not geom[0, 0]
    assert bin_index[0, 0] == -1


def test_compute_radial_profile_peaks_at_the_known_ring_radius():
    data = _synthetic_ring(ring_radius=100.0)

    profile = compute_radial_profile(data, center=(200.5, 200.5), smooth=False)

    peak_idx = int(np.nanargmax(profile.intensity))
    assert profile.r_axis[peak_idx] == pytest.approx(100.0, abs=0.6)


def test_compute_radial_profile_full_coverage_inside_the_frame():
    data = _synthetic_ring(ring_radius=100.0)

    profile = compute_radial_profile(data, center=(200.5, 200.5), smooth=False)

    peak_idx = int(np.nanargmax(profile.intensity))
    assert profile.coverage[peak_idx] == 1.0


def test_compute_radial_profile_smoothing_changes_the_curve_but_not_the_axis():
    data = _synthetic_ring(ring_radius=100.0)

    raw = compute_radial_profile(data, center=(200.5, 200.5), smooth=False)
    smoothed = compute_radial_profile(data, center=(200.5, 200.5), smooth=True)

    np.testing.assert_allclose(raw.r_axis, smoothed.r_axis)
    assert not np.allclose(raw.intensity, smoothed.intensity, equal_nan=True)
    # raw_intensity always carries the unsmoothed curve, even when smooth=True.
    np.testing.assert_allclose(smoothed.raw_intensity, raw.intensity, equal_nan=True)


def test_compute_radial_profile_handles_empty_geometry():
    data = np.zeros((5, 5), dtype=np.float32)
    geom_mask = np.zeros((5, 5), dtype=bool)

    profile = compute_radial_profile(data, center=(3.0, 3.0), geom_mask=geom_mask)

    assert profile.r_axis.size == 0
    assert profile.intensity.size == 0
