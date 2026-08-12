import numpy as np

from pgixs_python.averaging.angular import (
    angular_profile,
    find_coverage_drop_bins,
    make_sector_mask,
    sector_average,
)


def test_make_sector_mask_wraparound_and_full_circle():
    full = make_sector_mask((5, 5), center=(3, 3), angle_range=(0, 0))
    wrapped = make_sector_mask((5, 5), center=(3, 3), angle_range=(350, 10))

    assert np.all(full)
    assert wrapped.shape == (5, 5)
    assert wrapped[2, 1]
    assert not wrapped[0, 2]


def test_sector_average_reports_coverage_drop():
    q_map = np.ones((7, 7), dtype=np.float32)
    data = np.ones((7, 7), dtype=np.float32)
    data[1:4, 1:4] = np.nan

    result = sector_average(data, q_map, center=(4, 4), angle_range=(0, 0))

    assert result.is_valid
    assert result.sector_pixel_count == 49
    assert result.full_pixel_count == 49
    assert np.any(result.coverage < 1)
    assert result.low_coverage_bins.size >= 0


def test_find_coverage_drop_bins_detects_step():
    coverage = np.array([1.0, 0.98, 0.95, 0.2, 0.19, 0.18], dtype=np.float32)

    drops = find_coverage_drop_bins(coverage)

    assert 3 in drops


def test_angular_profile_uniform_ring_gives_flat_intensity():
    data = np.full((13, 13), 5.0, dtype=np.float32)
    center = (7.0, 7.0)

    result = angular_profile(data, center=center, radius_range=(2.0, 5.0), n_angle_bins=8)

    assert result.angle_axis.shape == (8,)
    assert result.intensity.shape == (8,)
    finite = np.isfinite(result.intensity)
    assert np.any(finite)
    assert np.allclose(result.intensity[finite], 5.0)
    assert np.all(result.coverage[finite] == 1.0)


def test_angular_profile_reports_low_coverage_for_masked_quadrant():
    data = np.full((13, 13), 5.0, dtype=np.float32)
    center = (7.0, 7.0)
    cols, rows = np.meshgrid(np.arange(1, 14, dtype=np.float64), np.arange(1, 14, dtype=np.float64))
    angle = np.mod(np.degrees(np.arctan2(-(rows - center[1]), -(cols - center[0]))), 360.0)
    data[(angle >= 0) & (angle < 90)] = np.nan

    result = angular_profile(data, center=center, radius_range=(2.0, 5.0), n_angle_bins=4)

    assert result.coverage[0] == 0.0
