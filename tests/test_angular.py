import numpy as np

from pgixs_python.averaging.angular import find_coverage_drop_bins, make_sector_mask, sector_average


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
