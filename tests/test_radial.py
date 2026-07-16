import numpy as np

from pgixs_python.averaging.radial import build_radius_bin_map, radius_average


def test_radius_bin_map_uses_one_based_center_like_matlab():
    q_map = np.ones((3, 3), dtype=np.float32)

    radius_map = build_radius_bin_map(q_map, center=(2, 2))

    assert radius_map.is_valid
    assert radius_map.n_bins == 2
    assert radius_map.bin_index[1, 1] == 0
    assert np.isclose(radius_map.r_axis[0], 0)


def test_radius_average_reports_gap_coverage():
    q_map = np.ones((3, 3), dtype=np.float32)
    data = np.ones((3, 3), dtype=np.float32)
    data[0, 1] = np.nan
    data[1, 0] = np.nan

    radius_map = build_radius_bin_map(q_map, center=(2, 2))
    result = radius_average(data, radius_map)

    assert result.intensity.shape == (2,)
    assert result.counts[0] == 1
    assert result.geom_counts[0] == 1
    assert result.counts[1] == 6
    assert result.geom_counts[1] == 8
    assert np.isclose(result.coverage[1], 0.75)


def test_radius_average_respects_geom_mask():
    q_map = np.ones((3, 3), dtype=np.float32)
    data = np.ones((3, 3), dtype=np.float32)
    geom_mask = np.ones((3, 3), dtype=bool)
    geom_mask[0, :] = False

    radius_map = build_radius_bin_map(q_map, center=(2, 2), geom_mask=geom_mask)
    result = radius_average(data, radius_map)

    assert np.sum(result.geom_counts) == 6
    assert np.all(result.coverage == 1)
