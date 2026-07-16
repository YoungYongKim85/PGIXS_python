import numpy as np

from pgixs_python.averaging.cake import angle_in_relative_sector, cake_average, normalize_angle180, pixel_angle_map


def test_angle_in_relative_sector_wraps_around_180():
    angles = np.array([-179, -170, 0, 170, 179], dtype=np.float32)

    mask = angle_in_relative_sector(angles, 170, -170)

    assert np.array_equal(mask, [True, True, False, True, True])


def test_normalize_angle180_matches_expected_range():
    values = normalize_angle180(np.array([-540, -181, -180, 0, 180, 181, 540]))

    assert np.all(values >= -180)
    assert np.all(values < 180)
    assert values[2] == -180
    assert values[4] == -180


def test_cake_average_pixel_mode_returns_coverage():
    q_map = np.ones((5, 5), dtype=np.float32)
    data = np.ones((5, 5), dtype=np.float32)
    data[2, 1] = np.nan
    center = (3, 3)
    angle_map = pixel_angle_map(q_map.shape, center)

    result = cake_average(data, q_map, angle_map, center, reference_angle=0, angle_range=(-20, 20))

    assert result.is_valid
    assert result.selected_pixel_count > 0
    assert np.any(result.coverage < 1)
    assert result.q_axis.shape == result.intensity.shape


def test_cake_average_full_relative_range():
    q_map = np.ones((3, 3), dtype=np.float32)
    data = np.ones((3, 3), dtype=np.float32)
    center = (2, 2)
    angle_map = pixel_angle_map(q_map.shape, center)

    result = cake_average(data, q_map, angle_map, center, reference_angle=0, angle_range=(-180, 180))

    assert result.is_valid
    assert result.selected_pixel_count == 9
