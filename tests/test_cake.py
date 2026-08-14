import numpy as np
import pytest

from pgixs_python.averaging.cake import (
    accumulation_average,
    angle_in_relative_sector,
    cake_average,
    normalize_angle180,
    pixel_angle_map,
)


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


def _synthetic_frame(rng, rows=40, cols=50):
    data = rng.uniform(1, 100, size=(rows, cols)).astype(np.float32)
    beam_y, beam_z = 20.0, 25.0
    cols_idx, rows_idx = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    q_map = (np.hypot(cols_idx - beam_y, rows_idx - beam_z) * 0.01).astype(np.float32)
    return data, q_map, (beam_y, beam_z)


def test_accumulation_average_matches_manual_band_mean():
    """The row/column band this averages over is already pixel-grid-aligned
    (unlike cake_average's fuzzy floor-radius bins), so its output must
    match a plain np.nanmean over the equivalent hand-picked slice exactly."""

    rng = np.random.default_rng(0)
    data, q_map, (beam_y, beam_z) = _synthetic_frame(rng)

    result = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=0.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0)

    assert result.is_valid
    manual_band_mean = np.nanmean(data[24:27, :], axis=0)
    # Nearest output point to the beam is the last one kept (angle 0 keeps
    # the left half, sorted ascending by distance from the beam).
    assert result.intensity[0] == pytest.approx(manual_band_mean[19])


def test_accumulation_average_perp_offset_shifts_the_band():
    rng = np.random.default_rng(0)
    data, q_map, (beam_y, beam_z) = _synthetic_frame(rng)

    result = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=0.0, perp_offset=5.0, band_minus=-1.0, band_plus=1.0)

    manual_band_mean = np.nanmean(data[24 - 5 : 27 - 5, :], axis=0)
    assert result.intensity[0] == pytest.approx(manual_band_mean[19])


def test_accumulation_average_rejects_non_axis_aligned_angle():
    rng = np.random.default_rng(0)
    data, q_map, center = _synthetic_frame(rng)

    with pytest.raises(ValueError, match="0, 90, or 180"):
        accumulation_average(data, q_map, center, angle_deg=45.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0)
