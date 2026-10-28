import warnings
from concurrent.futures import ThreadPoolExecutor

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


def test_cake_average_thread_safe_no_warning_race():
    """Cakecut's whole reduction chain (cake_average -> build_radius_bin_map
    -> radius_average -> qbin_average) never calls np.nanmean/np.nanmedian
    on a possibly-empty/all-NaN slice in the first place -- every bin's
    values are pre-filtered to only its actually-present finite pixels
    (np.bincount-based sums, or values[mask] before np.mean/np.median), so
    there is no warnings.catch_warnings()-style suppression to race on
    (unlike accumulation_average's old implementation -- see
    test_accumulation_average_thread_safe_no_warning_race). Verified here
    with a wedge that deliberately has a zero-coverage radius bin (a
    detector gap masked out mid-wedge), run concurrently with
    warnings.simplefilter("error") active globally."""

    rng = np.random.default_rng(0)
    shape = (120, 120)
    center = (60.5, 60.5)
    data = rng.uniform(1.0, 100.0, size=shape).astype(np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    q_map = (np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0)) * 0.01).astype(np.float32)
    angle_map = pixel_angle_map(shape, center)
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    data[(r > 15) & (r < 25)] = np.nan  # a gap ring mid-wedge -> a zero-coverage bin

    def run(_i):
        return cake_average(data, q_map, angle_map, center=center, reference_angle=0.0, angle_range=(-2.0, 2.0))

    baseline = run(0)
    assert baseline.is_valid
    assert np.nanmin(baseline.coverage) == 0.0  # confirms the empty-bin path is actually exercised

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run, range(40)))

    for result in results:
        np.testing.assert_array_equal(result.intensity, baseline.intensity)


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
    manual_band_mean = np.nanmean(data[23:26, :], axis=0)  # 1-based rows 24-26 around beam_z=25
    # The beam pixel itself has q = 0 and is dropped, so the first output
    # point is the column just left of the beam (angle 0 keeps the left half).
    assert result.intensity[0] == pytest.approx(manual_band_mean[18])


def test_accumulation_average_90_band_is_centred_on_beam_column():
    rng = np.random.default_rng(0)
    data, q_map, (beam_y, beam_z) = _synthetic_frame(rng)

    result = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=90.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0)

    # 1-based columns 19-21 around beam_y=20; first point is the row just
    # above the beam (the beam row has q = 0).
    manual_band_mean = np.nanmean(data[:, 18:21], axis=1)
    assert result.intensity[0] == pytest.approx(manual_band_mean[23])


def test_accumulation_average_perp_offset_shifts_the_band():
    rng = np.random.default_rng(0)
    data, q_map, (beam_y, beam_z) = _synthetic_frame(rng)

    result = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=0.0, perp_offset=5.0, band_minus=-1.0, band_plus=1.0)

    manual_band_mean = np.nanmean(data[23 - 5 : 26 - 5, :], axis=0)
    assert result.intensity[0] == pytest.approx(manual_band_mean[18])


def test_accumulation_average_q_is_read_on_the_beam_line_not_the_shifted_band():
    """MATLAB `Accumpix_gi` labels the band with Q on the beam row/column.
    Averaging Q over a band moved off the beam would add the offset's own
    perpendicular component and push every peak to higher q."""

    rng = np.random.default_rng(0)
    data, q_map, (beam_y, beam_z) = _synthetic_frame(rng)

    shifted = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=0.0, perp_offset=5.0, band_minus=-1.0, band_plus=1.0)
    centred = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=0.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0)
    np.testing.assert_allclose(shifted.q_axis, centred.q_axis)
    np.testing.assert_allclose(shifted.q_axis, q_map[24, 18::-1])

    vertical = accumulation_average(data, q_map, (beam_y, beam_z), angle_deg=90.0, perp_offset=7.0, band_minus=-1.0, band_plus=1.0)
    np.testing.assert_allclose(vertical.q_axis, q_map[23::-1, 19])


def test_accumulation_average_rejects_non_axis_aligned_angle():
    rng = np.random.default_rng(0)
    data, q_map, center = _synthetic_frame(rng)

    with pytest.raises(ValueError, match="0, 90, or 180"):
        accumulation_average(data, q_map, center, angle_deg=45.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0)


def test_accumulation_average_all_nan_band_emits_no_warning():
    """A band that lands entirely on gap/masked pixels (e.g. right on the
    beam column) is expected/harmless -- must come back as
    `is_valid=False`, not print/raise numpy's "Mean of empty slice"."""

    rng = np.random.default_rng(0)
    data, q_map, (_beam_y, beam_z) = _synthetic_frame(rng)
    data[:, :3] = np.nan  # columns the band below will land on

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = accumulation_average(
            data, q_map, (1.0, beam_z), angle_deg=90.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0
        )

    assert not result.is_valid


def test_accumulation_average_thread_safe_no_warning_race():
    """Regression: the previous implementation suppressed "Mean of empty
    slice" with `warnings.catch_warnings()`, which mutates a *global*
    filter list -- not safe across the concurrent threads
    `process_linecut_stack`'s `parallel=True` option now calls this
    function from. Under a race, one thread's filter could be installed or
    torn down while another thread's triggering call is in flight, leaking
    the warning to the console. The current implementation excludes the
    all-NaN slice from the reducer call entirely instead, so no warning is
    ever raised in the first place -- verified here by running many calls
    concurrently with `warnings.simplefilter("error")` active globally."""

    rng = np.random.default_rng(0)
    rows, cols = 40, 50
    data = rng.uniform(1, 100, size=(rows, cols)).astype(np.float32)
    data[:, :3] = np.nan
    cols_idx, rows_idx = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    beam_y, beam_z = 1.0, 25.0  # band sits right on the all-NaN columns
    q_map = (np.hypot(cols_idx - beam_y, rows_idx - beam_z) * 0.01).astype(np.float32)

    def run(_i):
        return accumulation_average(
            data, q_map, (beam_y, beam_z), angle_deg=90.0, perp_offset=0.0, band_minus=-1.0, band_plus=1.0
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run, range(40)))

    assert not results[0].is_valid
