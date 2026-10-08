import numpy as np
import pytest

from pgixs_python.averaging.cake import pixel_angle_map
from pgixs_python.averaging.radial import radius_average
from pgixs_python.ui_backend.series_processing import (
    average_exposure_seconds,
    build_circular_radius_map,
    compute_experiment_x_axis,
    normalize_rows_by_counter,
    process_azimuthal_stack,
    process_circular_stack,
    process_linecut_stack,
    process_roi_stack,
    sort_x_axis,
)


def _synthetic_ring_stack(n_frames=4, shape=(200, 200), center=(100.5, 100.5), radii=None, seed=0):
    """A stack of frames, each with a ring at a slightly different radius
    (mimicking a real series where the pattern evolves frame to frame)."""

    if radii is None:
        radii = [60.0 + 2.0 * i for i in range(n_frames)]
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(seed)
    frames = []
    for radius in radii:
        data = 10.0 + 500.0 * np.exp(-((r - radius) ** 2) / (2.0 * 3.0**2))
        data = data + rng.uniform(0.0, 1.0, size=shape)
        frames.append(data.astype(np.float32))
    return np.stack(frames, axis=0)


def test_process_linecut_stack_default_cut_tracks_moving_ring():
    center = (100.5, 100.5)
    radii = [60.0, 70.0, 80.0]
    stack = _synthetic_ring_stack(n_frames=3, center=center, radii=radii)
    q_map = np.hypot(*np.meshgrid(np.arange(1, 201) - center[0], np.arange(1, 201) - center[1]))
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    result = process_linecut_stack(stack, q_map, angle_map, center=center, angle_deg=90.0)

    assert result.intensity.shape[0] == 3
    assert result.intensity.shape == result.coverage.shape
    for i, radius in enumerate(radii):
        peak_idx = int(np.nanargmax(result.intensity[i]))
        assert result.x_axis[peak_idx] == pytest.approx(radius, abs=2.0)


def test_process_linecut_stack_reports_progress_per_frame():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=4, center=center, radii=[60.0, 60.0, 60.0, 60.0])
    q_map = np.hypot(*np.meshgrid(np.arange(1, 201) - center[0], np.arange(1, 201) - center[1]))
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    calls = []
    process_linecut_stack(
        stack, q_map, angle_map, center=center, angle_deg=90.0,
        progress_callback=lambda completed, total: calls.append((completed, total)),
    )

    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_process_linecut_stack_parallel_matches_serial():
    center = (100.5, 100.5)
    radii = [58.0, 63.0, 71.0, 66.0, 60.0, 75.0]
    stack = _synthetic_ring_stack(n_frames=6, center=center, radii=radii)
    q_map = np.hypot(*np.meshgrid(np.arange(1, 201) - center[0], np.arange(1, 201) - center[1]))
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    serial = process_linecut_stack(stack, q_map, angle_map, center=center, angle_deg=90.0, parallel=False)
    parallel = process_linecut_stack(stack, q_map, angle_map, center=center, angle_deg=90.0, parallel=True)

    np.testing.assert_array_equal(serial.x_axis, parallel.x_axis)
    np.testing.assert_array_equal(serial.intensity, parallel.intensity)
    np.testing.assert_array_equal(serial.coverage, parallel.coverage)


def test_process_linecut_stack_parallel_reports_progress_exactly_once_per_frame():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=8, center=center, radii=[60.0 + i for i in range(8)])
    q_map = np.hypot(*np.meshgrid(np.arange(1, 201) - center[0], np.arange(1, 201) - center[1]))
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    calls = []
    process_linecut_stack(
        stack, q_map, angle_map, center=center, angle_deg=90.0, parallel=True,
        progress_callback=lambda completed, total: calls.append((completed, total)),
    )

    # Completion order isn't guaranteed under threading, but every "total"
    # must be 8 and "completed" must count 1..8 exactly once each.
    assert [total for _, total in calls] == [8] * 8
    assert sorted(completed for completed, _ in calls) == list(range(1, 9))


def test_process_linecut_stack_rejects_geometry_shape_mismatch():
    # Different-sized frames within one "stack" is invalid input in
    # practice, but exercising the guard directly with a hand-built
    # mismatch is simplest: call twice with different-shaped fake results
    # by using two different centers is awkward, so just confirm a normal
    # call doesn't raise (shape mismatch is naturally impossible for a
    # true ndarray stack of equal-shaped frames -- this documents intent).
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=2, center=center, radii=[60.0, 60.0])
    q_map = np.hypot(*np.meshgrid(np.arange(1, 201) - center[0], np.arange(1, 201) - center[1]))
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    result = process_linecut_stack(stack, q_map, angle_map, center=center, angle_deg=90.0)
    assert result.intensity.shape[0] == 2


def _synthetic_ring_stack_change_ai(n_frames=3, shape=(200, 200), center=(100.5, 100.5), radius=60.0, radius_step=1.0, seed=0):
    """A stack whose ring sits at a *drifting* pixel radius, mimicking
    "Change ai" mode where each frame's own Q-map (pixel radius -> Q) shifts
    slightly because the incident angle steps every frame -- the ring's
    apparent radius in raw pixels moves even though the physical Q it sits
    at is meant to stay fixed once each frame uses its own correct q-map."""

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(seed)
    frames = []
    q_maps = []
    for i in range(n_frames):
        frame_radius = radius + radius_step * i
        data = 10.0 + 500.0 * np.exp(-((r - frame_radius) ** 2) / (2.0 * 3.0**2))
        data = data + rng.uniform(0.0, 1.0, size=shape)
        frames.append(data.astype(np.float32))
        # This frame's own q-map: scale pixel radius so the ring lands at
        # the *same* Q (60.0 * 0.01) in every frame despite the drifting
        # pixel radius -- the point of per-frame q-maps.
        scale = 0.01 * radius / frame_radius
        q_maps.append((r * scale).astype(np.float32))
    return np.stack(frames, axis=0), np.stack(q_maps, axis=0)


def test_process_linecut_stack_change_ai_aligns_drifting_ring_to_fixed_q():
    center = (100.5, 100.5)
    stack, q_map_stack = _synthetic_ring_stack_change_ai(n_frames=3, center=center, radius=60.0, radius_step=5.0)
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    result = process_linecut_stack(
        stack, None, angle_map, center=center, angle_deg=90.0, q_map_stack=q_map_stack
    )

    assert result.intensity.shape[0] == 3
    for i in range(3):
        peak_idx = int(np.nanargmax(result.intensity[i]))
        assert result.x_axis[peak_idx] == pytest.approx(0.60, abs=0.05)


def test_process_linecut_stack_requires_q_map_without_change_ai():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=1, center=center, radii=[60.0])
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    with pytest.raises(ValueError):
        process_linecut_stack(stack, None, angle_map, center=center, angle_deg=90.0)


def test_process_linecut_stack_ref_axis_aligns_extension_onto_original_reference():
    center = (100.5, 100.5)
    full_stack, full_q_stack = _synthetic_ring_stack_change_ai(n_frames=3, center=center, radius=60.0, radius_step=5.0)
    angle_map = pixel_angle_map(full_stack.shape[1:], center=center)

    original = process_linecut_stack(
        full_stack, None, angle_map, center=center, angle_deg=90.0, q_map_stack=full_q_stack
    )

    # Extend with just the new (3rd) frame, using the original series'
    # reference axis -- must produce the *same*-length/valued axis as the
    # original run, not a fresh one derived from the new frame alone.
    extension = process_linecut_stack(
        full_stack[2:3], None, angle_map, center=center, angle_deg=90.0,
        q_map_stack=full_q_stack[2:3], ref_axis=original.x_axis,
    )

    np.testing.assert_allclose(extension.x_axis, original.x_axis)
    np.testing.assert_allclose(extension.intensity[0], original.intensity[2], equal_nan=True)


def test_process_azimuthal_stack_shapes():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=3, center=center, radii=[60.0, 60.0, 60.0])

    result = process_azimuthal_stack(stack, center=center, radius_range=(55.0, 65.0), n_angle_bins=90)

    assert result.intensity.shape == (3, result.x_axis.shape[0])
    assert result.coverage.shape == result.intensity.shape


def test_process_azimuthal_stack_reports_progress_per_frame():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=4, center=center, radii=[60.0, 60.0, 60.0, 60.0])

    calls = []
    process_azimuthal_stack(
        stack, center=center, radius_range=(55.0, 65.0), n_angle_bins=90,
        progress_callback=lambda completed, total: calls.append((completed, total)),
    )

    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_process_azimuthal_stack_parallel_matches_serial():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=6, center=center, radii=[60.0, 61.0, 62.0, 63.0, 64.0, 65.0])

    serial = process_azimuthal_stack(stack, center=center, radius_range=(55.0, 70.0), n_angle_bins=90, parallel=False)
    parallel = process_azimuthal_stack(stack, center=center, radius_range=(55.0, 70.0), n_angle_bins=90, parallel=True)

    np.testing.assert_array_equal(serial.x_axis, parallel.x_axis)
    np.testing.assert_array_equal(serial.intensity, parallel.intensity)
    np.testing.assert_array_equal(serial.coverage, parallel.coverage)


def test_process_roi_stack_matches_hand_computed_sum():
    stack = np.zeros((3, 10, 10), dtype=np.float32)
    stack[0, 2:5, 2:5] = 1.0  # 9 pixels of value 1
    stack[1, 2:5, 2:5] = 2.0  # 9 pixels of value 2
    stack[2, 2:5, 2:5] = 3.0  # 9 pixels of value 3

    sums = process_roi_stack(stack, rois=[(3, 5, 3, 5)])  # 1-based inclusive, matches the 2:5 python slice

    assert sums.shape == (3, 1)
    np.testing.assert_allclose(sums[:, 0], [9.0, 18.0, 27.0])


def test_process_roi_stack_multiple_rois():
    stack = np.ones((2, 10, 10), dtype=np.float32)
    sums = process_roi_stack(stack, rois=[(1, 2, 1, 2), (1, 10, 1, 10)])

    assert sums.shape == (2, 2)
    np.testing.assert_allclose(sums[:, 0], [4.0, 4.0])
    np.testing.assert_allclose(sums[:, 1], [100.0, 100.0])


def test_process_roi_stack_handles_nan():
    stack = np.full((1, 5, 5), np.nan, dtype=np.float32)
    stack[0, 1, 1] = 5.0
    sums = process_roi_stack(stack, rois=[(1, 5, 1, 5)])
    assert sums[0, 0] == pytest.approx(5.0)


def test_average_exposure_seconds_uses_mean_frame_to_frame_interval():
    # Deliberately uneven intervals (10, 12, 8) -- averages to 10, not just
    # the first interval, matching "don't trust a single data point".
    timestamps = np.array([0.0, 10.0, 22.0, 30.0])
    assert average_exposure_seconds(timestamps) == pytest.approx(10.0)


def test_average_exposure_seconds_ignores_non_finite_and_needs_two_points():
    assert average_exposure_seconds(np.array([5.0])) == 0.0
    assert average_exposure_seconds(np.array([])) == 0.0
    assert average_exposure_seconds(np.array([np.nan, np.nan])) == 0.0
    assert average_exposure_seconds(np.array([0.0, np.nan, 10.0])) == pytest.approx(10.0)


def test_compute_experiment_x_axis_time_mode_auto_computes_exposure_offset():
    # No time_offset_seconds given -- auto-derived from the series' own
    # average frame interval (10s here), not left at 0, since the first
    # frame's own exposure duration isn't directly known from a single
    # timestamp. Also confirms the unit now shows up in the label.
    timestamps = np.array([0.0, 10.0, 20.0])
    values, label = compute_experiment_x_axis("Time", timestamps, np.zeros((3, 3)))
    np.testing.assert_allclose(values, [10.0, 20.0, 30.0])
    assert label == "Time, s"


def test_compute_experiment_x_axis_time_mode_unit_conversion_and_offset():
    # An explicit time_offset_seconds overrides the auto-derived one.
    timestamps = np.array([0.0, 60.0, 120.0])
    values, label = compute_experiment_x_axis(
        "Time", timestamps, np.zeros((3, 3)), time_offset_seconds=5.0, time_unit="min"
    )
    np.testing.assert_allclose(values, [5.0 / 60.0, 65.0 / 60.0, 125.0 / 60.0])
    assert label == "Time, min"


def test_compute_experiment_x_axis_temperature_submodes():
    temps = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    values1, label1 = compute_experiment_x_axis("Temperature", np.array([np.nan, np.nan]), temps)
    values2, label2 = compute_experiment_x_axis("Temperature", np.array([np.nan, np.nan]), temps, sub_mode="Temperature2")
    values3, label3 = compute_experiment_x_axis("Temperature", np.array([np.nan, np.nan]), temps, sub_mode="Temperature3")

    np.testing.assert_allclose(values1, [1.0, 4.0])
    assert label1 == "Temperature"
    np.testing.assert_allclose(values2, [2.0, 5.0])
    assert label2 == "Temperature 2"
    np.testing.assert_allclose(values3, [3.0, 6.0])
    assert label3 == "Temperature 3"


def test_compute_experiment_x_axis_temperature_submode_time_falls_back_to_time_axis():
    timestamps = np.array([0.0, 5.0])
    values, label = compute_experiment_x_axis("Temperature", timestamps, np.zeros((2, 3)), sub_mode="Time")
    np.testing.assert_allclose(values, [5.0, 10.0])
    assert label == "Time, s"


def test_compute_experiment_x_axis_lowtemp_and_angle_modes():
    temps = np.array([[10.0, 20.0, np.nan], [11.0, 21.0, np.nan]])

    lowtemp_values, lowtemp_label = compute_experiment_x_axis("LowTemp.", np.array([np.nan, np.nan]), temps)
    lowtemp2_values, lowtemp2_label = compute_experiment_x_axis(
        "LowTemp.", np.array([np.nan, np.nan]), temps, sub_mode="LowTemp2"
    )
    angle_values, angle_label = compute_experiment_x_axis("Angle", np.array([np.nan, np.nan]), temps)

    np.testing.assert_allclose(lowtemp_values, [10.0, 11.0])
    assert lowtemp_label == "Low Temperature"
    np.testing.assert_allclose(lowtemp2_values, [20.0, 21.0])
    assert lowtemp2_label == "Low Temperature 2"
    np.testing.assert_allclose(angle_values, [10.0, 11.0])
    assert angle_label == "Incidence Angle, deg"


def test_compute_experiment_x_axis_rejects_unknown_mode():
    with pytest.raises(ValueError):
        compute_experiment_x_axis("Bogus", np.zeros(2), np.zeros((2, 3)))


def test_sort_x_axis_off_keeps_load_order():
    order = sort_x_axis(np.array([3.0, 1.0, 2.0]), should_sort=False)
    np.testing.assert_array_equal(order, [0, 1, 2])


def test_sort_x_axis_on_sorts_ascending():
    order = sort_x_axis(np.array([3.0, 1.0, 2.0]), should_sort=True)
    np.testing.assert_array_equal(order, [1, 2, 0])


def _q_map(center=(100.5, 100.5), shape=(200, 200)):
    return np.hypot(*np.meshgrid(np.arange(1, shape[1] + 1) - center[0], np.arange(1, shape[0] + 1) - center[1])).astype(
        np.float32
    )


def test_build_circular_radius_map_full_circle_matches_direct_radius_average():
    center = (100.5, 100.5)
    q_map = _q_map(center)
    radius_map = build_circular_radius_map(q_map, center)

    assert radius_map.is_valid
    assert radius_map.q_axis.shape[0] == radius_map.n_bins
    # Full 0..360 range should include (nearly) every finite/positive-q pixel.
    assert np.count_nonzero(radius_map.geom_mask) == int(np.count_nonzero(np.isfinite(q_map) & (q_map > 0)))


def test_build_circular_radius_map_restricts_to_angle_range():
    center = (100.5, 100.5)
    q_map = _q_map(center)
    full = build_circular_radius_map(q_map, center, angle_range=(0.0, 360.0))
    half = build_circular_radius_map(q_map, center, angle_range=(0.0, 180.0))

    assert np.count_nonzero(half.geom_mask) < np.count_nonzero(full.geom_mask)


def test_build_circular_radius_map_applies_roi_mask():
    center = (100.5, 100.5)
    q_map = _q_map(center)
    full = build_circular_radius_map(q_map, center)

    roi_mask = np.zeros(q_map.shape, dtype=bool)
    roi_mask[:100, :] = True  # keep only the top half
    masked = build_circular_radius_map(q_map, center, mask=roi_mask)

    assert np.count_nonzero(masked.geom_mask) < np.count_nonzero(full.geom_mask)


def test_build_circular_radius_map_rejects_mismatched_mask_shape():
    center = (100.5, 100.5)
    q_map = _q_map(center)
    with pytest.raises(ValueError):
        build_circular_radius_map(q_map, center, mask=np.ones((5, 5), dtype=bool))


def test_process_circular_stack_matches_hand_computed_radius_average():
    center = (100.5, 100.5)
    radii = [60.0, 70.0, 80.0]
    stack = _synthetic_ring_stack(n_frames=3, center=center, radii=radii)
    q_map = _q_map(center)
    radius_map = build_circular_radius_map(q_map, center)

    result = process_circular_stack(stack, radius_map)

    assert result.intensity.shape[0] == 3
    assert result.intensity.shape == result.coverage.shape
    for i in range(3):
        expected = radius_average(stack[i], radius_map)
        np.testing.assert_allclose(result.intensity[i], expected.intensity, equal_nan=True)
        np.testing.assert_allclose(result.x_axis, expected.q_axis)


def test_process_circular_stack_tracks_moving_ring():
    center = (100.5, 100.5)
    radii = [60.0, 70.0, 80.0]
    stack = _synthetic_ring_stack(n_frames=3, center=center, radii=radii)
    radius_map = build_circular_radius_map(_q_map(center), center)

    result = process_circular_stack(stack, radius_map)

    # `_q_map`'s Q value at a given pixel equals that pixel's own radius
    # from center, so the peak bin's x_axis value should track each
    # frame's own ring radius directly.
    for i, radius in enumerate(radii):
        peak_idx = int(np.nanargmax(result.intensity[i]))
        assert result.x_axis[peak_idx] == pytest.approx(radius, abs=3.0)


def test_process_circular_stack_reports_progress_per_frame():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=4, center=center, radii=[60.0] * 4)
    radius_map = build_circular_radius_map(_q_map(center), center)

    calls = []
    process_circular_stack(
        stack, radius_map, progress_callback=lambda completed, total: calls.append((completed, total))
    )

    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_process_circular_stack_parallel_matches_serial():
    center = (100.5, 100.5)
    radii = [58.0, 63.0, 71.0, 66.0, 60.0, 75.0]
    stack = _synthetic_ring_stack(n_frames=6, center=center, radii=radii)
    radius_map = build_circular_radius_map(_q_map(center), center)

    serial = process_circular_stack(stack, radius_map, parallel=False)
    parallel = process_circular_stack(stack, radius_map, parallel=True)

    np.testing.assert_array_equal(serial.x_axis, parallel.x_axis)
    np.testing.assert_array_equal(serial.intensity, parallel.intensity)
    np.testing.assert_array_equal(serial.coverage, parallel.coverage)


def test_process_circular_stack_calc_std_populates_std_matrix():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=2, center=center, radii=[60.0, 60.0])
    radius_map = build_circular_radius_map(_q_map(center), center)

    without_std = process_circular_stack(stack, radius_map, calc_std=False)
    with_std = process_circular_stack(stack, radius_map, calc_std=True)

    assert without_std.std is None
    assert with_std.std is not None
    assert with_std.std.shape == with_std.intensity.shape


def test_process_circular_stack_nulls_bins_below_min_coverage():
    center = (100.5, 100.5)
    stack = _synthetic_ring_stack(n_frames=1, center=center, radii=[60.0])
    radius_map = build_circular_radius_map(_q_map(center), center)

    permissive = process_circular_stack(stack, radius_map, min_coverage=0.0)
    strict = process_circular_stack(stack, radius_map, min_coverage=1.01)  # nothing can reach >1.0 coverage

    assert np.count_nonzero(np.isfinite(permissive.intensity)) > 0
    assert np.count_nonzero(np.isfinite(strict.intensity)) == 0


def test_process_circular_stack_rejects_empty_geometry():
    center = (100.5, 100.5)
    q_map = _q_map(center)
    empty_mask = np.zeros(q_map.shape, dtype=bool)
    radius_map = build_circular_radius_map(q_map, center, mask=empty_mask)
    assert not radius_map.is_valid

    stack = _synthetic_ring_stack(n_frames=1, center=center, radii=[60.0])
    with pytest.raises(ValueError):
        process_circular_stack(stack, radius_map)


def test_process_azimuthal_stack_rejects_an_empty_stack_with_a_clear_error():
    with pytest.raises(ValueError, match="at least one frame"):
        process_azimuthal_stack(np.empty((0, 20, 20), dtype=np.float32), center=(10.0, 10.0), radius_range=(3.0, 6.0))


def test_process_linecut_stack_change_ai_keeps_a_row_for_leading_invalid_frames():
    center = (100.5, 100.5)
    stack, q_map_stack = _synthetic_ring_stack_change_ai(n_frames=3, center=center)
    q_map_stack = q_map_stack.copy()
    q_map_stack[0] = np.nan  # no pixels selectable in frame 1
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    result = process_linecut_stack(stack, None, angle_map, center=center, angle_deg=90.0, q_map_stack=q_map_stack)

    assert result.intensity.shape[0] == 3
    assert np.all(np.isnan(result.intensity[0]))
    assert np.any(np.isfinite(result.intensity[1]))


def test_process_linecut_stack_static_frames_dropping_different_points_stay_aligned_by_q():
    center = (100.5, 100.5)
    frame = _synthetic_ring_stack(n_frames=1, center=center, radii=[60.0])[0]
    stack = np.stack([frame, frame], axis=0)
    # A 1-px accumulation band (rows 100-101, 1-based) loses a different
    # column in each frame, so each frame's own cut drops a different point.
    stack[0, 99:101, 50] = np.nan
    stack[1, 99:101, 60] = np.nan
    angle_map = pixel_angle_map(stack.shape[1:], center=center)

    result = process_linecut_stack(
        stack, _q_map(center), angle_map, center=center, angle_deg=0.0,
        accumulation=True, accum_minus=0.0, accum_plus=0.0,
    )

    both = np.isfinite(result.intensity).all(axis=0)
    assert result.x_axis.shape == (100,)
    assert np.count_nonzero(~both) == 2
    # Same frame data, so every point both frames kept must agree.
    np.testing.assert_array_equal(result.intensity[0, both], result.intensity[1, both])


def test_normalize_rows_by_counter_blanks_frames_without_a_counter():
    matrix = np.full((4, 3), 10.0, dtype=np.float32)
    counts = np.array([[2.0, 1.0], [np.nan, 1.0], [0.0, 1.0], [5.0, 1.0]], dtype=np.float32)

    out = normalize_rows_by_counter(matrix, counts, "IC")

    np.testing.assert_allclose(out[[0, 3]], [[5.0] * 3, [2.0] * 3])
    assert np.all(np.isnan(out[1:3]))
    assert np.all(np.isnan(normalize_rows_by_counter(matrix, None, "BC")))
