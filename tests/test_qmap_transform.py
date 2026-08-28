import numpy as np
import pytest

from pgixs_python.qmap.transform import (
    image_transform_q_gi,
    image_transform_q_tr,
    pixel_point_from_q_space,
    polar_transform_q_grid,
    q_space_point,
)


def _identity_setup():
    """4x4 grid where dQ == 1.0 and the per-pixel Q maps exactly match the
    target grid axes, so each interior pixel forward-maps onto exactly one
    target cell with full weight (an easy-to-verify identity mapping)."""

    rows = cols = 4
    beam_pos = (2.0, 2.0)
    pixel_mm = 1.0
    wavelength_a = 1.0
    sdd_mm = 2.0 * np.pi  # dQ = 2*pi*pixel/(wavelength*sdd) = 1.0

    cols_idx = np.arange(1, cols + 1, dtype=np.float32)
    rows_idx = np.arange(1, rows + 1, dtype=np.float32)
    horizontal_axis = 1.0 * (cols_idx - beam_pos[0])
    vertical_axis = 1.0 * (beam_pos[1] - rows_idx)

    horizontal_q_map = np.broadcast_to(horizontal_axis, (rows, cols)).astype(np.float32)
    vertical_q_map = np.broadcast_to(vertical_axis[:, None], (rows, cols)).astype(np.float32)

    data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    return data, horizontal_q_map, vertical_q_map, beam_pos, pixel_mm, wavelength_a, sdd_mm


def test_image_transform_q_gi_identity_mapping_reproduces_source_pixels():
    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()

    result = image_transform_q_gi(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm, mask_beamstop=False)

    # Each pixel's Q value sits exactly on its own target grid node, so the
    # bilinear splat degenerates to full weight on a single cell -- the
    # transformed image exactly reproduces the source pixel-space image.
    np.testing.assert_allclose(result.image, data)


def test_q_space_point_matches_image_transform_q_gi_own_axes():
    """q_space_point's (qx, qy) at each grid-aligned (col, row) must equal
    exactly what image_transform_q_gi's own horizontal_axis/vertical_axis
    report for that same index -- it's meant to be the same uniform-grid
    coordinate system, computed independently of actually regridding an
    image (used to draw a cut overlay directly in Q-space)."""

    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()
    result = image_transform_q_gi(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm, mask_beamstop=False)

    for col in range(1, 5):
        for row in range(1, 5):
            qx, qy = q_space_point(col, row, beam_pos, pixel_mm, wavelength_a, sdd_mm)
            assert qx == pytest.approx(result.horizontal_axis[col - 1])
            assert qy == pytest.approx(result.vertical_axis[row - 1])


def test_pixel_point_from_q_space_is_the_exact_inverse_of_q_space_point():
    """Regression: "Qxy/Qz 맵에서 90도 line cut을... 해보면 일직선
    라인이 아니라 휘어져 나와" -- console_widget.py draws a cut overlay
    as a straight line in Q-space via q_space_point, and inverts a
    Q-space click back to a pixel point via pixel_point_from_q_space;
    those two must be exact inverses of each other (this uniform grid's
    axes are each linear in pixel position) for the overlay and the
    click that sets it up to stay consistent."""

    beam_pos = (12.3, 45.6)
    pixel_mm = 0.075
    wavelength_a = 1.3
    sdd_mm = 300.0

    for col, row in [(1.0, 1.0), (50.0, 30.0), (12.3, 45.6), (-10.0, 200.0)]:
        qx, qy = q_space_point(col, row, beam_pos, pixel_mm, wavelength_a, sdd_mm)
        back_col, back_row = pixel_point_from_q_space(qx, qy, beam_pos, pixel_mm, wavelength_a, sdd_mm)
        assert back_col == pytest.approx(col)
        assert back_row == pytest.approx(row)


def test_image_transform_q_gi_masks_beamstop_column():
    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()

    result = image_transform_q_gi(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm, mask_beamstop=True)

    beam_col = int(round(beam_pos[0])) - 1
    assert np.all(np.isnan(result.image[:, beam_col]))
    # A non-beamstop interior column is untouched by the masking.
    assert not np.isnan(result.image[0, 0])


def test_image_transform_q_tr_does_not_mask_beamstop_column():
    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()

    result = image_transform_q_tr(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm)

    beam_col = int(round(beam_pos[0])) - 1
    assert not np.isnan(result.image[0, beam_col])
    np.testing.assert_allclose(result.image[:3, :3], data[:3, :3])


def test_image_transform_q_gi_axes_match_expected_spacing():
    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()

    result = image_transform_q_gi(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm, mask_beamstop=False)

    np.testing.assert_allclose(result.horizontal_axis, [-1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.vertical_axis, [1.0, 0.0, -1.0, -2.0])


def test_image_transform_q_gi_out_of_range_pixels_excluded():
    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()
    # Push one pixel's Q value far outside the target grid range.
    hmap = hmap.copy()
    hmap[1, 1] = 999.0

    result = image_transform_q_gi(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm, mask_beamstop=False)

    assert np.isnan(result.image[1, 1])


def test_image_transform_q_gi_handles_no_valid_pixels():
    data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm = _identity_setup()
    hmap = np.full_like(hmap, 999.0)

    result = image_transform_q_gi(data, hmap, vmap, beam_pos, pixel_mm, wavelength_a, sdd_mm, mask_beamstop=False)

    assert np.all(np.isnan(result.image))
    assert result.image.shape == data.shape


def test_polar_transform_q_grid_puts_constant_ring_in_one_row():
    """A uniform value at a fixed radius (regardless of angle) must land in
    a single r-bin (one row) no matter which angle-bin (column) it lands
    in -- the whole point of the polar remap is separating (angle, |Q|)
    onto independent axes."""

    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    h = 5.0 * np.cos(theta)
    v = 5.0 * np.sin(theta)
    data = np.full(theta.shape, 7.0, dtype=np.float32)

    result = polar_transform_q_grid(data, h, v, n_angle_bins=36, n_r_bins=10)

    finite_rows = np.any(np.isfinite(result.image), axis=1)
    assert np.count_nonzero(finite_rows) == 1
    row = result.image[finite_rows][0]
    np.testing.assert_allclose(row[np.isfinite(row)], 7.0)


def test_polar_transform_q_grid_matches_the_app_wide_clock_convention():
    """Regression: "angle 방향이 바뀐거 같아... 9시방향이 0도, 12시방향이 90도 3시방향이
    180도" -- polar must use the same clockwise, 9/12/3-o'clock angle
    convention as everything else in this codebase (pixel_angle_map, the
    click-to-angle formula, Cakecut, Azimuthal Process), not the plain-math
    atan2(v, h) convention (which puts 0deg at 3 o'clock, counterclockwise)."""

    n_angle_bins = 360

    def _angle_of(h: float, v: float) -> float:
        data = np.array([1.0], dtype=np.float32)
        result = polar_transform_q_grid(
            data, np.array([h]), np.array([v]), n_angle_bins=n_angle_bins, n_r_bins=1
        )
        (col,) = np.flatnonzero(np.isfinite(result.image[0]))
        return float(result.horizontal_axis[col])

    assert _angle_of(-1.0, 0.0) == pytest.approx(0.0, abs=1.0)  # 9 o'clock (left) -> 0deg
    assert _angle_of(0.0, 1.0) == pytest.approx(90.0, abs=1.0)  # 12 o'clock (up) -> 90deg
    assert _angle_of(1.0, 0.0) == pytest.approx(180.0, abs=1.0)  # 3 o'clock (right) -> 180deg
    assert _angle_of(0.0, -1.0) == pytest.approx(270.0, abs=1.0)  # 6 o'clock (down) -> 270deg


def test_polar_transform_q_grid_separates_two_angles_at_same_radius():
    """Two different values at the same radius but opposite angles (0deg
    and 180deg) must land in different columns (angle bins), not get
    averaged together into one cell."""

    h = np.array([10.0, -10.0], dtype=np.float64)
    v = np.array([0.0, 0.0], dtype=np.float64)
    data = np.array([1.0, 9.0], dtype=np.float32)

    result = polar_transform_q_grid(data, h, v, n_angle_bins=4, n_r_bins=4)

    values = result.image[np.isfinite(result.image)]
    assert sorted(values.tolist()) == [1.0, 9.0]


def test_polar_transform_q_grid_axes_span_the_data():
    h = np.array([3.0, 4.0], dtype=np.float64)  # r = 5 for both (3-4-5 triangle)
    v = np.array([4.0, 3.0], dtype=np.float64)
    data = np.array([1.0, 2.0], dtype=np.float32)

    result = polar_transform_q_grid(data, h, v, n_angle_bins=36, n_r_bins=5)

    assert result.horizontal_axis.shape == (36,)
    assert result.vertical_axis.shape == (5,)
    assert result.horizontal_axis.min() >= 0.0
    assert result.horizontal_axis.max() < 360.0
    assert result.vertical_axis.max() <= 5.0 + 1e-6


def test_polar_transform_q_grid_handles_no_valid_pixels():
    h = np.full((3, 3), np.nan)
    v = np.full((3, 3), np.nan)
    data = np.zeros((3, 3), dtype=np.float32)

    result = polar_transform_q_grid(data, h, v, n_angle_bins=8, n_r_bins=4)

    assert result.image.shape == (4, 8)
    assert np.all(np.isnan(result.image))


def test_polar_transform_q_grid_angle_range_restricts_sweep_to_half_circle():
    """GIXS callers pass angle_range=(0, 180) (grazing incidence only ever
    scatters above the horizon, v >= 0) -- the returned angle axis must
    cover just that half, and points below the horizon (v < 0) must be
    excluded entirely rather than wrapped/clipped into an edge bin."""

    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    h = 5.0 * np.cos(theta)
    v = 5.0 * np.sin(theta)
    data = np.where(v >= 0, 3.0, 1000.0).astype(np.float32)  # below-horizon points would blow up the mean if leaked in

    result = polar_transform_q_grid(data, h, v, n_angle_bins=18, n_r_bins=4, angle_range=(0.0, 180.0))

    assert result.horizontal_axis.shape == (18,)
    assert result.horizontal_axis.min() >= 0.0
    assert result.horizontal_axis.max() <= 180.0
    finite = np.isfinite(result.image)
    assert np.any(finite)
    assert np.allclose(result.image[finite], 3.0)
