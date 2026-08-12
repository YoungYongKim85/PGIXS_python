import numpy as np

from pgixs_python.qmap.transform import image_transform_q_gi, image_transform_q_tr


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
