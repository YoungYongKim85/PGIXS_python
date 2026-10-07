import numpy as np
import pytest

from pgixs_python.processing import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    apply_detector_mask,
    detector_correction_factor,
    detector_mask_for_type,
    grid_remove,
    kapton_efficiency,
    load_attenuation_table,
    mask_zero_to_nan,
    polarization,
    silicon_efficiency,
    solid_angle,
)


def test_load_attenuation_table_kapton_matches_matlab_source():
    table = load_attenuation_table("kapton")
    assert table.shape == (221, 2)
    np.testing.assert_allclose(table[0], [8.0, 0.12046956], rtol=1e-6)
    np.testing.assert_allclose(table[-1], [30.0, 9.22952295], rtol=1e-6)


def test_load_attenuation_table_silicon_matches_matlab_source():
    table = load_attenuation_table("silicon")
    assert table.shape == (221, 2)
    np.testing.assert_allclose(table[0], [8.0, 6.97258535e-03], rtol=1e-6)
    np.testing.assert_allclose(table[-1], [30.0, 0.37614752], rtol=1e-6)


def test_load_attenuation_table_rejects_unknown_name():
    with pytest.raises(ValueError):
        load_attenuation_table("bogus")


def test_kapton_and_silicon_efficiency_run_with_real_tables():
    shape = (20, 20)
    center = (10.5, 10.5)
    k0 = 2.0 * np.pi / 1.0
    y, z = np.meshgrid(np.arange(1, shape[1] + 1, dtype=np.float64), np.arange(1, shape[0] + 1, dtype=np.float64))
    q_map = np.hypot(y - center[0], z - center[1]).astype(np.float32) * 0.01

    kapton_table = load_attenuation_table("kapton")
    silicon_table = load_attenuation_table("silicon")

    kapton_eff = kapton_efficiency(q_map, kapton_table, k0, energy_kev=9.5, data_size=shape, beam_position=center, pixel_size_mm=0.075)
    silicon_eff = silicon_efficiency(q_map, k0, silicon_table, energy_kev=9.5)

    assert kapton_eff.shape == shape
    assert silicon_eff.shape == shape
    assert np.all(np.isfinite(kapton_eff))
    assert np.all(np.isfinite(silicon_eff))
    assert np.nanmax(kapton_eff) == pytest.approx(1.0, abs=1e-5)
    assert np.nanmin(silicon_eff) == pytest.approx(1.0, abs=1e-5)


def test_apply_detector_mask_preserves_gap_as_nan():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mask = np.array([[1, 0], [np.nan, 1]], dtype=np.float32)

    out = apply_detector_mask(data, mask)

    assert out[0, 0] == 1
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 0])
    assert out[1, 1] == 4


def test_mask_zero_to_nan():
    out = mask_zero_to_nan(np.array([[1, 0, -1]], dtype=np.float32))

    assert out[0, 0] == 1
    assert np.isnan(out[0, 1])
    assert np.isnan(out[0, 2])


def test_detector_mask_unknown_reports_fallback_without_mask():
    result = detector_mask_for_type("unknown")

    assert result.mask is None
    assert result.pixel_size_mm == 0.075
    assert result.report.fallback_used


def test_solid_angle_and_polarization_shapes_and_normalization():
    shape = (3, 4)

    sol = solid_angle(shape, beam_position=(2, 2), pixel_size_mm=0.075, sdd_mm=200.0)
    pol = polarization(shape, beam_position=(2, 2), pixel_size_mm=0.075, sdd_mm=200.0)

    assert sol.shape == shape
    assert pol.shape == shape
    assert np.isclose(np.nanmax(sol), 1.0)
    assert np.all(np.isfinite(pol))


def test_detector_correction_factor_and_application_preserve_nan():
    data = np.array([[10.0, np.nan], [30.0, 40.0]], dtype=np.float32)
    q = np.full((2, 2), 0.1, dtype=np.float32)
    inputs = DetectorCorrectionInputs(
        data_size=(2, 2),
        beam_position=(1.0, 1.0),
        pixel_size_mm=0.075,
        sdd_mm=200.0,
        q_map=q,
        energy_kev=10.0,
        detector_type="Eiger4M",
    )

    factor = detector_correction_factor(inputs, k0=5.0)
    out = apply_detector_correction(data, factor)

    assert factor.shape == data.shape
    assert np.all(np.isfinite(factor))
    assert np.isnan(out[0, 1])
    assert np.isfinite(out[0, 0])


def test_grid_remove_only_applies_to_eiger4m():
    data = np.ones((2167, 2070), dtype=np.float32)
    data[:, [255, 258, 513, 516, 771, 774, 1295, 1298, 1553, 1556, 1811, 1814]] = 2
    data[[255, 258, 806, 809, 1357, 1360, 1908, 1911], :] = 2

    skipped, skipped_report = grid_remove(data, detector_type="Other")
    corrected, report = grid_remove(data, detector_type="Eiger4M")

    assert not skipped_report.applied
    np.testing.assert_allclose(skipped, data)
    assert report.applied
    assert np.isfinite(report.correction_factor)
    assert corrected.shape == data.shape


def test_grid_remove_restores_grid_lines_in_loader_orientation():
    # Build a MATLAB-orientation (2167, 2070) frame whose chip-boundary
    # lines read half, then hand it over the way callers do: transposed to
    # `load_h5_frame`'s (2070, 2167).
    matlab = np.ones((2167, 2070), dtype=np.float32)
    for start in (256, 807, 1358, 1909):
        matlab[start - 1 : start + 3, :] = 0.5
    for start in (256, 514, 772, 1296, 1554, 1812):
        matlab[:, start - 1 : start + 3] = 0.5

    corrected, report = grid_remove(matlab.T, detector_type="Eiger4M")

    assert report.applied
    assert report.correction_factor == pytest.approx(2.0)
    assert corrected.shape == (2070, 2167)
    np.testing.assert_allclose(corrected, 1.0)


def _correction_inputs(detector_type):
    shape = (12, 12)
    center = (6.5, 6.5)
    y, z = np.meshgrid(
        np.arange(1, shape[1] + 1, dtype=np.float64),
        np.arange(1, shape[0] + 1, dtype=np.float64),
    )
    q_map = (np.hypot(y - center[0], z - center[1]) * 0.02).astype(np.float32)
    return DetectorCorrectionInputs(
        data_size=shape,
        beam_position=center,
        pixel_size_mm=0.075,
        sdd_mm=250.0,
        q_map=q_map,
        energy_kev=9.5,
        detector_type=detector_type,
    ), shape, center, q_map


def test_detector_correction_factor_always_includes_kapton():
    """Regression: MATLAB's `computePGIXSDetectorCorrectionFactor.m` loads the
    Kapton table itself and multiplies it in unconditionally, so every q-Cal
    path gets it. This port used to apply it only when a caller passed the
    table explicitly -- which only the GIXS-series "Change ai" path ever did,
    silently dropping the correction from the single-frame consoles, the plain
    series q-Cal, Auto Refresh and the CLI workflow."""

    inputs, shape, center, q_map = _correction_inputs("MarCCD")
    k0 = 2.0 * np.pi / 1.3

    factor = detector_correction_factor(inputs, k0=k0)

    base = solid_angle(shape, center, 0.075, 250.0) * polarization(shape, center, 0.075, 250.0)
    expected = base * kapton_efficiency(
        q_map, load_attenuation_table("kapton"), k0, 9.5, shape, center, 0.075
    )
    np.testing.assert_allclose(factor, expected, rtol=1e-6)
    assert not np.allclose(factor, base), "Kapton correction was not applied"


def test_detector_correction_factor_adds_silicon_only_for_eiger4m():
    """`Eff_correc2` is Eiger4M-only in MATLAB (its silicon sensor chip)."""

    k0 = 2.0 * np.pi / 1.3
    eiger_inputs, _, _, q_map = _correction_inputs("Eiger4M")
    other_inputs, _, _, _ = _correction_inputs("MarCCD")

    eiger = detector_correction_factor(eiger_inputs, k0=k0)
    other = detector_correction_factor(other_inputs, k0=k0)

    expected_ratio = silicon_efficiency(q_map, k0, load_attenuation_table("silicon"), 9.5)
    np.testing.assert_allclose(eiger / other, expected_ratio, rtol=1e-5)


def test_detector_masks_load_from_the_packaged_files_not_the_fallback():
    """Regression: `mask_dir=None` used to resolve a bare relative filename
    against the process CWD, so the mask file was never found and every
    detector silently ran on the all-ones fallback -- MATLAB applies the real
    mask (`applyPGIXSMask`) on every load. The files are now shipped in
    `pgixs_python/processing/data`, which packaging/pgixs.spec already bundles
    wholesale."""

    eiger = detector_mask_for_type("Eiger4M")
    assert not eiger.report.fallback_used, eiger.report.message
    # Stored in MATLAB's orientation (2167, 2070); the loaders hand us frames
    # the other way round, so the mask has to come back transposed.
    assert eiger.mask.shape == (2070, 2167)
    assert 0.05 < np.mean(~np.isfinite(eiger.mask)) < 0.08

    marccd = detector_mask_for_type("MarCCD")
    assert not marccd.report.fallback_used, marccd.report.message
    assert marccd.mask.shape == (2048, 2048)
    assert 0.20 < np.mean(~np.isfinite(marccd.mask)) < 0.23


def test_dilate_gap_mask_grows_the_exclusion_by_n_pad():
    """Port of MATLAB `dilatePGIXSGapMask.m`: 4-neighbour dilation, seeded
    from NaN/non-positive mask entries."""

    from pgixs_python.processing.masking import dilate_gap_mask

    mask = np.ones((7, 7), dtype=np.float32)
    mask[3, :] = np.nan

    assert sorted(set(np.nonzero(dilate_gap_mask(mask, 0))[0].tolist())) == [3]
    assert sorted(set(np.nonzero(dilate_gap_mask(mask, 1))[0].tolist())) == [2, 3, 4]
    assert sorted(set(np.nonzero(dilate_gap_mask(mask, 2))[0].tolist())) == [1, 2, 3, 4, 5]

    # Zero and negative entries seed the exclusion too, same as NaN.
    zero_seeded = np.ones((5, 5), dtype=np.float32)
    zero_seeded[2, 2] = 0.0
    assert dilate_gap_mask(zero_seeded, 1)[2, 2]
    assert dilate_gap_mask(zero_seeded, 1)[1, 2]
    assert not dilate_gap_mask(zero_seeded, 1)[0, 0]
