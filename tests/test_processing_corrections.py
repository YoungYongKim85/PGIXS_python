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
