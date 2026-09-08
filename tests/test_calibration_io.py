import importlib.util

import numpy as np
import pytest

from pgixs_python.calibration.calibration_io import (
    StandardCalibration,
    load_standard_calibration,
    save_standard_calibration_json,
    save_standard_calibration_mat,
    save_standard_calibration_std,
)


def test_load_standard_calibration_json_roundtrip(tmp_path):
    cal = StandardCalibration(
        beam_center=np.array([100.0, 200.0]),
        wavelength=1.3051,
        pixel_size=0.075,
        sdd=300.0,
        detector="Eiger4M",
        standard_peak=np.array([1.0, 2.0]),
        standard_q_value=0.1052,
        detector_index="eiger4m",
        image_shape=np.array([2167, 2070]),
        diagnostics={"AcceptedCount": 2},
    )
    json_file = tmp_path / "standard.json"

    save_standard_calibration_json(json_file, cal)
    loaded, report = load_standard_calibration(json_file, expected_detector="Eiger4M")

    assert report.ok
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [100.0, 200.0])
    assert loaded.detector == "Eiger4M"
    assert loaded.detector_index == "eiger4m"
    assert loaded.diagnostics["AcceptedCount"] == 2


def test_load_standard_calibration_reports_detector_mismatch(tmp_path):
    cal = StandardCalibration(beam_center=np.array([1.0, 2.0]), wavelength=1.0, pixel_size=0.1, sdd=100.0, detector="SX165")
    json_file = tmp_path / "standard.json"
    save_standard_calibration_json(json_file, cal)

    loaded, report = load_standard_calibration(json_file, expected_detector="Eiger4M")

    assert loaded is None
    assert not report.ok
    assert report.detector_mismatch


def test_load_standard_calibration_mat(tmp_path):
    if importlib.util.find_spec("scipy") is None:
        import pytest

        pytest.skip("scipy is not installed")

    import scipy.io

    mat_file = tmp_path / "standard.mat"
    scipy.io.savemat(
        mat_file,
        {
            "pc": np.array([10.0, 20.0]),
            "wavelength": 1.3051,
            "pixelsize": 0.075,
            "sdd": 300.0,
            "dttypename": "Eiger4M",
            "STqvalue": 0.1052,
        },
    )

    loaded, report = load_standard_calibration(mat_file, expected_detector="Eiger4M")

    assert report.ok
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [10.0, 20.0])
    assert loaded.standard_q_value == 0.1052


def test_save_standard_calibration_mat_roundtrip(tmp_path):
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("scipy is not installed")

    cal = StandardCalibration(
        beam_center=np.array([398.0, 1846.0]),
        wavelength=1.3051,
        pixel_size=0.075,
        sdd=214.84,
        detector="Eiger4M",
        standard_peak=np.array([187.84]),
        standard_q_value=0.1052,
        detector_index="eiger4m",
        image_shape=np.array([2167, 2070]),
    )
    mat_file = tmp_path / "standard.mat"

    save_standard_calibration_mat(mat_file, cal)
    loaded, report = load_standard_calibration(mat_file, expected_detector="Eiger4M")

    assert report.ok, report.message
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [398.0, 1846.0])
    assert loaded.wavelength == pytest.approx(1.3051)
    assert loaded.sdd == pytest.approx(214.84)
    assert loaded.detector == "Eiger4M"
    assert loaded.detector_index == "eiger4m"
    np.testing.assert_allclose(loaded.standard_peak, [187.84])
    assert loaded.standard_q_value == pytest.approx(0.1052)
    np.testing.assert_allclose(loaded.image_shape, [2167, 2070])


def test_load_standard_calibration_std_reads_a_real_matlab_file():
    """A real `.std` written by the calibration GUI, parsed end to end.

    Skips where the fixture is absent: `testdata/` is not committed (the real
    frames are far too large), so this only runs on a machine that has the
    beamline data.
    """

    std_file = "testdata/SAXSBattery/standard/20260416_sb_0001_standard.std"

    loaded, report = load_standard_calibration(std_file, expected_detector="Eiger4M")

    if not report.ok and "File not found" in report.message:
        pytest.skip("real test fixture not available")
    assert report.ok, report.message
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [724.51, 1393.59])
    assert loaded.wavelength == pytest.approx(1.385298)
    assert loaded.pixel_size == pytest.approx(0.075)
    assert loaded.sdd == pytest.approx(889.6241)
    assert loaded.detector == "Eiger4M"
    assert loaded.detector_index == "eiger4m"
    np.testing.assert_allclose(loaded.standard_peak, [275.177])
    assert loaded.standard_q_value == pytest.approx(0.1052)
    assert loaded.energy_kev == pytest.approx(8.95, abs=1.0e-4)


def test_load_standard_calibration_std_reads_the_original_two_value_layout(tmp_path):
    """`stdsave.m`'s own layout writes section 3 as (wavelength, pixel size)
    with no Energy field; this app's own writer adds Energy in the middle.

    This used to be covered only by a real beamline `.std` under `testdata/`,
    which was later re-saved in the extended layout -- silently taking the
    original-layout coverage with it. Inline here so it cannot drift again,
    and so it runs where `testdata/` is not checked out.
    """

    std_file = tmp_path / "legacy_standard.std"
    std_file.write_text(
        """%1 center coordinate(yc,zc)
  7.243465E+02 ,   1.392226E+03
%2. Distances from center to peak  and q value ( h1st, q1st)
  2.752347E+02 ,   1.052000E-01
%3. Wave length[A] and Pixel size[mm]
  1.385309E+00  ,   7.500000E-02
%4. Sample to detector distance[mm]
  8.898037E+02
%5. detector type and name
eiger4m,Eiger4M
""",
        encoding="utf-8",
    )

    loaded, report = load_standard_calibration(std_file, expected_detector="Eiger4M")

    assert report.ok, report.message
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [724.3465, 1392.226])
    assert loaded.wavelength == pytest.approx(1.385309)
    assert loaded.pixel_size == pytest.approx(0.075)
    assert loaded.sdd == pytest.approx(889.8037)
    # Energy is absent from this layout and has to come from the wavelength,
    # the same way every other format derives it.
    assert loaded.energy_kev == pytest.approx(8.94993, abs=1.0e-4)


def test_save_standard_calibration_std_roundtrip(tmp_path):
    cal = StandardCalibration(
        beam_center=np.array([724.3465, 1392.226]),
        wavelength=1.385309,
        pixel_size=0.075,
        sdd=889.8037,
        detector="Eiger4M",
        energy_kev=8.9499,
        standard_peak=np.array([275.2347]),
        standard_q_value=0.1052,
        detector_index="eiger4m",
    )
    std_file = tmp_path / "standard.std"

    save_standard_calibration_std(std_file, cal)
    loaded, report = load_standard_calibration(std_file, expected_detector="Eiger4M")

    assert report.ok, report.message
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [724.3465, 1392.226])
    assert loaded.wavelength == pytest.approx(1.385309)
    assert loaded.pixel_size == pytest.approx(0.075)
    assert loaded.sdd == pytest.approx(889.8037)
    assert loaded.detector == "Eiger4M"
    assert loaded.detector_index == "eiger4m"
    np.testing.assert_allclose(loaded.standard_peak, [275.2347])
    assert loaded.standard_q_value == pytest.approx(0.1052)
    # The extended field this app adds over the original stdsave.m format.
    assert loaded.raw["Energy"] == pytest.approx(8.9499)
