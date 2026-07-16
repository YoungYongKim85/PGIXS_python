import importlib.util

import numpy as np

from pgixs_python.calibration.calibration_io import (
    StandardCalibration,
    load_standard_calibration,
    save_standard_calibration_json,
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
