import argparse

import numpy as np

from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json
from pgixs_python.workflows.common import format_qmap_summary, load_workflow_qmap


def test_load_workflow_qmap_trxs_from_json_standard(tmp_path):
    standard_file = tmp_path / "standard.json"
    save_standard_calibration_json(
        standard_file,
        StandardCalibration(
            beam_center=np.array([3.0, 3.0]),
            wavelength=1.3051,
            pixel_size=0.075,
            sdd=300.0,
            detector="Eiger4M",
        ),
    )
    args = argparse.Namespace(standard=str(standard_file), detector="Eiger4M")

    result = load_workflow_qmap(args, image_shape=(5, 5), mode="trxs")

    assert result is not None
    assert result.q_map.shape == (5, 5)
    assert result.qx_map is not None
    assert result.qy_map is not None
    assert result.q_map[2, 2] == 0
    assert "qmap: built" in format_qmap_summary(result)


def test_load_workflow_qmap_gixs_from_json_standard(tmp_path):
    standard_file = tmp_path / "standard.json"
    save_standard_calibration_json(
        standard_file,
        StandardCalibration(
            beam_center=np.array([3.0, 3.0]),
            wavelength=1.3051,
            pixel_size=0.075,
            sdd=300.0,
            detector="Eiger4M",
        ),
    )
    args = argparse.Namespace(standard=str(standard_file), detector="Eiger4M", incident_angle=0.2)

    result = load_workflow_qmap(args, image_shape=(5, 5), mode="gixs")

    assert result is not None
    assert result.q_map.shape == (5, 5)
    assert result.qr_map is not None
    assert result.tth2d is not None
    assert result.af2d is not None
