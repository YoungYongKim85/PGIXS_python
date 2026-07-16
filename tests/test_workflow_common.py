import argparse
from pathlib import Path

import numpy as np

from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json
from pgixs_python.io.h5_loader import H5RetryReport
from pgixs_python.io.h5_series import H5FrameRecord, H5SeriesLoadResult
from pgixs_python.workflows.common import (
    build_two_d_info,
    format_processing_summary,
    format_qmap_summary,
    load_workflow_qmap,
    process_workflow_stack,
)


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


def test_build_two_d_info_from_stack_and_qmap(tmp_path):
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
    qmap = load_workflow_qmap(args, image_shape=(5, 5), mode="trxs")
    stack = np.stack(
        [
            np.ones((5, 5), dtype=np.float32),
            np.ones((5, 5), dtype=np.float32) * 3,
        ],
        axis=0,
    )
    stack_result = H5SeriesLoadResult(
        stack=stack,
        records=[
            H5FrameRecord(Path("a.h5"), 1, True, H5RetryReport(True, 1)),
            H5FrameRecord(Path("b.h5"), 2, True, H5RetryReport(True, 1)),
        ],
        counts=np.ones((2, 2), dtype=np.float32),
        timestamps=np.array([0.0, 1.0], dtype=np.float64),
        temperatures=np.full((2, 3), np.nan, dtype=np.float32),
        retry_failed_frames=np.array([], dtype=np.int32),
        metadata_failed_frames=np.array([2], dtype=np.int32),
    )

    assert qmap is not None
    info = build_two_d_info("TrXS", stack_result, qmap)

    assert info.measurement_mode == "TrXS"
    assert info.detector == "Eiger4M"
    assert info.data2d_pix.shape == (5, 5)
    assert np.all(info.data2d_pix == 2)
    assert info.q_map.shape == (5, 5)
    assert info.source_files == ["a.h5", "b.h5"]
    np.testing.assert_array_equal(info.metadata_failed_frames, np.array([2], dtype=np.int32))


def test_process_workflow_stack_applies_normalization_and_diagnostics():
    stack = np.stack(
        [
            np.array([[10, 20]], dtype=np.float32),
            np.array([[30, 40]], dtype=np.float32),
        ],
        axis=0,
    )
    stack_result = H5SeriesLoadResult(
        stack=stack,
        records=[
            H5FrameRecord(Path("a.h5"), 1, True, H5RetryReport(True, 1)),
            H5FrameRecord(Path("b.h5"), 2, True, H5RetryReport(True, 1)),
        ],
        counts=np.array([[10, 1], [5, 1]], dtype=np.float32),
        timestamps=np.array([0.0, 1.0], dtype=np.float64),
        temperatures=np.full((2, 3), np.nan, dtype=np.float32),
        retry_failed_frames=np.array([], dtype=np.int32),
        metadata_failed_frames=np.array([], dtype=np.int32),
    )
    args = argparse.Namespace(
        normalization="IC",
        background=None,
        no_clamp_negative=False,
        grid_remove=False,
        detector_correction=False,
        detector="Eiger4M",
    )

    processed = process_workflow_stack(args, stack_result, qmap_result=None)

    np.testing.assert_allclose(processed.stack_result.stack[0], np.array([[1, 2]], dtype=np.float32))
    np.testing.assert_allclose(processed.stack_result.stack[1], np.array([[6, 8]], dtype=np.float32))
    assert "normalization: IC" in format_processing_summary(processed)
