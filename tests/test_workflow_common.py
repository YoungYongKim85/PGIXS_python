import argparse
import json
from pathlib import Path

import numpy as np

from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json
from pgixs_python.io.h5_loader import H5RetryReport
from pgixs_python.io.h5_series import H5FrameRecord, H5SeriesLoadResult
from pgixs_python.workflows.common import (
    build_two_d_info,
    export_workflow_1d,
    format_1d_export_summary,
    format_processing_summary,
    format_qmap_summary,
    format_report_summary,
    format_manifest_summary,
    load_workflow_config,
    load_workflow_qmap,
    parser_config_keys,
    parse_workflow_args,
    process_workflow_stack,
    save_workflow_manifest,
    save_workflow_report,
    validate_workflow_config_values,
    WorkflowExportResult,
    WorkflowReportResult,
    _nanmean_no_warning,
)
from pgixs_python.workflows.trxs_process import build_parser as build_trxs_parser


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


def test_nanmean_no_warning_keeps_all_nan_pixels():
    stack = np.array(
        [
            [[1.0, np.nan], [np.nan, 4.0]],
            [[3.0, np.nan], [np.nan, 6.0]],
        ],
        dtype=np.float32,
    )

    out = _nanmean_no_warning(stack, axis=0)

    np.testing.assert_allclose(out[:, 0], np.array([2.0, np.nan], dtype=np.float32), equal_nan=True)
    np.testing.assert_allclose(out[:, 1], np.array([np.nan, 5.0], dtype=np.float32), equal_nan=True)


def test_export_workflow_1d_writes_csv(tmp_path):
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
    args_q = argparse.Namespace(standard=str(standard_file), detector="Eiger4M")
    qmap = load_workflow_qmap(args_q, image_shape=(5, 5), mode="trxs")
    stack_result = H5SeriesLoadResult(
        stack=np.ones((1, 5, 5), dtype=np.float32),
        records=[H5FrameRecord(Path("sample.h5"), 1, True, H5RetryReport(True, 1))],
        counts=np.ones((1, 2), dtype=np.float32),
        timestamps=np.array([0.0], dtype=np.float64),
        temperatures=np.full((1, 3), np.nan, dtype=np.float32),
        retry_failed_frames=np.array([], dtype=np.int32),
        metadata_failed_frames=np.array([], dtype=np.int32),
    )
    args = argparse.Namespace(
        export_1d=True,
        out=str(tmp_path),
        export_format="csv",
        q_bins=8,
        q_min=None,
        q_max=None,
        q_scale="linear",
        export_std=True,
        export_diagnostics=True,
        export_figure=False,
    )

    assert qmap is not None
    result = export_workflow_1d(args, "TrXS", stack_result, qmap)

    assert result.data_path is not None
    assert result.data_path.exists()
    assert result.diagnostics is not None
    assert result.diagnostics["one_d_bin_count"] == 8
    assert "one_d_coverage_min" in result.diagnostics
    assert "1D export: saved" in format_1d_export_summary(result)
    assert result.data_path.read_text(encoding="utf-8").splitlines()[0] == (
        "q,intensity,standard_deviation,valid_count,coverage"
    )


def test_save_workflow_report_writes_json(tmp_path):
    stack_result = H5SeriesLoadResult(
        stack=np.ones((1, 2, 2), dtype=np.float32),
        records=[H5FrameRecord(Path("sample.h5"), 1, True, H5RetryReport(True, 1))],
        counts=np.ones((1, 2), dtype=np.float32),
        timestamps=np.array([0.0], dtype=np.float64),
        temperatures=np.full((1, 3), np.nan, dtype=np.float32),
        retry_failed_frames=np.array([], dtype=np.int32),
        metadata_failed_frames=np.array([], dtype=np.int32),
    )
    processed = process_workflow_stack(
        argparse.Namespace(
            normalization="None",
            background=None,
            no_clamp_negative=False,
            grid_remove=False,
            detector_correction=False,
            detector="Eiger4M",
        ),
        stack_result,
        qmap_result=None,
    )
    args = argparse.Namespace(
        report_json=True,
        out=str(tmp_path),
        dataset="/entry/data/data",
        detector="Eiger4M",
        metadata_mode="",
        reference_timestamp=None,
        retry_count=1,
        normalization="None",
        background=None,
        background_ic=None,
        background_bc=None,
        no_clamp_negative=False,
        mask_detector=False,
        grid_remove=False,
        detector_correction=False,
        export_1d=False,
        export_format="csv",
        q_bins=600,
        q_min=None,
        q_max=None,
        q_scale="linear",
        export_std=False,
        export_diagnostics=False,
        export_figure=False,
    )

    report = save_workflow_report(
        args,
        "TrXS",
        stack_result,
        processed.stack_result,
        None,
        processed,
        None,
        export_result=WorkflowExportResult(),
    )

    assert report.report_path is not None
    assert report.report_path.exists()
    assert "report: saved" in format_report_summary(report)


def test_save_workflow_manifest_appends_jsonl(tmp_path):
    stack_result = H5SeriesLoadResult(
        stack=np.ones((1, 2, 2), dtype=np.float32),
        records=[H5FrameRecord(Path("sample.h5"), 1, True, H5RetryReport(True, 1))],
        counts=np.ones((1, 2), dtype=np.float32),
        timestamps=np.array([0.0], dtype=np.float64),
        temperatures=np.full((1, 3), np.nan, dtype=np.float32),
        retry_failed_frames=np.array([], dtype=np.int32),
        metadata_failed_frames=np.array([1], dtype=np.int32),
    )
    processed = process_workflow_stack(
        argparse.Namespace(
            normalization="None",
            background=None,
            no_clamp_negative=False,
            grid_remove=False,
            detector_correction=False,
            detector="Eiger4M",
        ),
        stack_result,
        qmap_result=None,
    )
    args = argparse.Namespace(manifest=str(tmp_path / "manifest.jsonl"))
    export = WorkflowExportResult(
        data_path=tmp_path / "sample_1D.csv",
        diagnostics={"one_d_bin_count": 4, "one_d_coverage_min": 0.75},
    )

    result = save_workflow_manifest(
        args,
        "TrXS",
        stack_result,
        processed.stack_result,
        None,
        processed,
        tmp_path / "sample_2Dinfo.h5",
        export,
        WorkflowReportResult(report_path=tmp_path / "sample_report.json"),
    )

    assert result.manifest_path is not None
    assert "manifest: appended" in format_manifest_summary(result)
    row = json.loads(result.manifest_path.read_text(encoding="utf-8").strip())
    assert row["measurement_mode"] == "TrXS"
    assert row["metadata_failed_count"] == 1
    assert row["one_d_bin_count"] == 4
    assert row["one_d_coverage_min"] == 0.75


def test_load_workflow_config_normalizes_keys(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"q-bins": 100, "export_1d": True}), encoding="utf-8")

    config = load_workflow_config(config_file)

    assert config["q_bins"] == 100
    assert config["export_1d"] is True


def test_parse_workflow_args_uses_config_defaults_and_cli_override(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "input": "from_config.h5",
                "standard": "standard.json",
                "detector": "Eiger4M",
                "normalization": "IC",
                "q_bins": 100,
                "export_1d": True,
            }
        ),
        encoding="utf-8",
    )

    args = parse_workflow_args(
        build_trxs_parser(),
        ["--config", str(config_file), "--normalization", "BC", "--q-bins", "200"],
    )

    assert args.input == "from_config.h5"
    assert args.standard == "standard.json"
    assert args.detector == "Eiger4M"
    assert args.export_1d is True
    assert args.normalization == "BC"
    assert args.q_bins == 200


def test_parser_config_keys_include_expected_options():
    keys = parser_config_keys(build_trxs_parser())

    assert "input" in keys
    assert "standard" in keys
    assert "q_bins" in keys


def test_parse_workflow_args_rejects_unknown_config_key(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"q_bnis": 100}), encoding="utf-8")

    try:
        parse_workflow_args(build_trxs_parser(), ["--config", str(config_file)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("unknown config key should stop parsing")


def test_validate_workflow_config_values_accepts_valid_values():
    validate_workflow_config_values(
        {
            "input": "sample.h5",
            "standard": "standard.json",
            "export_1d": True,
            "export_format": "dat",
            "q_bins": 100,
            "q_min": 0.01,
            "q_max": 0.3,
            "q_scale": "log",
            "normalization": "IC",
            "retry_count": 1,
        }
    )


def test_parse_workflow_args_rejects_invalid_config_q_bins(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"q_bins": 0}), encoding="utf-8")

    try:
        parse_workflow_args(build_trxs_parser(), ["--config", str(config_file)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid q_bins should stop parsing")


def test_parse_workflow_args_rejects_invalid_config_q_range(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"q_min": 0.3, "q_max": 0.1}), encoding="utf-8")

    try:
        parse_workflow_args(build_trxs_parser(), ["--config", str(config_file)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid q range should stop parsing")


def test_parse_workflow_args_rejects_invalid_config_bool(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"export_1d": "yes"}), encoding="utf-8")

    try:
        parse_workflow_args(build_trxs_parser(), ["--config", str(config_file)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid bool config value should stop parsing")


def test_parse_workflow_args_rejects_invalid_config_frame_index(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"frame_index": -1}), encoding="utf-8")

    try:
        parse_workflow_args(build_trxs_parser(), ["--config", str(config_file)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid frame_index should stop parsing")
