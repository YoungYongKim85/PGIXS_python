import importlib.util

import numpy as np

from pgixs_python.io.two_d_info import TwoDInfo


def test_two_d_info_preview_payload_reports_stats_and_nan_preview(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    from pgixs_python.io.two_d_info import save_two_d_info_h5
    from pgixs_python.ui_backend.preview import two_d_info_preview_payload

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    data[1, 1] = np.nan
    info = TwoDInfo(
        measurement_mode="GIXS",
        data2d_pix=data,
        data2d_q=data * 2,
        q_map=np.linspace(0.01, 0.16, 16, dtype=np.float32).reshape(4, 4),
        qz_map=np.ones((4, 4), dtype=np.float32),
        qz_axis=np.linspace(0.0, 1.0, 4, dtype=np.float32),
        beam_position=np.array([2.0, 3.0], dtype=np.float32),
        wavelength=1.24,
        sdd=1200.0,
        pixel=0.075,
        detector="Eiger4M",
        source_files=["raw.h5"],
        retry_failed_frames=np.array([2], dtype=np.int32),
    )
    out_file = tmp_path / "sample_2Dinfo.h5"
    save_two_d_info_h5(out_file, info)

    payload = two_d_info_preview_payload(out_file, max_points=4)

    assert payload["ok"]
    assert payload["measurement_mode"] == "GIXS"
    assert payload["shape"] == [4, 4]
    assert payload["nan_fraction"] > 0
    assert payload["intensity"]["finite_count"] == 15
    assert payload["calibration"]["beam_position"] == [2.0, 3.0]
    assert payload["retry_failed_frames"] == [2]
    assert payload["preview"]["data2d_q"][1][1] is None


def test_one_d_export_preview_payload_reads_csv_with_coverage(tmp_path):
    from pgixs_python.ui_backend.preview import one_d_export_preview_payload

    csv_file = tmp_path / "curve_1D.csv"
    csv_file.write_text(
        "q,intensity,standard_deviation,valid_count,coverage\n"
        "0.01,10,1,100,1\n"
        "0.02,20,2,80,0.8\n"
        "0.03,12,3,4,0.05\n"
        "0.04,11,4,3,0.04\n",
        encoding="utf-8",
    )

    payload = one_d_export_preview_payload(csv_file)

    assert payload["ok"]
    assert payload["point_count"] == 4
    assert payload["finite_count"] == 4
    assert payload["has_standard_deviation"]
    assert payload["has_coverage"]
    assert payload["coverage_diagnostics"]["warning_count"] >= 1
    assert "Coverage warning" in payload["coverage_diagnostics"]["message"]
    assert round(payload["q"]["min"], 4) == 0.01
    assert payload["intensity"]["max"] == 20.0
    assert [round(value, 4) for value in payload["preview"]["q"]] == [0.01, 0.02, 0.03, 0.04]


def test_one_d_export_preview_payload_reads_dat_without_header(tmp_path):
    from pgixs_python.ui_backend.preview import one_d_export_preview_payload

    dat_file = tmp_path / "curve_1D.dat"
    dat_file.write_text("0.01 10\n0.02 20\n", encoding="utf-8")

    payload = one_d_export_preview_payload(dat_file)

    assert payload["ok"]
    assert payload["point_count"] == 2
    assert not payload["has_standard_deviation"]
