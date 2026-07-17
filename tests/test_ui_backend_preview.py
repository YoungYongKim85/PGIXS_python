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
