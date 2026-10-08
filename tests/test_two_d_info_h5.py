import importlib.util

import numpy as np

from pgixs_python.io.two_d_info import TwoDInfo


def test_two_d_info_h5_roundtrip(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    from pgixs_python.io.two_d_info import load_two_d_info_h5, save_two_d_info_h5

    q = np.ones((3, 4), dtype=np.float32)
    info = TwoDInfo(
        measurement_mode="TrXS",
        data2d_pix=np.arange(12, dtype=np.float32).reshape(3, 4),
        data2d_q=np.arange(12, dtype=np.float32).reshape(3, 4),
        q_map=q,
        qx_map=np.zeros((3, 4), dtype=np.float32),
        qy_map=np.ones((3, 4), dtype=np.float32),
        qz_map=np.ones((3, 4), dtype=np.float32) * 2,
        qy_axis=np.linspace(-1, 1, 4, dtype=np.float32),
        qz_axis=np.linspace(-2, 2, 3, dtype=np.float32),
        beam_position=np.array([2.0, 2.0], dtype=np.float32),
        wavelength=1.2,
        sdd=3000.0,
        pixel=0.075,
        detector="Eiger4M",
        source_files=["sample.h5"],
        retry_failed_frames=np.array([3, 7], dtype=np.int32),
        diagnostics={"coverage": np.array([1.0, 0.5], dtype=np.float32)},
    )
    out_file = tmp_path / "roundtrip_2dinfo.h5"
    save_two_d_info_h5(out_file, info)
    loaded = load_two_d_info_h5(out_file)

    assert loaded.measurement_mode == "TrXS"
    assert loaded.detector == "Eiger4M"
    assert loaded.source_files == ["sample.h5"]
    assert np.array_equal(loaded.retry_failed_frames, np.array([3, 7], dtype=np.int32))
    assert np.allclose(loaded.q_map, q)
    assert "coverage" in loaded.diagnostics



def test_two_d_info_h5_saves_diagnostics_of_any_shape(tmp_path):
    import json

    import pytest

    pytest.importorskip("h5py")
    from pgixs_python.io.two_d_info import load_two_d_info_h5, save_two_d_info_h5

    grid = np.zeros((2, 2), dtype=np.float32)
    info = TwoDInfo(
        measurement_mode="GIXS", data2d_pix=grid, data2d_q=grid, q_map=grid, qx_map=grid, qy_map=grid,
        qz_map=grid, qy_axis=np.zeros(2), qz_axis=np.zeros(2), beam_position=np.zeros(2), wavelength=1.0,
        sdd=1.0, pixel=0.075, detector="Eiger4M",
        diagnostics={
            "zero_d": np.array(4.5),
            "labels": ["a", "b"],
            "options": {"bins": 64, "mode": "linear"},
            "rows": [{"i": 1}, {"i": 2}],
        },
    )
    save_two_d_info_h5(tmp_path / "d.h5", info)

    diagnostics = load_two_d_info_h5(tmp_path / "d.h5").diagnostics
    assert diagnostics["zero_d"] == 4.5
    assert json.loads(diagnostics["labels"]) == ["a", "b"]
    assert json.loads(diagnostics["options"]) == {"bins": 64, "mode": "linear"}
    assert json.loads(diagnostics["rows"]) == [{"i": 1}, {"i": 2}]
