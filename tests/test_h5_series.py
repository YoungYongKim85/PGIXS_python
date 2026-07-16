import importlib.util

import numpy as np


def test_load_h5_series_reads_stack_and_metadata(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    import h5py

    from pgixs_python.io.h5_series import load_h5_series

    files = []
    for idx in range(2):
        h5_file = tmp_path / f"frame_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            raw = np.full((2, 3), idx + 1, dtype=np.float32)
            h5.create_dataset("/entry/data/data", data=raw)
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([10.0 + idx], dtype=np.float32))
            base.create_dataset("Counter3", data=np.array([20.0 + idx], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([100.0 + idx], dtype=np.float64))
        files.append(h5_file)

    result = load_h5_series(files)

    assert result.stack.shape == (2, 3, 2)
    assert np.all(result.stack[0] == 1)
    assert np.all(result.stack[1] == 2)
    np.testing.assert_allclose(result.counts[:, 0], [10.0, 11.0])
    np.testing.assert_allclose(result.counts[:, 1], [20.0, 21.0])
    assert result.retry_failed_frames.size == 0
    assert result.metadata_failed_frames.size == 0


def test_load_h5_series_marks_failed_frame_with_nan_fallback(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    import h5py

    from pgixs_python.io.h5_series import load_h5_series

    ok_file = tmp_path / "ok.h5"
    missing_file = tmp_path / "missing.h5"
    with h5py.File(ok_file, "w") as h5:
        h5.create_dataset("/entry/data/data", data=np.ones((2, 3), dtype=np.float32))

    result = load_h5_series([ok_file, missing_file], retry_count=1)

    assert result.stack.shape == (2, 3, 2)
    assert np.all(np.isnan(result.stack[1]))
    np.testing.assert_array_equal(result.retry_failed_frames, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(result.metadata_failed_frames, np.array([2], dtype=np.int32))
