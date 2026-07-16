import importlib.util

import numpy as np


def test_load_h5_frame_transposes_and_masks(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    import h5py

    from pgixs_tab1.io.h5_loader import load_h5_frame

    h5_file = tmp_path / "frame.h5"
    raw = np.array([[1, 2, 3], [4, 5, 10_000_000_001]], dtype=np.float32)
    with h5py.File(h5_file, "w") as h5:
        h5.create_dataset("/entry/data/data", data=raw)

    mask = np.ones((3, 2), dtype=np.float32)
    mask[0, 0] = 0
    data = load_h5_frame(h5_file, mask=mask)

    assert data.shape == (3, 2)
    assert data[0, 0] == 0
    assert np.isnan(data[2, 1])


def test_load_h5_frame_with_retry_returns_nan_fallback(tmp_path):
    from pgixs_tab1.io.h5_loader import load_h5_frame_with_retry

    data, report = load_h5_frame_with_retry(tmp_path / "missing.h5", fallback_size=(2, 3), retry_count=1)
    assert not report.ok
    assert report.attempts == 2
    assert data.shape == (2, 3)
    assert np.all(np.isnan(data))


def test_read_h5_metadata_counter_fallback(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    import h5py

    from pgixs_tab1.io.h5_metadata import read_h5_metadata

    h5_file = tmp_path / "meta.h5"
    with h5py.File(h5_file, "w") as h5:
        base = h5.create_group("/entry/instrument/user/params")
        base.create_dataset("Counter1", data=np.array([10.0], dtype=np.float32))
        base.create_dataset("Counter2", data=np.array([20.0], dtype=np.float32))
        base.create_dataset("TimeStamp", data=np.array([123.5], dtype=np.float64))
        base.create_dataset("Temp1", data=np.array([300.0], dtype=np.float32))

    meta = read_h5_metadata(h5_file, mode="Temperature", reference_timestamp=100.0)
    assert meta.counter1 == 10.0
    assert meta.counter_bc == 20.0
    assert meta.bc_counter == "Counter2"
    assert meta.timestamp == 23.5
    assert meta.temperature[0] == 300.0

