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


def test_load_h5_series_passes_frame_index(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    import h5py

    from pgixs_python.io.h5_series import load_h5_series

    h5_file = tmp_path / "stack.h5"
    raw = np.stack(
        [
            np.full((4, 5), 1, dtype=np.float32),
            np.full((4, 5), 5, dtype=np.float32),
        ],
        axis=0,
    )
    with h5py.File(h5_file, "w") as h5:
        h5.create_dataset("/entry/data/data", data=raw)

    result = load_h5_series([h5_file], frame_index=1)

    assert result.stack.shape == (1, 5, 4)
    assert np.all(result.stack[0] == 5)


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


def test_load_h5_series_calls_progress_callback_per_file(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    import h5py

    from pgixs_python.io.h5_series import load_h5_series

    files = []
    for idx in range(3):
        h5_file = tmp_path / f"frame_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=np.full((2, 2), idx, dtype=np.float32))
        files.append(h5_file)

    calls = []
    load_h5_series(files, progress_callback=lambda completed, total, name: calls.append((completed, total, name)))

    assert calls == [(1, 3, "frame_0001.h5"), (2, 3, "frame_0002.h5"), (3, 3, "frame_0003.h5")]


def _write_series_files(tmp_path, n_files=6):
    import h5py

    files = []
    for idx in range(n_files):
        h5_file = tmp_path / f"frame_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=np.full((3, 4), float(idx), dtype=np.float32))
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([10.0 + idx], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([1000.0 + idx], dtype=np.float64))
        files.append(h5_file)
    return files


def test_load_h5_series_parallel_matches_serial_output(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    from pgixs_python.io.h5_series import load_h5_series

    files = _write_series_files(tmp_path, n_files=6)

    serial = load_h5_series(files, parallel=False)
    parallel = load_h5_series(files, parallel=True, max_workers=4)

    # Output must be identical and in input order regardless of which
    # worker happened to finish first -- this is the property that
    # actually matters for correctness (a fast frame 5 must not land in
    # frame 2's stack slot).
    np.testing.assert_array_equal(serial.stack, parallel.stack)
    np.testing.assert_array_equal(serial.counts, parallel.counts)
    np.testing.assert_array_equal(serial.timestamps, parallel.timestamps)
    for idx in range(6):
        assert parallel.stack[idx, 0, 0] == float(idx)


def test_load_h5_series_parallel_reports_progress_for_every_file(tmp_path):
    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    from pgixs_python.io.h5_series import load_h5_series

    files = _write_series_files(tmp_path, n_files=6)
    calls = []
    load_h5_series(
        files, parallel=True, max_workers=3,
        progress_callback=lambda completed, total, name: calls.append((completed, total)),
    )

    # Order of individual file names isn't guaranteed under parallel
    # execution, but every file must still report exactly once, and the
    # completed-count must reach the total.
    assert len(calls) == 6
    assert [c[0] for c in calls] == list(range(1, 7))
    assert all(c[1] == 6 for c in calls)


def test_load_h5_series_parallel_single_file_does_not_use_pool(tmp_path):
    """A 1-file series takes the serial path even with parallel=True
    (nothing to parallelize) -- confirms that branch doesn't error out."""

    if importlib.util.find_spec("h5py") is None:
        import pytest

        pytest.skip("h5py is not installed")

    from pgixs_python.io.h5_series import load_h5_series

    files = _write_series_files(tmp_path, n_files=1)
    result = load_h5_series(files, parallel=True)
    assert result.stack.shape[0] == 1


def test_load_h5_series_raises_when_no_frame_can_be_read(tmp_path):
    import pytest

    h5py = pytest.importorskip("h5py")
    from pgixs_python.io.h5_series import load_h5_series

    paths = []
    for idx in range(2):
        paths.append(tmp_path / f"s_{idx}.h5")
        with h5py.File(paths[-1], "w") as h5:
            h5.create_dataset("/entry/data/data", data=np.ones((3, 4), dtype=np.float32))

    with pytest.raises(RuntimeError, match="No H5 frame"):
        load_h5_series(paths, dataset_path="/wrong/path", mask=np.ones((4, 3), dtype=np.float32), retry_count=0)


def test_natural_sort_key_orders_frames_past_the_zero_padding():
    from pgixs_python.io.h5_series import natural_sort_key

    names = ["ABCD_10000.h5", "ABCD_9999.h5", "ABCD_0001.h5"]

    assert sorted(names, key=natural_sort_key) == ["ABCD_0001.h5", "ABCD_9999.h5", "ABCD_10000.h5"]
