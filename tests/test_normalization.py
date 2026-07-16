import numpy as np

from pgixs_python.processing.normalization import normalize_counters, process_normalization


def test_process_normalization_none_subtracts_background_and_clamps():
    data = np.array([[5, 1]], dtype=np.float32)
    background = np.array([[2, 4]], dtype=np.float32)

    out, report = process_normalization(data, background, method="None")

    assert report.success
    np.testing.assert_allclose(out, [[3, 0]])


def test_process_normalization_ic_with_background():
    data = np.array([[10, 20]], dtype=np.float32)
    background = np.array([[2, 4]], dtype=np.float32)

    out, report = process_normalization(data, background, counts=[10, 5], background_counts=[2, 1], method="IC")

    assert report.success
    np.testing.assert_allclose(out, [[0, 0]])


def test_process_normalization_invalid_counter_returns_original():
    data = np.array([[10, 20]], dtype=np.float32)

    out, report = process_normalization(data, counts=[0, 1], method="IC")

    assert not report.success
    np.testing.assert_allclose(out, data)


def test_process_normalization_bc_no_background():
    data = np.array([[10, 20]], dtype=np.float32)

    out, report = process_normalization(data, counts=[2, 5], method="BC")

    assert report.success
    np.testing.assert_allclose(out, [[2, 4]])


def test_normalize_counters_safe_fallbacks():
    out = normalize_counters([10, 20], [2, 0])

    np.testing.assert_allclose(out, [5, 20])
