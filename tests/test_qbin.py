import numpy as np

from pgixs_python.averaging.qbin import mean_by_bin, qbin_average


def test_mean_by_bin_omits_nan():
    data = np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float32)
    bins = np.array([0, 0, 1, 1])
    mean, counts = mean_by_bin(data, bins, 2)
    assert np.allclose(mean, [1.0, 4.0])
    assert np.array_equal(counts, [1, 2])


def test_qbin_average_reports_coverage():
    data = np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float32)
    bins = np.array([0, 0, 1, 1])

    result = qbin_average(data, bins, 2)

    assert np.allclose(result.intensity, [1.0, 4.0])
    assert np.array_equal(result.counts, [1, 2])
    assert np.array_equal(result.geom_counts, [2, 2])
    assert np.allclose(result.coverage, [0.5, 1.0])


def test_qbin_average_huber_resists_outlier():
    data = np.array([1.0, 1.1, 1.2, 100.0], dtype=np.float32)
    bins = np.array([0, 0, 0, 0])

    mean, _ = mean_by_bin(data, bins, 1)
    robust = qbin_average(data, bins, 1, use_median=False, calc_std=True)

    assert mean[0] > 20
    assert robust.intensity[0] < 10
    assert robust.std[0] > 0


def test_qbin_average_median_mode():
    data = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    bins = np.array([0, 0, 0])

    result = qbin_average(data, bins, 1, use_median=True, calc_std=True)

    assert result.intensity[0] == 2.0
    assert result.std[0] > 0
