import numpy as np

from pgixs_python.averaging.qbin import mean_by_bin


def test_mean_by_bin_omits_nan():
    data = np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float32)
    bins = np.array([0, 0, 1, 1])
    mean, counts = mean_by_bin(data, bins, 2)
    assert np.allclose(mean, [1.0, 4.0])
    assert np.array_equal(counts, [1, 2])

