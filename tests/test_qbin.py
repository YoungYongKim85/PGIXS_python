import numpy as np

from pgixs_python.averaging.qbin import _median_by_bin, mean_by_bin, qbin_average, summarize_coverage


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


def test_qbin_average_mean_mode_is_arithmetic_mean():
    data = np.array([1.0, 1.1, 1.2, 100.0, 5.0], dtype=np.float32)
    bins = np.array([0, 0, 0, 0, 1])

    result = qbin_average(data, bins, 2, use_median=False, calc_std=True)

    assert np.isclose(result.intensity[0], np.mean(data[:4]))
    assert np.isclose(result.std[0], np.std(data[:4], ddof=1))
    # One pixel has no spread to report, as MATLAB's std over a single value.
    assert result.intensity[1] == 5.0
    assert np.isnan(result.std[1])


def test_qbin_average_keeps_signal_on_a_small_part_of_the_ring():
    """A textured Bragg ring lights up only a few per cent of its azimuth.
    The old Huber default read those pixels as outliers and kept about the
    background; the mean must carry the ring's full average intensity."""

    background = np.full(200, 10.0)
    background[:6] = 5000.0  # 3% of the ring is the Bragg spot
    bins = np.zeros(background.size, dtype=np.int64)

    result = qbin_average(background, bins, 1)

    assert np.isclose(result.intensity[0], background.mean())
    assert result.intensity[0] > 150


def test_qbin_average_median_mode():
    data = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    bins = np.array([0, 0, 0])

    result = qbin_average(data, bins, 1, use_median=True, calc_std=True)

    assert result.intensity[0] == 2.0
    assert result.std[0] > 0


def test_summarize_coverage_reports_low_and_drop_bins():
    coverage = np.array([1.0, 1.0, 1.0, 1.0, 0.05, 0.04, 0.04, 1.0, 1.0, 1.0], dtype=np.float32)

    summary = summarize_coverage(coverage, low_threshold=0.15)

    assert summary["coverage_min"] < 0.1
    np.testing.assert_array_equal(summary["coverage_low_bins"], np.array([4, 5, 6], dtype=np.int64))
    assert np.asarray(summary["coverage_drop_bins"]).size >= 1


def test_median_by_bin_is_independent_of_within_bin_value_order():
    """`_median_by_bin` groups by bin with a stable argsort rather than
    lexsorting the values too -- a median doesn't depend on input order, and
    the cheaper sort is ~5x faster on a full detector frame (it runs twice per
    median/MAD reduction, on every frame of a series). Shuffling the values
    within each bin must not move the result."""

    rng = np.random.default_rng(7)
    bins = np.repeat(np.arange(40), 25)
    values = rng.uniform(0.0, 100.0, size=bins.size)

    baseline = _median_by_bin(values, bins, 40)

    shuffle = rng.permutation(values.size)
    shuffled = _median_by_bin(values[shuffle], bins[shuffle], 40)

    np.testing.assert_array_equal(baseline, shuffled)
    for b in range(40):
        np.testing.assert_allclose(baseline[b], np.median(values[bins == b]))
