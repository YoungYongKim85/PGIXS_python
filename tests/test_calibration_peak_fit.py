import numpy as np
import pytest

from pgixs_python.calibration.peak_fit import fit_gaussian_peak


def _synthetic_gaussian(center=60.0, sigma=4.0, amplitude=500.0, offset=10.0, n=200, span=(0.0, 100.0)):
    x = np.linspace(span[0], span[1], n)
    y = amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset
    return x, y


def test_fit_gaussian_peak_recovers_noiseless_parameters():
    x, y = _synthetic_gaussian()

    result = fit_gaussian_peak(x, y)

    assert result.success
    assert result.center == pytest.approx(60.0)
    assert result.sigma == pytest.approx(4.0)
    assert result.amplitude == pytest.approx(500.0)
    assert result.offset == pytest.approx(10.0)
    assert result.r_square > 0.999
    assert result.fit_x.size > 0
    assert result.fit_y.size == result.fit_x.size


def test_fit_gaussian_peak_windowed_ignores_data_outside_window():
    x, y = _synthetic_gaussian(center=60.0)
    # Corrupt data far outside the intended fit window -- must not affect the result.
    y = y.copy()
    y[x < 10.0] = 1.0e6

    result = fit_gaussian_peak(x, y, window_center=60.0, half_width=20.0)

    assert result.success
    assert result.center == pytest.approx(60.0, abs=0.5)


def test_fit_gaussian_peak_fails_gracefully_on_degenerate_input():
    result = fit_gaussian_peak(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    assert not result.success
    assert "Not enough finite points" in result.message
    assert np.isnan(result.center)


def test_fit_gaussian_peak_fails_gracefully_on_flat_data():
    x = np.linspace(0.0, 10.0, 20)
    y = np.full_like(x, 5.0)

    result = fit_gaussian_peak(x, y)

    assert not result.success
    assert "amplitude guess" in result.message.lower()


def test_fit_gaussian_peak_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        fit_gaussian_peak(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_fit_gaussian_peak_fits_a_peak_on_a_negative_baseline():
    x = np.linspace(-20.0, 20.0, 81)
    y = 100.0 * np.exp(-((x - 2.0) ** 2) / (2.0 * 3.0**2)) - 5.0

    result = fit_gaussian_peak(x, y)

    assert result.success, result.message
    assert result.center == pytest.approx(2.0, abs=1e-3)
    assert result.offset == pytest.approx(-5.0, abs=1e-2)
