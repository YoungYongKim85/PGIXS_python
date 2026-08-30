import pytest

from pgixs_python.calibration.sdd import compute_sdd_from_ring, predicted_ring_radius
from pgixs_python.calibration.standards import standard_q_value


def test_compute_sdd_from_ring_positive():
    result = compute_sdd_from_ring(pixel_radius=200, pixel_size_mm=0.075, energy_kev=9.5, q_value=0.1052)
    assert result.sdd > 0
    assert result.wavelength > 0
    assert result.two_theta_deg > 0


def test_standard_q_value_scales_order():
    assert standard_q_value("Silver behenate", order=2) == 0.2104


def test_predicted_ring_radius_is_the_exact_inverse_of_compute_sdd_from_ring():
    pixel_size_mm = 0.075
    energy_kev = 9.5
    q_value = 0.1052

    result = compute_sdd_from_ring(pixel_radius=200.0, pixel_size_mm=pixel_size_mm, energy_kev=energy_kev, q_value=q_value)
    radius = predicted_ring_radius(result.sdd, pixel_size_mm, energy_kev, q_value)

    assert radius == pytest.approx(200.0)
