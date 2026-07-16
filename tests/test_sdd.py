from pgixs_python.calibration.sdd import compute_sdd_from_ring
from pgixs_python.calibration.standards import standard_q_value


def test_compute_sdd_from_ring_positive():
    result = compute_sdd_from_ring(pixel_radius=200, pixel_size_mm=0.075, energy_kev=9.5, q_value=0.1052)
    assert result.sdd > 0
    assert result.wavelength > 0
    assert result.two_theta_deg > 0


def test_standard_q_value_scales_order():
    assert standard_q_value("Silver behenate", order=2) == 0.2104
