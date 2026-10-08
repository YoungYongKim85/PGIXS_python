import numpy as np
import pytest

from pgixs_python.ui_backend import penetration_session as ps


def setup_function(_fn):
    ps.reset_session()


def test_penetration_calculate_happy_path():
    payload = {"chem_formula": "Si", "mass_density": 2.329, "energy_kev": 11.3, "angle_min": 0.0, "angle_max": 1.0}
    result = ps.penetration_calculate(payload)

    assert result["ok"], result.get("message")
    x = np.asarray(result["x"])
    y = np.asarray(result["y"])
    assert x.shape == (200,)
    assert y.shape == (200,)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)
    assert np.all(np.isfinite(y))
    assert np.all(y > 0.0)
    assert result["critical_angle"] > 0.0
    assert result["delta"] > 0.0
    assert result["beta"] > 0.0


def test_penetration_calculate_matches_refrac_directly():
    from pgixs_python.physics.refrac import refrac

    result = ps.penetration_calculate(
        {"chem_formula": "Si", "mass_density": 2.329, "energy_kev": 11.3, "angle_min": 0.0, "angle_max": 1.0}
    )
    ref = refrac("Si", 11.3, 2.329)

    assert result["critical_angle"] == pytest.approx(ref.critical_angle[0])
    assert result["delta"] == pytest.approx(ref.dispersion[0])
    assert result["beta"] == pytest.approx(ref.absorption[0])
    assert result["wavelength"] == pytest.approx(ref.wavelength[0])


def test_penetration_depth_increases_with_angle_above_critical():
    result = ps.penetration_calculate(
        {"chem_formula": "Si", "mass_density": 2.329, "energy_kev": 11.3, "angle_min": 0.5, "angle_max": 5.0}
    )
    y = np.asarray(result["y"])
    # Well above the critical angle (~0.18 deg for Si at 11.3 keV), the
    # penetration depth should increase monotonically with incidence angle.
    assert np.all(np.diff(y) > 0)


def test_penetration_calculate_rejects_bad_angle_range():
    result = ps.penetration_calculate(
        {"chem_formula": "Si", "mass_density": 2.329, "energy_kev": 11.3, "angle_min": 1.0, "angle_max": 0.5}
    )
    assert not result["ok"]


def test_penetration_calculate_rejects_invalid_formula():
    result = ps.penetration_calculate(
        {"chem_formula": "Xx2", "mass_density": 1.0, "energy_kev": 11.3, "angle_min": 0.0, "angle_max": 1.0}
    )
    assert not result["ok"]


def test_penetration_save_requires_calculation_first():
    result = ps.penetration_save({"directory": "."})
    assert not result["ok"]


def test_penetration_save_writes_a_file(tmp_path):
    ps.penetration_calculate({"chem_formula": "Si", "mass_density": 2.329, "energy_kev": 11.3, "angle_min": 0.0, "angle_max": 1.0})
    result = ps.penetration_save({"directory": str(tmp_path)})

    assert result["ok"], result.get("message")
    assert (tmp_path / "PenetrationDepth_Si_11.3keV_0_to_1.dat").exists()


def test_set_session_isolates_two_independent_sessions():
    """Regression for any caller serving more than one user at a time:
    each owns a PenetrationSession and swaps it in before dispatching, so
    one user's Save Data can never write the other's curve."""

    alice = ps.PenetrationSession()
    bob = ps.PenetrationSession()

    ps.set_session(alice)
    result = ps.penetration_calculate(
        {"chem_formula": "Si", "mass_density": 2.329, "energy_kev": 11.3, "angle_min": 0.0, "angle_max": 1.0}
    )
    assert result["ok"], result.get("message")

    ps.set_session(bob)
    assert bob.last_result is None
    assert ps.penetration_save({"directory": "."})["ok"] is False

    ps.set_session(alice)
    assert ps.get_session() is alice
    assert alice.last_result["chem_formula"] == "Si"
