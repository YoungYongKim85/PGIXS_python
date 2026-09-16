import numpy as np
import pytest

from pgixs_python.ui_backend import xtrans_session as xs


def setup_function(_fn):
    xs.reset_session()


def test_xtrans_calculate_transmittance_mode():
    payload = {
        "chem_formula": "C8H8",
        "mass_density": 1.05,
        "thickness_um": 100,
        "energy_min": 8.5,
        "energy_max": 15.0,
        "n_points": 20,
        "mode": "Transmittance",
    }
    result = xs.xtrans_calculate(payload)

    assert result["ok"], result.get("message")
    x = np.asarray(result["x"])
    y = np.asarray(result["y"])
    assert x.shape == (20,)
    assert y.shape == (20,)
    assert x[0] == pytest.approx(8.5)
    assert x[-1] == pytest.approx(15.0)
    # Transmittance must be in (0, 1] for a real absorbing material.
    assert np.all(y > 0.0) and np.all(y <= 1.0)
    # Higher energy -> less absorption -> higher transmittance.
    assert y[-1] > y[0]


def test_xtrans_calculate_critical_angle_mode():
    payload = {
        "chem_formula": "Si",
        "mass_density": 2.329,
        "energy_min": 8.0,
        "energy_max": 12.0,
        "n_points": 10,
        "mode": "Critical Angle",
    }
    result = xs.xtrans_calculate(payload)

    assert result["ok"], result.get("message")
    y = np.asarray(result["y"])
    # Silicon's critical angle at ~10 keV is a well-known small number
    # around 0.15-0.2 deg.
    assert np.all((y > 0.05) & (y < 0.5))


def test_xtrans_calculate_sld_modes_are_scaled_by_1e6():
    from pgixs_python.physics.refrac import refrac

    payload_base = {"chem_formula": "SiO2", "mass_density": 2.2, "energy_min": 8.0, "energy_max": 8.0001, "n_points": 2}
    real = xs.xtrans_calculate({**payload_base, "mode": "SLD (Real)"})
    imag = xs.xtrans_calculate({**payload_base, "mode": "SLD (Imag)"})
    assert real["ok"] and imag["ok"]

    ref = refrac("SiO2", 8.0, 2.2)
    assert real["y"][0] == pytest.approx(ref.re_sld[0] * 1e6, rel=1e-2)
    assert imag["y"][0] == pytest.approx(ref.im_sld[0] * 1e6, rel=1e-2)


def test_xtrans_calculate_rejects_invalid_formula():
    result = xs.xtrans_calculate(
        {"chem_formula": "Xx2", "mass_density": 1.0, "energy_min": 8.0, "energy_max": 10.0, "n_points": 5, "mode": "Transmittance"}
    )
    assert not result["ok"]


def test_xtrans_calculate_rejects_bad_energy_range():
    result = xs.xtrans_calculate(
        {"chem_formula": "Si", "mass_density": 2.329, "energy_min": 10.0, "energy_max": 8.0, "n_points": 5, "mode": "Transmittance"}
    )
    assert not result["ok"]


def test_xtrans_save_requires_calculation_first():
    result = xs.xtrans_save({"directory": "."})
    assert not result["ok"]


def test_xtrans_save_writes_a_file(tmp_path):
    xs.xtrans_calculate(
        {"chem_formula": "Si", "mass_density": 2.329, "energy_min": 8.0, "energy_max": 10.0, "n_points": 5, "mode": "Transmittance"}
    )
    result = xs.xtrans_save({"directory": str(tmp_path)})

    assert result["ok"], result.get("message")
    assert (tmp_path / "Transmittance_8_to_10.dat").exists()


def test_set_session_isolates_two_independent_sessions():
    """Regression for any caller serving more than one user at a time:
    each owns an XTransSession and swaps it in before dispatching, so one
    user's Save Data can never write the other's curve."""

    alice = xs.XTransSession()
    bob = xs.XTransSession()

    xs.set_session(alice)
    result = xs.xtrans_calculate(
        {
            "chem_formula": "C8H8",
            "mass_density": 1.05,
            "energy_min": 8.5,
            "energy_max": 15.0,
            "n_points": 5,
            "mode": "Transmittance",
        }
    )
    assert result["ok"], result.get("message")

    xs.set_session(bob)
    assert bob.last_result is None
    assert xs.xtrans_save({"directory": "."})["ok"] is False

    xs.set_session(alice)
    assert xs.get_session() is alice
    assert alice.last_result["mode"] == "Transmittance"
