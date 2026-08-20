import numpy as np
import pytest

from pgixs_python.physics.refrac import parse_chemical_formula, refrac


def test_parse_chemical_formula_simple():
    assert parse_chemical_formula("Si") == {"Si": 1.0}
    assert parse_chemical_formula("SiO2") == {"Si": 1.0, "O": 2.0}
    assert parse_chemical_formula("C8H8") == {"C": 8.0, "H": 8.0}


def test_parse_chemical_formula_case_sensitive_co_vs_CO():
    assert parse_chemical_formula("Co") == {"Co": 1.0}
    assert parse_chemical_formula("CO") == {"C": 1.0, "O": 1.0}


def test_parse_chemical_formula_decimal_stoichiometry():
    assert parse_chemical_formula("Si1.5O2") == {"Si": 1.5, "O": 2.0}


def test_parse_chemical_formula_repeated_element_accumulates():
    # Not a normal formula, but matches refrac.m's own per-character scan:
    # a repeated element symbol just adds onto the same accumulator.
    assert parse_chemical_formula("HH") == {"H": 2.0}


@pytest.mark.parametrize("bad", ["", "2Si", "si", "Si-O2"])
def test_parse_chemical_formula_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        parse_chemical_formula(bad)


def test_refrac_matches_matlab_docstring_example_si3n4():
    """Zhang Jiang's refrac.m docstring's own worked example:
    refrac('Si3N4', 8.04778, 3.44)."""

    r = refrac("Si3N4", 8.04778, 3.44)

    assert r.molecular_weight == pytest.approx(140.28, abs=0.01)
    assert r.number_of_electrons == pytest.approx(70)
    assert r.electron_density == pytest.approx(1.0337, abs=1e-4)
    assert r.wavelength[0] == pytest.approx(1.5406, abs=1e-4)
    assert r.dispersion[0] == pytest.approx(1.1164e-5, rel=1e-3)
    assert r.absorption[0] == pytest.approx(1.6477e-7, rel=1e-3)
    assert r.critical_angle[0] == pytest.approx(0.27074, abs=1e-4)
    assert r.att_length[0] == pytest.approx(0.0074403, rel=1e-3)
    assert r.re_sld[0] == pytest.approx(2.9554e-5, rel=1e-3)
    assert r.im_sld[0] == pytest.approx(4.362e-7, rel=1e-3)


def test_refrac_sorts_energy_ascending():
    r = refrac("Si", [12.0, 8.0, 10.0], 2.329)
    assert list(r.energy) == [8.0, 10.0, 12.0]


def test_refrac_rejects_energy_out_of_range():
    with pytest.raises(ValueError):
        refrac("Si", 0.01, 2.329)
    with pytest.raises(ValueError):
        refrac("Si", 31.0, 2.329)


def test_refrac_rejects_unknown_element():
    with pytest.raises(ValueError):
        refrac("Xx2", 10.0, 1.0)


def test_refrac_vectorized_over_energy_array():
    energies = np.linspace(8.0, 15.0, 20)
    r = refrac("C8H8", energies, 1.05)
    assert r.energy.shape == (20,)
    assert r.dispersion.shape == (20,)
    assert np.all(np.isfinite(r.dispersion))
    assert np.all(np.isfinite(r.absorption))
    # Dispersion should decrease with increasing energy away from edges
    # (delta ~ 1/E^2 broadly) for a simple organic compound with no
    # absorption edges in this range.
    assert r.dispersion[0] > r.dispersion[-1]
