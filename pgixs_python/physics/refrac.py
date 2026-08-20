"""X-ray/matter refraction properties: a port of Zhang Jiang's MATLAB
`refrac.m` (2004-2008, CXRO/Henke atomic scattering factor tables).

Given a chemical formula, mass density, and one or more photon energies
(0.03-30 keV), computes the complex index of refraction `n = 1 - delta -
i*beta` and everything derived from it: critical angle, attenuation
length, and real/imaginary scattering length density (SLD). Used by the
XTrans and Penetration Depth tabs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

# Constants exactly as hard-coded in refrac.m (SI units).
_THOMPSON = 2.81794092e-15  # classical electron radius, m
_SPEEDOFLIGHT = 299792458.0  # m/s
_PLANCK = 6.626068e-34  # J*s
_ELEMENTCHARGE = 1.60217646e-19  # C
_AVOGADRO = 6.02214199e23  # 1/mol

_DATA_DIR = Path(__file__).parent / "data" / "atomic_scattering_factor"

# (atomic_number, atomic_weight [g/mol]) -- dumped from periodictable.mat.
PERIODIC_TABLE: dict[str, tuple[int, float]] = {
    "H": (1, 1.00794),
    "He": (2, 4.0026),
    "Li": (3, 6.941),
    "Be": (4, 9.01218),
    "B": (5, 10.81),
    "C": (6, 12.0107),
    "N": (7, 14.0067),
    "O": (8, 15.9994),
    "F": (9, 18.9984),
    "Ne": (10, 20.1797),
    "Na": (11, 22.9898),
    "Mg": (12, 24.305),
    "Al": (13, 26.9815),
    "Si": (14, 28.0855),
    "P": (15, 30.9738),
    "S": (16, 32.066),
    "Cl": (17, 35.4527),
    "Ar": (18, 39.948),
    "K": (19, 39.0983),
    "Ca": (20, 40.078),
    "Sc": (21, 44.9559),
    "Ti": (22, 47.867),
    "V": (23, 50.9415),
    "Cr": (24, 51.9961),
    "Mn": (25, 54.938),
    "Fe": (26, 55.845),
    "Co": (27, 58.9332),
    "Ni": (28, 58.6934),
    "Cu": (29, 63.546),
    "Zn": (30, 65.39),
    "Ga": (31, 69.723),
    "Ge": (32, 72.61),
    "As": (33, 74.9216),
    "Se": (34, 78.96),
    "Br": (35, 79.904),
    "Kr": (36, 83.8),
    "Rb": (37, 85.4678),
    "Sr": (38, 87.62),
    "Y": (39, 88.9059),
    "Zr": (40, 91.224),
    "Nb": (41, 92.9064),
    "Mo": (42, 95.94),
    "Tc": (43, 98.0),
    "Ru": (44, 101.07),
    "Rh": (45, 102.906),
    "Pd": (46, 106.42),
    "Ag": (47, 107.868),
    "Cd": (48, 112.411),
    "In": (49, 114.818),
    "Sn": (50, 118.71),
    "Sb": (51, 121.76),
    "Te": (52, 127.6),
    "I": (53, 126.904),
    "Xe": (54, 131.29),
    "Cs": (55, 132.905),
    "Ba": (56, 137.327),
    "La": (57, 138.906),
    "Ce": (58, 140.116),
    "Pr": (59, 140.908),
    "Nd": (60, 144.24),
    "Pm": (61, 145.0),
    "Sm": (62, 150.36),
    "Eu": (63, 151.964),
    "Gd": (64, 157.25),
    "Tb": (65, 158.925),
    "Dy": (66, 162.5),
    "Ho": (67, 164.93),
    "Er": (68, 167.26),
    "Tm": (69, 168.934),
    "Yb": (70, 173.04),
    "Lu": (71, 174.967),
    "Hf": (72, 178.49),
    "Ta": (73, 180.948),
    "W": (74, 183.84),
    "Re": (75, 186.207),
    "Os": (76, 190.23),
    "Ir": (77, 192.217),
    "Pt": (78, 195.078),
    "Au": (79, 196.967),
    "Hg": (80, 200.59),
    "Tl": (81, 204.383),
    "Pb": (82, 207.2),
    "Bi": (83, 208.98),
    "Po": (84, 209.0),
    "At": (85, 210.0),
    "Rn": (86, 222.0),
    "Fr": (87, 223.0),
    "Ra": (88, 226.0),
    "Ac": (89, 227.0),
    "Th": (90, 232.038),
    "Pa": (91, 231.036),
    "U": (92, 238.029),
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]*)([0-9.]*)")


@dataclass
class RefracResult:
    chem_formula: str
    molecular_weight: float
    number_of_electrons: float
    mass_density: float
    electron_density: float
    energy: np.ndarray  # keV
    wavelength: np.ndarray  # Angstrom
    dispersion: np.ndarray  # delta
    absorption: np.ndarray  # beta
    critical_angle: np.ndarray  # degrees
    att_length: np.ndarray  # cm
    re_sld: np.ndarray  # 1/Angstrom^2
    im_sld: np.ndarray  # 1/Angstrom^2


def parse_chemical_formula(formula: str) -> dict[str, float]:
    """Parse a case-sensitive chemical formula ("SiO2", "C8H8") into
    {element_symbol: atom_count}. Matches refrac.m's character scan: an
    empty/zero count defaults to 1, no parentheses or hydrate support."""

    if not formula or not isinstance(formula, str):
        raise ValueError("Invalid chemical formula.")

    matched_length = 0
    counts: dict[str, float] = {}
    for match in _FORMULA_TOKEN.finditer(formula):
        if match.start() != matched_length:
            raise ValueError("Invalid chemical formula.")
        matched_length = match.end()
        symbol, count_str = match.group(1), match.group(2)
        try:
            count = float(count_str) if count_str not in ("", ".") else 0.0
        except ValueError as exc:
            raise ValueError("Invalid chemical formula.") from exc
        if count == 0.0:
            count = 1.0
        counts[symbol] = counts.get(symbol, 0.0) + count

    if matched_length != len(formula) or not counts:
        raise ValueError("Invalid chemical formula.")
    return counts


@lru_cache(maxsize=None)
def _load_f1f2_table(symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read `<symbol>.nff` (CXRO/Henke table: E[eV], f1, f2), dropping
    rows below 29 eV where f1 is a -9999 sentinel -- matches refrac.m."""

    path = _DATA_DIR / f"{symbol.lower()}.nff"
    if not path.exists():
        raise ValueError(f"Element '{symbol}' is NOT in the table list.")
    data = np.loadtxt(path, skiprows=1)
    energy_ev, f1, f2 = data[:, 0], data[:, 1], data[:, 2]
    valid = energy_ev >= 29.0
    return energy_ev[valid], f1[valid], f2[valid]


@lru_cache(maxsize=None)
def _interpolators(symbol: str) -> tuple[PchipInterpolator, PchipInterpolator]:
    energy_ev, f1, f2 = _load_f1f2_table(symbol)
    return PchipInterpolator(energy_ev, f1, extrapolate=True), PchipInterpolator(energy_ev, f2, extrapolate=True)


def refrac(chem_formula: str, energy_kev: np.ndarray | float, mass_density: float) -> RefracResult:
    """Port of `refrac.m`'s `subrefrac`. `energy_kev` may be scalar or an
    array in [0.03, 30] keV; the result's arrays follow energy sorted
    ascending, matching the MATLAB function's own `sort(energy)`."""

    energy = np.atleast_1d(np.asarray(energy_kev, dtype=np.float64))
    if energy.size == 0 or not np.all(np.isfinite(energy)):
        raise ValueError("Invalid x-ray energy.")
    if np.min(energy) < 0.03 or np.max(energy) > 30.0:
        raise ValueError("Energy is out of range 0.03KeV~30KeV.")
    energy = np.sort(energy)

    counts = parse_chemical_formula(chem_formula)
    for symbol in counts:
        if symbol not in PERIODIC_TABLE:
            raise ValueError(f"Element '{symbol}' is NOT in the table list.")

    wavelength_m = (_SPEEDOFLIGHT * _PLANCK / _ELEMENTCHARGE) / (energy * 1000.0)

    molecular_weight = 0.0
    number_of_electrons = 0.0
    for symbol, n_atoms in counts.items():
        atomic_number, atomic_weight = PERIODIC_TABLE[symbol]
        molecular_weight += n_atoms * atomic_weight
        number_of_electrons += atomic_number * n_atoms

    dispersion = np.zeros_like(energy)
    absorption = np.zeros_like(energy)
    energy_ev = energy * 1000.0
    prefactor = wavelength_m**2 / (2.0 * np.pi) * _THOMPSON * _AVOGADRO * mass_density * 1.0e6 / molecular_weight
    for symbol, n_atoms in counts.items():
        f1_interp, f2_interp = _interpolators(symbol)
        dispersion += prefactor * n_atoms * f1_interp(energy_ev)
        absorption += prefactor * n_atoms * f2_interp(energy_ev)

    critical_angle = np.sqrt(2.0 * dispersion) * 180.0 / np.pi
    att_length = wavelength_m / absorption / (4.0 * np.pi) * 1.0e2
    re_sld = dispersion * 2.0 * np.pi / wavelength_m**2 / 1.0e20
    im_sld = absorption * 2.0 * np.pi / wavelength_m**2 / 1.0e20
    electron_density = 1.0e6 * mass_density / molecular_weight * _AVOGADRO * number_of_electrons / 1.0e30

    return RefracResult(
        chem_formula=chem_formula,
        molecular_weight=molecular_weight,
        number_of_electrons=number_of_electrons,
        mass_density=mass_density,
        electron_density=electron_density,
        energy=energy,
        wavelength=wavelength_m * 1.0e10,
        dispersion=dispersion,
        absorption=absorption,
        critical_angle=critical_angle,
        att_length=att_length,
        re_sld=re_sld,
        im_sld=im_sld,
    )
