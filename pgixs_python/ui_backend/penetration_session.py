"""Backend for the Penetration Depth tab: a port of MATLAB
`Penetration_GUI.m` on top of `physics.refrac`.

Computes the evanescent-wave penetration depth vs. incidence angle for a
single photon energy, using the standard GIXS formula (valid both above
and below the critical angle).
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.export.data_export import OneDExport, save_1d_data
from pgixs_python.physics.refrac import refrac


@dataclass
class PenetrationSession:
    last_result: dict[str, Any] | None = None


_SESSION = PenetrationSession()


def get_session() -> PenetrationSession:
    return _SESSION


def set_session(session: PenetrationSession) -> None:
    """Install `session` as the active module-level session.

    Every `penetration_*` function operates on whatever session is
    installed here. One desktop tab never needed to swap, but a caller
    serving more than one user at a time does -- without this, two of them
    would share `last_result` and one's Save Data would write the other's
    curve. Mirrors `console_session.set_session` and the two series
    modules'.
    """

    global _SESSION
    _SESSION = session


def reset_session() -> None:
    global _SESSION
    _SESSION = PenetrationSession()


def penetration_calculate(payload: dict[str, Any]) -> dict[str, Any]:
    chem_formula = str(payload.get("chem_formula", "")).strip()
    try:
        mass_density = float(payload["mass_density"])
        energy_kev = float(payload["energy_kev"])
        angle_min = float(payload["angle_min"])
        angle_max = float(payload["angle_max"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid input: {exc}"}

    if angle_max <= angle_min:
        return {"ok": False, "message": "Angle range must satisfy min < max."}

    try:
        result = refrac(chem_formula, energy_kev, mass_density)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    delta = float(result.dispersion[0])
    beta = float(result.absorption[0])
    critical_angle = float(result.critical_angle[0])
    wavelength_a = float(result.wavelength[0])

    alpha_i_deg = np.linspace(angle_min, angle_max, 200)
    th_i = np.radians(alpha_i_deg)
    th_c = np.radians(critical_angle)

    inner_sqrt = np.sqrt((th_i**2 - th_c**2) ** 2 + 4.0 * beta**2)
    lp_angstrom = wavelength_a / ((4.0 * np.pi / np.sqrt(2.0)) * np.sqrt(inner_sqrt - (th_i**2 - th_c**2)))
    depth_nm = lp_angstrom / 10.0

    _SESSION.last_result = {
        "x": alpha_i_deg,
        "y": depth_nm,
        "x_label": "incidence_angle_deg",
        "y_label": "penetration_depth_nm",
        "angle_min": angle_min,
        "angle_max": angle_max,
        "chem_formula": chem_formula,
        "energy_kev": energy_kev,
    }

    return {
        "ok": True,
        "x": alpha_i_deg.tolist(),
        "y": depth_nm.tolist(),
        "delta": delta,
        "beta": beta,
        "critical_angle": critical_angle,
        "wavelength": wavelength_a,
    }


def penetration_save(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.last_result is None:
        return {"ok": False, "message": "Run a calculation first."}

    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    r = _SESSION.last_result
    # Material and energy are in the name so another calculation over the
    # same angle range doesn't overwrite this one.
    formula = re.sub(r"[^\w.-]+", "_", r["chem_formula"])
    filename = f"PenetrationDepth_{formula}_{r['energy_kev']:g}keV_{r['angle_min']:g}_to_{r['angle_max']:g}.dat"
    path = Path(directory) / filename

    export = OneDExport(x=r["x"], intensity=r["y"], x_label=r["x_label"], intensity_label=r["y_label"])
    try:
        save_1d_data(path, export)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}

    return {"ok": True, "path": str(path)}


__all__ = ["PenetrationSession", "get_session", "penetration_calculate", "penetration_save", "reset_session", "set_session"]
