"""Backend for the XTrans tab: X-ray transmittance / critical angle / SLD
calculator, a port of MATLAB `XTrans_GUI.m` on top of `physics.refrac`.

Mirrors the rest of `ui_backend`'s pattern: a module-level session holding
the last computed curve (for Save Data), and functions that each return
`{"ok": bool, ...}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pgixs_python.export.data_export import OneDExport, save_1d_data
from pgixs_python.physics.refrac import refrac

_MODES = ("Transmittance", "Critical Angle", "SLD (Real)", "SLD (Imag)")


@dataclass
class XTransSession:
    last_result: dict[str, Any] | None = None


_SESSION = XTransSession()


def get_session() -> XTransSession:
    return _SESSION


def set_session(session: XTransSession) -> None:
    """Install `session` as the active module-level session.

    Every `xtrans_*` function operates on whatever session is installed
    here. One desktop tab never needed to swap, but a caller serving more
    than one user at a time does -- without this, two of them would share
    `last_result` and one's Save Data would write the other's curve.
    Mirrors `console_session.set_session` and the two series modules'.
    """

    global _SESSION
    _SESSION = session


def reset_session() -> None:
    global _SESSION
    _SESSION = XTransSession()


def xtrans_calculate(payload: dict[str, Any]) -> dict[str, Any]:
    chem_formula = str(payload.get("chem_formula", "")).strip()
    mode = str(payload.get("mode", "Transmittance"))
    if mode not in _MODES:
        return {"ok": False, "message": f"Unknown mode: {mode}"}

    try:
        mass_density = float(payload["mass_density"])
        thickness_um = float(payload.get("thickness_um", 0.0))
        energy_min = float(payload["energy_min"])
        energy_max = float(payload["energy_max"])
        n_points = int(payload.get("n_points", 50))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid input: {exc}"}

    if n_points < 2:
        return {"ok": False, "message": "# of points must be at least 2."}
    if energy_min <= 0 or energy_max <= energy_min:
        return {"ok": False, "message": "Energy range must satisfy 0 < min < max."}

    energy = np.linspace(energy_min, energy_max, n_points)
    try:
        result = refrac(chem_formula, energy, mass_density)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    # `y_label` uses matplotlib mathtext for a proper 10^-6 / A^-2
    # superscript on the plot axis; `y_label_plain` is the same unit
    # spelled out in plain ASCII for the Data Table header and the saved
    # .dat file's column header, where mathtext syntax would just show up
    # as literal "$...$" text.
    if mode == "Transmittance":
        att_mm = result.att_length * 10.0
        thickness_mm = thickness_um * 1.0e-3
        y = np.exp(-thickness_mm / att_mm)
        y_label = y_label_plain = "Transmittance"
    elif mode == "Critical Angle":
        y = result.critical_angle
        y_label = y_label_plain = "Critical Angle, deg"
    elif mode == "SLD (Real)":
        y = result.re_sld * 1.0e6
        y_label = "Re(SLD), $10^{-6}$ Å$^{-2}$"
        y_label_plain = "Re(SLD), 1e-6 A^-2"
    else:
        y = result.im_sld * 1.0e6
        y_label = "Im(SLD), $10^{-6}$ Å$^{-2}$"
        y_label_plain = "Im(SLD), 1e-6 A^-2"

    _SESSION.last_result = {
        "x": result.energy,
        "y": y,
        "x_label": "energy_keV",
        "y_label": y_label_plain,
        "mode": mode,
        "chem_formula": chem_formula,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "molecular_weight": result.molecular_weight,
    }

    return {
        "ok": True,
        "x": result.energy.tolist(),
        "y": y.tolist(),
        "x_label": "Photon Energy, keV",
        "y_label": y_label,
        "molecular_weight": result.molecular_weight,
        "number_of_electrons": result.number_of_electrons,
    }


def xtrans_save(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION.last_result is None:
        return {"ok": False, "message": "Run a calculation first."}

    directory = str(payload.get("directory", "")).strip()
    if not directory:
        return {"ok": False, "message": "Choose a save directory."}

    r = _SESSION.last_result
    mode_slug = r["mode"].replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{mode_slug}_{r['energy_min']:g}_to_{r['energy_max']:g}.dat"
    path = Path(directory) / filename

    export = OneDExport(x=r["x"], intensity=r["y"], x_label=r["x_label"], intensity_label=r["y_label"])
    try:
        save_1d_data(path, export)
    except Exception as exc:  # noqa: BLE001 - surface as a visible failure.
        return {"ok": False, "message": str(exc)}

    return {"ok": True, "path": str(path)}


__all__ = ["XTransSession", "get_session", "reset_session", "set_session", "xtrans_calculate", "xtrans_save"]
