"""SDD calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SDDRingResult:
    sdd: float
    wavelength: float
    two_theta_deg: float


def wavelength_from_energy_kev(energy_kev: float) -> float:
    """Return wavelength in Angstrom from photon energy in keV."""

    return 12.398419843320026 / float(energy_kev)


def compute_sdd_from_ring(pixel_radius: float, pixel_size_mm: float, energy_kev: float, q_value: float) -> SDDRingResult:
    """Compute sample-detector distance from one standard ring."""

    wavelength = wavelength_from_energy_kev(energy_kev)
    theta = np.arcsin(float(q_value) * wavelength / (4.0 * np.pi))
    two_theta_deg = float(np.degrees(2.0 * theta))
    if not np.isfinite(two_theta_deg) or two_theta_deg <= 0:
        raise ValueError("Invalid q/energy combination for SDD calculation.")
    sdd = float(pixel_radius) * float(pixel_size_mm) / np.tan(np.radians(two_theta_deg))
    return SDDRingResult(sdd=sdd, wavelength=wavelength, two_theta_deg=two_theta_deg)

