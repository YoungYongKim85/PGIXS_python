"""Detector intensity correction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorCorrectionInputs:
    data_size: tuple[int, int]
    beam_position: tuple[float, float]
    pixel_size_mm: float
    sdd_mm: float
    q_map: np.ndarray
    energy_kev: float
    detector_type: str = ""
    kapton_attenuation: np.ndarray | None = None
    silicon_attenuation: np.ndarray | None = None


def apply_detector_correction(data: np.ndarray, correction_factor: np.ndarray) -> np.ndarray:
    """Divide intensity by a correction factor while preserving invalid pixels."""

    image = np.asarray(data, dtype=np.float32)
    factor = np.asarray(correction_factor, dtype=np.float32)
    if image.shape != factor.shape:
        raise ValueError(f"Data shape {image.shape} does not match correction shape {factor.shape}.")

    out = np.full(image.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(image) & np.isfinite(factor) & (factor != 0)
    out[valid] = image[valid] / factor[valid]
    return out


def detector_correction_factor(inputs: DetectorCorrectionInputs, k0: float) -> np.ndarray:
    """Build the MATLAB-style total detector correction factor map."""

    shape = tuple(int(v) for v in inputs.data_size)
    if np.asarray(inputs.q_map).shape != shape:
        raise ValueError("q_map shape must match data_size.")

    total = solid_angle(shape, inputs.beam_position, inputs.pixel_size_mm, inputs.sdd_mm)
    total *= polarization(shape, inputs.beam_position, inputs.pixel_size_mm, inputs.sdd_mm)

    if inputs.kapton_attenuation is not None:
        total *= kapton_efficiency(
            inputs.q_map,
            inputs.kapton_attenuation,
            k0,
            inputs.energy_kev,
            shape,
            inputs.beam_position,
            inputs.pixel_size_mm,
        )

    if str(inputs.detector_type) == "Eiger4M" and inputs.silicon_attenuation is not None:
        total *= silicon_efficiency(inputs.q_map, k0, inputs.silicon_attenuation, inputs.energy_kev)

    return total.astype(np.float32, copy=False)


def solid_angle(
    data_size: tuple[int, int],
    beam_position: tuple[float, float],
    pixel_size_mm: float,
    sdd_mm: float,
) -> np.ndarray:
    """Solid-angle correction normalized to max 1, matching MATLAB indexing."""

    y, z = _pixel_grid(data_size)
    ry = (y - float(beam_position[0])) * float(pixel_size_mm)
    rz = (float(beam_position[1]) - z) * float(pixel_size_mm)
    r = np.hypot(ry, rz)
    distance = np.sqrt(r * r + float(sdd_mm) ** 2)
    theta = np.arctan(r / float(sdd_mm))
    domega = float(pixel_size_mm) * float(pixel_size_mm) * np.cos(theta) / (distance * distance)
    finite = np.isfinite(domega)
    if np.any(finite):
        domega = domega / np.nanmax(domega)
    return domega.astype(np.float32)


def polarization(
    data_size: tuple[int, int],
    beam_position: tuple[float, float],
    pixel_size_mm: float,
    sdd_mm: float,
    horizontal_fraction: float = 1.0,
) -> np.ndarray:
    """Horizontal-source polarization correction."""

    y, z = _pixel_grid(data_size)
    ry = (y - float(beam_position[0])) * float(pixel_size_mm)
    rz = (float(beam_position[1]) - z) * float(pixel_size_mm)
    gamma = np.arctan(ry / float(sdd_mm))
    delta = np.arctan(rz / np.sqrt(float(sdd_mm) ** 2 + ry * ry))
    ch = 1.0 - (np.cos(delta) ** 2) * (np.sin(gamma) ** 2)
    cv = np.cos(delta) ** 2
    pol = float(horizontal_fraction) * ch + (1.0 - float(horizontal_fraction)) * cv
    return pol.astype(np.float32)


def kapton_efficiency(
    q_map: np.ndarray,
    attenuation_table: np.ndarray,
    k0: float,
    energy_kev: float,
    data_size: tuple[int, int],
    beam_position: tuple[float, float],
    pixel_size_mm: float,
    curve_radius_mm: float = 250.0,
    thickness_mm: float = 0.50,
) -> np.ndarray:
    """Kapton transmission efficiency normalized to max 1."""

    attenuation_length_cm = _attenuation_length(attenuation_table, energy_kev)
    theta = _safe_scattering_angle(q_map, k0)
    y, z = _pixel_grid(data_size)
    det = np.hypot(y - float(beam_position[0]), float(beam_position[1]) - z)
    phia = np.arctan(det * float(pixel_size_mm) / float(curve_radius_mm))
    phib = np.arctan(det * float(pixel_size_mm) / (float(curve_radius_mm) + float(thickness_mm)))
    psi = np.arctan(float(curve_radius_mm) * np.abs(phia - phib) / float(thickness_mm))
    attenuation_length_mm = attenuation_length_cm * 10.0
    c1 = np.exp(-float(thickness_mm) / np.cos(theta + psi) / attenuation_length_mm)
    return _normalize_max(c1)


def silicon_efficiency(
    q_map: np.ndarray,
    k0: float,
    attenuation_table: np.ndarray,
    energy_kev: float,
    thickness_mm: float = 0.45,
) -> np.ndarray:
    """Eiger silicon chip attenuation correction normalized to min 1."""

    attenuation_length_cm = _attenuation_length(attenuation_table, energy_kev)
    theta = _safe_scattering_angle(q_map, k0)
    attenuation_length_mm = attenuation_length_cm * 10.0
    c2 = 1.0 - np.exp(-float(thickness_mm) / np.cos(theta) / attenuation_length_mm)
    return _normalize_min(c2)


def _pixel_grid(data_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = int(data_size[0]), int(data_size[1])
    y, z = np.meshgrid(
        np.arange(1, cols + 1, dtype=np.float32),
        np.arange(1, rows + 1, dtype=np.float32),
    )
    return y, z


def _attenuation_length(table: np.ndarray, energy_kev: float) -> float:
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("attenuation_table must have at least two columns: energy, attenuation_length.")
    idx = int(np.argmin(np.abs(arr[:, 0] - round(float(energy_kev), 1))))
    value = float(arr[idx, 1])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("attenuation length must be finite and positive.")
    return value


def _safe_scattering_angle(q_map: np.ndarray, k0: float) -> np.ndarray:
    q = np.asarray(q_map, dtype=np.float64)
    arg = np.clip(q / (2.0 * float(k0)), -1.0, 1.0)
    return 2.0 * np.arcsin(arg)


def _normalize_max(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(out)
    if np.any(finite):
        scale = np.nanmax(out)
        if np.isfinite(scale) and scale != 0:
            out = out / scale
    return out.astype(np.float32)


def _normalize_min(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(out)
    if np.any(finite):
        scale = np.nanmin(out[finite])
        if np.isfinite(scale) and scale != 0:
            out = out / scale
    return out.astype(np.float32)
