"""Transmission scattering Q-map calculations."""

from __future__ import annotations

import numpy as np


def qmap_trxs(
    det_y: int,
    det_z: int,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    sdd_mm: float,
    k0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute transmission Q maps using the MATLAB `Qmaping_tr` convention."""

    det_y_px, det_z_px, beam, pixel, sdd, wave_number = _validate_inputs(
        det_y, det_z, beam_pos, pixel_mm, sdd_mm, k0
    )
    y, z = np.meshgrid(
        np.arange(1, det_y_px + 1, dtype=np.float32),
        np.arange(1, det_z_px + 1, dtype=np.float32),
    )

    det_y_rel = y - beam[0]
    det_z_rel = beam[1] - z
    sdd_pix = sdd / pixel
    dist_total_proj = np.hypot(sdd_pix, det_y_rel)

    tth = np.degrees(np.arctan2(det_y_rel, sdd_pix))
    af = np.degrees(np.arctan2(det_z_rel, dist_total_proj))

    cos_tth = np.cos(np.radians(tth))
    sin_tth = np.sin(np.radians(tth))
    cos_af = np.cos(np.radians(af))
    sin_af = np.sin(np.radians(af))

    qx = wave_number * cos_tth * cos_af - wave_number
    qy = wave_number * sin_tth * cos_af
    qz = wave_number * sin_af
    q = np.sqrt(qx**2 + qy**2 + qz**2)
    return qx.astype(np.float32), qy.astype(np.float32), qz.astype(np.float32), q.astype(np.float32)


def _validate_inputs(
    det_y: int,
    det_z: int,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    sdd_mm: float,
    k0: float,
) -> tuple[int, int, np.ndarray, np.float32, np.float32, np.float32]:
    det_y_px = int(det_y)
    det_z_px = int(det_z)
    if det_y_px <= 0 or det_z_px <= 0:
        raise ValueError("Detector dimensions must be positive.")
    pixel = np.float32(pixel_mm)
    sdd = np.float32(sdd_mm)
    wave_number = np.float32(k0)
    if pixel <= 0 or sdd <= 0 or wave_number <= 0:
        raise ValueError("pixel_mm, sdd_mm, and k0 must be positive.")
    beam = np.asarray(beam_pos, dtype=np.float32).reshape(-1)
    if beam.size != 2 or not np.all(np.isfinite(beam)):
        raise ValueError("beam_pos must contain two finite values.")
    return det_y_px, det_z_px, beam, pixel, sdd, wave_number
