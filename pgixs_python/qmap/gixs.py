"""Grazing-incidence scattering Q-map calculations."""

from __future__ import annotations

import numpy as np

from pgixs_python.qmap.trxs import _validate_inputs


def qmap_gixs(
    det_y: int,
    det_z: int,
    beam_pos: tuple[float, float],
    pixel_mm: float,
    sdd_mm: float,
    k0: float,
    incident_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute GI Q maps using the MATLAB `Qmaping_gi` convention."""

    det_y_px, det_z_px, beam, pixel, sdd, wave_number = _validate_inputs(
        det_y, det_z, beam_pos, pixel_mm, sdd_mm, k0
    )
    ai = np.float32(incident_angle_deg)

    y, z = np.meshgrid(
        np.arange(1, det_y_px + 1, dtype=np.float32),
        np.arange(1, det_z_px + 1, dtype=np.float32),
    )

    det_y_rel = y - beam[0]
    det_z_rel = beam[1] - z
    sdd_pix = sdd / pixel
    dist_total_proj = np.hypot(sdd_pix, det_y_rel)

    tth = np.degrees(np.arctan2(det_y_rel * pixel, sdd))
    af = np.degrees(np.arctan2(det_z_rel, dist_total_proj))

    cos_tth = np.cos(np.radians(tth))
    sin_tth = np.sin(np.radians(tth))
    cos_af = np.cos(np.radians(af))
    sin_af = np.sin(np.radians(af))

    kfx = wave_number * cos_tth * cos_af
    kfy = wave_number * sin_tth * cos_af
    kfz = wave_number * sin_af

    qx_lab = kfx - wave_number
    qy_lab = kfy
    qz_lab = kfz

    c_ai = np.cos(np.radians(ai))
    s_ai = np.sin(np.radians(ai))

    qx = c_ai * qx_lab + s_ai * qz_lab
    qy = qy_lab
    qz = -s_ai * qx_lab + c_ai * qz_lab

    qr = np.sign(qy) * np.hypot(qx, qy)
    q = np.hypot(qr, qz)

    af2d = np.degrees(np.arcsin(np.clip(qz / wave_number - s_ai, -1.0, 1.0)))
    tth2d = np.degrees(np.arcsin(np.clip(qy_lab / wave_number / cos_af, -1.0, 1.0)))
    eps = np.finfo(np.float32).eps
    af2d = np.where(af2d == 0, eps, af2d)
    tth2d = np.where(tth2d == 0, eps, tth2d)

    return (
        qy.astype(np.float32),
        qr.astype(np.float32),
        qz.astype(np.float32),
        q.astype(np.float32),
        tth2d.astype(np.float32),
        af2d.astype(np.float32),
    )
