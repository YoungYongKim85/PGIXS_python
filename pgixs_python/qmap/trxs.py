"""Transmission scattering Q-map calculations."""

from __future__ import annotations

import numpy as np


def qmap_trxs(det_y: int, det_z: int, beam_pos: tuple[float, float], pixel_mm: float, sdd_mm: float, k0: float):
    """Port target for MATLAB `Qmaping_tr`.

    This initial implementation keeps the public shape contract and will be
    numerically aligned against MATLAB in the next step.
    """

    y = (np.arange(det_y, dtype=np.float64) + 1.0) - float(beam_pos[0])
    z = -((np.arange(det_z, dtype=np.float64) + 1.0) - float(beam_pos[1]))
    yy, zz = np.meshgrid(y, z)
    rx = yy * float(pixel_mm)
    rz = zz * float(pixel_mm)
    rr = np.sqrt(rx**2 + rz**2 + float(sdd_mm) ** 2)
    qy = float(k0) * rx / rr
    qz = float(k0) * rz / rr
    qx = float(k0) * (float(sdd_mm) / rr - 1.0)
    q = np.sqrt(qx**2 + qy**2 + qz**2)
    return qx.astype(np.float32), qy.astype(np.float32), qz.astype(np.float32), q.astype(np.float32)

