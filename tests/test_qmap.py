import numpy as np

from pgixs_python.qmap.gixs import qmap_gixs
from pgixs_python.qmap.trxs import qmap_trxs


def test_qmap_trxs_center_is_zero():
    qx, qy, qz, q = qmap_trxs(det_y=5, det_z=5, beam_pos=(3, 3), pixel_mm=0.075, sdd_mm=300, k0=5.0)

    assert q.shape == (5, 5)
    assert qx.dtype == np.float32
    assert qx[2, 2] == 0
    assert qy[2, 2] == 0
    assert qz[2, 2] == 0
    assert q[2, 2] == 0


def test_qmap_trxs_sign_convention():
    _, qy, qz, _ = qmap_trxs(det_y=5, det_z=5, beam_pos=(3, 3), pixel_mm=0.075, sdd_mm=300, k0=5.0)

    assert qy[2, 4] > 0
    assert qy[2, 0] < 0
    assert qz[0, 2] > 0
    assert qz[4, 2] < 0


def test_qmap_gixs_shape_and_center_offset():
    qy, qr, qz, q, tth2d, af2d = qmap_gixs(
        det_y=5,
        det_z=5,
        beam_pos=(3, 3),
        pixel_mm=0.075,
        sdd_mm=300,
        k0=5.0,
        incident_angle_deg=0.2,
    )

    assert q.shape == (5, 5)
    assert qy.dtype == np.float32
    assert qr[2, 2] == 0
    assert qz[2, 2] == 0
    assert q[2, 2] == 0
    assert tth2d[2, 2] > 0
    assert af2d[2, 2] < 0
