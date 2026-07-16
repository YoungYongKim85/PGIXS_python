from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgixs_tab1.averaging.qbin import mean_by_bin
from pgixs_tab1.calibration.sdd import compute_sdd_from_ring


def main() -> int:
    result = compute_sdd_from_ring(pixel_radius=200, pixel_size_mm=0.075, energy_kev=9.5, q_value=0.1052)
    if not (result.sdd > 0 and result.wavelength > 0 and result.two_theta_deg > 0):
        raise AssertionError("SDD smoke calculation failed.")

    mean, counts = mean_by_bin(
        np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float32),
        np.array([0, 0, 1, 1]),
        2,
    )
    if not np.allclose(mean, [1.0, 4.0], equal_nan=True):
        raise AssertionError("Q-bin smoke mean mismatch.")
    if not np.array_equal(counts, [1, 2]):
        raise AssertionError("Q-bin smoke count mismatch.")

    print("PGIXS Tab1 Python smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
