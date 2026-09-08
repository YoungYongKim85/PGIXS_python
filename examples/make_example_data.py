"""Generate a small synthetic dataset so the quickstart can be followed without
beamline data.

    python examples/make_example_data.py

Writes into `examples/data/` (git-ignored):

    calibration/agbeh_0001.h5          silver behenate powder rings
    gixs_series/example_gi_0001.h5     three grazing-incidence frames
    gixs_series/example_gi_0002.h5     with a temperature ramp
    gixs_series/example_gi_0003.h5

The frames use the real Eiger4M geometry (2167 x 2070 pixels, 75 um pitch) and
the same HDF5 layout the 3C beamline writes, so they load through exactly the
same path as real measurements -- including the detector mask, which is what
puts the module gaps in the images.

These are simulated patterns, not measurements. They exist so that a new user
(or a reviewer) can exercise the whole workflow and check that the numbers
that come back are the ones that went in: the calibration frame is built at a
known sample-detector distance, so running the SDD calibration on it should
return that distance.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# --- the geometry the frames are built at -----------------------------------
DETECTOR_SHAPE = (2167, 2070)   # rows, columns -- Eiger4M, as written by the beamline
PIXEL_MM = 0.075
SDD_MM = 220.0                  # what the SDD calibration should recover
WAVELENGTH_A = 1.1979
# 1-based, as every PGIXS calibration file stores it. Deliberately clear of
# the Eiger4M module gaps: the mask has a dead column band at 1029-1041, so a
# beam centre there would put every vertical cut straight down a dead column.
# Row 1600 sits above the 1616-1654 gap band, leaving ~500 px of clean
# detector above the beam before the next one (1065-1103).
BEAM_COL = 900.0
BEAM_ROW = 1600.0
INCIDENT_ANGLE_DEG = 0.12

AGBEH_Q1 = 0.1052               # silver behenate first order, 1/A
N_ORDERS = 8

OUT_DIR = Path(__file__).resolve().parent / "data"


def radius_for_q(q: np.ndarray | float) -> np.ndarray | float:
    """Pixel radius from the beam centre at which a given |q| lands."""

    sin_theta = np.asarray(q) * WAVELENGTH_A / (4.0 * np.pi)
    two_theta = 2.0 * np.arcsin(np.clip(sin_theta, -1.0, 1.0))
    return SDD_MM * np.tan(two_theta) / PIXEL_MM


def pixel_grid() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = DETECTOR_SHAPE
    col_idx, row_idx = np.meshgrid(
        np.arange(1, cols + 1, dtype=np.float32),
        np.arange(1, rows + 1, dtype=np.float32),
    )
    return col_idx - BEAM_COL, BEAM_ROW - row_idx   # +y right, +z up


def write_frame(path: Path, image: np.ndarray, counter1: int, timestamp: float,
                temperature: float) -> None:
    """One HDF5 frame in the layout `load_h5_frame` expects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.clip(image, 0.0, None)
    rng = np.random.default_rng(int(counter1))
    counts = rng.poisson(counts).astype(np.uint32)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("/entry/data/data", data=counts, compression="gzip", compression_opts=6)
        params = h5.create_group("/entry/instrument/user/params")
        # Counter1/Counter2 are the ion chamber and beam counter the
        # Normalize (IC/BC) options divide by; TimeStamp drives the Time axis.
        params.create_dataset("Counter1", data=np.array([counter1], dtype=np.int64))
        params.create_dataset("Counter2", data=np.array([counter1 // 2], dtype=np.int64))
        params.create_dataset("TimeStamp", data=np.array([timestamp], dtype=np.float64))
        params.create_dataset("Temp1", data=np.array([temperature], dtype=np.float32))
        params.create_dataset("Temp2", data=np.array([temperature - 2.0], dtype=np.float32))


def calibration_frame() -> np.ndarray:
    """Silver behenate: concentric powder rings at q = n * 0.1052 1/A."""

    dy, dz = pixel_grid()
    r = np.hypot(dy, dz)

    image = np.full(DETECTOR_SHAPE, 4.0, dtype=np.float32)
    for order in range(1, N_ORDERS + 1):
        r0 = float(radius_for_q(AGBEH_Q1 * order))
        if r0 > 1.05 * float(r.max()):
            break
        # Rings weaken with order, roughly as real powder patterns do.
        amplitude = 9000.0 / order**1.6
        image += amplitude * np.exp(-((r - r0) ** 2) / (2.0 * 2.2**2))

    # Direct beam and beamstop shadow.
    image += 60000.0 * np.exp(-(r**2) / (2.0 * 6.0**2))
    image *= 1.0 - 0.98 * np.exp(-(r**2) / (2.0 * 22.0**2))
    return image


def gixs_frame(frame_index: int) -> np.ndarray:
    """A grazing-incidence pattern: a horizon, an out-of-plane Bragg rod that
    grows across the series, and an in-plane reflection that does not."""

    dy, dz = pixel_grid()
    k0 = 2.0 * np.pi / WAVELENGTH_A
    # Small-angle reciprocal-space coordinates, with the incidence angle
    # shifting qz the way real GI geometry does.
    d_q = k0 * PIXEL_MM / SDD_MM
    qxy = dy * d_q
    qz = dz * d_q + k0 * np.sin(np.radians(INCIDENT_ANGLE_DEG))

    image = np.full(DETECTOR_SHAPE, 3.0, dtype=np.float32)

    # Yoneda / horizon band just above the direct beam.
    image += 900.0 * np.exp(-((dz - 8.0) ** 2) / (2.0 * 5.0**2))

    # Out-of-plane rod at qxy ~ 0, growing with frame index (the "ordering"
    # a temperature ramp would drive).
    growth = 1.0 + 1.4 * frame_index
    image += growth * 2600.0 * np.exp(-(qxy**2) / (2.0 * 0.012**2)) \
        * np.exp(-((qz - 0.42) ** 2) / (2.0 * 0.020**2))

    # In-plane reflections at fixed |qxy|, constant across the series.
    for sign in (-1.0, 1.0):
        image += 1800.0 * np.exp(-((qxy - sign * 0.38) ** 2) / (2.0 * 0.013**2)) \
            * np.exp(-((qz - 0.10) ** 2) / (2.0 * 0.035**2))

    # Diffuse background falling with |q|.
    q = np.hypot(qxy, qz)
    image += 260.0 * np.exp(-q / 0.45)

    r = np.hypot(dy, dz)
    image += 40000.0 * np.exp(-(r**2) / (2.0 * 5.0**2))
    image *= 1.0 - 0.98 * np.exp(-(r**2) / (2.0 * 20.0**2))
    return image


def main() -> int:
    print(f"detector      : {DETECTOR_SHAPE[0]} x {DETECTOR_SHAPE[1]}, {PIXEL_MM} mm pixels")
    print(f"geometry      : SDD {SDD_MM} mm, lambda {WAVELENGTH_A} A, beam ({BEAM_COL}, {BEAM_ROW})")
    print(f"AgBeh order 1 : q = {AGBEH_Q1} 1/A  ->  r = {radius_for_q(AGBEH_Q1):.1f} px")

    cal_path = OUT_DIR / "calibration" / "agbeh_0001.h5"
    write_frame(cal_path, calibration_frame(), counter1=100000, timestamp=0.0, temperature=25.0)
    print(f"wrote {cal_path.relative_to(OUT_DIR.parent)}  "
          f"({cal_path.stat().st_size / 1e6:.1f} MB)")

    for i in range(3):
        path = OUT_DIR / "gixs_series" / f"example_gi_{i + 1:04d}.h5"
        write_frame(path, gixs_frame(i), counter1=100000 - 500 * i,
                    timestamp=60.0 * i, temperature=25.0 + 30.0 * i)
        print(f"wrote {path.relative_to(OUT_DIR.parent)}  "
              f"({path.stat().st_size / 1e6:.1f} MB)")

    print("\nNext: docs/quickstart.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
