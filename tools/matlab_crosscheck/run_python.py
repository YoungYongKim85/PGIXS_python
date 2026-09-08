"""Reduce one real frame with PGIXS_python, through the normal console API.

    python tools/matlab_crosscheck/run_python.py tools/matlab_crosscheck/config.example.json

Reads the same JSON config `run_matlab.m` reads and writes CSVs with the same
column layout into the same output directory, so `compare.py` can read both.
Everything goes through `console_session`'s public entry points -- the point of
the cross-check is to exercise the code the app actually runs, not a
reimplementation of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Running this as a script puts tools/matlab_crosscheck on sys.path, not the
# repo root, so the package would not import unless it happens to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pgixs_python.ui_backend.console_session as cs  # noqa: E402

# Same h*c PGIXS_python uses everywhere (see calibration/sdd.py). MATLAB's own
# constant differs by 7.8 ppm; run_matlab.m explains why both sides convert
# from energy separately instead of sharing a wavelength.
HC_KEV_ANGSTROM = 12.398419843320026


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PGIXS_python reductions for the MATLAB cross-check.")
    parser.add_argument("config", help="JSON config, shared with run_matlab.m.")
    return parser


def save(path: Path, header: str, columns: list[np.ndarray]) -> None:
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, fmt="%.10g")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = Path(cfg["frame"])
    mode = str(cfg["mode"]).lower()
    beam = (float(cfg["beam_center"][0]), float(cfg["beam_center"][1]))
    energy_kev = HC_KEV_ANGSTROM / float(cfg["wavelength_a"])

    cs.reset_session()
    loaded = cs.console_load({
        "directory": str(frame.parent),
        "filename": frame.name,
        "detector": cfg.get("detector", "Eiger4M"),
        "mode": mode,
    })
    if not loaded.get("ok"):
        raise SystemExit(f"load failed: {loaded.get('message')}")

    session = cs.get_session()
    print(f"frame         : {frame}")
    print(f"data shape    : {session.data.shape}")
    print(f"energy        : {energy_kev:.10f} keV")

    # Load Info normally fills the calibration dict from a .mat/.std/.json;
    # here the numbers come from the config so both sides provably use the
    # same ones, so just mark the step done and pass them to q-Map Cal.
    session.calibration_loaded = True
    qmap = cs.console_qmap({
        "energy_kev": energy_kev,
        "sdd_mm": float(cfg["sdd_mm"]),
        "beam_y": beam[0],
        "beam_z": beam[1],
        "incident_angle_deg": float(cfg.get("incident_angle_deg", 0.1)),
    })
    if not qmap.get("ok"):
        raise SystemExit(f"q-Map Cal. failed: {qmap.get('message')}")

    reductions = cfg.get("reductions", {})

    line = reductions.get("line")
    if line:
        for angle in line["angles_deg"]:
            result = cs.console_line_1d({"angle_deg": float(angle), "mapping": "Pixel"})
            if not result.get("ok"):
                raise SystemExit(f"line cut {angle} deg failed: {result.get('message')}")
            curve = cs.get_session().last_curve
            q = np.asarray(curve["q"], dtype=float)
            i = np.asarray(curve["intensity"], dtype=float)
            save(out_dir / f"python_line{int(angle)}.csv", "q,intensity,coverage",
                 [q, i, np.asarray(curve["coverage"], dtype=float)])
            print(f"line ang={int(angle):3d}  : {q.size} q bins, "
                  f"{int(np.count_nonzero(~np.isfinite(i)))} NaN, "
                  f"q {np.nanmin(q):.6f} .. {np.nanmax(q):.6f}")

    azimuthal = reductions.get("azimuthal")
    if azimuthal:
        result = cs.console_azimuthal_1d({
            "pixel": float(azimuthal["pixel"]),
            "range_minus": float(azimuthal.get("range_minus", -1.0)),
            "range_plus": float(azimuthal.get("range_plus", 1.0)),
            "mean_or_median": "Median" if azimuthal.get("use_median") else "Mean",
            "mapping": "Pixel",
        })
        if not result.get("ok"):
            raise SystemExit(f"azimuthal failed: {result.get('message')}")
        curve = cs.get_session().last_curve
        angle = np.asarray(curve["angle"], dtype=float)
        i = np.asarray(curve["intensity"], dtype=float)
        save(out_dir / "python_azimuthal.csv", "angle_deg,intensity,coverage",
             [angle, i, np.asarray(curve["coverage"], dtype=float)])
        print(f"azimuthal     : {angle.size} angles, "
              f"{int(np.count_nonzero(~np.isfinite(i)))} NaN, "
              f"deg {angle[0]:.3f} .. {angle[-1]:.3f}")

    circular = reductions.get("circular")
    if circular:
        result = cs.console_circular_average({
            "angle_min": float(circular.get("angle_min", 0.0)),
            "angle_max": float(circular.get("angle_max", 360.0)),
            "min_coverage": float(circular.get("min_coverage", 0.8)),
            "use_median": bool(circular.get("use_median", False)),
            "calc_std": True,
        })
        if not result.get("ok"):
            raise SystemExit(f"circular average failed: {result.get('message')}")
        curve = cs.get_session().last_curve
        q = np.asarray(curve["q"], dtype=float)
        i = np.asarray(curve["intensity"], dtype=float)
        save(out_dir / "python_circular.csv", "q,intensity,coverage",
             [q, i, np.asarray(curve["coverage"], dtype=float)])
        print(f"circular      : {q.size} bins, "
              f"{int(np.count_nonzero(~np.isfinite(i)))} NaN, "
              f"q {np.nanmin(q):.6f} .. {np.nanmax(q):.6f}")

    print("PYTHON DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
