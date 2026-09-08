"""Compare the CSVs run_matlab.m and run_python.py wrote for the same config.

    python tools/matlab_crosscheck/compare.py tools/matlab_crosscheck/config.example.json
    python tools/matlab_crosscheck/compare.py <config> --plot compare.png

Prints a per-reduction report and optionally renders an overlay + ratio figure.

Reading the report
------------------
`peak positions` is the real correctness check: it depends on beam centre,
SDD, energy and the detector corrections, and the two sides must agree to well
under one bin. `signed median` catches a systematic scale or offset error --
it should sit near zero. `|rel| median` is expected to be *nonzero* and
sometimes large, because MATLAB and Python do not use the same estimator or
the same sampling geometry (see README.md); it is a noise measure, not an
error measure. Judge a run by the first two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", comments="#"))


def resample_onto(src_x: np.ndarray, src_y: np.ndarray, dst_x: np.ndarray) -> np.ndarray:
    """Put the MATLAB curve on Python's x axis.

    Bins when the source is denser than the destination (MATLAB's 0.2-degree
    azimuthal grid, or its per-ray line-cut samples), and falls back to linear
    interpolation for destination bins the source does not populate. This keeps
    the comparison like-for-like instead of pretending a ray sample and a
    binned average live at the same coordinate.
    """

    order = np.argsort(src_x)
    src_x, src_y = src_x[order], src_y[order]
    if dst_x.size < 2:
        return np.interp(dst_x, src_x, src_y)

    mid = (dst_x[:-1] + dst_x[1:]) / 2.0
    edges = np.concatenate([[dst_x[0] - (dst_x[1] - dst_x[0]) / 2.0], mid,
                            [dst_x[-1] + (dst_x[-1] - dst_x[-2]) / 2.0]])
    idx = np.digitize(src_x, edges) - 1
    keep = (idx >= 0) & (idx < dst_x.size) & np.isfinite(src_y)
    sums = np.bincount(idx[keep], weights=src_y[keep], minlength=dst_x.size)
    counts = np.bincount(idx[keep], minlength=dst_x.size)
    out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    empty = ~np.isfinite(out)
    if empty.any():
        finite = np.isfinite(src_y)
        out[empty] = np.interp(dst_x[empty], src_x[finite], src_y[finite])
    return out


def top_peaks(x: np.ndarray, y: np.ndarray, n: int, exclusion: float,
              min_prominence: float = 3.0) -> list[float]:
    """The n strongest well-separated maxima that are real features.

    A candidate has to rise `min_prominence` times above the local median
    (over a window a few times the exclusion width) to count. A plain
    "strongest points" search is useless here: the flat, low-count stretches
    of these curves are a few photons per pixel, so their tallest points are
    Poisson flukes that land in different places on the two sides by
    construction and say nothing about whether the geometry agrees. Judging
    against the local level rather than the global maximum also keeps this
    working on curves whose dynamic range spans several decades.
    """

    ok = np.isfinite(y)
    x, y = x[ok], y[ok].copy()
    if y.size == 0:
        return []

    # A maximum sitting on either end of the swept range has no local context
    # on one side, and both sides' ranges end at slightly different places (a
    # ray sample at exactly 180 deg vs a bin centred on 179.5), so the
    # detector-edge rise there is a sampling artifact, not a feature.
    interior = (x >= x.min() + exclusion) & (x <= x.max() - exclusion)
    if interior.any():
        x, y = x[interior], y[interior]

    window = exclusion * 4.0
    found: list[float] = []
    for _ in range(n):
        if not np.isfinite(y).any():
            break
        j = int(np.nanargmax(y))
        if y[j] == -np.inf:
            break
        near = np.abs(x - x[j]) <= window
        local = float(np.nanmedian(y[near & np.isfinite(y)])) if near.any() else 0.0
        if not (local > 0 and y[j] >= min_prominence * local):
            break
        found.append(float(x[j]))
        y[np.abs(x - x[j]) < exclusion] = -np.inf
    return sorted(found)


def report(name: str, unit: str, mx, my, px, py, peak_exclusion: float) -> None:
    print(f"\n===== {name} =====")
    print(f"  MATLAB : {mx.size:5d} points   {mx.min():.5f} .. {mx.max():.5f} {unit}"
          f"   {int(np.count_nonzero(~np.isfinite(my)))} NaN")
    print(f"  Python : {px.size:5d} points   {np.nanmin(px):.5f} .. {np.nanmax(px):.5f} {unit}"
          f"   {int(np.count_nonzero(~np.isfinite(py)))} NaN")

    m_on_p = resample_onto(mx, my, px)
    both = np.isfinite(m_on_p) & np.isfinite(py) & (np.abs(py) > 0)
    if not both.any():
        print("  no overlapping finite bins -- nothing to compare")
        return

    signed = (py[both] - m_on_p[both]) / np.abs(m_on_p[both])
    rel = np.abs(signed)
    print(f"  overlap             : {int(both.sum())} bins")
    print(f"  signed median diff  : {np.median(signed) * 100:+.3f}%   "
          f"(a real scale/offset error would show here)")
    print(f"  |rel| diff          : median {np.median(rel) * 100:.2f}%  "
          f"p90 {np.percentile(rel, 90) * 100:.2f}%  max {rel.max() * 100:.1f}%"
          f"  at {px[both][int(np.argmax(rel))]:.4f} {unit}")

    # Match each MATLAB peak to its nearest Python peak rather than pairing by
    # rank: one side ranking a shoulder above a neighbouring peak would
    # otherwise shift every pair after it and manufacture a huge fake offset.
    mp = top_peaks(mx, my, 4, peak_exclusion)
    pp = top_peaks(px, py, 4, peak_exclusion)
    print("  peak positions      :")
    if not mp or not pp:
        print("      (no peaks above the noise floor)")
    for a in mp:
        b = min(pp, key=lambda v: abs(v - a))
        note = "" if abs(b - a) <= peak_exclusion else "   <- no counterpart found"
        print(f"      MATLAB {a:9.4f}   Python {b:9.4f}   diff {abs(a - b):.5f} {unit}{note}")

    below = int(np.count_nonzero(mx < np.nanmin(px)))
    if below:
        print(f"  MATLAB-only low end : {below} samples below Python's first point")


def make_plot(panels: list[dict], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6.8 * cols, 5.4 * rows))
    gs = fig.add_gridspec(rows * 2, cols, height_ratios=[1, 0.55] * rows,
                          hspace=0.55, wspace=0.24)

    for k, panel in enumerate(panels):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(gs[r * 2, c])
        axr = fig.add_subplot(gs[r * 2 + 1, c])
        mx, my, px, py = panel["mx"], panel["my"], panel["px"], panel["py"]
        ax.plot(mx, my, "-", color="#d1495b", lw=0.9, label="MATLAB", alpha=0.85)
        ax.plot(px, py, "-", color="#1b4965", lw=0.9, label="Python", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(panel["title"], fontsize=10, pad=6)
        ax.set_ylabel("Intensity", fontsize=9)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(alpha=0.15, lw=0.5)
        ax.tick_params(labelsize=8)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = py / resample_onto(mx, my, px)
        axr.axhline(1.0, color="0.4", lw=0.7, ls="--")
        axr.plot(px, ratio, "-", color="#2a9d8f", lw=0.8)
        axr.set_yscale("log")
        axr.set_ylim(0.2, 5)
        axr.set_yticks([0.25, 0.5, 1, 2, 4])
        axr.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        axr.set_ylabel("Python / MATLAB", fontsize=8)
        axr.set_xlabel(panel["xlabel"], fontsize=9)
        axr.grid(alpha=0.15, lw=0.5)
        axr.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=11.5, y=0.985)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MATLAB and Python cross-check output.")
    parser.add_argument("config", help="The JSON config both runners used.")
    parser.add_argument("--plot", help="Optional PNG path for an overlay + ratio figure.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    reductions = cfg.get("reductions", {})

    angstrom = "1/A"
    panels: list[dict] = []
    missing: list[str] = []

    def add(stem: str, name: str, unit: str, xlabel: str, peak_exclusion: float,
            m_cols=(0, 1), p_cols=(0, 1)) -> None:
        m_path, p_path = out_dir / f"matlab_{stem}.csv", out_dir / f"python_{stem}.csv"
        if not (m_path.exists() and p_path.exists()):
            missing.append(f"{m_path.name} / {p_path.name}")
            return
        m, p = load_csv(m_path), load_csv(p_path)
        mx, my = m[:, m_cols[0]], m[:, m_cols[1]]
        px, py = p[:, p_cols[0]], p[:, p_cols[1]]
        report(name, unit, mx, my, px, py, peak_exclusion)
        panels.append({"mx": mx, "my": my, "px": px, "py": py,
                       "title": name, "xlabel": xlabel})

    line = reductions.get("line")
    if line:
        for angle in line["angles_deg"]:
            add(f"line{int(angle)}", f"1D line cut, angle = {int(angle)} deg",
                angstrom, f"q  ({angstrom})", 0.05)
    if reductions.get("azimuthal"):
        add("azimuthal", "Azimuthal", "deg", "azimuthal angle  (deg)", 15.0)
    if reductions.get("circular"):
        add("circular", "Circular average", angstrom, f"q  ({angstrom})", 0.05)

    for item in missing:
        print(f"\n[skipped] {item} -- run run_matlab.m and run_python.py first")

    if args.plot and panels:
        make_plot(panels, Path(args.plot),
                  f"PGIXS_matlab vs PGIXS_python  -  {Path(cfg['frame']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
