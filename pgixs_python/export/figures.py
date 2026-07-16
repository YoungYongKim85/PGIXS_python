"""Publication-style figure export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pgixs_python.export.data_export import OneDExport


def save_1d_figure(
    file_path: str | Path,
    data: OneDExport,
    xscale: str = "log",
    yscale: str = "log",
    dpi: int = 300,
) -> Path:
    """Save a simple 1D curve figure using matplotlib when available."""

    plt = _import_pyplot()
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    x = np.asarray(data.x, dtype=np.float32)
    y = np.asarray(data.intensity, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[valid], y[valid], color="black", linewidth=1.2)
    ax.set_xlabel(data.x_label)
    ax.set_ylabel(data.intensity_label)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(direction="in", which="both")
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for figure export.") from exc
    return plt
