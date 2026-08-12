"""matplotlib-embedded plot widgets shared by every console tab.

Using real matplotlib axes (instead of the hand-rolled canvas/LUT/colorbar/
axis-tick code the web prototype needed) gives correct axis directions,
colorbars, log-scale normalization, the full colormap registry, and
pan/zoom/save for free -- `NavigationToolbar2QT`'s Home button *is* "reset
zoom", and its own zoom-rectangle/pan tools replace the custom Y-Range/
Z-Range crop math the web version needed to hand-write.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ImageCanvas(QWidget):
    """2D detector image display: imshow + colorbar + click-to-pixel + overlay."""

    pixel_clicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # constrained_layout (not tight_layout) is what correctly centers an
        # aspect='equal' image plus its colorbar -- tight_layout shrinks the
        # axes to make room for the colorbar but doesn't recenter the pair
        # in the figure, which is exactly the "pushed to one side" symptom.
        self.figure = Figure(figsize=(5, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(280)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self._image = None
        self._colorbar = None
        self._overlay_artists: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def render(
        self,
        data: np.ndarray,
        *,
        cmap: str = "viridis",
        log_scale: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        extent: tuple[float, float, float, float] | None = None,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        font_size: int = 10,
        keep_limits: bool = False,
    ) -> tuple[float, float]:
        """Render `data` and return the (vmin, vmax) actually used -- either
        the caller's own values or the auto-computed ones, so the caller can
        keep its own contrast fields in sync without re-deriving them."""

        prev_xlim = self.ax.get_xlim() if keep_limits else None
        prev_ylim = self.ax.get_ylim() if keep_limits else None

        # Rebuild the figure from scratch each render rather than clearing
        # and reusing `self.ax` -- a colorbar owns a second axes appended to
        # the figure, and incrementally clearing/reusing the main axes while
        # trying to remove/recreate that second axes across calls is fragile
        # (matplotlib's colorbar bookkeeping can end up pointing at a stale
        # subplotspec). A full `figure.clear()` sidesteps that entirely.
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._colorbar = None
        self._overlay_artists = []
        finite = data[np.isfinite(data)]
        if vmin is None or vmax is None:
            if finite.size:
                # Real detector frames routinely carry one or a handful of
                # hot/overflow pixels many orders of magnitude above the
                # actual signal (e.g. this project's own HZO test frame has
                # a raw max of ~4.6e8 against a 99.9th-percentile of ~860) --
                # true min/max auto-contrast crushes the whole image to
                # near-black around such an outlier. Percentiles are robust
                # to that and are the standard choice for detector-image
                # auto-contrast (DAWN/pyFAI/ImageJ all default this way).
                positive = finite[finite > 0]
                sample = positive if positive.size else finite
                auto_min = float(np.percentile(sample, 1.0))
                auto_max = float(np.percentile(sample, 99.5))
            else:
                auto_min, auto_max = 0.0, 1.0
            vmin = auto_min if vmin is None else vmin
            vmax = auto_max if vmax is None else vmax
        if vmax <= vmin:
            vmax = vmin + 1.0

        norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)
        masked = np.ma.masked_invalid(data)
        cmap_obj = self._resolve_cmap(cmap)
        cmap_obj.set_bad(color="white")

        self._image = self.ax.imshow(
            masked,
            cmap=cmap_obj,
            norm=norm,
            origin="upper",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        # `figure.colorbar(im, ax=self.ax, ...)` makes constrained_layout
        # re-anchor an aspect='equal' axes to hug the colorbar -- it sets
        # ax.set_anchor((1.0, 0.5)) internally, which shoves the whole
        # image+colorbar pair to one side of the figure instead of
        # centering it. Deriving the colorbar axes from `self.ax`'s own
        # divider sidesteps that auto-realignment entirely.
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        self._colorbar = self.figure.colorbar(self._image, cax=cax)
        self._colorbar.ax.tick_params(labelsize=max(7, font_size - 2))
        self.ax.set_anchor("C")

        self.ax.set_xlabel(xlabel, fontsize=font_size)
        self.ax.set_ylabel(ylabel, fontsize=font_size)
        if title:
            self.ax.set_title(title, fontsize=max(8, font_size - 1))
        self.ax.tick_params(labelsize=max(7, font_size - 2))

        if keep_limits and prev_xlim is not None:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)

        self.canvas.draw_idle()
        return vmin, vmax

    def set_axis_limits(self, xlim: tuple[float, float] | None, ylim: tuple[float, float] | None) -> None:
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def get_axis_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.ax.get_xlim(), self.ax.get_ylim()

    def reset_view(self) -> None:
        self.toolbar.home()

    def draw_beam_line(self, center: tuple[float, float], angle_deg: float) -> None:
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), -np.sin(rad)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        length = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        cx, cy = center
        (line,) = self.ax.plot(
            [cx - dx * length, cx + dx * length],
            [cy - dy * length, cy + dy * length],
            color="white",
            linewidth=1.6,
            linestyle="--",
        )
        self._overlay_artists.append(line)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def draw_beam_circle(self, center: tuple[float, float], radius: float) -> None:
        circle = Circle(center, radius, fill=False, edgecolor="white", linewidth=1.6)
        self.ax.add_patch(circle)
        self._overlay_artists.append(circle)
        self.canvas.draw_idle()

    def draw_column_marker(self, x: float) -> None:
        """Vertical dashed line at display x-coordinate `x`, spanning the
        current view -- marks which detector column a z-plot was taken
        along."""
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        (line,) = self.ax.plot([x, x], ylim, color="#ff5a1f", linewidth=1.6, linestyle="--")
        self._overlay_artists.append(line)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def clear_overlays(self) -> None:
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._overlay_artists = []
        self.canvas.draw_idle()

    def save_figure(self, path: str) -> None:
        self.figure.savefig(path, dpi=150, facecolor="white")

    @staticmethod
    def _resolve_cmap(name: str):
        import matplotlib.pyplot as plt

        try:
            return plt.get_cmap(name).copy()
        except ValueError:
            return plt.get_cmap("viridis").copy()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.pixel_clicked.emit(float(event.xdata), float(event.ydata))

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        new_xlim = (x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        new_ylim = (y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()


class CurveCanvas(QWidget):
    """1D result curve display, reused for Line/Azimuthal/Circular results
    and for the z-plot incidence-calibration peak pick (click on the curve)."""

    point_clicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 2.6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(160)
        self.ax = self.figure.add_subplot(111)
        self._data_x: np.ndarray = np.empty(0)
        self._data_y: np.ndarray = np.empty(0)
        self._hover_marker = None
        self._hover_text = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._hide_hover)

    def render(self, x: np.ndarray, y: np.ndarray, *, xlabel: str = "", ylabel: str = "Intensity, arb. units", title: str = "") -> None:
        self.ax.clear()
        self.ax.plot(x, y, color="#1d5c9e", linewidth=1.4)
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        if title:
            self.ax.set_title(title, fontsize=9)
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(labelsize=8)

        self._data_x = np.asarray(x)
        self._data_y = np.asarray(y)
        # Marker + coordinate readout for the nearest point under the mouse
        # (see _on_motion) -- recreated after every ax.clear() above, hidden
        # until the mouse actually moves over the axes.
        (self._hover_marker,) = self.ax.plot(
            [], [], marker="+", markersize=10, markeredgewidth=2, color="#ff5a1f", linestyle="None", zorder=5
        )
        self._hover_marker.set_visible(False)
        self._hover_text = self.ax.text(
            0.98, 0.96, "", transform=self.ax.transAxes, ha="right", va="top", fontsize=8,
            color="#ff5a1f", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ff5a1f", alpha=0.85),
            visible=False,
        )

        self.canvas.draw_idle()

    def clear(self) -> None:
        self.ax.clear()
        self._data_x = np.empty(0)
        self._data_y = np.empty(0)
        self._hover_marker = None
        self._hover_text = None
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.point_clicked.emit(float(event.xdata), float(event.ydata))

    def _on_motion(self, event) -> None:
        if self._hover_marker is None or self._data_x.size == 0:
            return
        if event.inaxes != self.ax or event.xdata is None:
            self._hide_hover()
            return
        idx = int(np.argmin(np.abs(self._data_x - event.xdata)))
        px, py = float(self._data_x[idx]), float(self._data_y[idx])
        self._hover_marker.set_data([px], [py])
        self._hover_marker.set_visible(True)
        self._hover_text.set_text(f"x={px:.4g}\ny={py:.4g}")
        self._hover_text.set_visible(True)
        self.canvas.draw_idle()

    def _hide_hover(self, _event=None) -> None:
        if self._hover_marker is None:
            return
        if not self._hover_marker.get_visible() and not self._hover_text.get_visible():
            return
        self._hover_marker.set_visible(False)
        self._hover_text.set_visible(False)
        self.canvas.draw_idle()
