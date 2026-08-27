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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import PolygonSelector, RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


def _resolve_cmap(name: str):
    import matplotlib.pyplot as plt

    try:
        return plt.get_cmap(name).copy()
    except ValueError:
        return plt.get_cmap("viridis").copy()


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
        self._polygon_selector: PolygonSelector | None = None
        self._rectangle_selector: RectangleSelector | None = None

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
        aspect: str = "equal",
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
        self.stop_polygon_selector()
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
        cmap_obj = _resolve_cmap(cmap)
        cmap_obj.set_bad(color="white")

        self._image = self.ax.imshow(
            masked,
            cmap=cmap_obj,
            norm=norm,
            origin="upper",
            extent=extent,
            aspect=aspect,
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
        else:
            # NavigationToolbar2's Home button restores whichever view sits
            # at the bottom of its own nav stack -- it has no idea we just
            # replaced the plotted data via ax.imshow() directly. We never
            # go through the toolbar's own pan/zoom handlers, which are what
            # normally populate that stack, so left alone it stays empty
            # forever and Home silently does nothing. `update()` clears any
            # stale stack from a previous dataset; `push_current()` then
            # seeds it with *this* full-extent view as the new baseline.
            self.toolbar.update()
            self.toolbar.push_current()

        self.canvas.draw_idle()
        return vmin, vmax

    def render_mesh(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        cmap: str = "viridis",
        log_scale: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        font_size: int = 10,
    ) -> tuple[float, float]:
        """Render a Q/Angle-vs-series-X-axis heatmap: `z` is `(len(y),
        len(x))`. Unlike `render()`'s `imshow`+`extent` (which assumes
        evenly-spaced pixels), `pcolormesh` maps each column to its own
        `x` coordinate -- correct when the series X-axis (Time/
        Temperature) isn't evenly spaced frame to frame, matching
        MATLAB's own `pcolor(hAX, X, Y, Z)` call this ports (GIXS2_gui.m's
        HeatmapQ_gui/HeatmapAzi_gui)."""

        self.stop_polygon_selector()
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._colorbar = None
        self._overlay_artists = []
        finite = z[np.isfinite(z)]
        if vmin is None or vmax is None:
            if finite.size:
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
        masked = np.ma.masked_invalid(z)
        cmap_obj = _resolve_cmap(cmap)
        cmap_obj.set_bad(color="white")

        self._image = self.ax.pcolormesh(x, y, masked, cmap=cmap_obj, norm=norm, shading="nearest")
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

        self.toolbar.update()
        self.toolbar.push_current()

        self.canvas.draw_idle()
        return vmin, vmax

    def render_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        mode: str = "surf",
        cmap: str = "viridis",
        log_scale: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        title: str = "",
        font_size: int = 10,
    ) -> tuple[float, float]:
        """3D companion to `render_mesh` -- `mode="surf"` is a shaded
        surface (`Axes3D.plot_surface`), `mode="waterfall"` is a stack of
        filled curtains, one per `y` row, each dropping to the surface's
        own minimum (`PolyCollection` + `add_collection3d`, the standard
        matplotlib recipe for what MATLAB's own `waterfall()` draws).
        Port of GIXS2_gui.m's HeatmapQ_gui/HeatmapAzi_gui "Plot Type"
        selector's other two modes (`render_mesh` already covers
        "Heatmap"). `z` is `(len(y), len(x))`, matching `render_mesh`.

        Colors both modes by intensity via the shared `cmap`/`norm` (a
        waterfall curtain gets one color, its own row's mean -- MATLAB's
        per-vertex height-gradient shading on each curtain isn't
        reproduced, a deliberate simplification). Log scale only affects
        that color mapping, not the Z axis itself -- `Axes3D`'s own log-
        scale support is fragile with NaN/non-positive data, which this
        app's masked regions routinely have."""

        self.stop_polygon_selector()
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self._colorbar = None
        self._overlay_artists = []
        finite = z[np.isfinite(z)]
        if vmin is None or vmax is None:
            if finite.size:
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
        cmap_obj = _resolve_cmap(cmap)
        cmap_obj.set_bad(color="white")
        z_floor = float(np.nanmin(finite)) if finite.size else 0.0

        if mode == "waterfall":
            from matplotlib.collections import PolyCollection

            polys = []
            row_values = []
            for i in range(z.shape[0]):
                row = z[i]
                filled_row = np.where(np.isfinite(row), row, z_floor)
                polys.append([(x[0], z_floor), *zip(x, filled_row), (x[-1], z_floor)])
                row_values.append(float(np.nanmean(row)) if np.isfinite(row).any() else z_floor)
            poly = PolyCollection(
                polys, facecolors=cmap_obj(norm(np.asarray(row_values))), edgecolors="k", linewidths=0.4
            )
            poly.set_alpha(0.85)
            self.ax.add_collection3d(poly, zs=y, zdir="y")
            self.ax.set_xlim(float(np.min(x)), float(np.max(x)))
            self.ax.set_ylim(float(np.min(y)), float(np.max(y)))
            self.ax.set_zlim(vmin, vmax)
            self._image = ScalarMappable(norm=norm, cmap=cmap_obj)
            self._image.set_array(z)
        else:
            XX, YY = np.meshgrid(x, y)
            masked = np.ma.masked_invalid(z)
            self._image = self.ax.plot_surface(
                XX, YY, masked, cmap=cmap_obj, norm=norm, linewidth=0, antialiased=True, rstride=1, cstride=1
            )

        self._colorbar = self.figure.colorbar(self._image, ax=self.ax, shrink=0.7, pad=0.1)
        self._colorbar.ax.tick_params(labelsize=max(7, font_size - 2))

        self.ax.set_xlabel(xlabel, fontsize=font_size)
        self.ax.set_ylabel(ylabel, fontsize=font_size)
        self.ax.set_zlabel(zlabel, fontsize=font_size)
        if title:
            self.ax.set_title(title, fontsize=max(8, font_size - 1))
        self.ax.tick_params(labelsize=max(7, font_size - 2))

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

    def draw_beam_line(self, center: tuple[float, float], angle_deg: float, bidirectional: bool = True) -> None:
        """Draw a line through `center` at `angle_deg`. `bidirectional=False`
        draws only the ray extending outward in that direction (angle_deg,
        not angle_deg+180) -- use this for a wedge boundary like Cakecut,
        where the region beyond center in the *opposite* direction was
        never actually selected by the cut and drawing it there is
        misleading (looks like an X/bowtie instead of a wedge/V)."""

        rad = np.radians(angle_deg)
        # Must satisfy the same convention _on_image_clicked's "line"
        # branch derives an angle FROM (atan2(-(y-beam_z), -(x-beam_y))),
        # so a redrawn line/ray at a given angle points back through
        # whatever pixel a click at that angle would have hit. Plain
        # cos(rad) is mirrored across the vertical axis relative to that
        # (confirmed: a ray "at 45 deg" landed on a pixel that reports back
        # as 135 deg) -- invisible for a *bidirectional* line at the
        # cardinal 0/90/180/270 angles this app used everywhere until
        # Cakecut's wedge boundaries needed a one-directional ray at
        # arbitrary (non-cardinal) angles, which is where the mirroring
        # first became visible.
        dx, dy = -np.cos(rad), -np.sin(rad)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        length = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        cx, cy = center
        start = 0.0 if not bidirectional else -length
        (line,) = self.ax.plot(
            [cx + dx * start, cx + dx * length],
            [cy + dy * start, cy + dy * length],
            color="white",
            linewidth=1.6,
            linestyle="--",
        )
        self._overlay_artists.append(line)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def draw_polyline(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Draw an arbitrary (possibly gapped -- NaN breaks the line)
        sequence of points, e.g. a pixel-space cut line/wedge/band edge
        mapped point-by-point into Q-space, where it may not stay straight.
        Preserves the current view like draw_beam_line, since the mapped
        path can run well outside the visible Q-range."""

        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        (line,) = self.ax.plot(xs, ys, color="white", linewidth=1.6, linestyle="--")
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

    def draw_roi_box(self, x1: float, x2: float, y1: float, y2: float, color: str, label: str) -> None:
        """Persistent colored rectangle + label overlay marking one drawn
        ROI (port of MATLAB `drawROIOverlay`/`ROIdef2_Callback`'s static
        box) -- `color` is the same one used for this ROI's curve in the
        Series Integration Viewer, so the box on the image and its curve
        are visually linked, matching the MATLAB behavior this ports."""

        left, right = sorted((x1, x2))
        bottom, top = sorted((y1, y2))
        rect = Rectangle((left, bottom), right - left, top - bottom, fill=False, edgecolor=color, linewidth=1.8)
        self.ax.add_patch(rect)
        text = self.ax.text(left, top, f" {label}", color=color, fontsize=9, fontweight="bold", va="bottom", ha="left")
        self._overlay_artists.append(rect)
        self._overlay_artists.append(text)
        self.canvas.draw_idle()

    def clear_overlays(self) -> None:
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._overlay_artists = []
        self.canvas.draw_idle()

    def start_rectangle_selector(self, on_complete, color: str = "yellow") -> None:
        """Let the user drag out a single rectangle on the image;
        `on_complete(x1, y1, x2, y2)` (data coordinates) fires once when the
        drag finishes. Used for Draw ROI's one-rectangle-per-ROI flow --
        the caller sets up a fresh selector for each subsequent ROI index,
        matching MATLAB's sequential `drawrectangle` calls in
        `ROIdef2_Callback`."""

        self.stop_rectangle_selector()

        def _handle(eclick, erelease) -> None:
            x1, y1, x2, y2 = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return
            self.stop_rectangle_selector()
            on_complete(x1, y1, x2, y2)

        self._rectangle_selector = RectangleSelector(
            self.ax, _handle, useblit=False, button=[1],
            props=dict(facecolor="none", edgecolor=color, linewidth=1.6),
            interactive=False,
        )

    def stop_rectangle_selector(self) -> None:
        if self._rectangle_selector is not None:
            self._rectangle_selector.disconnect_events()
            self._rectangle_selector = None

    def start_polygon_selector(self, on_complete) -> None:
        """Let the user click out a polygon on the image (double-click, or
        click the first vertex again, to close it) -- `on_complete` is
        called once with the list of (x, y) vertices in data coordinates.
        Used for the Circular Process mask: caller converts those vertices
        into a pixel-array boolean mask however its current display
        convention (raw vs. beam-centered) maps onto session.data."""

        self.stop_polygon_selector()

        def _handle(vertices: list) -> None:
            self.stop_polygon_selector()
            on_complete(list(vertices))

        self._polygon_selector = PolygonSelector(self.ax, onselect=_handle, useblit=False)

    def stop_polygon_selector(self) -> None:
        if self._polygon_selector is not None:
            self._polygon_selector.disconnect_events()
            self._polygon_selector = None

    def save_figure(self, path: str) -> None:
        self.figure.savefig(path, dpi=150, facecolor="white")

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
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self._data_x: np.ndarray = np.empty(0)
        self._data_y: np.ndarray = np.empty(0)
        self._hover_marker = None
        self._hover_text = None
        self._overlay_artists: list = []
        self._colorbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._hide_hover)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _reset_axes(self) -> None:
        """Full figure clear + recreate `self.ax`, rather than
        `self.ax.clear()` -- a colorbar (added by `render_series`) owns a
        second axes appended to the figure, and `ax.clear()` alone leaves
        it behind as a stale orphan on the next plain `render()` call.
        Same reasoning/approach as `ImageCanvas.render`."""

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._colorbar = None
        self._overlay_artists = []

    def render(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        xlabel: str = "",
        ylabel: str = "Intensity, arb. units",
        title: str = "",
        xlog: bool = False,
        ylog: bool = False,
    ) -> None:
        self._reset_axes()
        # Explicit float64 (not just np.asarray(x)) matters when a caller
        # hands us JSON-style data where masked/non-finite points were
        # serialized as None instead of NaN (e.g. preview.py's
        # downsample_1d_for_json, built for the web UI's JSON transport) --
        # np.asarray alone would silently produce an object-dtype array,
        # and indexing into it in _on_motion crashes float(None). Casting
        # to float64 here converts None to NaN instead, same as any other
        # non-finite point.
        self._data_x = np.asarray(x, dtype=np.float64)
        self._data_y = np.asarray(y, dtype=np.float64)
        self.ax.plot(self._data_x, self._data_y, color="#1d5c9e", linewidth=1.4)
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        if title:
            self.ax.set_title(title, fontsize=9)
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(labelsize=8)
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
        # set_scales after the plot exists so it can silently drop
        # non-positive points from a log axis the same way on first render
        # as on a later live toggle.
        self.set_scales(xlog, ylog)
        # Same reasoning as ImageCanvas.render(): we replot via ax.clear()
        # + ax.plot() directly rather than through the toolbar's own
        # pan/zoom handlers, so its nav stack never otherwise learns about
        # this new curve -- reset it here so Home/back always return to
        # *this* curve's full extent instead of a stale previous one.
        self.toolbar.update()
        self.toolbar.push_current()

    def render_series(
        self,
        x: np.ndarray,
        y_matrix: np.ndarray,
        color_values: np.ndarray,
        *,
        xlabel: str = "",
        ylabel: str = "Intensity, arb. units",
        colorbar_label: str = "",
        cmap: str = "viridis",
        xlog: bool = False,
        ylog: bool = False,
        max_curves: int = 200,
    ) -> None:
        """Plot every row of `y_matrix` (one curve per series frame) against
        the shared `x` axis, each colored by its own `color_values[i]` on a
        continuous colormap with a colorbar -- port of GIXS2_gui.m's
        `update1Dplot_Callback` (which color-codes by frame index through a
        raw `jet` colormap and a colorbar re-labeled with the X-axis
        variable's name/unit). This uses the real `color_values` on a
        continuous `Normalize` scale instead, so the colorbar's ticks are
        directly the Time/Temperature/Angle values rather than needing a
        separate index-to-value tick remap.

        `max_curves` downsamples evenly across the series (matches
        MATLAB's own `MAX_CURVES` guard) since a real series can run to
        thousands of frames -- plotting all of them is both unreadable and
        slow.
        """

        self._reset_axes()
        self._data_x = np.asarray(x, dtype=np.float64)
        y_matrix = np.asarray(y_matrix, dtype=np.float64)
        color_values = np.asarray(color_values, dtype=np.float64)
        n_curves = y_matrix.shape[0]

        if n_curves > max_curves:
            sel = np.unique(np.round(np.linspace(0, n_curves - 1, max_curves)).astype(np.int64))
        else:
            sel = np.arange(n_curves)

        finite_colors = color_values[np.isfinite(color_values)]
        if finite_colors.size:
            vmin, vmax = float(np.min(finite_colors)), float(np.max(finite_colors))
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = _resolve_cmap(cmap)

        self._data_y = y_matrix[sel[-1]] if sel.size else np.empty(0)
        for i in sel:
            color = cmap_obj(norm(color_values[i])) if np.isfinite(color_values[i]) else "#888888"
            self.ax.plot(self._data_x, y_matrix[i], color=color, linewidth=1.0)

        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(labelsize=8)
        (self._hover_marker,) = self.ax.plot(
            [], [], marker="+", markersize=10, markeredgewidth=2, color="#ff5a1f", linestyle="None", zorder=5
        )
        self._hover_marker.set_visible(False)
        self._hover_text = self.ax.text(
            0.98, 0.96, "", transform=self.ax.transAxes, ha="right", va="top", fontsize=8,
            color="#ff5a1f", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ff5a1f", alpha=0.85),
            visible=False,
        )
        self.set_scales(xlog, ylog)

        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        self._colorbar = self.figure.colorbar(sm, ax=self.ax)
        self._colorbar.set_label(colorbar_label, fontsize=9)
        self._colorbar.ax.tick_params(labelsize=8)

        self.toolbar.update()
        self.toolbar.push_current()

    def render_multi(
        self,
        x: np.ndarray,
        y_matrix: np.ndarray,
        labels: list[str],
        colors: list[str] | None = None,
        *,
        xlabel: str = "",
        ylabel: str = "Intensity, arb. units",
        xlog: bool = False,
        ylog: bool = False,
    ) -> None:
        """Plot several explicitly-labeled/colored curves at once with a
        legend -- for GIXS2's ROI Process, where each ROI has its own fixed
        color (matched to its rectangle overlay on the 2D image, see
        `ImageCanvas.draw_roi_box`) rather than a continuous colormap by
        series X-axis value (`render_series`). Port of MATLAB
        `PlotROI_Callback`'s per-ROI color + legend."""

        self._reset_axes()
        x = np.asarray(x, dtype=np.float64)
        y_matrix = np.asarray(y_matrix, dtype=np.float64)
        n_curves = y_matrix.shape[0]
        self._data_x = x
        self._data_y = y_matrix[0] if n_curves else np.empty(0)
        markers = ["o", "s", "^", "D", "v", "P", "X"]
        for i in range(n_curves):
            color = colors[i] if colors is not None and i < len(colors) else None
            label = labels[i] if i < len(labels) else f"Curve {i + 1}"
            self.ax.plot(x, y_matrix[i], marker=markers[i % len(markers)], markersize=5, color=color, label=label, linewidth=1.5)

        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(labelsize=8)
        if n_curves > 1:
            self.ax.legend(fontsize=8, loc="best")
        (self._hover_marker,) = self.ax.plot(
            [], [], marker="+", markersize=10, markeredgewidth=2, color="#ff5a1f", linestyle="None", zorder=5
        )
        self._hover_marker.set_visible(False)
        self._hover_text = self.ax.text(
            0.98, 0.96, "", transform=self.ax.transAxes, ha="right", va="top", fontsize=8,
            color="#ff5a1f", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ff5a1f", alpha=0.85),
            visible=False,
        )
        self.set_scales(xlog, ylog)
        self.toolbar.update()
        self.toolbar.push_current()

    def reset_view(self) -> None:
        self.toolbar.home()

    def mark_range(self, x1: float, x2: float, color: str = "b") -> None:
        """Two dashed vertical bound lines at `x1`/`x2` -- the SDD
        Calibration tab's range-select-then-fit workflow uses this to show
        which part of the curve a click-selected pair of points bounds
        (matches MATLAB `OneDplotpixel_gui.m`'s `getBounds` boundary
        lines), one color per side for the symmetric +/- fit mode."""

        (line,) = self.ax.plot([x1, x1], self.ax.get_ylim(), linestyle="--", color=color, linewidth=1.5)
        self._overlay_artists.append(line)
        (line2,) = self.ax.plot([x2, x2], self.ax.get_ylim(), linestyle="--", color=color, linewidth=1.5)
        self._overlay_artists.append(line2)
        self.canvas.draw_idle()

    def draw_vline(self, x: float, color: str = "r") -> None:
        """A single dashed vertical marker line spanning the current view
        -- e.g. the critical-angle marker on the Penetration Depth plot."""

        (line,) = self.ax.plot([x, x], self.ax.get_ylim(), linestyle="--", color=color, linewidth=1.5)
        self._overlay_artists.append(line)
        self.canvas.draw_idle()

    def overlay_line(self, x: np.ndarray, y: np.ndarray, color: str = "r") -> None:
        """Draw an arbitrary (x, y) curve on top of the current plot --
        used to show a Gaussian fit result over the data it was fit to."""

        (line,) = self.ax.plot(x, y, color=color, linewidth=2.0)
        self._overlay_artists.append(line)
        self.canvas.draw_idle()

    def clear_overlays(self) -> None:
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._overlay_artists = []
        self.canvas.draw_idle()

    def set_scales(self, xlog: bool, ylog: bool) -> None:
        """Toggle X/Y between linear and log without touching the plotted
        data -- lets ColorScale-style controls apply live, matching the
        Image Controller's ColorScale/Colormap. Points that are <= 0 simply
        don't appear on a log axis (matplotlib's normal behavior); real
        detector counts/q values are never negative, so this only ever
        drops zero-intensity or (rare) beam-center q=0 points, not silently
        distorting the rest of the curve."""

        self.ax.set_xscale("log" if xlog else "linear")
        self.ax.set_yscale("log" if ylog else "linear")
        self.canvas.draw_idle()

    def clear(self) -> None:
        self._reset_axes()
        self._data_x = np.empty(0)
        self._data_y = np.empty(0)
        self._hover_marker = None
        self._hover_text = None
        self.canvas.draw_idle()

    def save_figure(self, path: str) -> None:
        self.figure.savefig(path, dpi=150, facecolor="white")

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.point_clicked.emit(float(event.xdata), float(event.ydata))

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        self.ax.set_xlim(self._zoomed_limits(self.ax.get_xlim(), event.xdata, factor, self.ax.get_xscale() == "log"))
        self.ax.set_ylim(self._zoomed_limits(self.ax.get_ylim(), event.ydata, factor, self.ax.get_yscale() == "log"))
        self.canvas.draw_idle()

    @staticmethod
    def _zoomed_limits(
        limits: tuple[float, float], center: float, factor: float, log: bool
    ) -> tuple[float, float]:
        """Scale `limits` toward/away from `center` by `factor`, done in
        log10 space when the axis is log-scaled -- naive linear
        interpolation on a log axis's limits skews the zoom off-center and
        can walk a bound through zero/negative, which matplotlib then
        rejects outright."""

        lo, hi = limits
        if log and lo > 0 and hi > 0 and center > 0:
            log_lo, log_hi, log_c = np.log10(lo), np.log10(hi), np.log10(center)
            return (10 ** (log_c - (log_c - log_lo) * factor), 10 ** (log_c + (log_hi - log_c) * factor))
        return (center - (center - lo) * factor, center + (hi - center) * factor)

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
