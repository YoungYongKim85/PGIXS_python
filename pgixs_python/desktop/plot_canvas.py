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
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import PolygonSelector, RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


def _resolve_cmap(name: str):
    import matplotlib.pyplot as plt

    try:
        return plt.get_cmap(name).copy()
    except ValueError:
        return plt.get_cmap("viridis").copy()


def _log10_axis(values: np.ndarray) -> np.ndarray | None:
    """`Axes3D.set_xscale`/`set_yscale("log")` only relabels ticks -- it
    doesn't transform the actual 3D projection, so the plotted geometry
    and its "log" tick labels disagree and the surface collapses into a
    thin, off-center sliver with stray diagonal edges (a known matplotlib
    limitation, not something a render parameter can fix). Manually
    log10-transforming the coordinate *values* before plotting, then
    relabeling ticks back to the real value via a `FuncFormatter`, is the
    standard workaround and is what MATLAB's own OpenGL 3D renderer does
    correctly out of the box (`set(ax, 'XScale', 'log')` in
    HeatmapQ_gui.m/HeatmapAzi_gui.m). Returns None (caller falls back to
    linear) if any value is non-positive -- log of <=0 is undefined, same
    as a Temperature/Angle axis that dips to or below zero would give
    MATLAB's own log axis."""

    if values.size and np.all(values > 0):
        return np.log10(values)
    return None


def _log10_masked(values: np.ndarray) -> np.ndarray:
    """Elementwise log10 for a Z (intensity) array, used for Surf/
    Waterfall's Z-axis geometry when Z Scale is "Log" -- unlike
    `_log10_axis` (a coordinate array, expected uniformly positive), an
    intensity array routinely has a handful of zero/negative pixels
    (background/noise), so a single non-positive value shouldn't refuse
    the whole array; those entries just come back NaN (masked out),
    matching `LogNorm`'s own convention for non-positive color data."""

    out = np.full(values.shape, np.nan, dtype=np.float64)
    positive = values > 0
    out[positive] = np.log10(values[positive])
    return out


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
        # Surf/Waterfall's drag-to-rotate view has no built-in indicator of
        # the current orientation -- `_rotation_label` is a screen-space
        # text artist `render_surface` creates, kept in sync with the live
        # elev/azim by `_sync_rotation_label` polling on a timer rather
        # than hooking mouse events, since Axes3D's own internal rotate
        # handler (registered via `mouse_init()` when the 3D axes is
        # created) may update elev/azim before *or* after ours runs
        # depending on connection order, and a timer sidesteps that
        # entirely. Stays `None` (and the timer no-ops) for every other
        # `ImageCanvas` use in the app, which never renders a 3D axes.
        self._rotation_label = None
        self.axis_log_fallback: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._rotation_timer = QTimer(self)
        self._rotation_timer.timeout.connect(self._sync_rotation_label)
        self._rotation_timer.start(150)

    def _sync_rotation_label(self) -> None:
        if self._rotation_label is None or self.ax.name != "3d":
            return
        text = f"Elev {self.ax.elev:.0f}°, Azim {self.ax.azim:.0f}°"
        if self._rotation_label.get_text() != text:
            self._rotation_label.set_text(text)
            self.canvas.draw_idle()

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
        # Re-enable constrained_layout in case the previous render was a
        # Surf/Waterfall (render_surface turns it off -- see its own
        # comment on why).
        self.figure.set_layout_engine("constrained")
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
        xscale: str = "linear",
        yscale: str = "linear",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Render a Q/Angle-vs-series-X-axis heatmap: `z` is `(len(y),
        len(x))`. Unlike `render()`'s `imshow`+`extent` (which assumes
        evenly-spaced pixels), `pcolormesh` maps each column to its own
        `x` coordinate -- correct when the series X-axis (Time/
        Temperature) isn't evenly spaced frame to frame, matching
        MATLAB's own `pcolor(hAX, X, Y, Z)` call this ports (GIXS2_gui.m's
        HeatmapQ_gui/HeatmapAzi_gui). `xscale`/`yscale` ("linear"/"log")
        and `xlim`/`ylim` are that same reference's own X/Y Scale and
        X/Y Range controls -- `log_scale`/`vmin`/`vmax` stay the *color*
        (Z) scale, a separate MATLAB control."""

        self.stop_polygon_selector()
        self.figure.clear()
        # Re-enable constrained_layout in case the previous render was a
        # Surf/Waterfall (render_surface turns it off -- see its own
        # comment on why).
        self.figure.set_layout_engine("constrained")
        self.ax = self.figure.add_subplot(111)
        self._colorbar = None
        self._overlay_artists = []
        self._rotation_label = None
        self.axis_log_fallback = []
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

        # Scale before limits -- matplotlib's own recommended order, so a
        # switch to "log" doesn't first clip/reinterpret limits that were
        # set under "linear".
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

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
        color_by: str = "intensity",
        cmap: str = "viridis",
        log_scale: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        colorbar_label: str = "",
        title: str = "",
        font_size: int = 10,
        xscale: str = "linear",
        yscale: str = "linear",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """3D companion to `render_mesh` -- `mode="surf"` is a shaded
        surface (`Axes3D.plot_surface`), `mode="waterfall"` is a stack of
        filled curtains, one per `x` column (each column is one series
        frame's own y-vs-intensity curve), each dropping to the surface's
        own minimum (`PolyCollection` + `add_collection3d`, the standard
        matplotlib recipe for what MATLAB's own `waterfall()` draws).
        Port of GIXS2_gui.m's HeatmapQ_gui/HeatmapAzi_gui "Plot Type"
        selector's other two modes (`render_mesh` already covers
        "Heatmap"). `z` is `(len(y), len(x))`, matching `render_mesh`.

        `mode="surf"` always colors by intensity. `mode="waterfall"`
        additionally honors `color_by`: `"intensity"` colors each curtain
        by its own column's mean intensity (default, via the shared
        `cmap`/`norm`, same as `"surf"`); `"index"` instead colors each
        curtain by its own frame's position (0..n-1) along a *separate*
        0..n-1 scale, independent of the intensity contrast -- lets a
        long series read as a clean sequential gradient (reported: "커튼
        색을 프레임 인덱스 컬러맵으로도 선택할 수 있게 해줘") instead of
        every curtain clustering near the same color when frame-to-frame
        intensity barely varies. MATLAB's own per-vertex height-gradient
        shading on each curtain isn't reproduced either way (one solid
        color per curtain, a deliberate simplification). `log_scale`
        drives both the *color* mapping (`norm`) and the Z axis' own
        geometry (matching MATLAB's own combined `ZScale`+`ColorScale`
        control in HeatmapQ_gui.m/HeatmapAzi_gui.m) -- see the note below
        on how the Z geometry transform is done without double-logging
        the color.

        `xscale`/`yscale`/`xlim`/`ylim` are the same X/Y Scale and X/Y
        Range controls `render_mesh` takes. Unlike `render_mesh`, a "log"
        request on any of X/Y/Z here is honored via a manual `_log10_axis`/
        `_log10_masked` transform of the plotted values rather than
        `Axes3D.set_xscale`/`set_yscale`/`set_zscale` -- those only relabel
        ticks without transforming the 3D projection, corrupting the
        geometry (see `_log10_axis`'s own docstring). The Z transform
        additionally has to keep color separate from height (via an
        explicit `facecolors`/`ScalarMappable`, not `plot_surface`'s own
        norm-driven coloring) since color must come from `norm` applied to
        the *raw* intensity, not the already-logged plotted height. Check
        `axis_log_fallback` after calling if a caller needs to know a
        requested "log" axis silently stayed linear (non-positive data on
        that axis -- for X/Y a single non-positive coordinate blocks the
        whole axis; for Z, individual non-positive samples are just masked
        out and only an *entirely* non-positive Z array falls back)."""

        self.stop_polygon_selector()
        self.figure.clear()
        # `constrained_layout` (the figure's default -- see __init__)
        # doesn't handle `Axes3D` well in a wide/short dialog -- turn it
        # off for this render and position the axes manually instead;
        # `render`/`render_mesh` re-enable it for their own (2D,
        # unaffected) renders since this is a `Figure`-level setting on
        # the shared `self.figure`.
        self.figure.set_layout_engine(None)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_position((0.02, 0.08, 0.90, 0.84))
        self._colorbar = None
        self._overlay_artists = []
        self._rotation_label = None
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

        # `Axes3D.set_xscale("log")` doesn't work (see `_log10_axis`'s own
        # docstring) -- log10-transform the coordinate *values* instead and
        # relabel ticks back to the real value afterward. Falls back to
        # linear (`x_plot is x`) if a "log" request can't be honored (a
        # non-positive value on that axis), same as MATLAB's own log axis
        # would refuse a Temperature/Angle axis dipping to <= 0.
        self.axis_log_fallback = []
        x_plot, y_plot = x, y
        x_log_used = y_log_used = False
        if xscale == "log":
            transformed = _log10_axis(x)
            if transformed is not None:
                x_plot, x_log_used = transformed, True
            else:
                self.axis_log_fallback.append("X")
        if yscale == "log":
            transformed = _log10_axis(y)
            if transformed is not None:
                y_plot, y_log_used = transformed, True
            else:
                self.axis_log_fallback.append("Y")

        # `log_scale` ("Z Scale") already drives the *color* mapping above
        # (`norm`) -- it also drives the Z axis' own geometry here, the
        # same log10-transform-the-values workaround X/Y use just above
        # (Axes3D has no working native log axis; MATLAB's own ZScale +
        # ColorScale set together is what this ports -- HeatmapQ_gui.m/
        # HeatmapAzi_gui.m: `set(ax,'ZScale',zSc,'ColorScale',zSc)`).
        # Color still has to come from `norm` applied to the *raw* data,
        # not `z_plot` -- plotting already-logged heights through a
        # `LogNorm` color mapping would log it a second time.
        z_plot = z
        z_floor_plot = z_floor
        z_log_used = False
        if log_scale:
            if finite.size and np.any(finite > 0):
                z_plot = _log10_masked(z)
                z_floor_plot = float(np.log10(max(vmin, 1e-9)))
                z_log_used = True
            else:
                self.axis_log_fallback.append("Z")

        if mode == "waterfall":
            from matplotlib.collections import PolyCollection

            n_curtains = z.shape[1]
            by_index = color_by == "index"
            curtain_norm = Normalize(vmin=0, vmax=max(n_curtains - 1, 1)) if by_index else norm
            polys = []
            curtain_values = []
            for j in range(n_curtains):
                col_raw = z[:, j]
                col_plot = z_plot[:, j]
                filled_col = np.where(np.isfinite(col_plot), col_plot, z_floor_plot)
                polys.append([(y_plot[0], z_floor_plot), *zip(y_plot, filled_col), (y_plot[-1], z_floor_plot)])
                if by_index:
                    curtain_values.append(float(j))
                else:
                    curtain_values.append(float(np.nanmean(col_raw)) if np.isfinite(col_raw).any() else z_floor)
            poly = PolyCollection(
                polys, facecolors=cmap_obj(curtain_norm(np.asarray(curtain_values))), edgecolors="k", linewidths=0.4
            )
            poly.set_alpha(0.85)
            # zdir="x": each curtain's own 2D vertices (y, z) map onto the
            # axes' (y, z) plane, offset along the axes' x by `zs` -- keeps
            # x/y/z meaning the same as "surf"/"heatmap" (x=series axis,
            # y=Q/Angle, z=intensity) even though curtains are now built
            # column-by-column instead of row-by-row.
            self.ax.add_collection3d(poly, zs=x_plot, zdir="x")
            self.ax.set_xlim(float(np.min(x_plot)), float(np.max(x_plot)))
            self.ax.set_ylim(float(np.min(y_plot)), float(np.max(y_plot)))
            if z_log_used:
                self.ax.set_zlim(z_floor_plot, float(np.log10(max(vmax, 1e-9))))
            else:
                self.ax.set_zlim(vmin, vmax)
            self._image = ScalarMappable(norm=curtain_norm, cmap=cmap_obj)
            self._image.set_array(np.asarray(curtain_values))
        else:
            XX, YY = np.meshgrid(x_plot, y_plot)
            if z_log_used:
                # Height (`masked_plot`) and color (`facecolors`, from
                # `norm` applied to the raw data) come from different
                # arrays here -- see the comment above on why `norm` can't
                # just be handed to `plot_surface` directly in this case.
                masked_plot = np.ma.masked_invalid(z_plot)
                facecolors = cmap_obj(norm(np.ma.masked_invalid(z)))
                self.ax.plot_surface(
                    XX, YY, masked_plot, facecolors=facecolors, linewidth=0, antialiased=True, rstride=1, cstride=1
                )
                self._image = ScalarMappable(norm=norm, cmap=cmap_obj)
                self._image.set_array(z)
            else:
                masked = np.ma.masked_invalid(z)
                self._image = self.ax.plot_surface(
                    XX, YY, masked, cmap=cmap_obj, norm=norm, linewidth=0, antialiased=True, rstride=1, cstride=1
                )

        # `figure.colorbar(..., ax=self.ax, shrink=..., pad=...)` (the
        # simple form `render_mesh` can't use for a 3D axes) shrinks and
        # *reanchors* the parent axes to carve out room for the colorbar --
        # its default anchor for that reanchor is the parent's right edge,
        # not its center, which is what actually pushed the whole 3D box
        # off to one side (reported: "surf와 waterfall의 이미지가 가운데
        # 있지 않고"). An explicit `cax` in its own fixed-position axes
        # sidesteps that reanchor entirely -- `self.ax` keeps exactly the
        # position set above.
        cax = self.figure.add_axes((0.90, 0.15, 0.02, 0.70))
        self._colorbar = self.figure.colorbar(self._image, cax=cax)
        self._colorbar.ax.tick_params(labelsize=max(7, font_size - 2))
        if colorbar_label:
            self._colorbar.set_label(colorbar_label, fontsize=font_size)

        self.ax.set_xlabel(xlabel, fontsize=font_size)
        self.ax.set_ylabel(ylabel, fontsize=font_size)
        self.ax.set_zlabel(zlabel, fontsize=font_size)
        if title:
            self.ax.set_title(title, fontsize=max(8, font_size - 1))
        self.ax.tick_params(labelsize=max(7, font_size - 2))

        if x_log_used:
            self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{10.0 ** val:.3g}"))
        if y_log_used:
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{10.0 ** val:.3g}"))
        if z_log_used:
            self.ax.zaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{10.0 ** val:.3g}"))

        # `xlim`/`ylim` (from the caller's X/Y Range fields) are always in
        # real units -- transform them into the same log10 space as the
        # plotted geometry when that axis is log, otherwise the requested
        # range would silently clip against transformed data.
        if xlim is not None:
            lo, hi = xlim
            if x_log_used:
                if lo > 0 and hi > 0:
                    self.ax.set_xlim(float(np.log10(lo)), float(np.log10(hi)))
            else:
                self.ax.set_xlim(xlim)
        if ylim is not None:
            lo, hi = ylim
            if y_log_used:
                if lo > 0 and hi > 0:
                    self.ax.set_ylim(float(np.log10(lo)), float(np.log10(hi)))
            else:
                self.ax.set_ylim(ylim)

        self._rotation_label = self.ax.text2D(
            0.02, 0.96, f"Elev {self.ax.elev:.0f}°, Azim {self.ax.azim:.0f}°",
            transform=self.ax.transAxes, fontsize=max(7, font_size - 1), color="dimgray",
        )

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
