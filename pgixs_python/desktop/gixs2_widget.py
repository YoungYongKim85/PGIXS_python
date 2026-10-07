"""GIXS2 tab: port of MATLAB `GIXS2_gui.m`'s core series-analysis workflow,
including "Change ai" angle-scanning q-Cal and the Play (Frame Viewer)
button -- Q/Azimuthal Heatmap viewers, whole-stack save, and custom
colormaps remain deferred (see the implementation plan).

Loads a whole time/temperature/angle series into an in-memory stack and
computes per-frame 1D line-cut / azimuthal / ROI-sum curves across it,
plotted against the chosen experiment X-axis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pgixs_python.desktop.background_tasks import run_in_background
from pgixs_python.desktop.console_widget import (
    _COLORMAPS,
    _divider,
    _double_spin,
    _field,
    _muted,
    _row,
    _sci_spin,
    _size_combo,
    _split_row,
    _wrap_long_tokens,
)
from pgixs_python.desktop.plot_canvas import CurveCanvas, ImageCanvas
from pgixs_python.ui_backend import gixs2_session as gs

_DETECTOR_LIST = ["Eiger4M", "MarCCD", "MX170HS", "PI-CCD"]
_INSITU_LIST = ["Time", "Temperature", "Angle", "LowTemp."]
_SUB_MODE_ITEMS = {
    "Temperature": ["Temperature1", "Temperature2", "Temperature3", "Time"],
    "LowTemp.": ["LowTemp1", "LowTemp2", "Time"],
    "Angle": ["Angle", "Time"],
    "Time": ["Time"],
}
_NORMALIZE_LIST = ["None", "IC", "BC"]
# One color per ROI index (N. ROI tops out at 5) -- used for both the ROI's
# rectangle overlay on the 2D image and its curve in the Series Integration
# Viewer, so a box and its curve are visually linked at a glance. Port of
# MATLAB PlotROI_Callback's roiColors, swapped for a set that stays legible
# against this app's own light theme (MATLAB's 'y'/'w' entries wash out).
_ROI_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

# Shared content-length budget (see console_widget.py's own `_size_combo`
# docstring) for every Normalize/Mean-Median/Scale combo box across the
# four viewer panels and the 1D Plot Process/Azimuthal Process/ROI
# Process sections -- without it, a combo box alone on its own row (e.g.
# Mean/Median) stretches to fill the whole row width, while one sharing a
# row with an Exec button only gets what's left over, so otherwise-
# equivalent dropdowns ended up visibly different widths.
_COMBO_CHARS = 10


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    """Port of MATLAB's own `finiteRange` (HeatmapQ_gui/HeatmapAzi_gui):
    the actual min/max of `values`' finite entries, padded apart slightly
    if they'd otherwise coincide -- used to seed a Heatmap dialog's X/Y
    Range fields with real, editable bounds instead of console_widget.py's
    usual 0/0 "auto" sentinel (Contrast's own convention), since a Range
    field with nothing in it gives the user no starting point to narrow
    from."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo >= hi:
        delta = max(abs(lo) * 0.01, 1.0)
        lo -= delta
        hi += delta
    return lo, hi


def _control_row(layout: QHBoxLayout | QVBoxLayout) -> QWidget:
    """Wrap a viewer's control-row layout (a single QHBoxLayout for most
    viewers, or a QVBoxLayout of stacked QHBoxLayout rows for one that
    needs more than one line -- see the 2D Image Plot Viewer's own
    control block) in a plain container widget -- sized naturally at
    first; `_equalize_control_row_heights` then pins all four (2D Image/
    1D Plot/Azimuthal/Series Integration) to the same explicit height once
    every row's widgets actually exist, so their canvases come out the
    same size instead of the outer QVBoxLayout handing all the *extra*
    space to whichever expanding canvas has the shortest (or no) control
    row below it."""

    container = QWidget()
    container.setLayout(layout)
    return container


def _equalize_control_row_heights(*rows: QWidget) -> None:
    """Pin every row to the tallest one's natural (themed) sizeHint --
    guessing a fixed pixel height here would either clip a themed
    QComboBox/QDoubleSpinBox/QPushButton's padding (too short) or leave
    visible dead space (too tall); asking Qt/the stylesheet what these
    specific widgets actually need is the only value that's always right."""

    height = max(row.sizeHint().height() for row in rows)
    for row in rows:
        row.setFixedHeight(height)


class GIXS2Widget(QWidget):
    """Single-instance tab -- like SDDcalWidget, GIXS2 talks to its own
    module-level session directly (no per-instance session-swap dance,
    since there's only ever one GIXS2 tab)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_dir = str(Path.home())
        self._series_files: list[str] = []
        self._roi_dialog: QDialog | None = None
        self._load_thread = None
        self._exec_1d_thread = None
        self._exec_azi_thread = None
        self._auto_refresh_thread = None
        self._qcal_thread = None
        self._current_image_data: np.ndarray | None = None
        self._play_dialog: QDialog | None = None
        self._heatmap_dialog: QDialog | None = None
        # Set True once Load Info succeeds, matching console_widget.py's
        # own `_pixel_centered` (GIXS_gui.m's `Openinfo_Callback`): from
        # then on the 2D image's pixel axis re-centers on the beam
        # position instead of reading raw detector row/column indices.
        self._pixel_centered = False
        # None until the user actually touches a 1D Plot Process control
        # (angle/Cakecut/Accum) or an Azimuthal Process control (Position/
        # Draw/pix/Range), then "1d" or "azimuthal" -- guards
        # `_redraw_overlays` two ways: (1) neither cut preview auto-appears
        # just because Load Info/q-Cal happens to leave calibration in
        # place (reported: a white line through the beam center showed up
        # right after Load Info, before the user ever asked to see a cut);
        # (2) only ONE process-cut overlay is ever shown at a time, matching
        # console_widget.py's own Line/Azimuthal Draw buttons (each clears
        # the image and draws only its own geometry) -- switching to
        # Azimuthal work clears a 1D line/wedge/band still on screen,
        # instead of the two piling up together.
        self._active_overlay: str | None = None
        # Set by the Position button (mirrors console_widget.py's own
        # `_position_mode`); the next click on the 2D image then sets
        # `azi_pix_spin` from the clicked point instead of panning/zooming.
        self._position_mode: str | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._progress_timer: QTimer | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setFrameShape(QFrame.NoFrame)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1100])

        self._update_sub_mode_items()

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        group = QGroupBox("Serial Data Parameters")
        layout = QVBoxLayout(group)
        layout.setSpacing(7)

        self.dir_edit = QLineEdit(self._last_dir)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.dir_set_button = QPushButton("Set")
        # `_field` gives every trailing widget the same layout stretch as
        # the edit/spin box next to it, so a short-text button would
        # otherwise grow to match the box's width, leaving the box itself
        # cramped -- cap the button at its own comfortable width instead so
        # the edit/spin box gets the freed space. Applied to every button
        # that shares a `_field` row with a spin/edit box in this panel.
        self.dir_set_button.setMaximumWidth(56)
        self.dir_set_button.clicked.connect(self._on_set_directory_clicked)
        layout.addWidget(_field("Directory", self.dir_edit, self.dir_set_button))

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(_DETECTOR_LIST)
        self.insitu_combo = QComboBox()
        self.insitu_combo.addItems(_INSITU_LIST)
        # "Temperature" alone needs more room than any detector name here
        # -- cap Insitu's own budget down closer to Detector's natural
        # need instead of growing Detector's small one to match (see
        # sddcal_widget.py's Standard/Order for why that would only make
        # this row need *more* total width, not less).
        _size_combo(self.insitu_combo, 6)
        self.insitu_combo.currentTextChanged.connect(self._update_sub_mode_items)
        layout.addWidget(_split_row(_field("Detector", self.detector_combo, label_width=60), _field("Insitu", self.insitu_combo, label_width=48)))

        self.load_button = QPushButton("LOAD DATA")
        self.load_button.setProperty("variant", "primary")
        self.load_button.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_button)

        self.filename_edit = QLineEdit("ABCD")
        self.set_table_button = QPushButton("Set / Table")
        self.set_table_button.setMaximumWidth(96)
        self.set_table_button.clicked.connect(self._on_set_table_clicked)
        layout.addWidget(_field("Filename", self.filename_edit, self.set_table_button))

        self.n_files_label = _muted("# of Files: 0")
        self.parallel_check = QCheckBox("Parallel loading")
        self.parallel_check.setToolTip("Load series files concurrently on several threads instead of one at a time.")
        layout.addWidget(_row(self.n_files_label, self.parallel_check))

        self.load_info_button = QPushButton("Load Info")
        self.load_info_button.clicked.connect(self._on_load_info_clicked)
        self.grid_remove_button = QPushButton("Grid Rmv")
        self.grid_remove_button.clicked.connect(self._on_grid_remove_clicked)
        layout.addWidget(_row(self.load_info_button, self.grid_remove_button))

        layout.addWidget(_divider("Incidence Angle Calibration"))
        self.y_pos_spin = _double_spin(-1.0e6, 1.0e6, 2, 0.0, 1.0)
        self.zplot_button = QPushButton("Z-Plot")
        self.zplot_button.setMaximumWidth(72)
        self.zplot_button.clicked.connect(self._on_zplot_clicked)
        layout.addWidget(_field("y-pos", self.y_pos_spin, self.zplot_button))

        self.ai_pix_spin = _double_spin(0.0, 1.0e6, 2, 30.0, 1.0)
        self.ai_pix_spin.editingFinished.connect(self._on_ai_pix_changed)
        self.ai_deg_spin = _double_spin(-90.0, 90.0, 4, 0.1, 0.01, cyan=True)
        self.ai_deg_spin.editingFinished.connect(self._on_ai_deg_changed)
        layout.addWidget(_split_row(_field("pix", self.ai_pix_spin, label_width=32), _field("ai deg", self.ai_deg_spin, label_width=48)))

        # Port of `GUI.hChange`/`GX2.hAistep`: when the Insitu mode is
        # "Angle", the incident angle steps by `ai_step` deg every frame
        # instead of staying fixed, so q-Cal must recompute a per-frame
        # Q-map (see gixs2_session.py's Change-ai branch) rather than one
        # shared Q-map for the whole series.
        self.change_ai_check = QCheckBox("Change ai")
        self.ai_step_spin = _double_spin(-90.0, 90.0, 4, 0.01, 0.01, cyan=True)
        layout.addWidget(_split_row(self.change_ai_check, _field("step", self.ai_step_spin, label_width=32)))

        self.qcal_button = QPushButton("q-Cal.")
        self.qcal_button.setProperty("variant", "primary")
        self.qcal_button.clicked.connect(self._on_qcal_clicked)
        layout.addWidget(self.qcal_button)

        layout.addWidget(_divider("1D Plot Process"))
        self.phi_spin = _double_spin(-180.0, 180.0, 1, 90.0, 1.0)
        self.line_position_button = QPushButton("Position")
        self.line_position_button.setMaximumWidth(84)
        self.line_position_button.setToolTip("Click a point on the 2D Image Plot Viewer to set 1D Ang (the angle from the beam center).")
        self.line_position_button.clicked.connect(lambda: self._arm_position_mode("line"))
        layout.addWidget(_field("1D Ang", self.phi_spin, self.line_position_button))
        self.accum_minus_spin = _double_spin(-1.0e5, 1.0e5, 1, -1.0, 1.0)
        self.accum_plus_spin = _double_spin(-1.0e5, 1.0e5, 1, 1.0, 1.0)
        self.accum_check = QCheckBox("Accum On")
        # Same label_width (56) as z-pos mv/y-pos mv below -- all four of
        # this section's split-row fields line up on one shared left edge
        # instead of each pair drifting by its own few pixels.
        layout.addWidget(_split_row(_field("Accum", self.accum_minus_spin, self.accum_plus_spin, label_width=56), self.accum_check))
        self.cakecut_minus_spin = _double_spin(-180.0, 180.0, 1, -1.0, 1.0)
        self.cakecut_plus_spin = _double_spin(-180.0, 180.0, 1, 1.0, 1.0)
        self.cakecut_check = QCheckBox("Cakecut On")
        layout.addWidget(_split_row(_field("Cakecut", self.cakecut_minus_spin, self.cakecut_plus_spin, label_width=56), self.cakecut_check))
        self.z_pos_move_spin = _double_spin(-1.0e5, 1.0e5, 2, 0.0, 1.0)
        self.y_pos_move_spin = _double_spin(-1.0e5, 1.0e5, 2, 0.0, 1.0)
        layout.addWidget(_split_row(_field("z-pos mv", self.z_pos_move_spin, label_width=56), _field("y-pos mv", self.y_pos_move_spin, label_width=56)))

        # Cakecut (angular wedge) and Accumulation (perpendicular band) are
        # mutually exclusive cut modes -- enforce that directly in the UI
        # (matches console_widget.py) rather than letting both stay checked
        # while the backend silently only honors one.
        self.cakecut_check.toggled.connect(lambda checked: self.accum_check.setChecked(False) if checked else None)
        self.accum_check.toggled.connect(lambda checked: self.cakecut_check.setChecked(False) if checked else None)
        # Whichever geometry (angle / cakecut wedge / accumulation band) is
        # currently selected shows live as a line/wedge/band overlay on the
        # 2D image, matching the single-frame GIXS console's own behavior --
        # but only once the user has actually touched one of these controls
        # (see _arm_1d_overlay), not just because Load Info/q-Cal happened
        # to run and left calibration in place.
        for control in (
            self.phi_spin, self.accum_minus_spin, self.accum_plus_spin,
            self.cakecut_minus_spin, self.cakecut_plus_spin, self.z_pos_move_spin, self.y_pos_move_spin,
        ):
            control.valueChanged.connect(lambda _v: self._arm_1d_overlay())
        self.cakecut_check.toggled.connect(lambda _c: self._arm_1d_overlay())
        self.accum_check.toggled.connect(lambda _c: self._arm_1d_overlay())

        self.exec_1d_button = QPushButton("Exec Cut")
        self.exec_1d_button.clicked.connect(self._on_exec_1d_clicked)
        self.normalize_1d_combo = QComboBox()
        self.normalize_1d_combo.addItems(_NORMALIZE_LIST)
        # Same content-length budget as every other combo box in the 1D
        # Plot Process/Azimuthal Process sections (see _COMBO_CHARS' own
        # docstring) so they don't drift apart depending on whichever
        # button/label happens to share their row.
        _size_combo(self.normalize_1d_combo, _COMBO_CHARS)
        self.normalize_1d_combo.currentTextChanged.connect(self._render_1d)
        layout.addWidget(_row(self.exec_1d_button, self.normalize_1d_combo))
        self.exec_1d_parallel_check = QCheckBox("Parallel")
        self.exec_1d_parallel_check.setToolTip(
            "Compute each frame's cut concurrently on several threads instead of one at a time -- applies to plain angle, Accum, and Cakecut alike."
        )
        layout.addWidget(self.exec_1d_parallel_check)

        layout.addWidget(_divider("Azimuthal Process"))
        self.azi_pix_spin = _double_spin(0.0, 1.0e6, 1, 202.0, 1.0)
        self.azi_position_button = QPushButton("Position")
        self.azi_position_button.setMaximumWidth(84)
        self.azi_position_button.setToolTip("Click a point on the 2D Image Plot Viewer to set pix (the radius from the beam center).")
        self.azi_position_button.clicked.connect(lambda: self._arm_position_mode("azimuthal"))
        # label_width=88 on every full-row field in this section (pix,
        # Range (-/+), Mean/Median) -- shared left edge for their edit
        # boxes/combo, matching "Range (-/+)"'s own width (the longest
        # label here, needing the most room).
        layout.addWidget(_field("pix", self.azi_pix_spin, self.azi_position_button, label_width=88))
        self.azi_minus_spin = _double_spin(-1.0e5, 1.0e5, 1, -1.0, 1.0)
        self.azi_plus_spin = _double_spin(-1.0e5, 1.0e5, 1, 1.0, 1.0)
        self.azi_draw_button = QPushButton("Draw")
        self.azi_draw_button.setMaximumWidth(64)
        self.azi_draw_button.setToolTip("Show the pix +/- Range(-/+) ring on the 2D image; keeps updating live as pix/Range change.")
        self.azi_draw_button.clicked.connect(self._arm_azi_overlay)
        layout.addWidget(_field("Range (-/+)", self.azi_minus_spin, self.azi_plus_spin, self.azi_draw_button, label_width=88))
        for control in (self.azi_pix_spin, self.azi_minus_spin, self.azi_plus_spin):
            control.valueChanged.connect(lambda _v: self._arm_azi_overlay())
        self.azi_mean_median_combo = QComboBox()
        self.azi_mean_median_combo.addItems(["Mean", "Median"])
        _size_combo(self.azi_mean_median_combo, _COMBO_CHARS)
        layout.addWidget(_field("Mean/Median", self.azi_mean_median_combo, label_width=88))
        self.azi_reflect_missing_wedge_check = QCheckBox("Reflect missing wedge (Qxy/Qz)")
        self.azi_reflect_missing_wedge_check.setToolTip(
            "Regrid every frame onto the Qxy/Qz grid before cutting, so a real coverage gap near Qxy=0 shows up "
            "as a break in the curve -- off by default (plain pixel-space cut, no gap) since regridding every "
            "frame is slower and needs a plain q-Cal (not \"Change ai\")."
        )
        layout.addWidget(self.azi_reflect_missing_wedge_check)

        self.exec_azi_button = QPushButton("Exec Azi")
        self.exec_azi_button.clicked.connect(self._on_exec_azi_clicked)
        self.normalize_azi_combo = QComboBox()
        self.normalize_azi_combo.addItems(_NORMALIZE_LIST)
        _size_combo(self.normalize_azi_combo, _COMBO_CHARS)
        self.normalize_azi_combo.currentTextChanged.connect(self._render_azimuthal)
        layout.addWidget(_row(self.exec_azi_button, self.normalize_azi_combo))

        layout.addWidget(_divider("Region of Interest"))
        self.n_roi_combo = QComboBox()
        self.n_roi_combo.addItems(["1", "2", "3", "4", "5"])
        self.draw_roi_button = QPushButton("Draw ROI")
        self.draw_roi_button.setMaximumWidth(96)
        self.draw_roi_button.setToolTip(
            "Opens a separate window (matching MATLAB) -- drag out one rectangle per ROI, each in its own color."
        )
        self.draw_roi_button.clicked.connect(self._on_draw_roi_clicked)
        layout.addWidget(_field("N. ROI", self.n_roi_combo, self.draw_roi_button))
        self.normalize_roi_combo = QComboBox()
        self.normalize_roi_combo.addItems(_NORMALIZE_LIST)
        _size_combo(self.normalize_roi_combo, _COMBO_CHARS)
        self.normalize_roi_combo.currentTextChanged.connect(self._render_integration)
        layout.addWidget(_field("Normalize", self.normalize_roi_combo))

        layout.addWidget(_divider("X-Axis / Auto"))
        self.sub_mode_combo = QComboBox()
        self.order_check = QCheckBox("Order")
        layout.addWidget(_row(self.sub_mode_combo, self.order_check))
        self.set_x_axis_button = QPushButton("Set X-Axis")
        self.set_x_axis_button.clicked.connect(self._on_set_x_axis_clicked)
        layout.addWidget(self.set_x_axis_button)
        self.auto_refresh_button = QPushButton("Auto Refresh (Find New)")
        self.auto_refresh_button.clicked.connect(self._on_auto_refresh_clicked)
        layout.addWidget(self.auto_refresh_button)

        layout.addWidget(_divider("Save"))
        self.save_1d_button = QPushButton("Save 1D")
        self.save_1d_button.clicked.connect(
            lambda: self._save(gs.gixs2_save_1d, self.normalize_1d_combo.currentText(), self.oned_canvas)
        )
        self.save_azi_button = QPushButton("Save Azi")
        self.save_azi_button.clicked.connect(
            lambda: self._save(gs.gixs2_save_azimuthal, self.normalize_azi_combo.currentText(), self.azi_canvas)
        )
        layout.addWidget(_row(self.save_1d_button, self.save_azi_button))
        self.save_integ_button = QPushButton("Save Integ.")
        self.save_integ_button.clicked.connect(
            lambda: self._save(gs.gixs2_save_integration, self._integ_normalize(), self.integ_canvas)
        )
        self.save_xaxis_button = QPushButton("Save X-Axis")
        self.save_xaxis_button.clicked.connect(lambda: self._save(gs.gixs2_save_x_axis))
        layout.addWidget(_row(self.save_integ_button, self.save_xaxis_button))

        layout.addStretch(1)
        self.status_label = _muted("")
        self.status_label.setMinimumHeight(3 * self.status_label.fontMetrics().lineSpacing())
        layout.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(group)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(480)
        return scroll

    # ------------------------------------------------------------------
    # Right panel: 4-quadrant viewers
    # ------------------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        outer = QSplitter(Qt.Vertical)
        outer.setHandleWidth(2)
        outer.setFrameShape(QFrame.NoFrame)

        top = QSplitter(Qt.Horizontal)
        top.setHandleWidth(2)
        top.setFrameShape(QFrame.NoFrame)
        bottom = QSplitter(Qt.Horizontal)
        bottom.setHandleWidth(2)
        bottom.setFrameShape(QFrame.NoFrame)

        image_group = QGroupBox("2D Image Plot Viewer")
        image_layout = QVBoxLayout(image_group)
        self.image_canvas = ImageCanvas()
        self.image_canvas.pixel_clicked.connect(self._on_image_clicked)
        image_layout.addWidget(self.image_canvas)
        image_ctrl = QVBoxLayout()
        image_ctrl.setContentsMargins(0, 0, 0, 0)
        image_ctrl.setSpacing(2)
        self.image_scale_combo = QComboBox()
        self.image_scale_combo.addItems(["Linear", "Log"])
        self.image_colormap_combo = QComboBox()
        self.image_colormap_combo.addItems(_COLORMAPS)
        # Scale only ever shows "Linear"/"Log" (matches the other three
        # viewers' own Scale combos) -- an 8-char budget fits that with a
        # little breathing room, instead of the wider 12-char budget a
        # standalone combo doesn't need (see console_widget.py's
        # `_size_combo` docstring for why not setFixedWidth). Colormap
        # keeps a wider 10-char budget since its real entries run much
        # longer ("twilight_shifted"); the closed box still elides with
        # "..." past that, and the dropdown list itself always shows the
        # full name. Keeping both no wider than they need to be is what
        # lets this row -- and so the whole 2D Image Plot Viewer -- get
        # close to the other three viewers' own narrower control rows in
        # width, instead of forcing the quadrant splitter to always give
        # it more than an equal share.
        _size_combo(self.image_scale_combo, 8)
        _size_combo(self.image_colormap_combo, 10)
        default_index = self.image_colormap_combo.findText("viridis")
        if default_index >= 0:
            self.image_colormap_combo.setCurrentIndex(default_index)
        # Contrast: 0/0 is the "auto" sentinel (matches console_widget.py's
        # own Contrast field) -- ImageCanvas.render's own percentile-based
        # auto-contrast is used whenever cmax <= cmin, and whatever it picks
        # is reflected back into these fields so the user has a starting
        # point to nudge from instead of two blank zeros. Scientific
        # notation (not `_double_spin`'s fixed-decimal one) since detector
        # intensity spans a fraction of a count up to 1e6+ raw counts.
        self.image_cmin_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self.image_cmax_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self.image_cmin_spin.valueChanged.connect(lambda _v: self._render_image(redraw_overlays=False))
        self.image_cmax_spin.valueChanged.connect(lambda _v: self._render_image(redraw_overlays=False))
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip(
            "Step through every loaded frame one by one, all rendered with the current contrast/colormap locked, for visual comparison."
        )
        self.play_button.clicked.connect(self._on_play_clicked)
        # `_render_image` takes optional `force_reset`/`redraw_overlays`
        # bools -- route through a lambda rather than connecting the slot
        # directly, so the combo's emitted text never lands in either
        # parameter position. `redraw_overlays=False` for all four
        # Scale/Colormap/Contrast controls: these are pure display changes,
        # not a new cut/process/ROI request, so whatever was drawn on the
        # image (1D/Azimuthal process overlay, ROI boxes) is cleared rather
        # than carried over, back to the plain image -- touching the
        # relevant control again (or Draw ROI again) brings it back.
        self.image_scale_combo.currentTextChanged.connect(lambda _text: self._render_image(redraw_overlays=False))
        self.image_colormap_combo.currentTextChanged.connect(lambda _text: self._render_image(redraw_overlays=False))
        # Scale/Colormap/Contrast/Play used to sit on one line, but that
        # made this the widest of the four viewers' own control rows by a
        # large margin -- the 1D/Azimuthal/Series Integration rows have at
        # most two fields, and Contrast alone (two spin boxes) is already
        # about as wide as either of those rows on its own. A single wide
        # row forced the quadrant splitter below to give this viewer more
        # than half the window width to fit it (reported: "이 크기가
        # 맞았으면 좋겠어... 모니터에 따라 크기가 너무 다른데"), no matter
        # what sizes/stretch factors the splitter itself was given.
        # Splitting across two shorter rows keeps every viewer's control
        # row in the same rough width range, so the quadrants can actually
        # end up equal.
        image_ctrl_row1 = QHBoxLayout()
        image_ctrl_row1.addWidget(_field("Scale", self.image_scale_combo, label_width=40))
        image_ctrl_row1.addWidget(_field("Colormap", self.image_colormap_combo, label_width=60))
        image_ctrl_row2 = QHBoxLayout()
        image_ctrl_row2.addWidget(_field("Contrast", self.image_cmin_spin, self.image_cmax_spin, label_width=54))
        image_ctrl_row2.addWidget(self.play_button)
        image_ctrl.addLayout(image_ctrl_row1)
        image_ctrl.addLayout(image_ctrl_row2)
        image_ctrl_row = _control_row(image_ctrl)
        image_layout.addWidget(image_ctrl_row)
        top.addWidget(image_group)

        oned_group = QGroupBox("1D Plot Viewer")
        oned_layout = QVBoxLayout(oned_group)
        self.oned_canvas = CurveCanvas()
        # Matches ImageCanvas's own minimum (280, vs. CurveCanvas's default
        # 160) -- otherwise the top row (2D Image + this) has a taller
        # forced minimum than the bottom row (Azimuthal + Integration,
        # still at 160), and the outer splitter gives the top row whatever
        # extra height its minimum demands before honoring the 50/50 split,
        # squeezing the bottom row shorter regardless of setSizes([500,500]).
        self.oned_canvas.canvas.setMinimumHeight(280)
        oned_layout.addWidget(self.oned_canvas)
        self.oned_scale_combo = QComboBox()
        self.oned_scale_combo.addItems(["Linear", "Logy", "Logxy"])
        # Same budget as the left panel's Normalize/Mean-Median combos
        # (see _COMBO_CHARS' own docstring).
        _size_combo(self.oned_scale_combo, _COMBO_CHARS)
        self.oned_scale_combo.currentTextChanged.connect(lambda _t: self._render_1d())
        self.oned_colormap_combo = QComboBox()
        self.oned_colormap_combo.addItems(_COLORMAPS)
        _size_combo(self.oned_colormap_combo, 10)
        oned_cmap_default = self.oned_colormap_combo.findText("viridis")
        if oned_cmap_default >= 0:
            self.oned_colormap_combo.setCurrentIndex(oned_cmap_default)
        self.oned_colormap_combo.currentTextChanged.connect(lambda _t: self._render_1d())
        self.oned_heatmap_button = QPushButton("Heatmap")
        self.oned_heatmap_button.setToolTip("Open a Q vs. X-axis heatmap of every frame's 1D cut (port of MATLAB's HeatmapQ_gui).")
        self.oned_heatmap_button.clicked.connect(self._on_1d_heatmap_clicked)
        # Scale/Colormap/Heatmap on one row used to be the widest of the
        # four viewers' own control rows by a wide margin (adding the new
        # Colormap combo tipped it over) -- split across two shorter rows
        # instead, same fix already applied to the 2D Image Plot Viewer's
        # own control block for the same reason (see its own comment),
        # so the quadrant splitter's equal split actually holds (see
        # test_gixs2_widget_four_quadrants_have_equal_stretch).
        oned_ctrl = QVBoxLayout()
        oned_ctrl.setContentsMargins(0, 0, 0, 0)
        oned_ctrl.setSpacing(2)
        oned_ctrl_row1 = QHBoxLayout()
        oned_ctrl_row1.addWidget(_field("Scale", self.oned_scale_combo, label_width=40))
        oned_ctrl_row1.addWidget(_field("Colormap", self.oned_colormap_combo, label_width=60))
        oned_ctrl_row2 = QHBoxLayout()
        oned_ctrl_row2.addWidget(self.oned_heatmap_button)
        oned_ctrl.addLayout(oned_ctrl_row1)
        oned_ctrl.addLayout(oned_ctrl_row2)
        oned_ctrl_row = _control_row(oned_ctrl)
        oned_layout.addWidget(oned_ctrl_row)
        top.addWidget(oned_group)

        azi_group = QGroupBox("Azimuthal Plot Viewer")
        azi_layout = QVBoxLayout(azi_group)
        self.azi_canvas = CurveCanvas()
        self.azi_canvas.canvas.setMinimumHeight(280)
        azi_layout.addWidget(self.azi_canvas)
        self.azi_heatmap_button = QPushButton("Heatmap")
        self.azi_heatmap_button.setToolTip(
            "Open an Angle vs. X-axis heatmap of every frame's azimuthal profile (port of MATLAB's HeatmapAzi_gui)."
        )
        self.azi_heatmap_button.clicked.connect(self._on_azi_heatmap_clicked)
        self.azi_colormap_combo = QComboBox()
        self.azi_colormap_combo.addItems(_COLORMAPS)
        _size_combo(self.azi_colormap_combo, 10)
        azi_cmap_default = self.azi_colormap_combo.findText("viridis")
        if azi_cmap_default >= 0:
            self.azi_colormap_combo.setCurrentIndex(azi_cmap_default)
        self.azi_colormap_combo.currentTextChanged.connect(lambda _t: self._render_azimuthal())
        azi_ctrl = QHBoxLayout()
        azi_ctrl.addWidget(_field("Colormap", self.azi_colormap_combo, label_width=60))
        azi_ctrl.addWidget(self.azi_heatmap_button)
        azi_ctrl_row = _control_row(azi_ctrl)
        azi_layout.addWidget(azi_ctrl_row)
        bottom.addWidget(azi_group)

        integ_group = QGroupBox("Series Integration Viewer")
        integ_layout = QVBoxLayout(integ_group)
        self.integ_canvas = CurveCanvas()
        self.integ_canvas.canvas.setMinimumHeight(280)
        integ_layout.addWidget(self.integ_canvas)
        self.integ_scale_combo = QComboBox()
        self.integ_scale_combo.addItems(["Linear", "Logy", "Logxy"])
        # Same budget as the 1D Plot Viewer's own Scale combo -- the two
        # are the same kind of control, just on different viewers.
        _size_combo(self.integ_scale_combo, _COMBO_CHARS)
        self.integ_scale_combo.currentTextChanged.connect(lambda _t: self._render_integration())
        self.replot_button = QPushButton("Re-Plot Series")
        self.replot_button.clicked.connect(self._render_integration)
        integ_ctrl = QHBoxLayout()
        integ_ctrl.addWidget(_field("Scale", self.integ_scale_combo, label_width=40))
        integ_ctrl.addWidget(self.replot_button)
        integ_ctrl_row = _control_row(integ_ctrl)
        integ_layout.addWidget(integ_ctrl_row)
        bottom.addWidget(integ_group)

        _equalize_control_row_heights(image_ctrl_row, oned_ctrl_row, azi_ctrl_row, integ_ctrl_row)

        # Left to their own size hints, the four quadrants end up visibly
        # uneven (the 2D Image Plot Viewer's own colorbar/Contrast/Play
        # row and the Series Integration Viewer's wider default curve both
        # nudge their splitter section wider than its neighbor) and that
        # unevenness only gets worse at whatever window size the app
        # happens to open at on a given monitor -- explicit equal initial
        # sizes plus equal stretch factors on all three splitters make the
        # four quadrants start at a true 1:1:1:1 grid and keep that ratio
        # as the window is resized, rather than one section eating all the
        # extra/lost space (`setStretchFactor`'s default of 0 for every
        # section, unset here previously).
        for row in (top, bottom):
            row.setStretchFactor(0, 1)
            row.setStretchFactor(1, 1)
            row.setSizes([500, 500])

        outer.addWidget(top)
        outer.addWidget(bottom)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 1)
        outer.setSizes([500, 500])
        # Exposed for tests -- see the equal-quadrant-sizing comment above.
        self._quadrant_splitters = (outer, top, bottom)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(outer)
        return container

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------
    def _set_status(self, message: str, ok: bool) -> None:
        self.status_label.setText(_wrap_long_tokens(message))
        self.status_label.setProperty("role", "status-ok" if ok else "status-error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _mark_ok(self, button: QPushButton, ok: bool) -> None:
        """Light-green button background once a step succeeds, matching
        MATLAB's own `BackgroundColor = [0.75 0.95 0.75]` feedback for
        Load Info/q-Cal/Grid Remove (same `QPushButton[state="ok"]` QSS
        rule the single-frame console's `_mark_ok` already uses)."""

        button.setProperty("state", "ok" if ok else None)
        button.style().unpolish(button)
        button.style().polish(button)

    def _update_sub_mode_items(self) -> None:
        self.sub_mode_combo.clear()
        self.sub_mode_combo.addItems(_SUB_MODE_ITEMS.get(self.insitu_combo.currentText(), ["Time"]))

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    def _set_last_dir(self, directory: str) -> None:
        self._last_dir = directory
        self.dir_edit.setText(directory)

    def _on_set_directory_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select start directory", self._last_dir)
        if not directory:
            return
        self._set_last_dir(directory)

    def _on_directory_edited(self) -> None:
        text = self.dir_edit.text().strip()
        if text and Path(text).is_dir():
            self._last_dir = text
        else:
            self.dir_edit.setText(self._last_dir)

    # ------------------------------------------------------------------
    # Load Data / Set-Table / Load Info / Grid Remove
    # ------------------------------------------------------------------
    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a series frame", self._last_dir, "HDF5 files (*.h5);;All files (*)")
        if not path:
            return
        file_path = Path(path)
        self._set_last_dir(str(file_path.parent))
        payload = {"directory": str(file_path.parent), "filename": file_path.name, "detector": self.detector_combo.currentText()}
        result = gs.gixs2_load(payload)
        if not result.get("ok"):
            self._mark_ok(self.load_button, False)
            self._set_status(result.get("message", "Load failed."), False)
            return
        self.filename_edit.setText(result["prefix"])
        self._current_image_data = None
        # Fresh raw data invalidates whatever beam-centered calibration was
        # loaded for the *previous* dataset -- back to raw pixel indices
        # until Load Info is run again, matching console_widget.py.
        self._pixel_centered = False
        # And whatever process-cut geometry the user had set up applied to
        # the previous dataset too -- don't carry the overlay over onto
        # this new one until they touch those controls again.
        self._active_overlay = None
        self._position_mode = None
        self._render_image()
        self._mark_ok(self.load_button, True)
        # New raw data invalidates whatever q-Cal/Grid Remove/series import
        # was done against the *previous* dataset, matching MATLAB's own
        # Openimage2_Callback reset of GUI.btnQcal/bGrid to their
        # not-yet-done color.
        self._mark_ok(self.qcal_button, False)
        self._mark_ok(self.grid_remove_button, False)
        self._mark_ok(self.set_table_button, False)
        self._set_status(f"Loaded {'x'.join(str(v) for v in result['shape'])} from {result['path']}.", True)

    def _on_set_table_clicked(self) -> None:
        """Set / Table: list every file that shares the Filename field's
        series prefix, each with a checkbox (checked by default); Import
        loads whichever are still checked, matching MATLAB's file-list +
        "Export & Load" dialog (`OpenNameDef_callback`/`table2workspace`)."""

        result = gs.gixs2_list_series_files({"directory": self._last_dir, "filename": self.filename_edit.text().strip()})
        if not result.get("ok"):
            self._set_status(result.get("message", "No series files found."), False)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("File List (Series)")
        dialog.resize(360, 480)
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget()
        for name in result["files"]:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        select_row = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(lambda: self._set_all_checked(list_widget, True))
        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(lambda: self._set_all_checked(list_widget, False))
        select_row.addWidget(select_all_button)
        select_row.addWidget(deselect_all_button)
        layout.addLayout(select_row)

        button_row = QHBoxLayout()
        import_button = QPushButton("Import")
        import_button.setProperty("variant", "primary")
        import_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_row.addWidget(import_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        if dialog.exec() != QDialog.Accepted:
            return
        checked = [list_widget.item(i).text() for i in range(list_widget.count()) if list_widget.item(i).checkState() == Qt.Checked]
        if not checked:
            self._set_status("No files selected.", False)
            return
        self._start_series_load(checked)

    @staticmethod
    def _set_all_checked(list_widget: QListWidget, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(state)

    def _start_series_load(self, filenames: list[str]) -> None:
        self.set_table_button.setEnabled(False)
        self._set_status(f"Loading {len(filenames)} file(s)...", True)
        self._start_progress_dialog(f"Loading {len(filenames)} file(s)...")
        payload = {
            "directory": self._last_dir,
            "filenames": filenames,
            "insitu_mode": self.insitu_combo.currentText(),
            "parallel": self.parallel_check.isChecked(),
        }
        self._load_thread = run_in_background(gs.gixs2_load_series, payload, self._on_series_loaded)

    def _start_progress_dialog(self, title: str, window_title: str = "Loading Series") -> None:
        """Poll `gs.gixs2_load_progress()` on a QTimer and drive a
        `QProgressDialog` from it -- shared by every background op that
        reports progress through the session's `load_progress` field
        (series loading, Auto Refresh, and Exec Cut's per-frame 1D-cut
        loop), not just series loading despite the field's name."""

        dialog = QProgressDialog(title, None, 0, 100, self)
        dialog.setWindowTitle(window_title)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setCancelButton(None)
        dialog.show()
        self._progress_dialog = dialog
        timer = QTimer(self)
        timer.timeout.connect(self._poll_load_progress)
        timer.start(150)
        self._progress_timer = timer

    def _poll_load_progress(self) -> None:
        status = gs.gixs2_load_progress()
        if not status.get("active") or self._progress_dialog is None:
            return
        total = status.get("total") or 1
        current = status.get("current") or 0
        self._progress_dialog.setMaximum(total)
        self._progress_dialog.setValue(current)
        self._progress_dialog.setLabelText(f"{status.get('label', 'Processing')} ({current}/{total})")

    def _stop_progress_dialog(self) -> None:
        if self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer = None
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

    def _on_series_loaded(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.set_table_button.setEnabled(True)
        if not result.get("ok"):
            self._mark_ok(self.set_table_button, False)
            self._set_status(result.get("message", "Series load failed."), False)
            return
        self.n_files_label.setText(f"# of Files: {result['n_files']}")
        self._current_image_data = None
        self._render_image()
        self._mark_ok(self.set_table_button, True)
        # The new stack invalidates any q-Cal/Grid Remove already applied
        # to whatever was loaded before -- same reasoning as _on_load_clicked.
        self._mark_ok(self.qcal_button, False)
        self._mark_ok(self.grid_remove_button, False)
        warnings = result.get("warnings") or []
        if warnings:
            QMessageBox.warning(self, "GIXS2 Load Series", "\n\n".join(warnings))
        self._set_status(f"Loaded {result['n_files']} files, shape {tuple(result['shape'])}. Showing the average image.", True)

    def _on_load_info_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a calibration file", self._last_dir, "Calibration (*.json)")
        if not path:
            return
        file_path = Path(path)
        result = gs.gixs2_load_info({"directory": str(file_path.parent), "filename": file_path.name})
        if not result.get("ok"):
            self._set_status(result.get("message", "Load Info failed."), False)
            return
        cal = result["calibration"]
        self._mark_ok(self.load_info_button, True)
        # GIXS_gui.m Openinfo_Callback: param.Expinfo = 1; renderImage() --
        # the pixel axis re-centers on the beam position from here on.
        self._pixel_centered = True
        self._render_image(force_reset=True)
        if result.get("message"):
            # q-Cal ran against the previous calibration; the session dropped it.
            self._mark_ok(self.qcal_button, False)
            self._set_status(result["message"], True)
            return
        self._set_status(f"Loaded Info: SDD={cal['sdd_mm']:.1f} mm, Energy={cal['energy_kev']:.3f} keV.", True)

    def _on_grid_remove_clicked(self) -> None:
        result = gs.gixs2_grid_remove({})
        if not result.get("ok"):
            self._mark_ok(self.grid_remove_button, False)
            self._set_status(result.get("message", "Grid Remove failed."), False)
            return
        self._current_image_data = None
        self._render_image()
        self._mark_ok(self.grid_remove_button, True)
        self._set_status(result.get("message", "Grid removed."), True)

    # ------------------------------------------------------------------
    # 2D image render
    # ------------------------------------------------------------------
    def _render_image(self, force_reset: bool = False, redraw_overlays: bool = True) -> None:
        session = gs.get_session()
        if session.data is None:
            return
        # Callers that just loaded genuinely new data null out
        # `_current_image_data` first, so `had_previous` (checked *before*
        # the reassignment below) is False there -- resetting the zoom --
        # and True for a same-data cosmetic change (colormap/scale combo),
        # preserving whatever pan/zoom the user had. `force_reset` is for
        # Load Info specifically: the data is unchanged but the axis
        # coordinate system itself just shifted (raw pixel -> beam-
        # centered), so any previously-kept zoom limits no longer refer to
        # the same place on the detector and must be dropped too.
        had_previous = self._current_image_data is not None and not force_reset
        self._current_image_data = session.data
        rows, cols = session.data.shape
        n_frames = session.stack.shape[0] if session.stack is not None else 1
        extent = self._image_extent((rows, cols))
        cmin = self.image_cmin_spin.value()
        cmax = self.image_cmax_spin.value()
        vmin = cmin if cmax > cmin else None
        vmax = cmax if cmax > cmin else None
        used_vmin, used_vmax = self.image_canvas.render(
            session.data,
            cmap=self.image_colormap_combo.currentText(),
            log_scale=self.image_scale_combo.currentText() == "Log",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            xlabel="Y, pixel",
            ylabel="Z, pixel",
            title=f"Average of {n_frames} frame(s)" if n_frames > 1 else "",
            keep_limits=had_previous,
        )
        if vmin is None:
            self.image_cmin_spin.blockSignals(True)
            self.image_cmax_spin.blockSignals(True)
            self.image_cmin_spin.setValue(used_vmin)
            self.image_cmax_spin.setValue(used_vmax)
            self.image_cmin_spin.blockSignals(False)
            self.image_cmax_spin.blockSignals(False)
        # ImageCanvas.render() just fully rebuilt the figure/axes (see its
        # own docstring on why), which silently drops any previously-drawn
        # overlay -- including ROI boxes. For most callers (new data, Load
        # Info, q-Cal, Exec...) redraw whatever's still relevant so it
        # doesn't disappear; a pure Scale/Colormap/Contrast change instead
        # passes redraw_overlays=False so the plain image comes back with
        # nothing drawn on it at all (reported: ROI boxes should clear the
        # same way the 1D/Azimuthal process overlay already does on these
        # controls) -- see those controls' own wiring comment.
        if redraw_overlays:
            self._redraw_overlays()

    def _image_extent(self, shape: tuple[int, int]) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) for `ImageCanvas.render`'s `extent`,
        shared by the main 2D Image Plot Viewer and the Play dialog so both
        always agree on the current pixel-centered/raw axis convention."""

        rows, cols = shape
        if self._pixel_centered:
            # Mirrors console_widget.py's `_render_pixel_image`: HD = (1:cols)
            # - beamY, VD = -((1:rows) - beamZ), so the beam sits at (0, 0).
            beam_y, beam_z = gs.get_session().calibration["beam_pos"]
            h_start, h_end = 1 - beam_y, cols - beam_y
            v_row1, v_row_last = beam_z - 1, beam_z - rows
        else:
            h_start, h_end = 1, cols
            v_row1, v_row_last = 1, rows
        return (h_start, h_end, v_row_last, v_row1)

    # ------------------------------------------------------------------
    # 1D Plot Process geometry overlay -- port of console_widget.py's own
    # `_draw_line_overlay`/`_current_beam_center`/`_mode_corrected_angle`/
    # `_pos_move_display_offset`, simplified to their pixel-only branches
    # since GIXS2 (unlike the single-frame console) has no Q-space "Map
    # format" toggle -- the 2D Image Plot Viewer is always in pixel space,
    # raw or beam-centered.
    # ------------------------------------------------------------------
    def _current_beam_center(self) -> tuple[float, float]:
        if not self._pixel_centered:
            beam_y, beam_z = gs.get_session().calibration["beam_pos"]
            return (beam_y, beam_z)
        # Once Load Info has recentered the pixel axis, the beam sits at
        # (0, 0) on that axis by construction (see _image_extent).
        return (0.0, 0.0)

    def _mode_corrected_angle(self, angle_deg: float) -> float:
        """Beam-centered display flips the vertical axis convention
        relative to raw pixel mode (VD = beamZ - row increases going *up*
        the screen, instead of the row index increasing going down) -- the
        "0 left / 90 up / 180 right" angle legend needs the same mirror
        applied before drawing, or the overlay would point the wrong way
        once Load Info has recentered the axis."""

        if self._pixel_centered:
            return (-angle_deg) % 360.0
        return angle_deg % 360.0

    def _pos_move_display_offset(self, band_edge_y: float = 0.0, band_edge_z: float = 0.0) -> tuple[float, float]:
        beam_y, beam_z = self._current_beam_center()
        y_pos_move = self.y_pos_move_spin.value()
        z_pos_move = self.z_pos_move_spin.value()
        offset_y = y_pos_move + band_edge_y
        raw_offset_z = band_edge_z - z_pos_move
        offset_z = -raw_offset_z if self._pixel_centered else raw_offset_z
        return (beam_y + offset_y, beam_z + offset_z)

    def _redraw_overlays(self) -> None:
        """Full resync of every overlay layer: clear everything once, then
        draw whichever process-cut geometry is currently active (1D Plot
        Process OR Azimuthal Process, never both -- see `_active_overlay`'s
        own docstring in `__init__`, matching console_widget.py's own
        Line/Azimuthal Draw buttons), then ROI boxes on top -- ROI is a
        saved region rather than an in-progress cut preview, so it's always
        shown independent of which process overlay is active. Shared by
        `_render_image`'s tail (after a full figure rebuild wipes
        everything) and by each overlay's own "arm" handler."""

        self.image_canvas.clear_overlays()
        self._draw_1d_overlay()
        self._draw_azi_overlay()
        self._draw_roi_overlay()

    def _arm_1d_overlay(self) -> None:
        """Mark that the user has actually engaged with the 1D Plot
        Process (touched a control or run Exec Cut) and redraw -- see
        `_active_overlay`'s own docstring in `__init__`."""

        self._active_overlay = "1d"
        self._redraw_overlays()

    def _draw_1d_overlay(self) -> None:
        """Show the currently-selected 1D Plot Process geometry (plain
        angle cut / Cakecut wedge / Accumulation band) as a line overlay on
        the 2D image, live as any of its controls change -- matches the
        single-frame GIXS console's own behavior, requested for GIXS2 too.
        Additive (does not clear first) -- always called by way of
        `_redraw_overlays`, which clears once up front."""

        session = gs.get_session()
        if session.data is None:
            return
        # Before Load Info or q-Cal has actually run, `calibration["beam_pos"]`
        # is still the session dataclass's own placeholder default -- drawing
        # a "beam center" line through it right after a fresh LOAD DATA would
        # show a confident-looking marker that has nothing to do with this
        # detector frame's real beam position. Wait for real calibration.
        if not (self._pixel_centered or session.corrected):
            return
        # And even once calibrated, only show it while it's the active
        # overlay -- neither unasked-for right after Load Info/q-Cal, nor
        # still lingering after the user has since moved on to Azimuthal
        # Process work.
        if self._active_overlay != "1d":
            return

        angle = self.phi_spin.value()
        cakecut_on = self.cakecut_check.isChecked()
        accum_on = self.accum_check.isChecked() and angle in (0.0, 90.0, 180.0)

        if cakecut_on:
            # Cakecut always emanates from the true beam center and ignores
            # Pos move (matches console_widget.py/GIXS_gui.m) -- show its
            # actual angular wedge as two boundary rays instead of a single
            # line, so the wedge width is visible.
            center = self._current_beam_center()
            self.image_canvas.draw_beam_line(
                center, self._mode_corrected_angle(angle + self.cakecut_minus_spin.value()), bidirectional=False
            )
            self.image_canvas.draw_beam_line(
                center, self._mode_corrected_angle(angle + self.cakecut_plus_spin.value()), bidirectional=False
            )
            return

        if accum_on:
            # Accumulation is a perpendicular pixel band, not a single line
            # -- draw its two edges instead of one centerline.
            if angle in (0.0, 180.0):
                edge_minus = self._pos_move_display_offset(band_edge_z=self.accum_minus_spin.value())
                edge_plus = self._pos_move_display_offset(band_edge_z=self.accum_plus_spin.value())
            else:  # angle == 90.0
                edge_minus = self._pos_move_display_offset(band_edge_y=self.accum_minus_spin.value())
                edge_plus = self._pos_move_display_offset(band_edge_y=self.accum_plus_spin.value())
            display_angle = self._mode_corrected_angle(angle)
            self.image_canvas.draw_beam_line(edge_minus, display_angle)
            self.image_canvas.draw_beam_line(edge_plus, display_angle)
            return

        center = self._pos_move_display_offset()
        self.image_canvas.draw_beam_line(center, self._mode_corrected_angle(angle))

    # ------------------------------------------------------------------
    # Azimuthal Process geometry overlay -- port of console_widget.py's own
    # `_on_azi_draw_clicked`/Position button, using the same "armed" guard
    # pattern as `_arm_1d_overlay` above so the ring doesn't auto-appear
    # just because calibration happens to be in place.
    # ------------------------------------------------------------------
    def _arm_azi_overlay(self) -> None:
        self._active_overlay = "azimuthal"
        self._redraw_overlays()

    def _draw_azi_overlay(self) -> None:
        """Additive (does not clear first) -- always called by way of
        `_redraw_overlays`, which clears once up front."""

        session = gs.get_session()
        if session.data is None:
            return
        if not (self._pixel_centered or session.corrected):
            return
        if self._active_overlay != "azimuthal":
            return
        center = self._current_beam_center()
        pixel = self.azi_pix_spin.value()
        r_minus = max(0.0, pixel + self.azi_minus_spin.value())
        r_plus = max(0.0, pixel + self.azi_plus_spin.value())
        self.image_canvas.draw_beam_circle(center, r_minus)
        self.image_canvas.draw_beam_circle(center, r_plus)

    # ------------------------------------------------------------------
    # Click-to-position (1D Ang / Azimuthal pix) -- port of
    # console_widget.py's own `_arm_position_mode`/`_on_image_clicked`
    # line/azimuthal branches.
    # ------------------------------------------------------------------
    def _arm_position_mode(self, mode: str) -> None:
        self._position_mode = mode
        if mode == "line":
            self._set_status("Click a point on the 2D image to set 1D Ang.", True)
        elif mode == "azimuthal":
            self._set_status("Click a point on the 2D image to set the azimuthal pix.", True)

    def _on_image_clicked(self, x: float, y: float) -> None:
        if self._position_mode is None:
            return
        if self.image_canvas.toolbar.mode:
            return
        if self._position_mode == "line":
            beam_y, beam_z = self._current_beam_center()
            raw_angle = (math.degrees(math.atan2(-(y - beam_z), -(x - beam_y))) + 360.0) % 360.0
            angle = self._mode_corrected_angle(raw_angle)
            if angle > 180.0:
                # phi_spin's own range is -180..180 (unlike console's wider
                # -360..360 line_angle_spin) -- fold the equivalent angle
                # back into that range rather than letting setValue clamp
                # it (silently wrong) at the 180 boundary.
                angle -= 360.0
            self.phi_spin.setValue(angle)
            self._set_status(f"1D Ang position clicked at ({x:.1f}, {y:.1f}) -> angle {angle:.2f} deg.", True)
        elif self._position_mode == "azimuthal":
            beam_y, beam_z = self._current_beam_center()
            radius = math.hypot(x - beam_y, y - beam_z)
            self.azi_pix_spin.setValue(radius)
            self._set_status(f"Azimuthal position clicked at ({x:.1f}, {y:.1f}) -> pix {radius:.1f}.", True)

    # ------------------------------------------------------------------
    # Play (Frame Viewer) -- port of FrameImage.m: a separate slider-driven
    # window that steps through every loaded frame one at a time, all
    # rendered with the same contrast/colormap/scale/axis so frames are
    # visually comparable rather than each auto-contrasting to its own
    # range.
    # ------------------------------------------------------------------
    def _on_play_clicked(self) -> None:
        session = gs.get_session()
        # Raw frames plus the shared correction factor: q-Cal no longer
        # materializes a corrected copy of the whole series, so each
        # frame is corrected as it is displayed (gs.corrected_frame).
        stack = session.corrected_stack if session.corrected_stack is not None else session.stack
        if stack is None or stack.shape[0] == 0:
            self._set_status("Load a series first.", False)
            return

        n_frames = stack.shape[0]
        # Snapshot the main viewer's own current display settings rather
        # than staying live-synced while the dialog is open -- matches
        # FrameImage.m taking a one-time copy of `param.cmin/cmax/colorm/
        # colorscale` into its own window instead of reacting to further
        # edits on the main GUI.
        cmin = self.image_cmin_spin.value()
        cmax = self.image_cmax_spin.value()
        vmin = cmin if cmax > cmin else None
        vmax = cmax if cmax > cmin else None
        cmap = self.image_colormap_combo.currentText()
        log_scale = self.image_scale_combo.currentText() == "Log"
        extent = self._image_extent(stack.shape[1:])

        dialog = QDialog(self)
        dialog.setWindowTitle("Frame Viewer (Play)")
        dialog.resize(700, 680)
        layout = QVBoxLayout(dialog)

        canvas = ImageCanvas()
        layout.addWidget(canvas)
        frame_label = QLabel()
        frame_label.setObjectName("play_frame_label")
        frame_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(frame_label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(n_frames)
        slider.setValue(1)
        layout.addWidget(slider)

        def render_frame(value: int) -> None:
            idx = max(1, min(n_frames, value)) - 1
            frame_label.setText(f"Frame: {idx + 1} / {n_frames}")
            canvas.render(
                gs.corrected_frame(idx), cmap=cmap, log_scale=log_scale, vmin=vmin, vmax=vmax,
                extent=extent, xlabel="Y, pixel", ylabel="Z, pixel", keep_limits=idx > 0,
            )

        slider.valueChanged.connect(render_frame)
        render_frame(1)

        def _on_finished(_result: int) -> None:
            # Disconnect first so closing mid-drag can't queue another
            # matplotlib draw_idle() after this point, then defer the
            # actual C++ deletion by one more event-loop tick so any
            # draw_idle already queued by the last render_frame() gets a
            # chance to run against a still-live canvas -- otherwise it
            # fires after the dialog (and its canvas) is gone, and
            # shiboken raises "Internal C++ object already deleted".
            try:
                slider.valueChanged.disconnect(render_frame)
            except (TypeError, RuntimeError):
                pass
            if self._play_dialog is dialog:
                self._play_dialog = None
            QTimer.singleShot(0, dialog.deleteLater)

        dialog.finished.connect(_on_finished)
        self._play_dialog = dialog
        dialog.show()

    # ------------------------------------------------------------------
    # Incidence angle calibration
    # ------------------------------------------------------------------
    def _on_zplot_clicked(self) -> None:
        result = gs.gixs2_z_plot({"y_pos": self.y_pos_spin.value()})
        if not result.get("ok"):
            self._set_status(result.get("message", "Z-Plot failed."), False)
            return
        self.oned_canvas.render(np.asarray(result["x"]), np.asarray(result["y"]), xlabel="Z, pixel (relative)", ylabel="Intensity, arb. units")
        # Guard against a duplicate connection if Z-Plot is clicked again
        # before a peak was ever picked -- point_clicked.connect() would
        # otherwise stack a second handler and double-fire the next click.
        try:
            self.oned_canvas.point_clicked.disconnect(self._on_zplot_curve_clicked)
        except (TypeError, RuntimeError):
            pass
        self.oned_canvas.point_clicked.connect(self._on_zplot_curve_clicked)
        self._set_status("Click the specular peak on the 1D Plot Viewer.", True)

    def _on_zplot_curve_clicked(self, x: float, _y: float) -> None:
        try:
            self.oned_canvas.point_clicked.disconnect(self._on_zplot_curve_clicked)
        except (TypeError, RuntimeError):
            pass
        self.ai_pix_spin.setValue(abs(x))
        self._on_ai_pix_changed()

    def _on_ai_pix_changed(self) -> None:
        result = gs.gixs2_ai_from_pixel({"pix": self.ai_pix_spin.value()})
        if not result.get("ok"):
            self._set_status(result.get("message", "ai conversion failed."), False)
            return
        self.ai_deg_spin.blockSignals(True)
        self.ai_deg_spin.setValue(result["ai_deg"])
        self.ai_deg_spin.blockSignals(False)

    def _on_ai_deg_changed(self) -> None:
        result = gs.gixs2_ai_from_degree({"ai_deg": self.ai_deg_spin.value()})
        if not result.get("ok"):
            self._set_status(result.get("message", "ai conversion failed."), False)
            return
        self.ai_pix_spin.blockSignals(True)
        self.ai_pix_spin.setValue(result["pix"])
        self.ai_pix_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # q-Cal
    # ------------------------------------------------------------------
    def _on_qcal_clicked(self) -> None:
        # "Change ai" recomputes a whole new q-map per frame instead of
        # one shared map for the series -- run it on a background thread
        # with the same progress dialog Exec Cut/Exec Azi/Auto Refresh
        # already use, instead of freezing the UI for however long that
        # per-frame loop takes (reported: "Change ai가 check가 되어 step이
        # 있는 경우 q-map을 하나하나 계산해야돼... process box가 있으면
        # 좋을 것 같아"). The plain (non-Change-ai) path is a single shared
        # q-map + one broadcast multiply either way, so it just finishes
        # near-instantly with the dialog barely visible.
        payload = {
            "incident_angle_deg": self.ai_deg_spin.value(),
            "change_ai": self.change_ai_check.isChecked(),
            "ai_step": self.ai_step_spin.value(),
        }
        self.qcal_button.setEnabled(False)
        self._start_progress_dialog("Running q-Cal...", window_title="q-Cal")
        self._qcal_thread = run_in_background(gs.gixs2_qcal, payload, self._on_qcal_finished)

    def _on_qcal_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.qcal_button.setEnabled(True)
        if not result.get("ok"):
            self._mark_ok(self.qcal_button, False)
            self._set_status(result.get("message", "q-Cal failed."), False)
            return
        self._current_image_data = None
        self._render_image()
        self._mark_ok(self.qcal_button, True)
        note = " (Change ai: per-frame Q-map)" if result.get("change_ai") else ""
        self._set_status(f"q-Cal applied to the whole series.{note}", True)

    # ------------------------------------------------------------------
    # 1D / Azimuthal exec
    # ------------------------------------------------------------------
    def _on_exec_1d_clicked(self) -> None:
        # A real series can be hundreds to thousands of frames -- run the
        # per-frame cut loop on a background thread with a progress dialog
        # (same QThread + polling pattern series loading already uses)
        # instead of freezing the UI for however long that takes.
        self._arm_1d_overlay()
        payload = {
            "angle_deg": self.phi_spin.value(),
            "cakecut": self.cakecut_check.isChecked(),
            "range_minus": self.cakecut_minus_spin.value(),
            "range_plus": self.cakecut_plus_spin.value(),
            "y_pos_move": self.y_pos_move_spin.value(),
            "z_pos_move": self.z_pos_move_spin.value(),
            "accumulation": self.accum_check.isChecked(),
            "accum_minus": self.accum_minus_spin.value(),
            "accum_plus": self.accum_plus_spin.value(),
            "parallel": self.exec_1d_parallel_check.isChecked(),
        }
        self.exec_1d_button.setEnabled(False)
        self._start_progress_dialog("Running 1D cut...", window_title="Exec Cut")
        self._exec_1d_thread = run_in_background(gs.gixs2_linecut_exec, payload, self._on_exec_1d_finished)

    def _on_exec_1d_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.exec_1d_button.setEnabled(True)
        if not result.get("ok"):
            self._set_status(result.get("message", "Exec Cut failed."), False)
            return
        self._render_1d()
        self._set_status(f"1D series: {result['n_frames']} frames x {result['n_points']} points.", True)

    def _render_1d(self) -> None:
        """Port of `update1Dplot_Callback`: every frame's own cut plotted at
        once, color-coded by the series' X-axis value (Time/Temperature/
        Angle -- "Frame index" until Set X-Axis has run) with a colorbar,
        rather than only the series mean. Defaults to `viridis` in place of
        MATLAB's `jet`, which is not perceptually uniform (banding/false
        edges) and is a poor choice for the sequential Time/Temperature
        progression this plot exists to show -- the Colormap combo lets the
        user pick a different one."""

        session = gs.get_session()
        if session.last_1d is None:
            return
        matrix = gs.order_frames(
            gs.gixs2_normalized_matrix(session.last_1d["intensity"], self.normalize_1d_combo.currentText())
        )
        color_values, colorbar_label = self._series_color_values(matrix.shape[0])
        scale = self.oned_scale_combo.currentText()
        xlog = scale == "Logxy"
        ylog = scale in ("Logy", "Logxy")
        self.oned_canvas.render_series(
            session.last_1d["x_axis"], matrix, color_values,
            xlabel="Q, 1/A", ylabel="Intensity, arb. units", colorbar_label=colorbar_label,
            cmap=self.oned_colormap_combo.currentText(), xlog=xlog, ylog=ylog,
        )
        self._render_integration()

    def _series_color_values(self, n_frames: int) -> tuple[np.ndarray, str]:
        """The per-frame value/label a series-of-curves plot color-codes
        by -- the series X-axis (Time/Temperature/Angle), recomputed live
        every render.

        MATLAB's `update1Dplot_Callback`/`updateAziIntplot` call
        `getExperimentXAxis(param)` directly, inline, every time they
        redraw -- there is no separate "commit" step gating it. Recomputing
        via `gixs2_set_x_axis` here (using whichever sub-mode/Order the
        left panel is currently set to) matches that: Time/Temperature/
        Angle shows up as soon as a loaded series has those fields, with no
        need to remember to press Set X-Axis first. Falls back to a plain
        frame index (matches MATLAB's own `X_var = 1:size(...)` fallback)
        only when no series is loaded yet or the axis genuinely can't be
        computed (e.g. an unrecognized Insitu mode)."""

        session = gs.get_session()
        if session.timestamps is not None:
            payload = {"sub_mode": self.sub_mode_combo.currentText(), "order": self.order_check.isChecked()}
            result = gs.gixs2_set_x_axis(payload)
            if result.get("ok") and session.x_axis_values is not None and session.x_axis_values.shape[0] == n_frames:
                return session.x_axis_values, (session.x_axis_label or "Frame")
        return np.arange(1, n_frames + 1, dtype=np.float64), "Frame, #"

    def _on_exec_azi_clicked(self) -> None:
        # Same background-thread + progress-dialog pattern as Exec Cut
        # (_on_exec_1d_clicked) -- a real series can be hundreds to
        # thousands of frames, so run the per-frame azimuthal loop off the
        # UI thread instead of freezing the window for however long that
        # takes.
        self._arm_azi_overlay()
        payload = {
            "pixel": self.azi_pix_spin.value(),
            "range_minus": self.azi_minus_spin.value(),
            "range_plus": self.azi_plus_spin.value(),
            "mean_or_median": self.azi_mean_median_combo.currentText(),
            "reflect_missing_wedge": self.azi_reflect_missing_wedge_check.isChecked(),
        }
        self.exec_azi_button.setEnabled(False)
        self._start_progress_dialog("Running azimuthal profile...", window_title="Exec Azi")
        self._exec_azi_thread = run_in_background(gs.gixs2_azimuthal_exec, payload, self._on_exec_azi_finished)

    def _on_exec_azi_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.exec_azi_button.setEnabled(True)
        if not result.get("ok"):
            self._set_status(result.get("message", "Exec Azi failed."), False)
            return
        self._render_azimuthal()
        self._set_status(f"Azimuthal series: {result['n_frames']} frames x {result['n_points']} points.", True)

    def _render_azimuthal(self) -> None:
        """Same series-of-curves + colorbar treatment as `_render_1d`,
        applied to the Azimuthal Plot Viewer: every frame's own azimuthal
        profile plotted at once, color-coded by the series X-axis value."""

        session = gs.get_session()
        if session.last_azimuthal is None:
            return
        matrix = gs.order_frames(
            gs.gixs2_normalized_matrix(session.last_azimuthal["intensity"], self.normalize_azi_combo.currentText())
        )
        color_values, colorbar_label = self._series_color_values(matrix.shape[0])
        self.azi_canvas.render_series(
            session.last_azimuthal["x_axis"], matrix, color_values,
            xlabel="Angle, deg", ylabel="Intensity, arb. units", colorbar_label=colorbar_label,
            cmap=self.azi_colormap_combo.currentText(),
        )
        self._render_integration()

    # ------------------------------------------------------------------
    # Heatmap -- port of MATLAB's HeatmapQ_gui/HeatmapAzi_gui: a separate
    # window showing every frame's own curve stacked into a Q/Angle-vs-
    # series-X-axis 2D heatmap, instead of the overlaid-curves view the
    # main 1D/Azimuthal Plot Viewers show. Snapshots the data at open time
    # (matches _on_play_clicked's own Frame Viewer) rather than staying
    # live-linked to further Exec Cut/Exec Azi runs.
    # ------------------------------------------------------------------
    def _on_1d_heatmap_clicked(self) -> None:
        session = gs.get_session()
        if session.last_1d is None:
            self._set_status("Run Exec Cut first.", False)
            return
        matrix = gs.gixs2_normalized_matrix(session.last_1d["intensity"], self.normalize_1d_combo.currentText())
        self._open_heatmap_dialog(
            "1D Heatmap", session.last_1d["x_axis"], "Q, 1/A", matrix, "1D",
        )

    def _on_azi_heatmap_clicked(self) -> None:
        session = gs.get_session()
        if session.last_azimuthal is None:
            self._set_status("Run Exec Azi first.", False)
            return
        matrix = gs.gixs2_normalized_matrix(session.last_azimuthal["intensity"], self.normalize_azi_combo.currentText())
        self._open_heatmap_dialog(
            "Azimuthal Heatmap", session.last_azimuthal["x_axis"], "Angle, deg", matrix, "Azimuthal",
        )

    def _open_heatmap_dialog(self, title: str, y_axis: np.ndarray, y_label: str, matrix: np.ndarray, kind: str) -> None:
        x_axis, x_label = self._series_color_values(matrix.shape[0])
        if x_axis.shape[0] != matrix.shape[0] or y_axis.shape[0] != matrix.shape[1]:
            self._set_status("Heatmap: series X-axis doesn't match the curve data.", False)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(900, 760)
        layout = QVBoxLayout(dialog)

        canvas = ImageCanvas()
        layout.addWidget(canvas)

        plot_type_combo = QComboBox()
        plot_type_combo.addItems(["Heatmap", "Surf", "Waterfall"])
        colormap_combo = QComboBox()
        colormap_combo.addItems(_COLORMAPS)
        default_index = colormap_combo.findText("viridis")
        if default_index >= 0:
            colormap_combo.setCurrentIndex(default_index)
        # X/Y Scale/Range are the plot's own axes (the series X-axis and
        # Q/Angle) -- Z Scale/Contrast (below) are the separate color
        # scale, matching MATLAB's own distinct X/Y/Z Scale controls.
        xscale_combo = QComboBox()
        xscale_combo.addItems(["Linear", "Log"])
        yscale_combo = QComboBox()
        yscale_combo.addItems(["Linear", "Log"])
        scale_combo = QComboBox()
        scale_combo.addItems(["Linear", "Log"])
        # Waterfall-only: color each curtain by its own frame's position
        # (0..n-1) instead of its mean intensity -- a long series where
        # frame-to-frame intensity barely varies otherwise leaves every
        # curtain clustered near the same color.
        curtain_color_combo = QComboBox()
        curtain_color_combo.addItems(["Intensity", "Frame Index"])
        curtain_color_combo.setEnabled(False)
        for combo in (plot_type_combo, colormap_combo, xscale_combo, yscale_combo, scale_combo, curtain_color_combo):
            _size_combo(combo, 12)

        x_lo, x_hi = _finite_range(x_axis)
        y_lo, y_hi = _finite_range(y_axis)
        xmin_spin = _double_spin(-1.0e12, 1.0e12, 4, x_lo, 1.0)
        xmax_spin = _double_spin(-1.0e12, 1.0e12, 4, x_hi, 1.0)
        ymin_spin = _double_spin(-1.0e12, 1.0e12, 4, y_lo, 0.01)
        ymax_spin = _double_spin(-1.0e12, 1.0e12, 4, y_hi, 0.01)
        cmin_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        cmax_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        save_image_button = QPushButton("Save Image")
        save_image_button.setToolTip(
            "Save the current Heatmap/Surf/Waterfall view as a PNG (also available via the toolbar's own save icon)."
        )

        ctrl_row1 = QHBoxLayout()
        ctrl_row1.addWidget(_field("Plot Type", plot_type_combo, label_width=64))
        ctrl_row1.addWidget(_field("Colormap", colormap_combo, label_width=64))
        ctrl_row1.addWidget(_field("X Scale", xscale_combo, label_width=54))
        ctrl_row1.addWidget(_field("Y Scale", yscale_combo, label_width=54))
        ctrl_row1.addWidget(_field("Z Scale", scale_combo, label_width=54))
        ctrl_row1.addWidget(_field("Curtain Color", curtain_color_combo, label_width=78))
        layout.addLayout(ctrl_row1)

        ctrl_row2 = QHBoxLayout()
        ctrl_row2.addWidget(_field("X Range", xmin_spin, xmax_spin, label_width=54))
        ctrl_row2.addWidget(_field("Y Range", ymin_spin, ymax_spin, label_width=54))
        ctrl_row2.addWidget(_field("Contrast", cmin_spin, cmax_spin, label_width=54))
        ctrl_row2.addWidget(save_image_button)
        layout.addLayout(ctrl_row2)
        status_label = _muted("")
        layout.addWidget(status_label)

        def save_image() -> None:
            default_name = f"{title.replace(' ', '_')}.png"
            path, _ = QFileDialog.getSaveFileName(
                dialog, "Save Heatmap Image", str(Path(self._last_dir) / default_name), "PNG Image (*.png)"
            )
            if not path:
                return
            try:
                canvas.save_figure(path)
            except Exception as exc:  # noqa: BLE001
                status_label.setText(f"Image save failed: {exc}")
                return
            status_label.setText(f"Saved {path}.")

        save_image_button.clicked.connect(save_image)

        def update() -> None:
            cmin = cmin_spin.value()
            cmax = cmax_spin.value()
            vmin = cmin if cmax > cmin else None
            vmax = cmax if cmax > cmin else None
            xmin, xmax = xmin_spin.value(), xmax_spin.value()
            ymin, ymax = ymin_spin.value(), ymax_spin.value()
            plot_type = plot_type_combo.currentText()
            curtain_color_combo.setEnabled(plot_type == "Waterfall")
            common = dict(
                cmap=colormap_combo.currentText(),
                log_scale=scale_combo.currentText() == "Log",
                vmin=vmin, vmax=vmax,
                xlabel=x_label, ylabel=y_label,
                title=f"{kind}: {matrix.shape[0]} frame(s)",
                xscale=xscale_combo.currentText().lower(),
                yscale=yscale_combo.currentText().lower(),
                xlim=(xmin, xmax) if xmax > xmin else None,
                ylim=(ymin, ymax) if ymax > ymin else None,
            )
            if plot_type == "Heatmap":
                used_vmin, used_vmax = canvas.render_mesh(x_axis, y_axis, matrix.T, **common)
                status_label.setText("")
            else:
                by_index = plot_type == "Waterfall" and curtain_color_combo.currentText() == "Frame Index"
                used_vmin, used_vmax = canvas.render_surface(
                    x_axis, y_axis, matrix.T,
                    mode="surf" if plot_type == "Surf" else "waterfall",
                    color_by="index" if by_index else "intensity",
                    colorbar_label=f"Frame # (0-{matrix.shape[0] - 1})" if by_index else "",
                    zlabel="Intensity, arb. units", **common,
                )
                if canvas.axis_log_fallback:
                    axes_text = "/".join(canvas.axis_log_fallback)
                    status_label.setText(
                        f"{axes_text} axis has non-positive values -- Log Scale not applied, stayed Linear."
                    )
                else:
                    status_label.setText("")
            if vmin is None:
                cmin_spin.blockSignals(True)
                cmax_spin.blockSignals(True)
                cmin_spin.setValue(used_vmin)
                cmax_spin.setValue(used_vmax)
                cmin_spin.blockSignals(False)
                cmax_spin.blockSignals(False)

        for combo in (plot_type_combo, colormap_combo, xscale_combo, yscale_combo, scale_combo, curtain_color_combo):
            combo.currentTextChanged.connect(lambda _t: update())
        for spin in (xmin_spin, xmax_spin, ymin_spin, ymax_spin, cmin_spin, cmax_spin):
            spin.valueChanged.connect(lambda _v: update())

        update()

        def _on_finished(_result: int) -> None:
            # Same shiboken-crash guard as the Play dialog (_on_play_clicked):
            # disconnect every live control first so closing mid-interaction
            # can't queue another draw_idle(), then defer the actual C++
            # deletion by one event-loop tick so any already-queued redraw
            # runs against a still-live canvas.
            for combo in (plot_type_combo, colormap_combo, xscale_combo, yscale_combo, scale_combo, curtain_color_combo):
                try:
                    combo.currentTextChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass
            for spin in (xmin_spin, xmax_spin, ymin_spin, ymax_spin, cmin_spin, cmax_spin):
                try:
                    spin.valueChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass
            if self._heatmap_dialog is dialog:
                self._heatmap_dialog = None
            QTimer.singleShot(0, dialog.deleteLater)

        dialog.finished.connect(_on_finished)
        self._heatmap_dialog = dialog
        dialog.show()

    # ------------------------------------------------------------------
    # ROI -- port of MATLAB `ROIdef2_Callback`: a separate "Draw ROI"
    # window shows the averaged image, and rectangles are drawn one ROI at
    # a time (drag, not two clicks), each in its own fixed color -- the
    # same color then marks that ROI's box on the main 2D image and its
    # curve in the Series Integration Viewer, so a box and its curve are
    # visually linked (`_ROI_COLORS`).
    # ------------------------------------------------------------------
    def _display_xy_to_raw_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Convert a click's displayed (x, y) back to a raw (col, row) pixel
        index into the stack array -- identity when the axis is still raw
        pixel indices, or the inverse of `_render_image`'s beam-centered
        extent formula (`h = col - beamY`, `v = beamZ - row`) once Load Info
        has recentered the display on the beam position."""

        if not self._pixel_centered:
            return x, y
        beam_y, beam_z = gs.get_session().calibration["beam_pos"]
        return x + beam_y, beam_z - y

    def _raw_pixel_to_display_xy(self, col: float, row: float) -> tuple[float, float]:
        """Inverse of `_display_xy_to_raw_pixel` -- a raw (col, row) pixel
        index to the currently-displayed (x, y), for redrawing a stored
        ROI's box after the axis has (or hasn't) been beam-centered."""

        if not self._pixel_centered:
            return col, row
        beam_y, beam_z = gs.get_session().calibration["beam_pos"]
        return col - beam_y, beam_z - row

    def _draw_roi_overlay(self) -> None:
        """Redraw every defined ROI's colored box on the main 2D Image Plot
        Viewer -- called after Exec ROI succeeds and again at the tail of
        `_render_image` (which just fully rebuilt the figure/axes, wiping
        any previous overlay, same as `_draw_1d_overlay`)."""

        session = gs.get_session()
        if not session.roi_list or session.data is None:
            return
        for i, (r1, r2, c1, c2) in enumerate(session.roi_list):
            x1, y1 = self._raw_pixel_to_display_xy(c1, r1)
            x2, y2 = self._raw_pixel_to_display_xy(c2, r2)
            self.image_canvas.draw_roi_box(x1, x2, y1, y2, _ROI_COLORS[i % len(_ROI_COLORS)], f"ROI {i + 1}")

    def _on_draw_roi_clicked(self) -> None:
        session = gs.get_session()
        if session.data is None:
            self._set_status("Load a series first.", False)
            return

        n_roi = int(self.n_roi_combo.currentText())
        pending_rois: list[tuple[int, int, int, int]] = []

        dialog = QDialog(self)
        dialog.setWindowTitle("Draw ROI")
        dialog.resize(720, 760)
        layout = QVBoxLayout(dialog)
        canvas = ImageCanvas()
        layout.addWidget(canvas)
        status_label = _muted("")
        layout.addWidget(status_label)

        extent = self._image_extent(session.data.shape)
        canvas.render(
            session.data, cmap=self.image_colormap_combo.currentText(),
            log_scale=self.image_scale_combo.currentText() == "Log",
            extent=extent, xlabel="Y, pixel", ylabel="Z, pixel",
        )

        def advance(index: int) -> None:
            if index >= n_roi:
                dialog.accept()
                return
            color = _ROI_COLORS[index % len(_ROI_COLORS)]
            status_label.setText(f"ROI {index + 1} / {n_roi}: drag a rectangle.")

            def on_complete(x1: float, y1: float, x2: float, y2: float) -> None:
                col1, row1 = self._display_xy_to_raw_pixel(x1, y1)
                col2, row2 = self._display_xy_to_raw_pixel(x2, y2)
                r1, r2 = sorted((int(round(row1)), int(round(row2))))
                c1, c2 = sorted((int(round(col1)), int(round(col2))))
                pending_rois.append((r1, r2, c1, c2))
                canvas.draw_roi_box(x1, x2, y1, y2, color, f"ROI {index + 1}")
                advance(index + 1)

            canvas.start_rectangle_selector(on_complete, color=color)

        advance(0)
        self._roi_dialog = dialog
        accepted = dialog.exec() == QDialog.Accepted
        canvas.stop_rectangle_selector()
        self._roi_dialog = None
        # Same shiboken-crash guard as the Play/Heatmap dialogs: the last
        # completed ROI's draw_roi_box() queued a draw_idle() just before
        # accept() closed the dialog -- defer the canvas's actual deletion
        # by one event-loop tick so that redraw runs first.
        QTimer.singleShot(0, dialog.deleteLater)

        if not accepted or len(pending_rois) < n_roi:
            self._set_status("ROI drawing cancelled.", False)
            return

        result = gs.gixs2_roi_exec({"rois": pending_rois})
        if not result.get("ok"):
            self._set_status(result.get("message", "Exec ROI failed."), False)
            return
        # _redraw_overlays() clears every overlay before redrawing 1D/Azi/ROI
        # in order -- avoids stacking this run's ROI boxes on top of a
        # previous run's without ever clearing them.
        self._redraw_overlays()
        self._render_integration()
        self._set_status(f"ROI series: {result['n_frames']} frames x {result['n_roi']} ROI(s).", True)

    # ------------------------------------------------------------------
    # X-Axis / Auto Refresh
    # ------------------------------------------------------------------
    def _on_set_x_axis_clicked(self) -> None:
        payload = {"sub_mode": self.sub_mode_combo.currentText(), "order": self.order_check.isChecked()}
        result = gs.gixs2_set_x_axis(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "Set X-Axis failed."), False)
            return
        # The 1D and Azimuthal Plot Viewers both color-code their curves by
        # the series X-axis value/label -- re-render whichever has already
        # been run so they pick up the value just set instead of staying on
        # the "Frame, #" fallback until the next unrelated Exec Cut/Azi.
        # Each already re-renders the Series Integration Viewer too; if
        # neither has run yet (e.g. only ROI so far), refresh it directly.
        session = gs.get_session()
        rendered_integration = False
        if session.last_1d is not None:
            self._render_1d()
            rendered_integration = True
        if session.last_azimuthal is not None:
            self._render_azimuthal()
            rendered_integration = True
        if not rendered_integration:
            self._render_integration()
        self._set_status(f"X-Axis set to {result['label']} ({len(result['x_axis'])} points).", True)

    def _on_auto_refresh_clicked(self) -> None:
        # Loading + correcting + reprocessing however many new files showed
        # up can take a while for a real series -- run it on a background
        # thread with the same progress dialog series loading/Exec Cut use,
        # instead of freezing the UI with no feedback while it works.
        self.auto_refresh_button.setEnabled(False)
        self._start_progress_dialog("Checking for new files...", window_title="Auto Refresh")
        payload = {"directory": self._last_dir, "parallel": self.parallel_check.isChecked()}
        self._auto_refresh_thread = run_in_background(gs.gixs2_auto_refresh, payload, self._on_auto_refresh_finished)

    def _on_auto_refresh_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.auto_refresh_button.setEnabled(True)
        if not result.get("ok"):
            self._set_status(result.get("message", "Auto Refresh failed."), False)
            return
        if result["n_new"] == 0:
            self._set_status("No new files.", True)
            return
        self.n_files_label.setText(f"# of Files: {result['n_total']}")
        # Re-derive the series X-axis for the new frame count right away.
        # The plot paths recompute it themselves, but Save 1D/Azi/Integ. read
        # `x_axis_values` straight from the session -- left at its pre-refresh
        # length, `save_series_data` rejects the mismatch and the save fails.
        # MATLAB avoids this by recomputing the axis inline at every save.
        if gs.get_session().timestamps is not None:
            gs.gixs2_set_x_axis({"sub_mode": self.sub_mode_combo.currentText(), "order": self.order_check.isChecked()})
        self._current_image_data = None
        self._render_image()
        if result.get("curve_extended"):
            # Port of `Autoplot_Callback`'s `switch Auto.Met` dispatch:
            # re-render only whichever of 1D/Azimuthal/ROI was the process
            # actually just extended, not both unconditionally -- a
            # ROI-only workflow (no prior Exec Cut/Exec Azi) has neither
            # `last_1d` nor `last_azimuthal` set, so `_render_1d`/
            # `_render_azimuthal` would both silently no-op and the Series
            # Integration Viewer would never pick up the newly-appended
            # ROI sums.
            auto_met = gs.get_session().auto_met
            if auto_met == "linecut":
                self._render_1d()
            elif auto_met == "azimuthal":
                self._render_azimuthal()
            elif auto_met == "roi":
                self._render_integration()
        self._set_status(f"Auto Refresh: +{result['n_new']} frame(s), total {result['n_total']}.", True)

    # ------------------------------------------------------------------
    # Series Integration Viewer
    # ------------------------------------------------------------------
    def _render_integration(self) -> None:
        session = gs.get_session()
        # Recompute live, same as `_series_color_values` does for the 1D/
        # Azimuthal Plot Viewers -- otherwise a stale `x_axis_values` (still
        # sized to whatever it was when "Set X-Axis" was last clicked)
        # mismatches a `last_roi`/`last_1d`/`last_azimuthal` that Auto
        # Refresh just extended, and the shape guards below silently drop
        # the redraw instead of showing the newly-appended frames.
        if session.timestamps is not None:
            gs.gixs2_set_x_axis({"sub_mode": self.sub_mode_combo.currentText(), "order": self.order_check.isChecked()})
        if session.x_axis_values is None:
            return
        scale = self.integ_scale_combo.currentText()
        xlog = scale == "Logxy"
        ylog = scale in ("Logy", "Logxy")
        auto_met = session.auto_met
        if auto_met == "roi" and session.last_roi is not None:
            # Port of MATLAB PlotROI_Callback: each ROI is its own curve in
            # its own color (matching its box on the 2D image), not one
            # combined sum -- see render_multi.
            if session.last_roi.shape[0] != session.x_axis_values.shape[0]:
                return
            n_roi = session.last_roi.shape[1]
            labels = [f"ROI {i + 1}" for i in range(n_roi)]
            colors = [_ROI_COLORS[i % len(_ROI_COLORS)] for i in range(n_roi)]
            matrix = gs.order_frames(gs.gixs2_normalized_matrix(session.last_roi, self.normalize_roi_combo.currentText()))
            self.integ_canvas.render_multi(
                session.x_axis_values, matrix.T, labels, colors,
                xlabel=session.x_axis_label, ylabel="Sum(ROI), arb. units",
                xlog=xlog, ylog=ylog,
            )
            return
        elif auto_met == "azimuthal" and session.last_azimuthal is not None:
            matrix = gs.order_frames(gs.gixs2_normalized_matrix(session.last_azimuthal["intensity"], self.normalize_azi_combo.currentText()))
            y = np.nansum(matrix, axis=1)
            ylabel = "Integrated Azimuthal, arb. units"
        elif session.last_1d is not None:
            matrix = gs.order_frames(gs.gixs2_normalized_matrix(session.last_1d["intensity"], self.normalize_1d_combo.currentText()))
            y = np.nansum(matrix, axis=1)
            ylabel = "Integrated 1D, arb. units"
        else:
            return
        if y.shape[0] != session.x_axis_values.shape[0]:
            return
        self.integ_canvas.render(
            session.x_axis_values, y, xlabel=session.x_axis_label, ylabel=ylabel, xlog=xlog, ylog=ylog, marker="o"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _integ_normalize(self) -> str:
        # The same combo the Integration plot uses for the current mode.
        auto_met = gs.get_session().auto_met
        if auto_met == "roi":
            return self.normalize_roi_combo.currentText()
        if auto_met == "azimuthal":
            return self.normalize_azi_combo.currentText()
        return self.normalize_1d_combo.currentText()

    def _save(self, save_fn, normalize: str | None = None, canvas: CurveCanvas | None = None) -> None:
        payload = {"directory": self._last_dir}
        if normalize is not None:
            payload["normalize"] = normalize
        result = save_fn(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "Save failed."), False)
            return
        data_path = Path(result["path"])
        if canvas is None:
            self._set_status(f"Saved {data_path}.", True)
            return
        # Port of MATLAB's own `savePanelSnapshot` (called from
        # `Saveget3_Callback`/`Saveget2_Callback`/`Saveget4_Callback`, the
        # Save 1D/Azi/Integ. handlers): each data export gets a same-named
        # image alongside it, same convention as the single-frame
        # console's own `_on_curve_save_clicked`. MATLAB re-plots into a
        # throwaway figure to work around `uiaxes` not supporting
        # `copyobj`; that workaround doesn't apply here since `canvas`'s
        # own `figure` already *is* a real matplotlib Figure, showing
        # exactly what's on screen.
        image_path = data_path.with_name(f"{data_path.stem}.png")
        try:
            canvas.save_figure(str(image_path))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Saved {data_path}, but image export failed: {exc}", False)
            return
        self._set_status(f"Saved {data_path} and {image_path}.", True)
