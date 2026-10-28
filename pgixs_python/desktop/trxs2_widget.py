"""TrXS2 tab: port of MATLAB `TrXS2_gui.m`'s series-analysis workflow --
Circular Average (the primary 1D mode), 1D Line Ang, Azimuthal Process, ROI
Process, X-Axis/Auto Refresh, Play (Frame Viewer), Q/Azimuthal Heatmap
dialogs, and a live Background Subtraction pipeline with point-click
normalization.

Structured like `gixs2_widget.py` (its own direct architectural template --
`_build_left_panel`/`_build_right_panel`, the same `QSplitter` 4-quadrant
layout, the same `run_in_background` progress-dialog pattern), adapted for
TrXS's simpler geometry: no incidence-angle/"Change ai" calibration branch at
all (TrXS is normal-incidence transmission), no Cakecut/Accumulation/Pos-move
(TrXS's own single-frame console config, `trxs_config()`, already sets
`has_cakecut=False, has_pos_move_and_accumulation=False`), and one
genuinely new piece GIXS2 has no equivalent of: a Background series
(averaged once, corrected, cut once) that a live Subtraction checkbox
cheaply re-applies over the already-computed stack curve matrix.

Module-level helpers (`_control_row`, `_equalize_control_row_heights`,
`_finite_range`) are duplicated here rather than imported from
`gixs2_widget.py`, matching this codebase's established convention of
small-helper duplication over cross-widget coupling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
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
    QSizePolicy,
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
from pgixs_python.ui_backend import trxs2_session as ts


# Auto Refresh Watch: default poll interval, and how long a new file must sit
# unmodified before it is loaded -- the detector may still be writing it.
_WATCH_DEFAULT_S = 10.0
_WATCH_SETTLE_S = 3.0

_DETECTOR_LIST = ["Eiger4M", "MarCCD", "MX170HS", "PI_CCD"]
# TrXS is normal-incidence transmission -- only Time/Temperature series make
# sense (no Angle-scan/LowTemp. branch, unlike GIXS2's wider _INSITU_LIST).
_EXPMODE_LIST = ["Time", "Temperature"]
# The X-axis sub-mode ("Tmode" in TrXS2_gui.m) is a fixed list here, not
# dependent on the Exp.Mode combo above the way GIXS2's own
# `_SUB_MODE_ITEMS` is -- `compute_experiment_x_axis` already handles every
# combination correctly (exp_mode=="Time" or sub_mode=="Time" both force the
# time axis regardless of the other), so there is no invalid pairing to
# guard against by narrowing this list dynamically.
_TMODE_LIST = ["Temperature1", "Temperature2", "Time"]
_NORMALIZE_LIST = ["None", "IC", "BC"]
# One color per ROI index -- see gixs2_widget.py's own `_ROI_COLORS` for the
# full rationale (port of MATLAB PlotROI_Callback's roiColors).
_ROI_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
# Shared content-length budget for every Normalize/Mean-Median/Scale combo
# box -- see gixs2_widget.py's own `_COMBO_CHARS` docstring.
_COMBO_CHARS = 10


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    """Port of MATLAB's own `finiteRange` -- see gixs2_widget.py's own
    `_finite_range` docstring."""

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
    """See gixs2_widget.py's own `_control_row` docstring."""

    container = QWidget()
    container.setLayout(layout)
    return container


def _equalize_control_row_heights(*rows: QWidget) -> None:
    """See gixs2_widget.py's own `_equalize_control_row_heights` docstring."""

    height = max(row.sizeHint().height() for row in rows)
    for row in rows:
        row.setFixedHeight(height)


class TrXS2Widget(QWidget):
    """Single-instance tab -- like GIXS2Widget, TrXS2 talks to its own
    module-level session directly (no per-instance session-swap dance,
    since there's only ever one TrXS2 tab)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_dir = str(Path.home())
        self._load_thread = None
        self._background_thread = None
        self._qcal_thread = None
        self._circ_exec_thread = None
        self._line_exec_thread = None
        self._azi_exec_thread = None
        self._auto_refresh_thread = None
        self._roi_dialog: QDialog | None = None
        self._play_dialog: QDialog | None = None
        self._heatmap_dialog: QDialog | None = None
        self._mask_dialog: QDialog | None = None
        self._current_image_data: np.ndarray | None = None
        # Set True once Load Info succeeds -- see gixs2_widget.py's own
        # `_pixel_centered` docstring.
        self._pixel_centered = False
        # "circular" | "line" | "azimuthal" | None -- see gixs2_widget.py's
        # own `_active_overlay` docstring for the two gates this guards
        # (hidden until real calibration exists; only one process-cut
        # overlay ever shown at once).
        self._active_overlay: str | None = None
        # Set by the Position/Point Find buttons -- see gixs2_widget.py's
        # own `_position_mode` docstring.
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

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        group = QGroupBox("Serial Data Parameters (TrXS)")
        layout = QVBoxLayout(group)
        layout.setSpacing(7)

        self.dir_edit = QLineEdit(self._last_dir)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.dir_set_button = QPushButton("Set")
        self.dir_set_button.setMaximumWidth(56)
        self.dir_set_button.clicked.connect(self._on_set_directory_clicked)
        layout.addWidget(_field("Directory", self.dir_edit, self.dir_set_button))

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(_DETECTOR_LIST)
        self.expmode_combo = QComboBox()
        self.expmode_combo.addItems(_EXPMODE_LIST)
        _size_combo(self.expmode_combo, 6)
        layout.addWidget(_split_row(_field("Detector", self.detector_combo, label_width=60), _field("Exp.Mode", self.expmode_combo, label_width=60)))

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
        self.parallel_check = QCheckBox("Parallel")
        self.parallel_check.setToolTip(
            "Load/process series files concurrently on several threads instead of one at a time -- "
            "applies to Set/Table Import, Circular/Line/Azimuthal Exec, and Auto Refresh alike. "
            "ROI Process is already a single vectorized operation and has no separate parallel path."
        )
        layout.addWidget(_row(self.n_files_label, self.parallel_check))

        self.background_button = QPushButton("Background")
        self.background_button.setToolTip(
            "Pick one background frame -- every file sharing its series prefix is averaged into one "
            "background frame, corrected by the same detector-correction factor as the main stack."
        )
        self.background_button.clicked.connect(self._on_background_clicked)
        self.subtract_check = QCheckBox("Subtraction")
        self.subtract_check.setToolTip("Live: subtracts the background's own cut from every frame's curve, no recompute delay.")
        self.subtract_check.toggled.connect(lambda checked: self._refresh_all_plots({"subtract": checked}))
        layout.addWidget(_row(self.background_button, self.subtract_check))

        self.load_info_button = QPushButton("Load Info")
        self.load_info_button.clicked.connect(self._on_load_info_clicked)
        self.grid_remove_button = QPushButton("Grid Rmv")
        self.grid_remove_button.clicked.connect(self._on_grid_remove_clicked)
        layout.addWidget(_row(self.load_info_button, self.grid_remove_button))

        self.qcal_button = QPushButton("q-Cal.")
        self.qcal_button.setProperty("variant", "primary")
        self.qcal_button.clicked.connect(self._on_qcal_clicked)
        layout.addWidget(self.qcal_button)

        layout.addWidget(_divider("Circular Process"))
        self.circ_angle_min_spin = _double_spin(-360.0, 360.0, 1, 0.0, 1.0)
        self.circ_angle_max_spin = _double_spin(-360.0, 720.0, 1, 360.0, 1.0)
        layout.addWidget(_field("Ang. Avg (min/max)", self.circ_angle_min_spin, self.circ_angle_max_spin))
        self.circ_angle_min_spin.valueChanged.connect(lambda _v: self._arm_circular_overlay())
        self.circ_angle_max_spin.valueChanged.connect(lambda _v: self._arm_circular_overlay())
        self.circ_min_coverage_spin = _double_spin(0.0, 1.0, 2, 0.80, 0.05)
        layout.addWidget(_field("Min coverage", self.circ_min_coverage_spin))
        self.circ_median_check = QCheckBox("Median")
        self.circ_std_check = QCheckBox("Std")
        layout.addWidget(_row(self.circ_median_check, self.circ_std_check))
        self.circ_mask_draw_button = QPushButton("Draw Mask")
        self.circ_mask_draw_button.setToolTip(
            "Draw a polygon on the 2D Image Plot -- pixels inside it are excluded from Circular Average; "
            "everything outside it is kept. Click points, double-click to close."
        )
        self.circ_mask_clear_button = QPushButton("Clear Mask")
        self.circ_mask_draw_button.clicked.connect(self._on_circ_mask_draw_clicked)
        self.circ_mask_clear_button.clicked.connect(self._on_circ_mask_clear_clicked)
        layout.addWidget(_row(self.circ_mask_draw_button, self.circ_mask_clear_button))
        self.circ_mask_status_label = _muted("Mask: none")
        layout.addWidget(self.circ_mask_status_label)
        self.circ_exec_button = QPushButton("Exec.")
        self.circ_exec_button.clicked.connect(self._on_circular_exec_clicked)
        layout.addWidget(self.circ_exec_button)

        layout.addWidget(_divider("1D Line Ang"))
        self.line_angle_spin = _double_spin(0.0, 360.0, 2, 90.0, 0.5)
        self.line_position_button = QPushButton("Position")
        self.line_position_button.setMaximumWidth(84)
        self.line_position_button.setToolTip("Click a point on the 2D Image Plot Viewer to set the line angle.")
        self.line_position_button.clicked.connect(lambda: self._arm_position_mode("line"))
        layout.addWidget(_field("Line Ang", self.line_angle_spin, self.line_position_button))
        self.line_angle_spin.valueChanged.connect(lambda _v: self._arm_line_overlay())
        self.line_draw_button = QPushButton("Line")
        self.line_draw_button.clicked.connect(self._arm_line_overlay)
        self.line_exec_button = QPushButton("Exec.")
        self.line_exec_button.clicked.connect(self._on_line_exec_clicked)
        layout.addWidget(_row(self.line_draw_button, self.line_exec_button))

        layout.addWidget(_divider("1D Output (Circular + Line)"))
        self.normalize_1d_combo = QComboBox()
        self.normalize_1d_combo.addItems(_NORMALIZE_LIST)
        _size_combo(self.normalize_1d_combo, _COMBO_CHARS)
        self.normalize_1d_combo.currentTextChanged.connect(lambda text: self._refresh_1d({"normalize": text}))
        self.normalize_1d_combo.setToolTip(
            "Applies to whichever of Circular Average / 1D Line Ang was run last -- both share this one "
            "1D Wave Vector Plot output, this Normalize setting, and Save 1D."
        )
        self.plot_1d_button = QPushButton("Plot")
        self.plot_1d_button.setToolTip("Redraw the 1D Wave Vector Plot with the current Normalize/Subtraction settings.")
        self.plot_1d_button.clicked.connect(lambda: self._refresh_1d({}))
        layout.addWidget(_field("Normalize (1D)", self.normalize_1d_combo, self.plot_1d_button))

        layout.addWidget(_divider("Azimuthal Process"))
        self.azi_pixel_spin = _double_spin(0.0, 1.0e6, 1, 202.0, 1.0)
        self.azi_point_find_button = QPushButton("Point Find")
        self.azi_point_find_button.setToolTip("Click a point on the 2D Image Plot Viewer to set pix (the radius from the beam center).")
        self.azi_point_find_button.clicked.connect(lambda: self._arm_position_mode("azimuthal"))
        layout.addWidget(_field("pix", self.azi_pixel_spin, self.azi_point_find_button, label_width=88))
        self.azi_minus_spin = _double_spin(-1.0e5, 1.0e5, 1, -1.0, 1.0)
        self.azi_plus_spin = _double_spin(-1.0e5, 1.0e5, 1, 1.0, 1.0)
        self.azi_draw_button = QPushButton("Draw")
        self.azi_draw_button.setMaximumWidth(64)
        self.azi_draw_button.setToolTip("Show the pix +/- Range(-/+) ring on the 2D image; keeps updating live as pix/Range change.")
        self.azi_draw_button.clicked.connect(self._arm_azi_overlay)
        layout.addWidget(_field("Range (-/+)", self.azi_minus_spin, self.azi_plus_spin, self.azi_draw_button, label_width=88))
        for control in (self.azi_pixel_spin, self.azi_minus_spin, self.azi_plus_spin):
            control.valueChanged.connect(lambda _v: self._arm_azi_overlay())
        self.azi_mean_median_combo = QComboBox()
        self.azi_mean_median_combo.addItems(["Mean", "Median"])
        _size_combo(self.azi_mean_median_combo, _COMBO_CHARS)
        layout.addWidget(_field("Mean/Median", self.azi_mean_median_combo, label_width=88))
        self.azi_exec_button = QPushButton("Exec.")
        self.azi_exec_button.clicked.connect(self._on_azimuthal_exec_clicked)
        self.normalize_azi_combo = QComboBox()
        self.normalize_azi_combo.addItems(_NORMALIZE_LIST)
        _size_combo(self.normalize_azi_combo, _COMBO_CHARS)
        self.normalize_azi_combo.currentTextChanged.connect(lambda text: self._refresh_azimuthal({"normalize": text}))
        layout.addWidget(_row(self.azi_exec_button, self.normalize_azi_combo))

        layout.addWidget(_divider("Region of Interest"))
        self.n_roi_combo = QComboBox()
        self.n_roi_combo.addItems(["1", "2", "3", "4", "5"])
        self.draw_roi_button = QPushButton("Draw ROI")
        self.draw_roi_button.setMaximumWidth(96)
        self.draw_roi_button.setToolTip(
            "Opens a separate window -- drag out one rectangle per ROI, each in its own color."
        )
        self.draw_roi_button.clicked.connect(self._on_draw_roi_clicked)
        layout.addWidget(_field("N. ROI", self.n_roi_combo, self.draw_roi_button))

        layout.addWidget(_divider("Options"))
        self.tmode_combo = QComboBox()
        self.tmode_combo.addItems(_TMODE_LIST)
        _size_combo(self.tmode_combo, 12)
        self.temp_sort_check = QCheckBox("Temp.Sort")
        layout.addWidget(_row(self.tmode_combo, self.temp_sort_check))
        self.tmode_combo.currentTextChanged.connect(lambda _t: self._on_x_axis_option_changed())
        self.temp_sort_check.toggled.connect(lambda _c: self._on_x_axis_option_changed())
        # MATLAB's time offset field (default 1, in the chosen unit) and
        # time unit: the Time axis is (t - t1) / unit + offset.
        self.time_offset_spin = _double_spin(-1.0e6, 1.0e6, 2, 1.0, 1.0)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["second", "min", "hour"])
        layout.addWidget(_field("Time offset", self.time_offset_spin, self.time_unit_combo))
        self.time_offset_spin.editingFinished.connect(self._on_x_axis_option_changed)
        self.time_unit_combo.currentTextChanged.connect(lambda _t: self._on_x_axis_option_changed())
        self.auto_refresh_button = QPushButton("Auto Refresh (Find New)")
        self.auto_refresh_button.clicked.connect(self._on_auto_refresh_clicked)
        layout.addWidget(self.auto_refresh_button)
        # Watch: poll the folder during an in-situ run and Auto Refresh by
        # itself whenever new frames have landed.
        self.watch_check = QCheckBox("Watch every")
        self.watch_check.toggled.connect(self._on_watch_toggled)
        self.watch_interval_spin = _double_spin(2.0, 600.0, 0, _WATCH_DEFAULT_S, 1.0)
        self.watch_interval_spin.setSuffix(" s")
        self.watch_interval_spin.valueChanged.connect(
            lambda value: self._watch_timer.setInterval(int(value * 1000))
        )
        layout.addWidget(_row(self.watch_check, self.watch_interval_spin))
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._on_watch_tick)

        layout.addWidget(_divider("Save"))
        self.save_1d_button = QPushButton("Save 1D")
        self.save_1d_button.setToolTip(
            "If Circular Exec ran with Std checked, each frame's data and std columns are interleaved "
            "(Q, frame1, frame1_STD, frame2, frame2_STD, ...) in this one file."
        )
        self.save_1d_button.clicked.connect(lambda: self._save(ts.trxs2_save_1d, self.oned_canvas))
        layout.addWidget(self.save_1d_button)
        self.save_azi_button = QPushButton("Save Azi")
        self.save_azi_button.clicked.connect(lambda: self._save(ts.trxs2_save_azimuthal, self.azi_canvas))
        layout.addWidget(self.save_azi_button)
        self.save_integ_button = QPushButton("Save Integ.")
        self.save_integ_button.clicked.connect(
            lambda: self._save(
                lambda payload: ts.trxs2_save_integration({**payload, "range": self._integ_range()}), self.integ_canvas
            )
        )
        self.save_xaxis_button = QPushButton("Save X-Axis")
        self.save_xaxis_button.clicked.connect(lambda: self._save(ts.trxs2_save_x_axis))
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

        # -- 2D Image Plot Viewer --------------------------------------
        image_group = QGroupBox("2D Image Plot")
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
        _size_combo(self.image_scale_combo, 8)
        _size_combo(self.image_colormap_combo, 10)
        default_index = self.image_colormap_combo.findText("viridis")
        if default_index >= 0:
            self.image_colormap_combo.setCurrentIndex(default_index)
        # Scientific notation for the same reason as Q Min-Max: detector
        # intensity spans everything from a fraction of a count (background-
        # subtracted/normalized data) to 1e6+ raw counts -- a fixed-decimal
        # display either rounds the small end down to "0.000000" or wastes
        # width on trailing zeros for the large end.
        self.image_cmin_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self.image_cmax_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self.image_cmin_spin.valueChanged.connect(lambda _v: self._render_image(redraw_overlays=False))
        self.image_cmax_spin.valueChanged.connect(lambda _v: self._render_image(redraw_overlays=False))
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Step through every loaded frame one by one, all rendered with the current contrast/colormap locked.")
        self.play_button.clicked.connect(self._on_play_clicked)
        self.image_scale_combo.currentTextChanged.connect(lambda _text: self._render_image(redraw_overlays=False))
        self.image_colormap_combo.currentTextChanged.connect(lambda _text: self._render_image(redraw_overlays=False))
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

        # -- 1D Wave Vector Plot -----------------------------------------
        oned_group = QGroupBox("1D Wave Vector Plot")
        oned_layout = QVBoxLayout(oned_group)
        self.oned_canvas = CurveCanvas()
        self.oned_canvas.canvas.setMinimumHeight(280)
        oned_layout.addWidget(self.oned_canvas)
        self.oned_scale_combo = QComboBox()
        self.oned_scale_combo.addItems(["Linear", "Logy", "Logxy"])
        _size_combo(self.oned_scale_combo, _COMBO_CHARS)
        self.oned_scale_combo.currentTextChanged.connect(lambda _t: self._render_1d_curve())
        self.oned_colormap_combo = QComboBox()
        self.oned_colormap_combo.addItems(_COLORMAPS)
        _size_combo(self.oned_colormap_combo, 10)
        oned_cmap_default = self.oned_colormap_combo.findText("viridis")
        if oned_cmap_default >= 0:
            self.oned_colormap_combo.setCurrentIndex(oned_cmap_default)
        self.oned_colormap_combo.currentTextChanged.connect(lambda _t: self._render_1d_curve())
        # Q Min-Max: 0/0 auto sentinel, applied directly to the axes (see
        # `_apply_oned_range`) rather than through a full re-render. A
        # scientific-notation spin box (not `_double_spin`'s plain fixed-
        # decimal one) since Q spans several orders of magnitude -- a fixed
        # 4-decimal display silently rounds a real value like 0.0002 down to
        # "0.0000" (reported: "Q Min-Max의 텍스트 박스에 0.0000만 써 있어").
        self.oned_qmin_spin = _sci_spin(-1.0e6, 1.0e6, 0.0, 1.0e-4)
        self.oned_qmax_spin = _sci_spin(-1.0e6, 1.0e6, 0.0, 1.0e-4)
        self.oned_qmin_spin.valueChanged.connect(lambda _v: self._on_range_changed())
        self.oned_qmax_spin.valueChanged.connect(lambda _v: self._on_range_changed())
        self.oned_heatmap_button = QPushButton("Heatmap")
        self.oned_heatmap_button.setToolTip("Open a Q vs. X-axis heatmap of every frame's 1D cut.")
        self.oned_heatmap_button.clicked.connect(self._on_1d_heatmap_clicked)
        oned_ctrl = QVBoxLayout()
        oned_ctrl.setContentsMargins(0, 0, 0, 0)
        oned_ctrl.setSpacing(2)
        oned_ctrl_row1 = QHBoxLayout()
        oned_ctrl_row1.addWidget(_field("Scale", self.oned_scale_combo, label_width=40))
        oned_ctrl_row1.addWidget(_field("Colormap", self.oned_colormap_combo, label_width=60))
        oned_ctrl_row2 = QHBoxLayout()
        oned_ctrl_row2.addWidget(_field("Q Min-Max", self.oned_qmin_spin, self.oned_qmax_spin, label_width=64))
        oned_ctrl_row2.addWidget(self.oned_heatmap_button)
        oned_ctrl.addLayout(oned_ctrl_row1)
        oned_ctrl.addLayout(oned_ctrl_row2)
        oned_ctrl_row = _control_row(oned_ctrl)
        oned_layout.addWidget(oned_ctrl_row)
        top.addWidget(oned_group)

        # -- Azimuthal Stack Plot -----------------------------------------
        azi_group = QGroupBox("Azimuthal Stack Plot")
        azi_layout = QVBoxLayout(azi_group)
        self.azi_canvas = CurveCanvas()
        self.azi_canvas.canvas.setMinimumHeight(280)
        azi_layout.addWidget(self.azi_canvas)
        self.azi_scale_combo = QComboBox()
        self.azi_scale_combo.addItems(["Linear", "Logy", "Logxy"])
        _size_combo(self.azi_scale_combo, _COMBO_CHARS)
        self.azi_scale_combo.currentTextChanged.connect(lambda _t: self._render_azimuthal_curve())
        self.azi_colormap_combo = QComboBox()
        self.azi_colormap_combo.addItems(_COLORMAPS)
        _size_combo(self.azi_colormap_combo, 10)
        azi_cmap_default = self.azi_colormap_combo.findText("viridis")
        if azi_cmap_default >= 0:
            self.azi_colormap_combo.setCurrentIndex(azi_cmap_default)
        self.azi_colormap_combo.currentTextChanged.connect(lambda _t: self._render_azimuthal_curve())
        # Deg Min-Max defaults to the full 0-360deg sweep TrXS's azimuthal
        # cut always covers (vs. GIXS2's half-circle 0-180) -- no separate
        # "auto" sentinel needed since that default already is the full
        # data range; still guarded (max > min) before being applied.
        self.azi_degmin_spin = _double_spin(-360.0, 720.0, 1, 0.0, 1.0)
        self.azi_degmax_spin = _double_spin(-360.0, 720.0, 1, 360.0, 1.0)
        self.azi_degmin_spin.valueChanged.connect(lambda _v: self._on_range_changed())
        self.azi_degmax_spin.valueChanged.connect(lambda _v: self._on_range_changed())
        self.azi_heatmap_button = QPushButton("Heatmap")
        self.azi_heatmap_button.setToolTip("Open an Angle vs. X-axis heatmap of every frame's azimuthal profile.")
        self.azi_heatmap_button.clicked.connect(self._on_azi_heatmap_clicked)
        azi_ctrl = QVBoxLayout()
        azi_ctrl.setContentsMargins(0, 0, 0, 0)
        azi_ctrl.setSpacing(2)
        azi_ctrl_row1 = QHBoxLayout()
        azi_ctrl_row1.addWidget(_field("Scale", self.azi_scale_combo, label_width=40))
        azi_ctrl_row1.addWidget(_field("Colormap", self.azi_colormap_combo, label_width=60))
        azi_ctrl_row2 = QHBoxLayout()
        azi_ctrl_row2.addWidget(_field("Deg Min-Max", self.azi_degmin_spin, self.azi_degmax_spin, label_width=68))
        azi_ctrl_row2.addWidget(self.azi_heatmap_button)
        azi_ctrl.addLayout(azi_ctrl_row1)
        azi_ctrl.addLayout(azi_ctrl_row2)
        azi_ctrl_row = _control_row(azi_ctrl)
        azi_layout.addWidget(azi_ctrl_row)
        bottom.addWidget(azi_group)

        # -- Integration & ROI Plot ----------------------------------------
        integ_group = QGroupBox("Integration & ROI Plot")
        integ_layout = QVBoxLayout(integ_group)
        self.integ_canvas = CurveCanvas()
        self.integ_canvas.canvas.setMinimumHeight(280)
        integ_layout.addWidget(self.integ_canvas)
        self.integ_scale_combo = QComboBox()
        self.integ_scale_combo.addItems(["Linear", "Logy", "Logxy"])
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

        for row in (top, bottom):
            row.setStretchFactor(0, 1)
            row.setStretchFactor(1, 1)
            row.setSizes([500, 500])

        outer.addWidget(top)
        outer.addWidget(bottom)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 1)
        outer.setSizes([500, 500])
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
        button.setProperty("state", "ok" if ok else None)
        button.style().unpolish(button)
        button.style().polish(button)

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
        result = ts.trxs2_load(payload)
        if not result.get("ok"):
            self._mark_ok(self.load_button, False)
            self._set_status(result.get("message", "Load failed."), False)
            return
        self.filename_edit.setText(result["prefix"])
        self._current_image_data = None
        self._pixel_centered = False
        self._active_overlay = None
        self._position_mode = None
        self._reset_contrast()
        self._close_play_dialog()
        self._render_image()
        self._mark_ok(self.load_button, True)
        # New raw data invalidates whatever q-Cal/Grid Remove/series import
        # was done against the *previous* dataset -- background stays
        # loaded across a new series (see trxs2_session._reset_for_new_series's
        # own docstring), so background_button is deliberately left alone.
        self._mark_ok(self.qcal_button, False)
        self._mark_ok(self.grid_remove_button, False)
        self._mark_ok(self.set_table_button, False)
        self._set_status(f"Loaded {'x'.join(str(v) for v in result['shape'])} from {result['path']}.", True)

    def _on_set_table_clicked(self) -> None:
        result = ts.trxs2_list_series_files({"directory": self._last_dir, "filename": self.filename_edit.text().strip()})
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

        accepted = dialog.exec() == QDialog.Accepted
        checked = [list_widget.item(i).text() for i in range(list_widget.count()) if list_widget.item(i).checkState() == Qt.Checked]
        # Parented to this widget, so it would otherwise live until exit.
        dialog.deleteLater()
        if not accepted:
            return
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
            "insitu_mode": self.expmode_combo.currentText(),
            "detector": self.detector_combo.currentText(),
            "parallel": self.parallel_check.isChecked(),
        }
        self._load_thread = run_in_background(ts.trxs2_load_series, payload, self._on_series_loaded)

    def _start_progress_dialog(self, title: str, window_title: str = "Loading Series") -> None:
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
        status = ts.trxs2_load_progress()
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
        self._reset_contrast()
        self._close_play_dialog()
        self._render_image()
        self._mark_ok(self.set_table_button, True)
        self._mark_ok(self.qcal_button, False)
        self._mark_ok(self.grid_remove_button, False)
        warnings = result.get("warnings") or []
        if warnings:
            QMessageBox.warning(self, "TrXS2 Load Series", "\n\n".join(warnings))
        self._set_status(f"Loaded {result['n_files']} files, shape {tuple(result['shape'])}. Showing the average image.", True)

    def _on_load_info_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a calibration file", self._last_dir, "Calibration (*.json)")
        if not path:
            return
        file_path = Path(path)
        result = ts.trxs2_load_info({"directory": str(file_path.parent), "filename": file_path.name})
        if not result.get("ok"):
            self._set_status(result.get("message", "Load Info failed."), False)
            return
        cal = result["calibration"]
        self._mark_ok(self.load_info_button, True)
        self._pixel_centered = True
        self._render_image(force_reset=True)
        if result.get("message"):
            # q-Cal ran against the previous calibration; the session dropped it.
            self._mark_ok(self.qcal_button, False)
            self._set_status(result["message"], True)
            return
        self._set_status(f"Loaded Info: SDD={cal['sdd_mm']:.1f} mm, Energy={cal['energy_kev']:.3f} keV.", True)

    def _on_grid_remove_clicked(self) -> None:
        result = ts.trxs2_grid_remove({})
        if not result.get("ok"):
            self._mark_ok(self.grid_remove_button, False)
            self._set_status(result.get("message", "Grid Remove failed."), False)
            return
        self._current_image_data = None
        self._render_image()
        self._mark_ok(self.grid_remove_button, True)
        self._set_status(result.get("message", "Grid removed."), True)

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------
    def _on_background_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a background frame", self._last_dir, "HDF5 files (*.h5);;All files (*)")
        if not path:
            return
        file_path = Path(path)
        payload = {"directory": str(file_path.parent), "filename": file_path.name}
        self.background_button.setEnabled(False)
        self._start_progress_dialog("Loading background...", window_title="Background")
        self._background_thread = run_in_background(ts.trxs2_background_load, payload, self._on_background_finished)

    def _on_background_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.background_button.setEnabled(True)
        if not result.get("ok"):
            self._mark_ok(self.background_button, False)
            self._set_status(result.get("message", "Background load failed."), False)
            return
        self._mark_ok(self.background_button, True)
        message = f"Background loaded: {result['n_files']} frame(s) averaged."
        warning = result.get("warning")
        if warning:
            message += f" {warning}"
        self._set_status(message, True)
        # The session re-cut the new background; redraw what Subtraction shows.
        self._refresh_all_plots()

    # ------------------------------------------------------------------
    # 2D image render
    # ------------------------------------------------------------------
    def _reset_contrast(self) -> None:
        """Back to the 0/0 auto-contrast sentinel, so a newly loaded image
        is not drawn with the previous image's min/max (which can leave it
        looking blank or blown out)."""

        for spin in (self.image_cmin_spin, self.image_cmax_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)

    def _render_image(self, force_reset: bool = False, redraw_overlays: bool = True) -> None:
        session = ts.get_session()
        if session.data is None:
            return
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
        if redraw_overlays:
            self._redraw_overlays()

    def _image_extent(self, shape: tuple[int, int]) -> tuple[float, float, float, float]:
        rows, cols = shape
        if self._pixel_centered:
            beam_y, beam_z = ts.get_session().calibration["beam_pos"]
            h_start, h_end = 1 - beam_y, cols - beam_y
            v_row1, v_row_last = beam_z - 1, beam_z - rows
        else:
            h_start, h_end = 1, cols
            v_row1, v_row_last = 1, rows
        return (h_start, h_end, v_row_last, v_row1)

    # ------------------------------------------------------------------
    # Overlay geometry -- Circular sector rays / Line Ang line / Azimuthal
    # ring, mutually exclusive (only one active-mode overlay shown at once,
    # see `_active_overlay`'s own docstring), plus ROI boxes always drawn on
    # top independent of which is active.
    # ------------------------------------------------------------------
    def _current_beam_center(self) -> tuple[float, float]:
        if not self._pixel_centered:
            beam_y, beam_z = ts.get_session().calibration["beam_pos"]
            return (beam_y, beam_z)
        return (0.0, 0.0)

    def _mode_corrected_angle(self, angle_deg: float) -> float:
        if self._pixel_centered:
            return (-angle_deg) % 360.0
        return angle_deg % 360.0

    def _redraw_overlays(self) -> None:
        self.image_canvas.clear_overlays()
        self._draw_circular_overlay()
        self._draw_line_overlay()
        self._draw_azi_overlay()
        self._draw_roi_overlay()

    def _arm_circular_overlay(self) -> None:
        self._active_overlay = "circular"
        self._redraw_overlays()

    def _draw_circular_overlay(self) -> None:
        session = ts.get_session()
        if session.data is None:
            return
        if not (self._pixel_centered or session.corrected):
            return
        if self._active_overlay != "circular":
            return
        angle_min = self.circ_angle_min_spin.value()
        angle_max = self.circ_angle_max_spin.value()
        # A 0-360deg (or any full-turn) sweep has no meaningful boundary --
        # both rays land on the same angle, reading as a stray line rather
        # than "this is the whole detector".
        if (angle_max - angle_min) % 360.0 == 0.0:
            return
        center = self._current_beam_center()
        self.image_canvas.draw_beam_line(center, self._mode_corrected_angle(angle_min), bidirectional=False)
        self.image_canvas.draw_beam_line(center, self._mode_corrected_angle(angle_max), bidirectional=False)

    def _arm_line_overlay(self) -> None:
        self._active_overlay = "line"
        self._redraw_overlays()

    def _draw_line_overlay(self) -> None:
        session = ts.get_session()
        if session.data is None:
            return
        if not (self._pixel_centered or session.corrected):
            return
        if self._active_overlay != "line":
            return
        center = self._current_beam_center()
        angle = self.line_angle_spin.value()
        self.image_canvas.draw_beam_line(center, self._mode_corrected_angle(angle))

    def _arm_azi_overlay(self) -> None:
        self._active_overlay = "azimuthal"
        self._redraw_overlays()

    def _draw_azi_overlay(self) -> None:
        session = ts.get_session()
        if session.data is None:
            return
        if not (self._pixel_centered or session.corrected):
            return
        if self._active_overlay != "azimuthal":
            return
        center = self._current_beam_center()
        pixel = self.azi_pixel_spin.value()
        r_minus = max(0.0, pixel + self.azi_minus_spin.value())
        r_plus = max(0.0, pixel + self.azi_plus_spin.value())
        self.image_canvas.draw_beam_circle(center, r_minus)
        self.image_canvas.draw_beam_circle(center, r_plus)

    def _display_xy_to_raw_pixel(self, x: float, y: float) -> tuple[float, float]:
        if not self._pixel_centered:
            return x, y
        beam_y, beam_z = ts.get_session().calibration["beam_pos"]
        return x + beam_y, beam_z - y

    def _raw_pixel_to_display_xy(self, col: float, row: float) -> tuple[float, float]:
        if not self._pixel_centered:
            return col, row
        beam_y, beam_z = ts.get_session().calibration["beam_pos"]
        return col - beam_y, beam_z - row

    def _draw_roi_overlay(self) -> None:
        session = ts.get_session()
        if not session.roi_list or session.data is None:
            return
        for i, (r1, r2, c1, c2) in enumerate(session.roi_list):
            x1, y1 = self._raw_pixel_to_display_xy(c1, r1)
            x2, y2 = self._raw_pixel_to_display_xy(c2, r2)
            self.image_canvas.draw_roi_box(x1, x2, y1, y2, _ROI_COLORS[i % len(_ROI_COLORS)], f"ROI {i + 1}")

    # ------------------------------------------------------------------
    # Click-to-position (Line Ang / Azimuthal pix)
    # ------------------------------------------------------------------
    def _arm_position_mode(self, mode: str) -> None:
        self._position_mode = mode
        if mode == "line":
            self._set_status("Click a point on the 2D image to set the line angle.", True)
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
            self.line_angle_spin.setValue(angle)
            self._set_status(f"Line Ang position clicked at ({x:.1f}, {y:.1f}) -> angle {angle:.2f} deg.", True)
        elif self._position_mode == "azimuthal":
            beam_y, beam_z = self._current_beam_center()
            radius = math.hypot(x - beam_y, y - beam_z)
            self.azi_pixel_spin.setValue(radius)
            self._set_status(f"Azimuthal position clicked at ({x:.1f}, {y:.1f}) -> pix {radius:.1f}.", True)

    # ------------------------------------------------------------------
    # Circular Process mask (draw/clear polygon)
    # ------------------------------------------------------------------
    def _on_circ_mask_draw_clicked(self) -> None:
        """Port of MATLAB's `get_modern_mask`: draw the mask polygon in its
        own large window instead of the small "2D Image Plot" quadrant --
        that quadrant is too small to click out a precise polygon in a
        4-way-split window."""

        session = ts.get_session()
        if session.data is None:
            self._set_status("Load data first.", False)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Draw Circular Average Mask")
        layout = QVBoxLayout(dialog)

        hint = QLabel(
            "Click points to draw the mask polygon (the inside is excluded from Circular Average); "
            "double-click, or click the first point again, to close it."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        canvas = ImageCanvas()
        # ImageCanvas defaults to a Preferred size policy, which only grows
        # into leftover dialog space on a par with the hint label/Cancel
        # button rather than claiming it outright -- Expanding + an explicit
        # layout stretch factor makes it the one widget that actually fills
        # the large window this dialog is sized to (unlike the small "2D
        # Image Plot" quadrant, where the splitter's own stretch factors
        # already force this).
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(canvas, 1)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.close)
        layout.addWidget(cancel_button)

        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            dialog.resize(int(avail.width() * 0.85), int(avail.height() * 0.9))
        else:
            dialog.resize(1000, 900)

        cmin = self.image_cmin_spin.value()
        cmax = self.image_cmax_spin.value()
        vmin = cmin if cmax > cmin else None
        vmax = cmax if cmax > cmin else None
        canvas.render(
            session.data,
            cmap=self.image_colormap_combo.currentText(),
            log_scale=self.image_scale_combo.currentText() == "Log",
            vmin=vmin,
            vmax=vmax,
            extent=self._image_extent(session.data.shape),
            xlabel="Y, pixel",
            ylabel="Z, pixel",
        )

        def _on_complete(vertices: list[tuple[float, float]]) -> None:
            self._on_circ_mask_polygon_complete(vertices)
            dialog.close()

        canvas.start_polygon_selector(_on_complete)

        def _on_finished(_result: int) -> None:
            # Same shiboken-crash guard as the Play/Heatmap/ROI dialogs:
            # matplotlib's PolygonSelector queues its own draw_idle() on
            # every click (including the one that completes the polygon),
            # so stop it and defer the canvas's actual deletion by one
            # event-loop tick to let any already-queued redraw run first.
            canvas.stop_polygon_selector()
            if self._mask_dialog is dialog:
                self._mask_dialog = None
            QTimer.singleShot(0, dialog.deleteLater)

        dialog.finished.connect(_on_finished)
        self._mask_dialog = dialog
        dialog.show()

    def _on_circ_mask_polygon_complete(self, vertices: list[tuple[float, float]]) -> None:
        """The drawn polygon marks the region to *exclude* -- everything
        outside it survives into Circular Average (trxs2_set_circular_mask's
        `mask` is True = keep, so the polygon's inside is inverted below)."""

        session = ts.get_session()
        if session.data is None or len(vertices) < 3:
            self._set_status("Mask needs at least 3 points.", False)
            return
        rows, cols = session.data.shape
        if self._pixel_centered:
            beam_y, beam_z = session.calibration["beam_pos"]
            col_disp = np.arange(1, cols + 1, dtype=np.float64) - beam_y
            row_disp = beam_z - np.arange(1, rows + 1, dtype=np.float64)
        else:
            col_disp = np.arange(1, cols + 1, dtype=np.float64)
            row_disp = np.arange(1, rows + 1, dtype=np.float64)
        xx, yy = np.meshgrid(col_disp, row_disp)

        from matplotlib.path import Path as MplPath

        path = MplPath(vertices)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        inside = path.contains_points(points).reshape(xx.shape)
        keep = ~inside

        result = ts.trxs2_set_circular_mask({"mask": keep})
        if not result.get("ok"):
            self._set_status(result.get("message", "Mask failed."), False)
            return
        kept = result["pixel_count"]
        total = result["total_pixels"]
        self.circ_mask_status_label.setText(f"Mask: {kept}/{total} px kept")
        self._set_status(f"Mask set ({kept}/{total} px kept).", True)

    def _on_circ_mask_clear_clicked(self) -> None:
        if self._mask_dialog is not None:
            self._mask_dialog.close()
        ts.trxs2_set_circular_mask({"clear": True})
        self.circ_mask_status_label.setText("Mask: none")
        self._set_status("Mask cleared.", True)

    # ------------------------------------------------------------------
    # Play (Frame Viewer)
    # ------------------------------------------------------------------
    def _close_play_dialog(self) -> None:
        # The Play window captured the old series' frame count; left open
        # over a new series it shows the wrong frames and indexes past the end.
        if self._play_dialog is not None:
            self._play_dialog.close()

    def _on_play_clicked(self) -> None:
        session = ts.get_session()
        # Raw frames plus the shared correction factor: q-Cal no longer
        # materializes a corrected copy of the whole series, so each
        # frame is corrected as it is displayed (ts.corrected_frame).
        stack = session.corrected_stack if session.corrected_stack is not None else session.stack
        if stack is None or stack.shape[0] == 0:
            self._set_status("Load a series first.", False)
            return

        n_frames = stack.shape[0]
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
                ts.corrected_frame(idx), cmap=cmap, log_scale=log_scale, vmin=vmin, vmax=vmax,
                extent=extent, xlabel="Y, pixel", ylabel="Z, pixel", keep_limits=idx > 0,
            )

        slider.valueChanged.connect(render_frame)
        render_frame(1)

        def _on_finished(_result: int) -> None:
            # Same shiboken-crash guard as GIXS2's Play dialog
            # (gixs2_widget.py's _on_play_clicked): disconnect first so
            # closing mid-drag can't queue another draw_idle(), then defer
            # the actual C++ deletion by one event-loop tick.
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
    # q-Cal (static only -- no incidence-angle branch)
    # ------------------------------------------------------------------
    def _on_qcal_clicked(self) -> None:
        self.qcal_button.setEnabled(False)
        self._start_progress_dialog("Running q-Cal...", window_title="q-Cal")
        self._qcal_thread = run_in_background(ts.trxs2_qcal, {}, self._on_qcal_finished)

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
        self._set_status("q-Cal applied to the whole series.", True)

    # ------------------------------------------------------------------
    # Circular / Line / Azimuthal exec
    # ------------------------------------------------------------------
    def _on_circular_exec_clicked(self) -> None:
        self._arm_circular_overlay()
        payload = {
            "angle_min": self.circ_angle_min_spin.value(),
            "angle_max": self.circ_angle_max_spin.value(),
            "min_coverage": self.circ_min_coverage_spin.value(),
            "use_median": self.circ_median_check.isChecked(),
            "calc_std": self.circ_std_check.isChecked(),
            "parallel": self.parallel_check.isChecked(),
        }
        self.circ_exec_button.setEnabled(False)
        self._start_progress_dialog("Running circular average...", window_title="Circular Exec")
        self._circ_exec_thread = run_in_background(ts.trxs2_circular_exec, payload, self._on_circular_exec_finished)

    def _on_circular_exec_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.circ_exec_button.setEnabled(True)
        if not result.get("ok"):
            self._set_status(result.get("message", "Circular Exec failed."), False)
            return
        self._reset_range(self.oned_qmin_spin, self.oned_qmax_spin, 0.0, 0.0)
        self._refresh_1d({"normalize": self.normalize_1d_combo.currentText(), "subtract": self.subtract_check.isChecked()})
        self._set_status(f"Circular series: {result['n_frames']} frames x {result['n_points']} points.", True)

    def _on_line_exec_clicked(self) -> None:
        self._arm_line_overlay()
        payload = {
            "angle_deg": self.line_angle_spin.value(),
            "parallel": self.parallel_check.isChecked(),
        }
        self.line_exec_button.setEnabled(False)
        self._start_progress_dialog("Running 1D line cut...", window_title="Line Exec")
        self._line_exec_thread = run_in_background(ts.trxs2_line_exec, payload, self._on_line_exec_finished)

    def _on_line_exec_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.line_exec_button.setEnabled(True)
        if not result.get("ok"):
            self._set_status(result.get("message", "Line Exec failed."), False)
            return
        self._reset_range(self.oned_qmin_spin, self.oned_qmax_spin, 0.0, 0.0)
        self._refresh_1d({"normalize": self.normalize_1d_combo.currentText(), "subtract": self.subtract_check.isChecked()})
        self._set_status(f"1D Line series: {result['n_frames']} frames x {result['n_points']} points.", True)

    def _on_azimuthal_exec_clicked(self) -> None:
        self._arm_azi_overlay()
        payload = {
            "pixel": self.azi_pixel_spin.value(),
            "range_minus": self.azi_minus_spin.value(),
            "range_plus": self.azi_plus_spin.value(),
            "mean_or_median": self.azi_mean_median_combo.currentText(),
            "parallel": self.parallel_check.isChecked(),
        }
        self.azi_exec_button.setEnabled(False)
        self._start_progress_dialog("Running azimuthal profile...", window_title="Azimuthal Exec")
        self._azi_exec_thread = run_in_background(ts.trxs2_azimuthal_exec, payload, self._on_azimuthal_exec_finished)

    def _on_azimuthal_exec_finished(self, result: dict[str, Any]) -> None:
        self._stop_progress_dialog()
        self.azi_exec_button.setEnabled(True)
        if not result.get("ok"):
            self._set_status(result.get("message", "Azimuthal Exec failed."), False)
            return
        self._reset_range(self.azi_degmin_spin, self.azi_degmax_spin, 0.0, 360.0)
        self._refresh_azimuthal({"normalize": self.normalize_azi_combo.currentText(), "subtract": self.subtract_check.isChecked()})
        self._set_status(f"Azimuthal series: {result['n_frames']} frames x {result['n_points']} points.", True)

    # ------------------------------------------------------------------
    # Render / Subtraction -- thin wrappers over the backend's own
    # trxs2_render_1d/trxs2_render_azimuthal; the widget's own job is just
    # redrawing matplotlib with whatever `session.last_1d["rendered"]`/
    # `last_azimuthal["rendered"]` ends up holding.
    # ------------------------------------------------------------------
    def _refresh_1d(self, payload: dict[str, Any] | None = None) -> None:
        result = ts.trxs2_render_1d(dict(payload or {}))
        if not result.get("ok"):
            return
        if result.get("warning"):
            self._set_status(result["warning"], False)
        self._render_1d_curve()

    def _refresh_azimuthal(self, payload: dict[str, Any] | None = None) -> None:
        result = ts.trxs2_render_azimuthal(dict(payload or {}))
        if not result.get("ok"):
            return
        if result.get("warning"):
            self._set_status(result["warning"], False)
        self._render_azimuthal_curve()

    def _refresh_all_plots(self, payload: dict[str, Any] | None = None) -> None:
        """Port of `RefreshAllPlots_Callback`: re-render whichever of
        1D/Azimuthal already has data -- called by the Subtraction
        checkbox toggling live."""

        session = ts.get_session()
        if session.last_1d is not None:
            self._refresh_1d(payload)
        if session.last_azimuthal is not None:
            self._refresh_azimuthal(payload)

    def _series_color_values(self, n_frames: int) -> tuple[np.ndarray, str]:
        session = ts.get_session()
        if session.timestamps is not None:
            payload = self._x_axis_payload()
            result = ts.trxs2_set_x_axis(payload)
            if result.get("ok") and session.x_axis_values is not None and session.x_axis_values.shape[0] == n_frames:
                return session.x_axis_values, (session.x_axis_label or "Frame")
        return np.arange(1, n_frames + 1, dtype=np.float64), "Frame, #"

    def _x_axis_payload(self) -> dict[str, Any]:
        return {
            "sub_mode": self.tmode_combo.currentText(),
            "order": self.temp_sort_check.isChecked(),
            "time_offset_seconds": self.time_offset_spin.value(),
            "time_unit": self.time_unit_combo.currentText(),
        }

    def _on_x_axis_option_changed(self) -> None:
        """Tmode/Temp.Sort changing live-refreshes whichever plot(s)
        already have data -- there is no separate "Set X-Axis" button in
        TrXS2 (unlike GIXS2's own `set_x_axis_button`); the series X-axis is
        instead always recomputed inline on every render via
        `_series_color_values`/`_render_integration`, matching MATLAB's own
        `update1Dplot_Callback`/`updateAziIntplot`, which call
        `getExperimentXAxis(param)` directly with no separate commit step."""

        session = ts.get_session()
        if session.last_1d is not None:
            self._refresh_1d({})
        if session.last_azimuthal is not None:
            self._refresh_azimuthal({})
        if session.last_1d is None and session.last_azimuthal is None:
            self._render_integration()

    def _render_1d_curve(self) -> None:
        session = ts.get_session()
        if session.last_1d is None or session.last_1d.get("rendered") is None:
            return
        matrix = ts.order_frames(np.asarray(session.last_1d["rendered"], dtype=np.float64))
        color_values, colorbar_label = self._series_color_values(matrix.shape[0])
        scale = self.oned_scale_combo.currentText()
        xlog = scale == "Logxy"
        ylog = scale in ("Logy", "Logxy")
        self.oned_canvas.render_series(
            session.last_1d["x_axis"], matrix, color_values,
            xlabel="Q, 1/A", ylabel="Intensity, arb. units", colorbar_label=colorbar_label,
            cmap=self.oned_colormap_combo.currentText(), xlog=xlog, ylog=ylog,
        )
        self._apply_oned_range()
        self._render_integration()

    @staticmethod
    def _reset_range(lo_spin, hi_spin, lo: float, hi: float) -> None:
        for spin, value in ((lo_spin, lo), (hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _integ_range(self) -> tuple[float, float]:
        # The Q or angle window of whichever cut the Integration plot sums
        # (MATLAB's qMin/qMax, aziMin/aziMax).
        if ts.get_session().auto_met == "azimuthal":
            return self.azi_degmin_spin.value(), self.azi_degmax_spin.value()
        return self.oned_qmin_spin.value(), self.oned_qmax_spin.value()

    def _on_range_changed(self) -> None:
        self._apply_oned_range()
        self._apply_azi_range()
        self._render_integration()

    def _apply_oned_range(self) -> None:
        qmin, qmax = self.oned_qmin_spin.value(), self.oned_qmax_spin.value()
        if qmax > qmin:
            self.oned_canvas.ax.set_xlim(qmin, qmax)
            self.oned_canvas.canvas.draw_idle()
            return
        # 0/0 auto sentinel: reflect the actual displayed Q range back into
        # the fields, matching Contrast's own "shows the real auto-computed
        # value instead of staying at two blank zeros" convention (reported:
        # "Q Min-Max에서 값이 제대로 안나와" -- it was staying at 0.00e+00
        # forever since nothing ever wrote a real value into it).
        session = ts.get_session()
        if session.last_1d is None:
            return
        lo, hi = _finite_range(np.asarray(session.last_1d["x_axis"], dtype=np.float64))
        self.oned_qmin_spin.blockSignals(True)
        self.oned_qmax_spin.blockSignals(True)
        self.oned_qmin_spin.setValue(lo)
        self.oned_qmax_spin.setValue(hi)
        self.oned_qmin_spin.blockSignals(False)
        self.oned_qmax_spin.blockSignals(False)

    def _render_azimuthal_curve(self) -> None:
        session = ts.get_session()
        if session.last_azimuthal is None or session.last_azimuthal.get("rendered") is None:
            return
        matrix = ts.order_frames(np.asarray(session.last_azimuthal["rendered"], dtype=np.float64))
        color_values, colorbar_label = self._series_color_values(matrix.shape[0])
        scale = self.azi_scale_combo.currentText()
        xlog = scale == "Logxy"
        ylog = scale in ("Logy", "Logxy")
        self.azi_canvas.render_series(
            session.last_azimuthal["x_axis"], matrix, color_values,
            xlabel="Angle, deg", ylabel="Intensity, arb. units", colorbar_label=colorbar_label,
            cmap=self.azi_colormap_combo.currentText(), xlog=xlog, ylog=ylog,
        )
        self._apply_azi_range()
        self._render_integration()

    def _apply_azi_range(self) -> None:
        degmin, degmax = self.azi_degmin_spin.value(), self.azi_degmax_spin.value()
        if degmax > degmin:
            self.azi_canvas.ax.set_xlim(degmin, degmax)
            self.azi_canvas.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Heatmap -- port of MATLAB's HeatmapQ_gui/HeatmapAzi_gui, see
    # gixs2_widget.py's own `_open_heatmap_dialog` docstring for the full
    # rationale (snapshots at open time, doesn't stay live-linked).
    # ------------------------------------------------------------------
    def _on_1d_heatmap_clicked(self) -> None:
        session = ts.get_session()
        if session.last_1d is None or session.last_1d.get("rendered") is None:
            self._set_status("Run Circular/Line Exec first.", False)
            return
        matrix = ts.order_frames(np.asarray(session.last_1d["rendered"], dtype=np.float64))
        self._open_heatmap_dialog("1D Heatmap", session.last_1d["x_axis"], "Q, 1/A", matrix, "1D")

    def _on_azi_heatmap_clicked(self) -> None:
        session = ts.get_session()
        if session.last_azimuthal is None or session.last_azimuthal.get("rendered") is None:
            self._set_status("Run Azimuthal Exec first.", False)
            return
        matrix = ts.order_frames(np.asarray(session.last_azimuthal["rendered"], dtype=np.float64))
        self._open_heatmap_dialog("Azimuthal Heatmap", session.last_azimuthal["x_axis"], "Angle, deg", matrix, "Azimuthal")

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
        xscale_combo = QComboBox()
        xscale_combo.addItems(["Linear", "Log"])
        yscale_combo = QComboBox()
        yscale_combo.addItems(["Linear", "Log"])
        scale_combo = QComboBox()
        scale_combo.addItems(["Linear", "Log"])
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
                    status_label.setText(f"{axes_text} axis has non-positive values -- Log Scale not applied, stayed Linear.")
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
            # Same shiboken-crash guard as GIXS2's Heatmap dialog
            # (gixs2_widget.py's _open_heatmap_dialog): disconnect every
            # live control first so closing mid-interaction can't queue
            # another draw_idle(), then defer the actual C++ deletion by
            # one event-loop tick so any already-queued redraw runs first.
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
    # ROI -- copied near-verbatim from gixs2_widget.py's own
    # `_on_draw_roi_clicked` (unchanged behavior vs GIXS2, per the plan).
    # ------------------------------------------------------------------
    def _on_draw_roi_clicked(self) -> None:
        session = ts.get_session()
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
        # Same shiboken-crash guard as GIXS2's Draw ROI dialog: the last
        # completed ROI's draw_roi_box() queued a draw_idle() just before
        # accept() closed the dialog -- defer the canvas's actual deletion
        # by one event-loop tick so that redraw runs first.
        QTimer.singleShot(0, dialog.deleteLater)

        if not accepted or len(pending_rois) < n_roi:
            self._set_status("ROI drawing cancelled.", False)
            return

        result = ts.trxs2_roi_exec({"rois": pending_rois})
        if not result.get("ok"):
            self._set_status(result.get("message", "Exec ROI failed."), False)
            return
        self._redraw_overlays()
        self._render_integration()
        self._set_status(f"ROI series: {result['n_frames']} frames x {result['n_roi']} ROI(s).", True)

    # ------------------------------------------------------------------
    # Auto Refresh
    # ------------------------------------------------------------------
    def _on_watch_toggled(self, on: bool) -> None:
        if on:
            self._watch_timer.start(int(self.watch_interval_spin.value() * 1000))
            self._set_status(f"Watching for new files every {self.watch_interval_spin.value():g} s.", True)
        else:
            self._watch_timer.stop()
            self._set_status("Stopped watching for new files.", True)

    def _on_watch_tick(self) -> None:
        # Never start while anything else holds the session: a load, an
        # Exec, an Auto Refresh already running (all behind a modal
        # progress dialog) or a modal dialog such as ROI drawing.
        if QApplication.activeModalWidget() is not None or not self.auto_refresh_button.isEnabled():
            return
        payload = {"directory": self._last_dir, "settle_seconds": _WATCH_SETTLE_S}
        pending = ts.trxs2_pending_new_files(payload)
        if not pending.get("ok"):
            self.watch_check.setChecked(False)
            self._set_status(f"Watch stopped: {pending.get('message', 'no series to extend.')}", False)
            return
        if pending["n_new"]:
            self._start_auto_refresh(settle_seconds=_WATCH_SETTLE_S)

    def _on_auto_refresh_clicked(self) -> None:
        self._start_auto_refresh()

    def _start_auto_refresh(self, settle_seconds: float = 0.0) -> None:
        self.auto_refresh_button.setEnabled(False)
        self._start_progress_dialog("Checking for new files...", window_title="Auto Refresh")
        payload = {
            "directory": self._last_dir,
            "parallel": self.parallel_check.isChecked(),
            "settle_seconds": settle_seconds,
        }
        self._auto_refresh_thread = run_in_background(ts.trxs2_auto_refresh, payload, self._on_auto_refresh_finished)

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
        # Re-derive the series X-axis for the new frame count (see the same
        # step in gixs2_widget): Save 1D/Azi/Integ. read `x_axis_values`
        # straight from the session, so a pre-refresh length makes them fail.
        if ts.get_session().timestamps is not None:
            ts.trxs2_set_x_axis(self._x_axis_payload())
        self._current_image_data = None
        self._render_image()
        if result.get("curve_extended"):
            auto_met = ts.get_session().auto_met
            if auto_met in ("circular", "line1d"):
                self._refresh_1d({})
            elif auto_met == "azimuthal":
                self._refresh_azimuthal({})
            elif auto_met == "roi":
                self._render_integration()
        self._set_status(f"Auto Refresh: +{result['n_new']} frame(s), total {result['n_total']}.", True)

    # ------------------------------------------------------------------
    # Integration & ROI Plot
    # ------------------------------------------------------------------
    def _render_integration(self) -> None:
        session = ts.get_session()
        if session.timestamps is not None:
            ts.trxs2_set_x_axis(self._x_axis_payload())
        if session.x_axis_values is None:
            return
        scale = self.integ_scale_combo.currentText()
        xlog = scale == "Logxy"
        ylog = scale in ("Logy", "Logxy")
        auto_met = session.auto_met
        if auto_met == "roi" and session.last_roi is not None:
            if session.last_roi.shape[0] != session.x_axis_values.shape[0]:
                return
            n_roi = session.last_roi.shape[1]
            labels = [f"ROI {i + 1}" for i in range(n_roi)]
            colors = [_ROI_COLORS[i % len(_ROI_COLORS)] for i in range(n_roi)]
            self.integ_canvas.render_multi(
                session.x_axis_values, ts.order_frames(session.last_roi).T, labels, colors,
                xlabel=session.x_axis_label, ylabel="Sum(ROI), arb. units",
                xlog=xlog, ylog=ylog,
            )
            return
        elif auto_met == "azimuthal" and session.last_azimuthal is not None and session.last_azimuthal.get("rendered") is not None:
            with np.errstate(invalid="ignore"):
                y = ts.integrate_frames(
                    ts.order_frames(session.last_azimuthal["rendered"]), session.last_azimuthal["x_axis"], self._integ_range()
                )
            ylabel = "Integrated Azimuthal, arb. units"
        elif session.last_1d is not None and session.last_1d.get("rendered") is not None:
            with np.errstate(invalid="ignore"):
                y = ts.integrate_frames(ts.order_frames(session.last_1d["rendered"]), session.last_1d["x_axis"], self._integ_range())
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
    def _save(self, save_fn, canvas: CurveCanvas | None = None) -> None:
        payload = {"directory": self._last_dir}
        result = save_fn(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "Save failed."), False)
            return
        data_path = Path(result["path"])
        if canvas is None:
            self._set_status(f"Saved {data_path}.", True)
            return
        image_path = data_path.with_name(f"{data_path.stem}.png")
        try:
            canvas.save_figure(str(image_path))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Saved {data_path}, but image export failed: {exc}", False)
            return
        self._set_status(f"Saved {data_path} and {image_path}.", True)
