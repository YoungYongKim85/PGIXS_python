"""Shared GIXS/TrXS console widget.

One `ConsoleWidget` class, parametrized by `ConsoleConfig`, serves both the
"Console (GIXS)" and "Console (TrXS)" tabs -- mirroring the `createConsole(cfg)`
JS factory pattern from the earlier web prototype, which proved out sharing
~90% of the structure between the two modes. Every button handler here is a
thin wrapper: build a payload dict from widget values, call the matching
`console_*` function from `pgixs_python.ui_backend.console_session` directly
(in-process, no HTTP/JSON round trip), then render the result with real
matplotlib axes via `ImageCanvas`/`CurveCanvas`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QPointF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pgixs_python.desktop.background_tasks import run_in_background
from pgixs_python.desktop.plot_canvas import CurveCanvas, ImageCanvas
from pgixs_python.desktop.theme import BORDER, RING
from pgixs_python.ui_backend import console_session as cs

_COLORMAPS = sorted(plt.colormaps())


@dataclass
class ConsoleConfig:
    mode: str  # "gixs" | "trxs"
    title: str
    detector_list: list[str]
    has_incidence: bool
    has_footprint: bool
    has_cakecut: bool
    has_circular: bool
    map_formats: list[str] = field(default_factory=lambda: ["Pixel", "Qxy/Qz", "Qy/Qz"])


def gixs_config() -> ConsoleConfig:
    return ConsoleConfig(
        mode="gixs",
        title="Single Data Analysis",
        detector_list=["Eiger4M", "MarCCD", "MX170HS", "PI-CCD", "MAR345"],
        has_incidence=True,
        has_footprint=True,
        has_cakecut=True,
        has_circular=False,
        map_formats=["Pixel", "Qxy/Qz", "Qy/Qz"],
    )


def trxs_config() -> ConsoleConfig:
    return ConsoleConfig(
        mode="trxs",
        title="Single Data Analysis",
        detector_list=["Eiger4M", "MarCCD", "MX170HS", "PI_CCD"],
        has_incidence=False,
        has_footprint=False,
        has_cakecut=False,
        has_circular=True,
        map_formats=["Pixel", "Qy/Qz"],
    )


def _double_spin(
    minimum: float = -1.0e9,
    maximum: float = 1.0e9,
    decimals: int = 3,
    value: float = 0.0,
    step: float = 1.0,
    cyan: bool = False,
    readonly: bool = False,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setMaximumWidth(110)
    # QDoubleSpinBox's *minimum* size hint is based on fitting the widest
    # possible value in its range (e.g. range up to 1e6 needs room for
    # "-1000000.00"), which setMaximumWidth alone doesn't shrink -- fields
    # packed two/three to a row (Energy+SDD, Center+Footprint) blew past
    # the panel width because of this. An explicit minimum lets the layout
    # actually shrink them; the value still displays fine at this width.
    spin.setMinimumWidth(52)
    # The up/down buttons rendered illegibly small at this field height even
    # after fixing their hit-testing and drawing an explicit arrow glyph --
    # dropping them entirely is cleaner than fighting Qt's stylesheet
    # rendering further. Typing a value or scrolling over the field (both
    # still work with NoButtons) covers the same need.
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
    if cyan:
        spin.setProperty("cyan", "true")
    if readonly:
        spin.setReadOnly(True)
        spin.setProperty("readonly", "true")
    return spin


def _row(*widgets: QWidget) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    return container


def _field(label_text: str, *widgets: QWidget, label_width: int = 104) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = QLabel(label_text)
    label.setFixedWidth(label_width)
    layout.addWidget(label)
    for widget in widgets:
        layout.addWidget(widget, 1)
    return container


def _split_row(*fields: QWidget) -> QWidget:
    """Place several `_field(...)` groups side by side on one row, sharing
    the width evenly -- for packing e.g. Energy+SDD, or Center+Footprint,
    onto a single line instead of one field per row."""

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    for field in fields:
        layout.addWidget(field, 1)
    return container


def _ring_icon(size: int = 13, color: str = RING) -> QLabel:
    """A small concentric-ring mark -- a schematic cross-section of the
    scattering rings this app exists to measure -- used only to open each
    processing-stage divider. Drawn with QPainter instead of a bundled
    image asset so the app stays a single self-contained Python package."""

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    center = QPointF(size / 2, size / 2)
    pen = painter.pen()
    pen.setColor(QColor(color))
    pen.setWidthF(1.2)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(center, size / 2 - 1.4, size / 2 - 1.4)
    painter.drawEllipse(center, size / 2 - 4.6, size / 2 - 4.6)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(center, 1.0, 1.0)
    painter.end()

    label = QLabel()
    label.setPixmap(pixmap)
    label.setFixedSize(size, size)
    return label


def _divider(text: str) -> QWidget:
    """Section divider for the left panel: ring mark + label + a hairline
    rule that fills the rest of the row. Marks where one processing stage
    (1D Plot Process, Azimuthal Process, ...) ends and the next begins --
    real sequence in the workflow, not decoration."""

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 10, 0, 2)
    layout.setSpacing(7)

    layout.addWidget(_ring_icon())

    label = QLabel(text)
    label.setProperty("role", "divider")
    font = label.font()
    font.setCapitalization(QFont.AllUppercase)
    font.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
    label.setFont(font)
    layout.addWidget(label)

    rule = QFrame()
    rule.setFrameShape(QFrame.HLine)
    rule.setFixedHeight(1)
    rule.setStyleSheet(f"background: {BORDER}; border: none;")
    layout.addWidget(rule, 1)

    return container


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "muted")
    label.setWordWrap(True)
    return label


class ConsoleWidget(QWidget):
    busy_changed = Signal(bool)

    def __init__(self, config: ConsoleConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        # `console_session` keeps one module-level active session; each tab
        # owns its own `ConsoleSession` object and installs it via
        # `set_session()` before every call into that module (`_activate()`
        # below), so the GIXS and TrXS tabs don't clobber each other's
        # loaded data while both are alive in the QTabWidget.
        self._session = cs.ConsoleSession(mode=config.mode)
        self._pixel_mm = 0.075
        self._axis_mode = "pixel"  # "pixel" | "q"
        # Mirrors GIXS_gui.m's `param.Expinfo`: false until Load Info
        # succeeds once, then stays true for the rest of the session (even
        # across later Load Data calls) -- Openinfo_Callback re-derives the
        # pixel axis as beam-centered (HD = col - beamY, VD = beamZ - row)
        # instead of raw 1-based indices, and that never reverts.
        self._pixel_centered = False
        self._current_image_data: np.ndarray | None = None
        self._current_image_extent: tuple[float, float, float, float] | None = None
        self._current_xlabel = ""
        self._current_ylabel = ""
        self._position_mode: str | None = None
        self._zplot_active = False
        self._last_dir = str(Path.home())
        self._load_thread: QThread | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._progress_timer: QTimer | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])

        self._activate()

    def _activate(self) -> None:
        """Install this tab's session as the active `console_session` target.
        Must be the first line of every method that calls into `cs.`."""

        cs.set_session(self._session)

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        cfg = self.config
        group = QGroupBox(cfg.title)
        layout = QVBoxLayout(group)
        layout.setSpacing(7)

        self.dir_edit = QLineEdit(self._last_dir)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.dir_set_button = QPushButton("Set")
        self.dir_set_button.clicked.connect(self._on_set_directory_clicked)
        layout.addWidget(_field("Directory", self.dir_edit, self.dir_set_button))

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(cfg.detector_list)
        self.multishot_check = QCheckBox("Multishot")
        self.multishot_check.setToolTip(
            "Average every file in this file's series (same name with the trailing _NNNN index stripped)."
        )
        layout.addWidget(_field("Detector", self.detector_combo))
        layout.addWidget(self.multishot_check)

        self.load_button = QPushButton("LOAD DATA")
        self.load_button.setProperty("variant", "primary")
        self.load_button.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_button)

        self.ic_label = QLabel("0")
        self.ic_label.setProperty("role", "numeric")
        self.bc_label = QLabel("0")
        self.bc_label.setProperty("role", "numeric")
        counters = _row(QLabel("IC:"), self.ic_label, QLabel("   BC:"), self.bc_label)
        layout.addWidget(counters)

        self.energy_spin = _double_spin(0.1, 200.0, 4, 9.5, 0.1, cyan=True)
        self.sdd_spin = _double_spin(1.0, 1.0e6, 2, 3000.0, 1.0, cyan=True)
        self.energy_spin.setSuffix(" keV")
        self.sdd_spin.setSuffix(" mm")
        energy_field = _field("Energy", self.energy_spin, label_width=48)
        energy_field.setToolTip("Energy [keV]")
        sdd_field = _field("SDD", self.sdd_spin, label_width=32)
        sdd_field.setToolTip("SDD [mm]")
        layout.addWidget(_split_row(energy_field, sdd_field))

        self.beam_y_spin = _double_spin(-1.0e6, 1.0e6, 2, 1000.0, 1.0, cyan=True)
        self.beam_z_spin = _double_spin(-1.0e6, 1.0e6, 2, 1800.0, 1.0, cyan=True)
        self.beam_y_spin.setSuffix(" px")
        self.beam_z_spin.setSuffix(" px")
        # Sharing one row with 2 spinboxes (not 1, like the other compacted
        # fields) makes this pair the first to get squeezed below its
        # displayed text's width if the window is narrowed -- e.g.
        # "1000.00 px" needs ~95px, well past the general 52px floor.
        # Guarantee enough room to never clip; if the window gets narrower
        # than the panel can fit, the left panel's own scrollbar kicks in
        # instead of the number silently truncating.
        self.beam_y_spin.setMinimumWidth(100)
        self.beam_z_spin.setMinimumWidth(100)
        center_field = _field("Center", self.beam_y_spin, self.beam_z_spin, label_width=48)
        center_field.setToolTip("Center [y, z], pixels")
        if cfg.has_footprint:
            self.footprint_spin = _double_spin(0.0, 1.0e5, 3, 1.0, 0.1)
            self.footprint_spin.setSuffix(" mm")
            footprint_field = _field("FP", self.footprint_spin, label_width=24)
            footprint_field.setToolTip("Footprint [mm]")
            layout.addWidget(_split_row(center_field, footprint_field))
        else:
            self.footprint_spin = None
            layout.addWidget(center_field)

        self.info_button = QPushButton("Load Info")
        self.info_button.clicked.connect(self._on_load_info_clicked)
        layout.addWidget(self.info_button)

        self.grid_remove_button = QPushButton("Grid Remove")
        self.grid_remove_button.clicked.connect(self._on_grid_remove_clicked)
        layout.addWidget(self.grid_remove_button)

        if cfg.has_incidence:
            self.ai_ypix_spin = _double_spin(-1.0e6, 1.0e6, 1, 0.0, 1.0)
            self.zplot_button = QPushButton("z-plot")
            self.zplot_button.setMaximumWidth(70)
            self.zplot_button.clicked.connect(self._run_zplot)
            layout.addWidget(_field("ai [deg] / y-pos", self.ai_ypix_spin, self.zplot_button, label_width=104))

            self.ai_pix_spin = _double_spin(-1.0e6, 1.0e6, 2, 30.0, 1.0)
            self.ai_deg_spin = _double_spin(-90.0, 90.0, 4, 0.1, 0.01, cyan=True)
            self.ai_pix_spin.valueChanged.connect(self._incidence_from_pixel)
            self.ai_deg_spin.valueChanged.connect(self._incidence_from_degree)
            layout.addWidget(_field("pixel / degree", self.ai_pix_spin, self.ai_deg_spin))
        else:
            self.ai_ypix_spin = None
            self.zplot_button = None
            self.ai_pix_spin = None
            self.ai_deg_spin = None

        self.bkg_button = QPushButton("Load Bkg.")
        self.bkg_button.clicked.connect(self._on_load_bkg_clicked)
        layout.addWidget(self.bkg_button)

        self.plus_data_spin = _double_spin(-1.0e6, 1.0e6, 3, 1.0, 0.01)
        self.normal_combo = QComboBox()
        self.normal_combo.addItems(["None", "IC", "BC"])
        data_field = _field("*Data", self.plus_data_spin, label_width=48)
        normal_field = _field("Normalization", self.normal_combo, label_width=88)
        layout.addWidget(_split_row(data_field, normal_field))

        self.subtract_button = QPushButton("BKG sub. && Norm")
        self.subtract_button.clicked.connect(self._on_subtract_clicked)
        layout.addWidget(self.subtract_button)

        self.q_uncertain_check = QCheckBox("q uncertainty")
        self.mapping_combo = QComboBox()
        self.mapping_combo.addItems(["Qxy/Qz", "Qy/Qz"])
        layout.addWidget(_row(self.q_uncertain_check, self.mapping_combo))

        self.qmap_button = QPushButton("q-Map Cal.")
        self.qmap_button.clicked.connect(self._on_qmap_clicked)
        layout.addWidget(self.qmap_button)

        self.q_range_label = _muted("Q range: not computed yet.")
        layout.addWidget(self.q_range_label)

        layout.addWidget(_divider("1D Plot Process"))
        self.line_angle_spin = _double_spin(-360.0, 360.0, 2, 90.0, 0.5)
        layout.addWidget(_field("Line angle", self.line_angle_spin))
        layout.addWidget(_muted("0: ←,  90: ↑,  180: →"))

        if cfg.has_cakecut:
            self.cakecut_check = QCheckBox("Cakecut")
            self.cakecut_check.setToolTip("Cakecut angle range (-/+), degrees from the line angle.")
            self.cakecut_minus_spin = _double_spin(-360.0, 360.0, 1, -1.0, 0.5)
            self.cakecut_plus_spin = _double_spin(-360.0, 360.0, 1, 1.0, 0.5)
            layout.addWidget(_row(self.cakecut_check, self.cakecut_minus_spin, self.cakecut_plus_spin))
        else:
            self.cakecut_check = None
            self.cakecut_minus_spin = None
            self.cakecut_plus_spin = None

        self.line_median_check = QCheckBox("Median")
        layout.addWidget(self.line_median_check)

        self.line_position_button = QPushButton("Position")
        self.line_draw_button = QPushButton("Line")
        self.line_plot_button = QPushButton("1d Plot")
        self.line_position_button.clicked.connect(lambda: self._arm_position_mode("line"))
        self.line_draw_button.clicked.connect(self._on_line_draw_clicked)
        self.line_plot_button.clicked.connect(self._on_line_plot_clicked)
        layout.addWidget(_row(self.line_position_button, self.line_draw_button, self.line_plot_button))

        layout.addWidget(_divider("Azimuthal Process"))
        self.azi_pixel_spin = _double_spin(0.0, 1.0e6, 1, 202.0, 1.0)
        self.azi_range_minus_spin = _double_spin(-1.0e5, 1.0e5, 2, -1.0, 0.5)
        self.azi_range_plus_spin = _double_spin(-1.0e5, 1.0e5, 2, 1.0, 0.5)
        layout.addWidget(_field("Pixel / Range(-/+)", self.azi_pixel_spin, self.azi_range_minus_spin))
        layout.addWidget(_field("Range(+)", self.azi_range_plus_spin))

        self.azi_q_label = QLineEdit("0")
        self.azi_q_label.setReadOnly(True)
        self.azi_q_label.setProperty("readonly", "true")
        layout.addWidget(_field("Q [1/A]", self.azi_q_label))

        self.azi_position_button = QPushButton("Position")
        self.azi_meanmedian_combo = QComboBox()
        self.azi_meanmedian_combo.addItems(["Mean", "Median"])
        self.azi_position_button.clicked.connect(lambda: self._arm_position_mode("azimuthal"))
        layout.addWidget(_row(self.azi_position_button, self.azi_meanmedian_combo))

        self.azi_draw_button = QPushButton("Draw")
        self.azi_plot_button = QPushButton("Plot")
        self.azi_draw_button.clicked.connect(self._on_azi_draw_clicked)
        self.azi_plot_button.clicked.connect(self._on_azi_plot_clicked)
        layout.addWidget(_row(self.azi_draw_button, self.azi_plot_button))

        if cfg.has_circular:
            layout.addWidget(_divider("Circular Process"))
            self.circ_angle_min_spin = _double_spin(-360.0, 360.0, 1, 0.0, 1.0)
            self.circ_angle_max_spin = _double_spin(-360.0, 720.0, 1, 360.0, 1.0)
            self.circ_min_coverage_spin = _double_spin(0.0, 1.0, 2, 0.80, 0.05)
            self.circ_median_check = QCheckBox("Median")
            self.circ_std_check = QCheckBox("Std")
            layout.addWidget(_field("Ang. Avg (min/max)", self.circ_angle_min_spin, self.circ_angle_max_spin))
            layout.addWidget(_field("Min coverage", self.circ_min_coverage_spin))
            layout.addWidget(_row(self.circ_median_check, self.circ_std_check))
            self.circ_button = QPushButton("Plot (Circular Average)")
            self.circ_button.clicked.connect(self._on_circular_clicked)
            layout.addWidget(self.circ_button)
        else:
            self.circ_angle_min_spin = None
            self.circ_angle_max_spin = None
            self.circ_min_coverage_spin = None
            self.circ_median_check = None
            self.circ_std_check = None
            self.circ_button = None

        layout.addStretch(1)

        self.status_label = _muted("")
        # Reserve room for ~3 wrapped lines up front -- Load Data's status
        # message includes the full loaded file path (see _on_load_finished),
        # which routinely wraps to multiple lines at this panel's width. Left
        # unreserved, the label's height (and so the whole panel's height)
        # visibly jumps when that message appears and shrinks back down on
        # the next short message (e.g. Load Info's "Calibration loaded.").
        self.status_label.setMinimumHeight(3 * self.status_label.fontMetrics().lineSpacing())
        layout.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(group)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(500)
        return scroll

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        cfg = self.config
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        plot_group = QGroupBox("2D Image Plot")
        plot_group.setObjectName("plotGroup")
        plot_layout = QVBoxLayout(plot_group)
        self.image_canvas = ImageCanvas()
        self.image_canvas.pixel_clicked.connect(self._on_image_clicked)
        plot_layout.addWidget(self.image_canvas)
        splitter.addWidget(plot_group)

        controller_group = QGroupBox("Image Controller")
        controller_layout = QVBoxLayout(controller_group)

        # This panel has plenty of horizontal room (the right side of the
        # splitter, unlike the constrained left panel), so related controls
        # share a row instead of each taking its own line: render-style
        # dropdowns together, numeric display tuning together, axis ranges
        # together -- 7 rows down to 4.
        self.color_scale_combo = QComboBox()
        self.color_scale_combo.addItems(["Linear", "Log"])
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(_COLORMAPS)
        default_index = self.colormap_combo.findText("viridis")
        if default_index >= 0:
            self.colormap_combo.setCurrentIndex(default_index)
        self.map_format_combo = QComboBox()
        self.map_format_combo.addItems(cfg.map_formats)
        self.map_format_combo.currentTextChanged.connect(self._on_map_format_changed)
        # Every display-only control (ColorScale/Colormap/Contrast/Range/
        # Font) applies the moment it changes -- Update no longer needs to
        # be clicked for these to take effect. Spin boxes have
        # setKeyboardTracking(False) (see _double_spin), so valueChanged
        # fires on step/scroll/Enter/focus-loss, not on every keystroke.
        self.color_scale_combo.currentTextChanged.connect(self._apply_display_settings)
        self.colormap_combo.currentTextChanged.connect(self._apply_display_settings)
        controller_layout.addWidget(
            _split_row(
                _field("ColorScale", self.color_scale_combo, label_width=70),
                _field("Colormap", self.colormap_combo, label_width=64),
                _field("Map format", self.map_format_combo, label_width=76),
            )
        )

        self.cmin_spin = _double_spin(-1.0e12, 1.0e12, 3, 0.0, 1.0)
        self.cmax_spin = _double_spin(-1.0e12, 1.0e12, 3, 0.0, 1.0)
        self.yrange_min_spin = _double_spin(-1.0e9, 1.0e9, 4, 0.0, 1.0)
        self.yrange_max_spin = _double_spin(-1.0e9, 1.0e9, 4, 0.0, 1.0)
        self.zrange_min_spin = _double_spin(-1.0e9, 1.0e9, 4, 0.0, 1.0)
        self.zrange_max_spin = _double_spin(-1.0e9, 1.0e9, 4, 0.0, 1.0)
        self.font_sz_spin = _double_spin(6.0, 30.0, 0, 10.0, 1.0)
        for spin in (
            self.cmin_spin,
            self.cmax_spin,
            self.yrange_min_spin,
            self.yrange_max_spin,
            self.zrange_min_spin,
            self.zrange_max_spin,
            self.font_sz_spin,
        ):
            spin.valueChanged.connect(self._apply_display_settings)
        controller_layout.addWidget(
            _split_row(
                _field("Contrast", self.cmin_spin, self.cmax_spin, label_width=54),
                _field("Y-Range", self.yrange_min_spin, self.yrange_max_spin, label_width=54),
                _field("Z-Range", self.zrange_min_spin, self.zrange_max_spin, label_width=54),
                _field("Font", self.font_sz_spin, label_width=32),
            )
        )

        self.update_button = QPushButton("Update")
        self.update_button.setToolTip("Reset Y-Range/Z-Range to the full extent (everything else stays as set).")
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.update_button.clicked.connect(self._on_update_clicked)
        self.reset_zoom_button.clicked.connect(self._on_reset_zoom_clicked)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["png", "jpg", "pdf", "svg"])
        self.save_button = QPushButton("Save Img")
        self.save_button.clicked.connect(self._on_save_clicked)
        controller_layout.addWidget(
            _row(self.update_button, self.reset_zoom_button, QLabel("Format"), self.save_format_combo, self.save_button)
        )
        splitter.addWidget(controller_group)

        curve_group = QGroupBox("1D Result")
        curve_layout = QVBoxLayout(curve_group)
        self.curve_canvas = CurveCanvas()
        self.curve_canvas.point_clicked.connect(self._on_curve_clicked)
        self.curve_status_label = _muted("")
        curve_layout.addWidget(self.curve_canvas)
        curve_layout.addWidget(self.curve_status_label)
        splitter.addWidget(curve_group)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        # Without explicit initial sizes, QSplitter's default first-layout
        # negotiation can leave the plot area squeezed far smaller than the
        # stretch factors would suggest (they only govern *subsequent*
        # resizes). Give it a sane starting split so the 2D image is
        # visibly large on first show, not just after a manual drag.
        splitter.setSizes([520, 190, 320])
        return container

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def _set_status(self, message: str, ok: bool) -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("role", "status-ok" if ok else "status-error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _set_curve_status(self, message: str, ok: bool) -> None:
        self.curve_status_label.setText(message)
        self.curve_status_label.setProperty("role", "status-ok" if ok else "status-error")
        self.curve_status_label.style().unpolish(self.curve_status_label)
        self.curve_status_label.style().polish(self.curve_status_label)

    def _mark_ok(self, button: QPushButton, ok: bool) -> None:
        button.setProperty("state", "ok" if ok else None)
        button.style().unpolish(button)
        button.style().polish(button)

    # ------------------------------------------------------------------
    # Calibration field <-> session sync
    # ------------------------------------------------------------------
    def _apply_calibration(self, cal: dict[str, Any]) -> None:
        self._pixel_mm = float(cal.get("pixel_mm", self._pixel_mm)) or self._pixel_mm
        self.energy_spin.blockSignals(True)
        self.energy_spin.setValue(cal.get("energy_kev", self.energy_spin.value()))
        self.energy_spin.blockSignals(False)
        self.sdd_spin.blockSignals(True)
        self.sdd_spin.setValue(cal.get("sdd_mm", self.sdd_spin.value()))
        self.sdd_spin.blockSignals(False)
        beam = cal.get("beam_pos", [self.beam_y_spin.value(), self.beam_z_spin.value()])
        self.beam_y_spin.blockSignals(True)
        self.beam_y_spin.setValue(beam[0])
        self.beam_y_spin.blockSignals(False)
        self.beam_z_spin.blockSignals(True)
        self.beam_z_spin.setValue(beam[1])
        self.beam_z_spin.blockSignals(False)
        if self.footprint_spin is not None:
            self.footprint_spin.setValue(cal.get("footprint_mm", self.footprint_spin.value()))
        if self.ai_deg_spin is not None:
            self.ai_deg_spin.blockSignals(True)
            self.ai_deg_spin.setValue(cal.get("incident_angle_deg", self.ai_deg_spin.value()))
            self.ai_deg_spin.blockSignals(False)
            self._incidence_from_degree()

    def _incidence_from_pixel(self) -> None:
        if self.ai_pix_spin is None or self.ai_deg_spin is None:
            return
        sdd = self.sdd_spin.value()
        if sdd == 0:
            return
        ai = math.degrees(math.atan((self.ai_pix_spin.value() * self._pixel_mm) / sdd)) / 2.0
        self.ai_deg_spin.blockSignals(True)
        self.ai_deg_spin.setValue(ai)
        self.ai_deg_spin.blockSignals(False)

    def _incidence_from_degree(self) -> None:
        if self.ai_pix_spin is None or self.ai_deg_spin is None:
            return
        if self._pixel_mm == 0:
            return
        sdd = self.sdd_spin.value()
        pix = sdd * math.tan(math.radians(self.ai_deg_spin.value() * 2.0)) / self._pixel_mm
        self.ai_pix_spin.blockSignals(True)
        self.ai_pix_spin.setValue(pix)
        self.ai_pix_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _current_extent(self, h_start: float, h_end: float, v_start: float, v_end: float) -> tuple[float, float, float, float]:
        return (h_start, h_end, v_end, v_start)

    def _render_pixel_image(self, keep_limits: bool = False) -> None:
        self._activate()
        session = cs.get_session()
        if session.data is None:
            return
        data = session.corrected_data if session.corrected_data is not None else session.data
        rows, cols = data.shape
        self._axis_mode = "pixel"
        if self._pixel_centered:
            # GIXS_gui.m Openinfo_Callback: HD = (1:cols) - beamY,
            # VD = -((1:rows) - beamZ) -- row 1's VD value is beamZ-1 (near
            # the beam, large) and the last row's is beamZ-rows (very
            # negative), so the axis reads "beamZ-1" at the top down to
            # "beamZ-rows" at the bottom: the beam center sits at Y/Z = 0.
            beam_y = self.beam_y_spin.value()
            beam_z = self.beam_z_spin.value()
            h_start, h_end = 1 - beam_y, cols - beam_y
            v_row1, v_row_last = beam_z - 1, beam_z - rows
        else:
            h_start, h_end = 1, cols
            v_row1, v_row_last = 1, rows
        self.yrange_min_spin.blockSignals(True)
        self.yrange_max_spin.blockSignals(True)
        self.zrange_min_spin.blockSignals(True)
        self.zrange_max_spin.blockSignals(True)
        if not keep_limits:
            self.yrange_min_spin.setValue(min(h_start, h_end))
            self.yrange_max_spin.setValue(max(h_start, h_end))
            self.zrange_min_spin.setValue(min(v_row1, v_row_last))
            self.zrange_max_spin.setValue(max(v_row1, v_row_last))
        self.yrange_min_spin.blockSignals(False)
        self.yrange_max_spin.blockSignals(False)
        self.zrange_min_spin.blockSignals(False)
        self.zrange_max_spin.blockSignals(False)
        extent = self._current_extent(h_start, h_end, v_row1, v_row_last)
        self._render_image(data, extent, "Y, pixel", "Z, pixel", keep_limits=keep_limits)

    def _render_image(self, data: np.ndarray, extent, xlabel: str, ylabel: str, keep_limits: bool = False) -> None:
        self._current_image_data = data
        self._current_image_extent = extent
        self._current_xlabel = xlabel
        self._current_ylabel = ylabel
        cmin = self.cmin_spin.value()
        cmax = self.cmax_spin.value()
        vmin = cmin if cmax > cmin else None
        vmax = cmax if cmax > cmin else None
        used_vmin, used_vmax = self.image_canvas.render(
            data,
            cmap=self.colormap_combo.currentText(),
            log_scale=self.color_scale_combo.currentText() == "Log",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
            font_size=int(self.font_sz_spin.value()),
            keep_limits=keep_limits,
        )
        if vmin is None:
            # Reflect back whatever ImageCanvas actually auto-computed
            # (percentile-based, robust to hot/outlier pixels) instead of
            # re-deriving a separate -- and previously inconsistent --
            # min/max here.
            self.cmin_spin.blockSignals(True)
            self.cmax_spin.blockSignals(True)
            self.cmin_spin.setValue(used_vmin)
            self.cmax_spin.setValue(used_vmax)
            self.cmin_spin.blockSignals(False)
            self.cmax_spin.blockSignals(False)
        if self._position_mode == "line":
            self._draw_line_overlay()
        elif self._position_mode == "azimuthal":
            pass

    def _draw_line_overlay(self) -> None:
        center = self._current_beam_center()
        self.image_canvas.draw_beam_line(center, self.line_angle_spin.value())

    def _current_beam_center(self) -> tuple[float, float]:
        if self._axis_mode == "pixel" and not self._pixel_centered:
            return (self.beam_y_spin.value(), self.beam_z_spin.value())
        # Q-space is beam-centered by construction, and once the pixel axis
        # has been re-centered by Load Info (see _render_pixel_image), the
        # beam sits at (0, 0) on that axis too.
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # Directory ("Set" mirrors GIXS_gui.m's Directory field + Set button --
    # seeds where LOAD DATA / Load Info / Load Bkg. open, matching
    # `pathset_Callback`'s `uigetdir`).
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
    # LOAD DATA
    # ------------------------------------------------------------------
    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a file", self._last_dir, "All files (*)")
        if not path:
            return
        file_path = Path(path)
        self._set_last_dir(str(file_path.parent))
        multishot = self.multishot_check.isChecked()
        payload = {
            "directory": str(file_path.parent),
            "filename": file_path.name,
            "mode": self.config.mode,
            "detector": self.detector_combo.currentText(),
            "multishot": multishot,
        }
        self._set_status(f"Loading {file_path.name}...", True)
        self.load_button.setEnabled(False)
        self.busy_changed.emit(True)
        if multishot:
            self._start_progress_dialog("Loading multishot series...")
        self._activate()
        self._load_thread = run_in_background(cs.console_load, payload, self._on_load_finished)

    def _start_progress_dialog(self, title: str) -> None:
        dialog = QProgressDialog(title, None, 0, 100, self)
        dialog.setWindowTitle("Loading")
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
        self._activate()
        status = cs.console_load_progress()
        if not status.get("active") or self._progress_dialog is None:
            return
        total = status.get("total") or 1
        current = status.get("current") or 0
        self._progress_dialog.setMaximum(total)
        self._progress_dialog.setValue(current)
        self._progress_dialog.setLabelText(f"Loading {status.get('label', '')} ({current}/{total})")

    def _stop_progress_dialog(self) -> None:
        if self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer = None
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

    def _on_load_finished(self, payload: dict[str, Any]) -> None:
        self._activate()
        self._stop_progress_dialog()
        self.load_button.setEnabled(True)
        self.busy_changed.emit(False)
        if not payload.get("ok"):
            self._set_status(payload.get("message", "Load failed."), False)
            return
        self.ic_label.setText(str(payload.get("ic") if payload.get("ic") is not None else "0"))
        self.bc_label.setText(str(payload.get("bc") if payload.get("bc") is not None else "0"))
        self._apply_calibration(payload.get("calibration", {}))
        self._mark_ok(self.qmap_button, False)
        self._mark_ok(self.grid_remove_button, False)
        self.q_range_label.setText("Q range: not computed yet.")
        self.map_format_combo.blockSignals(True)
        self.map_format_combo.setCurrentText("Pixel")
        self.map_format_combo.blockSignals(False)
        self._axis_mode = "pixel"
        self.image_canvas.clear_overlays()
        self._position_mode = None
        self._render_pixel_image()
        self.image_canvas.reset_view()
        series_note = f" (averaged {payload['series_file_count']} files)" if payload.get("series_file_count") else ""
        shape = payload.get("shape", [])
        self._set_status(f"Loaded {'x'.join(str(s) for s in shape)} from {payload.get('path')}.{series_note}", True)

    # ------------------------------------------------------------------
    # Load Info
    # ------------------------------------------------------------------
    def _on_load_info_clicked(self) -> None:
        self._activate()
        path, _ = QFileDialog.getOpenFileName(self, "Select calibration file", self._last_dir, "Calibration JSON (*.json)")
        if not path:
            return
        file_path = Path(path)
        result = cs.console_load_info({"directory": str(file_path.parent), "filename": file_path.name})
        if not result.get("ok"):
            self._set_status(result.get("message", "Load Info failed."), False)
            return
        self._apply_calibration(result.get("calibration", {}))
        # GIXS_gui.m Openinfo_Callback: param.Expinfo = 1; renderImage() --
        # the pixel axis re-centers on the beam position from here on.
        self._pixel_centered = True
        if self._axis_mode == "pixel":
            self._render_pixel_image()
        self._set_status("Calibration loaded.", True)

    # ------------------------------------------------------------------
    # Grid Remove
    # ------------------------------------------------------------------
    def _on_grid_remove_clicked(self) -> None:
        self._activate()
        result = cs.console_grid_remove({})
        if not result.get("ok"):
            self._set_status(result.get("message", "Grid remove failed."), False)
            return
        self._mark_ok(self.grid_remove_button, True)
        self.map_format_combo.blockSignals(True)
        self.map_format_combo.setCurrentText("Pixel")
        self.map_format_combo.blockSignals(False)
        self._render_pixel_image()
        self._set_status(result.get("message", "Grid remove complete."), True)

    # ------------------------------------------------------------------
    # z-plot (incidence-angle peak pick)
    # ------------------------------------------------------------------
    def _run_zplot(self) -> None:
        self._activate()
        session = cs.get_session()
        if session.data is None or self.ai_ypix_spin is None:
            self._set_status("Load data first.", False)
            return
        rows, cols = session.data.shape
        target_col = int(round(self.beam_y_spin.value() + self.ai_ypix_spin.value()))
        target_col = max(1, min(cols, target_col))
        column = session.data[:, target_col - 1]
        # aiZplot_Callback plots against `m = -(row - beamZ)` (beam-centered,
        # same convention as VD), then feeds the clicked x straight into
        # GX.hAiPix.Value. GX.hAiPix is a pixel *offset from the beam* --
        # CalIncidence_Callback's atand(pixai*pixel/SDD)/2 only makes sense
        # for a small relative offset, not a raw detector row index -- so
        # the curve's x-axis has to be beam-centered too, or a click here
        # would feed a huge, wrong value into the incidence-angle formula.
        beam_z = self.beam_z_spin.value()
        row_pixels = beam_z - np.arange(1, rows + 1, dtype=np.float64)
        valid = np.isfinite(column)
        if not np.any(valid):
            self._set_status("No finite data along that column for z-plot.", False)
            return
        self.curve_canvas.render(row_pixels[valid], column[valid], xlabel="pixel (beam-centered, z)")
        self._zplot_active = True
        self._set_curve_status("Click the peak on the profile below to set the incidence pixel.", True)

        # Mark which detector column the z-plot was taken along, so the
        # "ai [deg] / y-pos" offset is visible on the 2D image itself.
        # Display coordinates depend on whether the pixel axis is currently
        # beam-centered (h = col - beamY) or raw (h = col), same formula
        # _render_pixel_image uses.
        self.image_canvas.clear_overlays()
        if self._axis_mode == "pixel":
            display_col = float(target_col) - self.beam_y_spin.value() if self._pixel_centered else float(target_col)
            self.image_canvas.draw_column_marker(display_col)

    def _on_curve_clicked(self, x: float, _y: float) -> None:
        if not self._zplot_active or self.ai_pix_spin is None:
            return
        self.ai_pix_spin.setValue(x)
        self._incidence_from_pixel()
        self._zplot_active = False
        self._set_curve_status(f"Incidence pixel set to {x:.2f} (ai = {self.ai_deg_spin.value():.4f} deg).", True)

    # ------------------------------------------------------------------
    # Background + Normalize
    # ------------------------------------------------------------------
    def _on_load_bkg_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a background file", self._last_dir, "All files (*)")
        if not path:
            return
        file_path = Path(path)
        multishot = self.multishot_check.isChecked()
        payload = {"directory": str(file_path.parent), "filename": file_path.name, "multishot": multishot}
        self._set_status("Loading background...", True)
        self.bkg_button.setEnabled(False)
        self.busy_changed.emit(True)
        if multishot:
            self._start_progress_dialog("Loading background series...")
        self._activate()
        self._load_thread = run_in_background(cs.console_background_load, payload, self._on_bkg_load_finished)

    def _on_bkg_load_finished(self, payload: dict[str, Any]) -> None:
        self._activate()
        self._stop_progress_dialog()
        self.bkg_button.setEnabled(True)
        self.busy_changed.emit(False)
        if not payload.get("ok"):
            self._set_status(payload.get("message", "Background load failed."), False)
            return
        self._set_status(f"Background loaded from {payload.get('path')}.", True)

    def _on_subtract_clicked(self) -> None:
        self._activate()
        payload = {"method": self.normal_combo.currentText(), "plus_data": self.plus_data_spin.value(), "confirm": False}
        preview = cs.console_normalize(payload)
        if not preview.get("ok"):
            self._set_status(preview.get("message", "Normalization failed."), False)
            return
        keep = QMessageBox.question(
            self,
            "Confirmation",
            "Preview computed. Keep this background-subtracted/normalized result?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if keep != QMessageBox.Yes:
            self._set_status("Normalization discarded.", True)
            return
        self._activate()
        committed = cs.console_normalize({**payload, "confirm": True})
        if not committed.get("ok"):
            self._set_status(committed.get("message", "Normalization failed."), False)
            return
        self.map_format_combo.blockSignals(True)
        self.map_format_combo.setCurrentText("Pixel")
        self.map_format_combo.blockSignals(False)
        self._render_pixel_image()
        self._set_status(committed.get("message", "Normalization complete."), True)

    # ------------------------------------------------------------------
    # q-Map Cal.
    # ------------------------------------------------------------------
    def _on_qmap_clicked(self) -> None:
        self._activate()
        payload = {
            "energy_kev": self.energy_spin.value(),
            "sdd_mm": self.sdd_spin.value(),
            "beam_y": self.beam_y_spin.value(),
            "beam_z": self.beam_z_spin.value(),
            "incident_angle_deg": self.ai_deg_spin.value() if self.ai_deg_spin is not None else 0.0,
            "footprint_mm": self.footprint_spin.value() if self.footprint_spin is not None else 0.0,
        }
        result = cs.console_qmap(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "q-Map Cal. failed."), False)
            return
        self._mark_ok(self.qmap_button, True)
        q_range = result.get("q_range") or {}
        q_min = q_range.get("min")
        q_max = q_range.get("max")
        self.q_range_label.setText(
            f"Q range: {q_min:.3f} to {q_max:.3f} 1/A" if q_min is not None and q_max is not None else "Q range: n/a"
        )
        self.map_format_combo.blockSignals(True)
        self.map_format_combo.setCurrentText("Pixel")
        self.map_format_combo.blockSignals(False)
        self._render_pixel_image()
        self._set_status("q-Map Cal. complete.", True)

    # ------------------------------------------------------------------
    # Map format (Pixel / Qxy/Qz / Qy/Qz)
    # ------------------------------------------------------------------
    def _on_map_format_changed(self, mode: str) -> None:
        self._activate()
        session = cs.get_session()
        if mode == "Pixel":
            if session.data is None:
                return
            self._render_pixel_image()
            self._set_status("Pixel view restored.", True)
            return
        if session.qmap is None:
            self._set_status("Run q-Map Cal. first.", False)
            self.map_format_combo.blockSignals(True)
            self.map_format_combo.setCurrentText("Pixel")
            self.map_format_combo.blockSignals(False)
            return
        result = cs.console_qspace_image({"mapping": mode})
        if not result.get("ok"):
            self._set_status(result.get("message", f"{mode} transform failed."), False)
            self.map_format_combo.blockSignals(True)
            self.map_format_combo.setCurrentText("Pixel")
            self.map_format_combo.blockSignals(False)
            return
        self._axis_mode = "q"
        preview = np.asarray(result["preview"], dtype=np.float64)
        hr = result["horizontal_range"]
        vr = result["vertical_range"]
        self.yrange_min_spin.blockSignals(True)
        self.yrange_max_spin.blockSignals(True)
        self.zrange_min_spin.blockSignals(True)
        self.zrange_max_spin.blockSignals(True)
        self.yrange_min_spin.setValue(hr["min"])
        self.yrange_max_spin.setValue(hr["max"])
        self.zrange_min_spin.setValue(vr["min"])
        self.zrange_max_spin.setValue(vr["max"])
        self.yrange_min_spin.blockSignals(False)
        self.yrange_max_spin.blockSignals(False)
        self.zrange_min_spin.blockSignals(False)
        self.zrange_max_spin.blockSignals(False)
        extent = self._current_extent(hr["min"], hr["max"], vr["max"], vr["min"])
        label = "q_{xy}, 1/A" if mode == "Qxy/Qz" else "q_y, 1/A"
        self._render_image(preview, extent, label, "q_z, 1/A")
        self.image_canvas.reset_view()
        self._set_status(f"{mode} image ready.", True)

    # ------------------------------------------------------------------
    # Display settings (live) / Update (reset ranges) / Reset Zoom / Save
    # ------------------------------------------------------------------
    def _apply_display_settings(self) -> None:
        # Re-render the *currently displayed* array (pixel- or Q-space --
        # whichever "Map format" is active) with the current ColorScale/
        # Colormap/Contrast/Font Sz, then apply the Y-Range/Z-Range fields
        # as axis limits. This deliberately does not re-fetch from the
        # backend in Q mode, so editing Y-Range/Z-Range actually crops the
        # view instead of being silently overwritten by a fresh
        # full-extent transform. Wired to every display control below so
        # each one takes effect immediately, not just on an explicit
        # button click.
        if self._current_image_data is None:
            return
        self._render_image(
            self._current_image_data,
            self._current_image_extent,
            self._current_xlabel,
            self._current_ylabel,
            keep_limits=False,
        )
        y_min = self.yrange_min_spin.value()
        y_max = self.yrange_max_spin.value()
        z_min = self.zrange_min_spin.value()
        z_max = self.zrange_max_spin.value()
        if y_max > y_min:
            self.image_canvas.set_axis_limits((y_min, y_max), None)
        if z_max != z_min:
            # Raw pixel mode reads top-to-bottom as small-to-large (row 1 at
            # top with the smallest value); Q-space and beam-centered pixel
            # mode both read top-to-bottom as large-to-small (the beam/q=0
            # origin is "up" from the far edge). See _render_pixel_image.
            axis_reversed = self._axis_mode == "pixel" and not self._pixel_centered
            ylim = (z_max, z_min) if axis_reversed else (z_min, z_max)
            self.image_canvas.set_axis_limits(None, ylim)

    def _on_update_clicked(self) -> None:
        # Every other display control now applies live (see
        # _apply_display_settings above), so Update's own job narrows to
        # just this: put Y-Range/Z-Range back to the full extent of the
        # currently displayed data, leaving ColorScale/Colormap/Contrast/
        # Font exactly as the user left them. `_current_image_extent` is
        # the (left, right, bottom, top) the last full render used --
        # stable across live Y/Z-Range edits, since those only crop the
        # view (set_axis_limits), never touch this stored extent.
        if self._current_image_extent is None:
            return
        left, right, bottom, top = self._current_image_extent
        self.yrange_min_spin.blockSignals(True)
        self.yrange_max_spin.blockSignals(True)
        self.zrange_min_spin.blockSignals(True)
        self.zrange_max_spin.blockSignals(True)
        self.yrange_min_spin.setValue(min(left, right))
        self.yrange_max_spin.setValue(max(left, right))
        self.zrange_min_spin.setValue(min(bottom, top))
        self.zrange_max_spin.setValue(max(bottom, top))
        self.yrange_min_spin.blockSignals(False)
        self.yrange_max_spin.blockSignals(False)
        self.zrange_min_spin.blockSignals(False)
        self.zrange_max_spin.blockSignals(False)
        self.image_canvas.set_axis_limits((left, right), (bottom, top))

    def _on_reset_zoom_clicked(self) -> None:
        self.image_canvas.reset_view()

    def _on_save_clicked(self) -> None:
        self._activate()
        result = cs.console_save({})
        if not result.get("ok"):
            self._set_status(result.get("message", "Save failed."), False)
            return
        session = cs.get_session()
        stem = Path(session.source_path).stem if session.source_path else "console"
        out_dir = Path(result["h5_path"]).parent
        fmt = self.save_format_combo.currentText()
        image_path = out_dir / f"{stem}_2d.{fmt}"
        try:
            self.image_canvas.save_figure(str(image_path))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Saved {result['h5_path']}, but image export failed: {exc}", False)
            return
        curve_note = f" and {result['curve_path']}" if result.get("curve_path") else ""
        self._set_status(f"Saved {result['h5_path']}{curve_note} and {image_path}.", True)

    # ------------------------------------------------------------------
    # Click-to-position (Line / Azimuthal)
    # ------------------------------------------------------------------
    def _arm_position_mode(self, mode: str) -> None:
        self._position_mode = mode
        if mode == "line":
            self._set_status("Click a point on the 2D image to set the line angle.", True)
        elif mode == "azimuthal":
            self._set_status("Click a point on the 2D image to set the azimuthal radius.", True)

    def _on_image_clicked(self, x: float, y: float) -> None:
        if self._position_mode is None:
            return
        if self.image_canvas.toolbar.mode:
            return
        beam_y, beam_z = self._current_beam_center()
        if self._position_mode == "line":
            angle = (math.degrees(math.atan2(-(y - beam_z), -(x - beam_y))) + 360.0) % 360.0
            self.line_angle_spin.setValue(angle)
            self._set_status(f"Line position clicked at ({x:.1f}, {y:.1f}) -> angle {angle:.2f} deg.", True)
        elif self._position_mode == "azimuthal":
            radius = math.hypot(x - beam_y, y - beam_z)
            self.azi_pixel_spin.setValue(radius)
            self._set_status(f"Azimuthal position clicked at ({x:.1f}, {y:.1f}) -> radius {radius:.1f} px.", True)

    # ------------------------------------------------------------------
    # 1D Line Process
    # ------------------------------------------------------------------
    def _on_line_draw_clicked(self) -> None:
        self.image_canvas.clear_overlays()
        self._draw_line_overlay()

    def _on_line_plot_clicked(self) -> None:
        self._activate()
        cfg = self.config
        payload = {
            "angle_deg": self.line_angle_spin.value(),
            "cakecut": self.cakecut_check.isChecked() if cfg.has_cakecut and self.cakecut_check is not None else False,
            "range_minus": self.cakecut_minus_spin.value() if cfg.has_cakecut and self.cakecut_minus_spin is not None else -1.0,
            "range_plus": self.cakecut_plus_spin.value() if cfg.has_cakecut and self.cakecut_plus_spin is not None else 1.0,
            "use_median": self.line_median_check.isChecked(),
        }
        result = cs.console_line_1d(payload)
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "1D line plot failed."), False)
            return
        preview = result["preview"]
        self.curve_canvas.render(np.asarray(preview["q"]), np.asarray(preview["intensity"]), xlabel="q [1/A]")
        diag = result.get("coverage_diagnostics", {})
        self._set_curve_status(
            f"Line 1D ready. Coverage median {diag.get('coverage_median', 'n/a')}, "
            f"low bins {diag.get('low_bin_count', 0)}, drops {diag.get('drop_bin_count', 0)}.",
            True,
        )

    # ------------------------------------------------------------------
    # Azimuthal Process
    # ------------------------------------------------------------------
    def _on_azi_draw_clicked(self) -> None:
        self.image_canvas.clear_overlays()
        center = self._current_beam_center()
        self.image_canvas.draw_beam_circle(center, self.azi_pixel_spin.value())

    def _on_azi_plot_clicked(self) -> None:
        self._activate()
        payload = {
            "pixel": self.azi_pixel_spin.value(),
            "range_minus": self.azi_range_minus_spin.value(),
            "range_plus": self.azi_range_plus_spin.value(),
            "mean_or_median": self.azi_meanmedian_combo.currentText(),
        }
        result = cs.console_azimuthal_1d(payload)
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Azimuthal plot failed."), False)
            return
        preview = result["preview"]
        self.curve_canvas.render(np.asarray(preview["angle"]), np.asarray(preview["intensity"]), xlabel="angle [deg]")
        diag = result.get("coverage_diagnostics", {})
        self._set_curve_status(
            f"Azimuthal profile ready. Coverage median {diag.get('coverage_median', 'n/a')}, "
            f"low bins {diag.get('low_bin_count', 0)}, drops {diag.get('drop_bin_count', 0)}.",
            True,
        )

    # ------------------------------------------------------------------
    # Circular Process (TrXS)
    # ------------------------------------------------------------------
    def _on_circular_clicked(self) -> None:
        self._activate()
        if self.circ_angle_min_spin is None:
            return
        payload = {
            "angle_min": self.circ_angle_min_spin.value(),
            "angle_max": self.circ_angle_max_spin.value(),
            "min_coverage": self.circ_min_coverage_spin.value(),
            "use_median": self.circ_median_check.isChecked(),
            "calc_std": self.circ_std_check.isChecked(),
        }
        result = cs.console_circular_average(payload)
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Circular average failed."), False)
            return
        preview = result["preview"]
        self.curve_canvas.render(np.asarray(preview["q"]), np.asarray(preview["intensity"]), xlabel="q [1/A]")
        diag = result.get("coverage_diagnostics", {})
        self._set_curve_status(
            f"Circular average ready. Coverage median {diag.get('coverage_median', 'n/a')}, "
            f"low bins {diag.get('low_bin_count', 0)}, drops {diag.get('drop_bin_count', 0)}.",
            True,
        )
