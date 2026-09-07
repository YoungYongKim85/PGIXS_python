"""Shared GIXS/TrXS console widget.

One `ConsoleWidget` class, parametrized by `ConsoleConfig`, serves both the
"GIXS" and "TrXS" tabs -- mirroring the `createConsole(cfg)`
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
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap, QValidator
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pgixs_python.desktop.background_tasks import run_in_background
from pgixs_python.desktop.plot_canvas import CurveCanvas, ImageCanvas
from pgixs_python.desktop.theme import BORDER, RING
from pgixs_python.ui_backend import console_session as cs

# Matplotlib ships 160+ colormaps; a curated shortlist is easier to pick
# from than the full registry. viridis is first/default.
_COLORMAPS = [
    "viridis",
    "turbo",
    "hot",
    "gray",
    "ocean",
    "magma",
    "RdBu",
    "RdBu_r",
    "binary",
    "gist_earth",
    "nipy_spectral",
    "twilight",
    "twilight_shifted",
]


@dataclass
class ConsoleConfig:
    mode: str  # "gixs" | "trxs"
    title: str
    detector_list: list[str]
    has_incidence: bool
    has_footprint: bool
    has_cakecut: bool
    has_circular: bool
    # Pos move (y,z) + Accumulation are GIXS-only in GIXS_gui.m (Accumpix_gi
    # etc.) -- TrXS_gui.m's Lineplot_Callback/Anglecut1d always cuts through
    # the beam center with no offset, so TrXS has neither field.
    has_pos_move_and_accumulation: bool = True
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
        has_pos_move_and_accumulation=True,
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
        has_pos_move_and_accumulation=False,
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


class _ScientificDoubleSpinBox(QDoubleSpinBox):
    """A QDoubleSpinBox that displays/accepts scientific notation (e.g.
    "2.00e-04") instead of plain fixed-decimal formatting -- for fields
    spanning several orders of magnitude (Q ranges, etc.), where a normal
    `decimals()` setting either silently rounds a small value like 0.0002
    down to "0.0000" at 4 decimals, or wastes width on trailing zeros for
    larger ones. Reported: "Q Min-Max의 텍스트 박스에 0.0000만 써 있어...
    1e-3 이런식으로 표기되게 하면 더 낫지 않을까"."""

    def __init__(self, sig_figs: int = 2, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sig_figs = max(0, sig_figs)
        # High internal precision so the spin box's own decimals()-based
        # rounding (independent of the sig_figs display format below) never
        # clips a typed/step value before textFromValue ever sees it.
        self.setDecimals(12)

    def textFromValue(self, value: float) -> str:
        return f"{value:.{self._sig_figs}e}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.strip())
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        stripped = text.strip()
        if stripped in ("", "-", "+", ".", "-.", "+."):
            return (QValidator.Intermediate, text, pos)
        try:
            float(stripped)
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            # Allow partial scientific-notation entry mid-typing (e.g. "1e",
            # "1e-", "1.5e+") rather than rejecting each keystroke outright.
            if stripped.lower().endswith(("e", "e+", "e-")):
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)


def _sci_spin(
    minimum: float = -1.0e6,
    maximum: float = 1.0e6,
    value: float = 0.0,
    step: float = 1.0e-4,
    sig_figs: int = 2,
) -> QDoubleSpinBox:
    spin = _ScientificDoubleSpinBox(sig_figs=sig_figs)
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setMaximumWidth(110)
    spin.setMinimumWidth(72)
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
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


def _size_combo(combo: QComboBox, chars: int = 12) -> QComboBox:
    """Give a combo box a content-length width budget instead of letting
    its single longest item (e.g. "twilight_shifted", "Symmetric Peaks
    (+/-)", "Mesoporous Silica") dictate both its preferred AND its real
    *minimum* size.

    QComboBox's default size-adjust policy (AdjustToContentsOnFirstShow)
    makes `minimumSizeHint()` track the longest item exactly -- a plain
    `setFixedWidth`/`setMaximumWidth` call only caps how wide the box
    *renders* when there's room; it does nothing to the layout's real
    minimum footprint, since Qt still reserves `minimumSizeHint()` worth
    of space regardless. That's why aligning several combos that way
    still made a narrow left panel need horizontal scrolling (reported:
    "다 같은 사이즈로 하니까 약간씩 삐져나가네").

    `AdjustToMinimumContentsLengthWithIcon` + a fixed `chars` budget makes
    both the preferred size and the real minimum exactly `chars`
    characters wide, independent of any one item's length -- longer
    entries elide with "..." once selected instead of forcing extra
    width, and same-`chars` siblings end up genuinely equal-width even
    under space pressure, not just when the window happens to be wide."""

    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    return combo


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


def _wrap_long_tokens(text: str) -> str:
    """Insert a zero-width-space break opportunity after every backslash.

    Qt's word-wrap only breaks at whitespace, and a Windows path (e.g. the
    full source/output paths Load Data and Save Img report) has none -- an
    unbreakable ~800px-wide token forces the status label, and with it the
    whole scroll area, wider instead of wrapping, which visibly resizes the
    left panel. This keeps every character of the message intact; it just
    gives the layout somewhere to break a long path across lines."""

    zero_width_space = chr(0x200B)
    return text.replace("\\", "\\" + zero_width_space)


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
        self._current_aspect = "equal"
        self._position_mode: str | None = None
        self._zplot_active = False
        self._mask_dialog: QDialog | None = None
        self._last_dir = str(Path.home())
        self._load_thread: QThread | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._progress_timer: QTimer | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal)
        # Fusion paints the handle's own rect with a hard-coded fill that
        # ignores both QSS "background" and QApplication's QPalette --
        # shrinking the handle to a hairline is what actually keeps it from
        # rendering as a thick black bar (see theme.py's QSplitter::handle
        # rule, which only controls the *color* once the rect is this small).
        splitter.setHandleWidth(2)
        splitter.setFrameShape(QFrame.NoFrame)
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
        _size_combo(self.detector_combo, 12)
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
        # Same content-length budget as the left panel's other "choice"
        # combos (Mean/Median, Qxy/Qz<->Qy/Qz) -- see _size_combo's own
        # docstring for why this (not setFixedWidth) actually keeps the
        # panel from needing to scroll horizontally.
        _size_combo(self.normal_combo, 8)
        data_field = _field("*Data", self.plus_data_spin, label_width=48)
        normal_field = _field("Normalization", self.normal_combo, label_width=88)
        layout.addWidget(_split_row(data_field, normal_field))

        self.subtract_button = QPushButton("BKG sub. && Norm")
        self.subtract_button.clicked.connect(self._on_subtract_clicked)
        layout.addWidget(self.subtract_button)

        if "Qxy/Qz" in cfg.map_formats:
            # Qxy/Qz vs Qy/Qz only exists as a GIXS distinction (TrXS's qr
            # is always plain Qy -- map_formats never even offers Qxy/Qz
            # for it), so this row -- and the "q uncertainty" checkbox that
            # was never wired to anything -- has nothing to do for TrXS.
            self.q_uncertain_check = QCheckBox("q uncertainty")
            self.mapping_combo = QComboBox()
            self.mapping_combo.addItems(["Qxy/Qz", "Qy/Qz"])
            _size_combo(self.mapping_combo, 10)
            layout.addWidget(_row(self.q_uncertain_check, self.mapping_combo))
        else:
            self.q_uncertain_check = None
            self.mapping_combo = None

        self.qmap_button = QPushButton("q-Map Cal.")
        self.qmap_button.clicked.connect(self._on_qmap_clicked)
        layout.addWidget(self.qmap_button)

        self.q_range_label = _muted("Q range: not computed yet.")
        layout.addWidget(self.q_range_label)

        # Editing any q-Map input after q-Map Cal. has run leaves the cached
        # map (and corrected data) describing the *old* geometry, while the
        # button still reads as done -- clear that indicator so it's visible
        # the map needs recomputing. `_apply_calibration` blocks these signals
        # for its own programmatic writes, so Load Info doesn't trip it.
        for _spin in (self.energy_spin, self.sdd_spin, self.beam_y_spin, self.beam_z_spin, self.footprint_spin):
            if _spin is not None:
                _spin.valueChanged.connect(self._mark_qmap_stale)
        if self.ai_deg_spin is not None:
            self.ai_deg_spin.valueChanged.connect(self._mark_qmap_stale)
        if self.q_uncertain_check is not None:
            self.q_uncertain_check.toggled.connect(self._mark_qmap_stale)

        if cfg.has_circular:
            layout.addWidget(_divider("Circular Process"))
            self.circ_angle_min_spin = _double_spin(-360.0, 360.0, 1, 0.0, 1.0)
            self.circ_angle_max_spin = _double_spin(-360.0, 720.0, 1, 360.0, 1.0)
            self.circ_min_coverage_spin = _double_spin(0.0, 1.0, 2, 0.80, 0.05)
            self.circ_median_check = QCheckBox("Median")
            self.circ_std_check = QCheckBox("Std")
            ang_avg_field = _field("Ang. Avg (min/max)", self.circ_angle_min_spin, self.circ_angle_max_spin)
            ang_avg_field.setToolTip(
                "The sector's boundary rays draw live on the 2D Image Plot as you edit\n"
                "these -- see which wedge of the detector Circular Average will use.\n"
                "Nothing to draw for a full 0-360deg sweep."
            )
            layout.addWidget(ang_avg_field)
            # Live, not a button -- redraws the boundary rays the moment
            # either value changes (spin boxes have setKeyboardTracking(False),
            # so this fires on step/scroll/Enter/focus-loss, not every keystroke).
            self.circ_angle_min_spin.valueChanged.connect(self._update_circ_sector_overlay)
            self.circ_angle_max_spin.valueChanged.connect(self._update_circ_sector_overlay)
            layout.addWidget(_field("Min coverage", self.circ_min_coverage_spin))
            layout.addWidget(_row(self.circ_median_check, self.circ_std_check))
            self.circ_mask_draw_button = QPushButton("Draw Mask")
            self.circ_mask_draw_button.setToolTip(
                "Draw a polygon on the 2D Image Plot (Pixel view) -- pixels inside it are\n"
                "excluded from Circular Average; everything outside it is kept.\n"
                "Click points, double-click to close."
            )
            self.circ_mask_clear_button = QPushButton("Clear Mask")
            self.circ_mask_draw_button.clicked.connect(self._on_circ_mask_draw_clicked)
            self.circ_mask_clear_button.clicked.connect(self._on_circ_mask_clear_clicked)
            layout.addWidget(_row(self.circ_mask_draw_button, self.circ_mask_clear_button))
            self.circ_mask_status_label = _muted("Mask: none")
            layout.addWidget(self.circ_mask_status_label)
            self.circ_button = QPushButton("Plot (Circular Average)")
            self.circ_button.clicked.connect(self._on_circular_clicked)
            layout.addWidget(self.circ_button)
        else:
            self.circ_angle_min_spin = None
            self.circ_angle_max_spin = None
            self.circ_min_coverage_spin = None
            self.circ_median_check = None
            self.circ_std_check = None
            self.circ_mask_draw_button = None
            self.circ_mask_clear_button = None
            self.circ_mask_status_label = None
            self.circ_button = None

        layout.addWidget(_divider("1D Plot Process"))
        self.line_angle_spin = _double_spin(-360.0, 360.0, 2, 90.0, 0.5)
        layout.addWidget(_field("Line angle", self.line_angle_spin))
        layout.addWidget(_muted("0: ←,  90: ↑,  180: →"))

        # Shared first-column width for Pos move's label and the Cakecut/
        # Accumulation checkboxes right below it, so all three rows' text
        # boxes line up at the same x position -- sized to fit
        # "Accumulation", the widest of the three (measured sizeHint ~164px).
        first_column_width = 168
        if cfg.has_pos_move_and_accumulation:
            self.y_pos_move_spin = _double_spin(-1.0e5, 1.0e5, 1, 0.0, 1.0)
            self.z_pos_move_spin = _double_spin(-1.0e5, 1.0e5, 1, 0.0, 1.0)
            pos_move_field = _field(
                "Pos move (y,z)", self.y_pos_move_spin, self.z_pos_move_spin, label_width=first_column_width
            )
            pos_move_field.setToolTip(
                "At angle 0/180, only z moves the cut (y stays 0).\n"
                "At angle 90, only y moves the cut (z stays 0).\n"
                "Every other angle ignores both (y=0, z=0).\n"
                "Has no effect on Cakecut."
            )
            layout.addWidget(pos_move_field)
            # Only one field ever applies at a time (see _update_pos_move_availability),
            # so keep them in sync with Line angle live rather than letting a
            # stale nonzero value sit in a field that's silently being ignored.
            self.line_angle_spin.valueChanged.connect(self._update_pos_move_availability)
            self._update_pos_move_availability()
        else:
            self.y_pos_move_spin = None
            self.z_pos_move_spin = None

        if cfg.has_cakecut:
            self.cakecut_check = QCheckBox("Cakecut")
            self.cakecut_check.setFixedWidth(first_column_width)
            self.cakecut_check.setToolTip("Cakecut angle range (-/+), degrees from the line angle.")
            self.cakecut_minus_spin = _double_spin(-360.0, 360.0, 1, -1.0, 0.5)
            self.cakecut_plus_spin = _double_spin(-360.0, 360.0, 1, 1.0, 0.5)
            layout.addWidget(_row(self.cakecut_check, self.cakecut_minus_spin, self.cakecut_plus_spin))
        else:
            self.cakecut_check = None
            self.cakecut_minus_spin = None
            self.cakecut_plus_spin = None

        if cfg.has_pos_move_and_accumulation:
            self.accum_check = QCheckBox("Accumulation")
            self.accum_check.setFixedWidth(first_column_width)
            self.accum_check.setToolTip("Perpendicular pixel-band mean around the cut line. Angle must be 0, 90, or 180.")
            self.accum_minus_spin = _double_spin(-360.0, 360.0, 1, -1.0, 0.5)
            self.accum_plus_spin = _double_spin(-360.0, 360.0, 1, 1.0, 0.5)
            layout.addWidget(_row(self.accum_check, self.accum_minus_spin, self.accum_plus_spin))
        else:
            self.accum_check = None
            self.accum_minus_spin = None
            self.accum_plus_spin = None

        # Cakecut (angular wedge) and Accumulation (perpendicular band) are
        # mutually exclusive cut modes -- enforce that directly in the UI
        # (uncheck the other one) rather than letting the user hit the
        # backend's "Turn off Cakecut before using Accumulation" error.
        if self.cakecut_check is not None and self.accum_check is not None:
            self.cakecut_check.toggled.connect(
                lambda checked: self.accum_check.setChecked(False) if checked else None
            )
            self.accum_check.toggled.connect(
                lambda checked: self.cakecut_check.setChecked(False) if checked else None
            )

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
        layout.addWidget(
            _field("Pixel / Range(-/+)", self.azi_pixel_spin, self.azi_range_minus_spin, self.azi_range_plus_spin, label_width=110)
        )
        self.azi_pixel_spin.valueChanged.connect(self._update_azi_q_label)

        self.azi_q_label = QLineEdit("0")
        self.azi_q_label.setReadOnly(True)
        self.azi_q_label.setProperty("readonly", "true")
        layout.addWidget(_field("Q [1/A]", self.azi_q_label))

        self.azi_position_button = QPushButton("Position")
        self.azi_meanmedian_combo = QComboBox()
        self.azi_meanmedian_combo.addItems(["Mean", "Median"])
        _size_combo(self.azi_meanmedian_combo, 10)
        self.azi_position_button.clicked.connect(lambda: self._arm_position_mode("azimuthal"))
        layout.addWidget(_row(self.azi_position_button, self.azi_meanmedian_combo))

        self.azi_draw_button = QPushButton("Draw")
        self.azi_plot_button = QPushButton("Plot")
        self.azi_draw_button.clicked.connect(self._on_azi_draw_clicked)
        self.azi_plot_button.clicked.connect(self._on_azi_plot_clicked)
        layout.addWidget(_row(self.azi_draw_button, self.azi_plot_button))

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
        splitter.setHandleWidth(2)
        splitter.setFrameShape(QFrame.NoFrame)
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
        # ColorScale ("Linear"/"Log") and Map format ("Pixel"/"Qxy/Qz"/
        # "Qy/Qz") otherwise size to their own much shorter content next to
        # Colormap (entries as long as "twilight_shifted") on the same row
        # -- give all three the same content-length budget so they line up
        # (see _size_combo's own docstring for why not setFixedWidth).
        for combo in (self.color_scale_combo, self.colormap_combo, self.map_format_combo):
            _size_combo(combo, 12)
        # Every display-only control (ColorScale/Colormap/Contrast/Range/
        # Font) applies the moment it changes -- Update no longer needs to
        # be clicked for these to take effect. Spin boxes have
        # setKeyboardTracking(False) (see _double_spin), so valueChanged
        # fires on step/scroll/Enter/focus-loss, not on every keystroke.
        self.color_scale_combo.currentTextChanged.connect(self._apply_display_settings)
        self.colormap_combo.currentTextChanged.connect(self._apply_display_settings)
        self.polar_check = QCheckBox("Polar (angle vs q)")
        self.polar_check.setToolTip(
            "View the current Qxy/Qz or Qy/Qz plot as a polar (angle vs |Q|) "
            "remap, centered at q=0. Unchecking recovers the Cartesian plot."
        )
        self.polar_check.setEnabled(False)
        self.polar_check.toggled.connect(self._on_polar_toggled)
        controller_layout.addWidget(
            _split_row(
                _field("ColorScale", self.color_scale_combo, label_width=70),
                _field("Colormap", self.colormap_combo, label_width=64),
                _field("Map format", self.map_format_combo, label_width=76),
                self.polar_check,
            )
        )

        # QDoubleSpinBox.setValue() *rounds* to `decimals`, not just
        # display-formats -- a fixed-decimal display silently collapsed
        # background-subtracted contrast values (routinely << 1, e.g.
        # 0.000123) to 0.000, and when both cmin and cmax rounded to the
        # same value the box read back cmax <= cmin, which
        # _apply_display_settings treats as "auto" and silently threw away
        # the real computed range. `_sci_spin` sidesteps this at any
        # magnitude (small or large) with a scientific-notation display and
        # a high internal precision independent of that display.
        self.cmin_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self.cmax_spin = _sci_spin(-1.0e12, 1.0e12, 0.0, 1.0)
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
        self.save_format_combo.setMaximumWidth(70)
        self.save_button = QPushButton("Save Img")
        self.save_button.clicked.connect(self._on_save_clicked)
        # Built directly (not via _row()) so the "Format" label and its
        # combo can get stretch=0 -- glued together regardless of window
        # width -- while the three buttons share the leftover space evenly.
        # _row()'s plain addWidget() with no stretch factors was instead
        # splitting all five items into equal-width cells, stranding the
        # short "Format" label in its own ~200px cell with a visible gap
        # before the combo that actually belongs next to it.
        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        bottom_layout.addWidget(self.update_button, 1)
        bottom_layout.addWidget(self.reset_zoom_button, 1)
        bottom_layout.addWidget(QLabel("Format"), 0)
        bottom_layout.addWidget(self.save_format_combo, 0)
        bottom_layout.addWidget(self.save_button, 1)
        controller_layout.addWidget(bottom_row)
        splitter.addWidget(controller_group)

        curve_group = QGroupBox("1D Result")
        curve_layout = QVBoxLayout(curve_group)
        self.curve_canvas = CurveCanvas()
        self.curve_canvas.point_clicked.connect(self._on_curve_clicked)

        self.curve_xscale_combo = QComboBox()
        self.curve_xscale_combo.addItems(["Linear", "Log"])
        self.curve_yscale_combo = QComboBox()
        self.curve_yscale_combo.addItems(["Linear", "Log"])
        self.curve_xscale_combo.currentTextChanged.connect(self._apply_curve_scales)
        self.curve_yscale_combo.currentTextChanged.connect(self._apply_curve_scales)
        self.curve_reset_zoom_button = QPushButton("Reset Zoom")
        self.curve_reset_zoom_button.setToolTip("Reset the 1D Result plot to its full data range.")
        self.curve_reset_zoom_button.clicked.connect(self._on_curve_reset_zoom_clicked)
        self.curve_save_button = QPushButton("Save")
        self.curve_save_button.setToolTip("Save the current 1D Result curve as a CSV.")
        self.curve_save_button.clicked.connect(self._on_curve_save_clicked)
        curve_controls = QWidget()
        curve_controls_layout = QHBoxLayout(curve_controls)
        curve_controls_layout.setContentsMargins(0, 0, 0, 0)
        curve_controls_layout.setSpacing(6)
        curve_controls_layout.addWidget(QLabel("X"), 0)
        curve_controls_layout.addWidget(self.curve_xscale_combo, 0)
        curve_controls_layout.addWidget(QLabel("Y"), 0)
        curve_controls_layout.addWidget(self.curve_yscale_combo, 0)
        curve_controls_layout.addStretch(1)
        curve_controls_layout.addWidget(self.curve_reset_zoom_button, 0)
        curve_controls_layout.addWidget(self.curve_save_button, 0)
        curve_layout.addWidget(curve_controls)

        curve_layout.addWidget(self.curve_canvas)
        self.curve_status_label = _muted("")
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
        self.status_label.setText(_wrap_long_tokens(message))
        self.status_label.setProperty("role", "status-ok" if ok else "status-error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _set_curve_status(self, message: str, ok: bool) -> None:
        self.curve_status_label.setText(_wrap_long_tokens(message))
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

    def _mark_qmap_stale(self, *_args) -> None:
        """Reset the q-Map Cal. "done" indicator once a q-Map input changes.
        Only meaningful once a map exists; before that the button is already
        in its neutral state."""

        if cs.get_session().qmap is not None:
            self._mark_ok(self.qmap_button, False)

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

    def _render_pixel_image(self, keep_limits: bool = False, data_override: np.ndarray | None = None) -> None:
        self._activate()
        session = cs.get_session()
        if session.data is None:
            return
        # `data_override` shows something other than the committed session
        # state -- currently just BKG sub./Norm.'s own pre-confirm preview
        # (see _on_subtract_clicked) -- without needing to actually commit
        # it to `session.data` first (and then have to un-commit it again
        # if the user discards).
        data = data_override if data_override is not None else (
            session.corrected_data if session.corrected_data is not None else session.data
        )
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

    def _render_image(
        self, data: np.ndarray, extent, xlabel: str, ylabel: str, keep_limits: bool = False, aspect: str = "equal"
    ) -> None:
        self._current_image_data = data
        self._current_image_extent = extent
        self._current_xlabel = xlabel
        self._current_ylabel = ylabel
        self._current_aspect = aspect
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
            aspect=self._current_aspect,
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

    def _pos_move_display_offset(self, band_edge_y: float = 0.0, band_edge_z: float = 0.0) -> tuple[float, float]:
        """Display-coordinate center for a line offset by Pos move, plus an
        optional extra perpendicular shift on top of it (Accumulation's
        band edges) -- pass only whichever of band_edge_y/band_edge_z
        matches the cut's axis (0/180 -> z, 90 -> y), the other stays 0.

        Pos move only ever has one of y_pos_move/z_pos_move nonzero at a
        time (the other is force-zeroed by _update_pos_move_availability),
        so both are added unconditionally rather than re-branching on angle
        here. z's sign flips between the two pixel-mode conventions: raw
        mode's display y *is* the row index (so shifting the cut's row by
        -z_pos_move shifts the displayed line by -z_pos_move too), while
        beam-centered mode's display y is VD = beam_z - row (inverted
        relative to row), so the same row shift shows up as +z_pos_move on
        screen. y needs no such flip -- raw mode's display x is the column
        index directly and centered mode's is HD = col - beam_y, and both
        equal y_pos_move + band_edge_y measured from the beam center either
        way. Matches the offsets console_line_1d's accumulation_average
        call actually cuts at. Only called in pixel display mode --
        `_draw_line_overlay` routes to `_draw_line_overlay_q_space`/
        `_draw_q_space_line` instead when Map format is Qxy/Qz or Qy/Qz,
        since these axes aren't a linear function of pixel position.

        Pos move doesn't exist for TrXS (see has_pos_move_and_accumulation
        -- TrXS_gui.m's line cut always goes through the beam center with
        no offset), so y_pos_move/z_pos_move are 0 whenever those spins
        are None."""

        beam_y, beam_z = self._current_beam_center()
        y_pos_move = self.y_pos_move_spin.value() if self.y_pos_move_spin is not None else 0.0
        z_pos_move = self.z_pos_move_spin.value() if self.z_pos_move_spin is not None else 0.0
        offset_y = y_pos_move + band_edge_y
        raw_offset_z = band_edge_z - z_pos_move
        offset_z = -raw_offset_z if self._pixel_centered else raw_offset_z
        return (beam_y + offset_y, beam_z + offset_z)

    def _pos_move_pixel_point(self, band_edge_y: float = 0.0, band_edge_z: float = 0.0) -> tuple[float, float]:
        """The same offset as `_pos_move_display_offset`, but as a true
        detector pixel (col, row) in raw 1-based coordinates -- independent
        of what the 2D image currently displays. Used to sample a cut
        line/band-edge in real pixel space before mapping each sample
        point through the per-pixel Q maps for a Q-space overlay (see
        `_draw_q_space_line`)."""

        y_pos_move = self.y_pos_move_spin.value() if self.y_pos_move_spin is not None else 0.0
        z_pos_move = self.z_pos_move_spin.value() if self.z_pos_move_spin is not None else 0.0
        col = self.beam_y_spin.value() + y_pos_move + band_edge_y
        row = self.beam_z_spin.value() - z_pos_move + band_edge_z
        return (col, row)

    def _draw_q_space_line(self, col0: float, row0: float, angle_deg: float, bidirectional: bool = True) -> None:
        """Draw a straight line through pixel (col0, row0) at angle_deg,
        directly in the same uniform Q-space coordinates the Qxy/Qz or
        Qy/Qz display and cut geometry actually use (console_q_space_point
        -- `qx = d_q * (col - beam_y)`, `qy = d_q * (beam_z - row)`, a
        single linear scale applied identically to both axes).

        This used to sample many points along the pixel-space line and
        look each one up in the true per-pixel Q maps instead, which
        curved a straight cut once mapped (reported: "Qxy/Qz 맵에서 90도
        line cut을... 해보면 일직선 라인이 아니라 휘어져 나와... 그냥
        일직선 라인이 되면 좋을 것 같아") -- console_line_1d's own
        Qxy/Qz cut no longer runs against those true per-pixel maps
        either (see console_session.py's `_q_space_cut_inputs`), so a
        straight line in this same uniform grid is what actually matches
        what gets cut, not an accurate-but-inconsistent curved path.

        `angle_deg` is mirrored via `_mode_corrected_angle` before
        drawing -- Q-space's own y-axis increases upward (see this
        method's own docstring), the same convention `_mode_corrected_angle`
        already exists to handle for beam-centered pixel display."""

        result = cs.console_q_space_point({"col": col0, "row": row0})
        if not result.get("ok"):
            return
        draw_angle = self._mode_corrected_angle(angle_deg)
        self.image_canvas.draw_beam_line((result["qx"], result["qy"]), draw_angle, bidirectional=bidirectional)

    def _draw_line_overlay_q_space(self, angle: float, cakecut_on: bool, accum_on: bool) -> None:
        if cakecut_on:
            # Cakecut always emanates from the true beam center and ignores
            # Pos move (matching console_line_1d/GIXS_gui.m).
            beam_y = self.beam_y_spin.value()
            beam_z = self.beam_z_spin.value()
            self._draw_q_space_line(beam_y, beam_z, angle + self.cakecut_minus_spin.value(), bidirectional=False)
            self._draw_q_space_line(beam_y, beam_z, angle + self.cakecut_plus_spin.value(), bidirectional=False)
            return
        if accum_on:
            if angle in (0.0, 180.0):
                col_m, row_m = self._pos_move_pixel_point(band_edge_z=self.accum_minus_spin.value())
                col_p, row_p = self._pos_move_pixel_point(band_edge_z=self.accum_plus_spin.value())
            else:  # angle == 90.0
                col_m, row_m = self._pos_move_pixel_point(band_edge_y=self.accum_minus_spin.value())
                col_p, row_p = self._pos_move_pixel_point(band_edge_y=self.accum_plus_spin.value())
            self._draw_q_space_line(col_m, row_m, angle)
            self._draw_q_space_line(col_p, row_p, angle)
            return
        col0, row0 = self._pos_move_pixel_point()
        self._draw_q_space_line(col0, row0, angle)

    def _draw_line_overlay(self) -> None:
        angle = self.line_angle_spin.value()
        cakecut_on = self.cakecut_check is not None and self.cakecut_check.isChecked()
        accum_on = self.accum_check is not None and self.accum_check.isChecked() and angle in (0.0, 90.0, 180.0)

        if self._axis_mode != "pixel":
            self._draw_line_overlay_q_space(angle, cakecut_on, accum_on)
            return

        if cakecut_on:
            # Cakecut always emanates from the true beam center and ignores
            # Pos move entirely (matching console_line_1d/GIXS_gui.m) --
            # show its actual angular wedge as two boundary lines instead
            # of a single line at the reference angle, so the wedge width
            # is actually visible.
            center = self._current_beam_center()
            self.image_canvas.draw_beam_line(
                center, self._mode_corrected_angle(angle + self.cakecut_minus_spin.value()), bidirectional=False
            )
            self.image_canvas.draw_beam_line(
                center, self._mode_corrected_angle(angle + self.cakecut_plus_spin.value()), bidirectional=False
            )
            return

        if accum_on:
            # Accumulation is a perpendicular pixel band, not a single
            # line -- draw its two edges (Pos move's offset plus the
            # band's own minus/plus range) instead of one center line, so
            # the band width is visible, not just its centerline.
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

    def _current_beam_center(self) -> tuple[float, float]:
        if self._axis_mode == "pixel" and not self._pixel_centered:
            return (self.beam_y_spin.value(), self.beam_z_spin.value())
        # Q-space is beam-centered by construction, and once the pixel axis
        # has been re-centered by Load Info (see _render_pixel_image), the
        # beam sits at (0, 0) on that axis too.
        return (0.0, 0.0)

    def _mode_corrected_angle(self, angle_deg: float) -> float:
        """Beam-centered pixel display flips the vertical axis convention
        relative to raw pixel mode: raw mode's display-y *is* the row
        index (increases going down the screen, matching the underlying
        array), while centered mode's display-y is VD = beam_z - row
        (increases going *up* the screen). The "0 left / 90 up / 180
        right" legend and the click-to-angle formula
        (_on_image_clicked's "line" branch) both assume raw mode's
        convention -- in centered/Q-space mode they need the angle
        mirrored across the horizontal axis to still mean the same
        physical direction on screen. This mirror is its own inverse, so
        the same call converts a physical angle into what draw_beam_line
        needs to draw it correctly, and converts a raw click-formula angle
        back into the physical one.

        Q-space mode's own display-y, `console_q_space_point`'s `qy = d_q
        * (beam_z - row)`, increases upward exactly like centered pixel
        mode's VD -- so it needs the same mirror, applied inside
        `_draw_q_space_line` for the draw direction (every Q-space overlay
        call site routes through it) rather than at each call site
        individually."""

        if (self._axis_mode == "pixel" and self._pixel_centered) or self._axis_mode == "q":
            return (-angle_deg) % 360.0
        return angle_deg % 360.0

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
        self.azi_q_label.setText("0")
        if self.circ_mask_status_label is not None:
            self.circ_mask_status_label.setText("Mask: none")
        self.map_format_combo.blockSignals(True)
        self.map_format_combo.setCurrentText("Pixel")
        self.map_format_combo.blockSignals(False)
        self.polar_check.blockSignals(True)
        self.polar_check.setChecked(False)
        self.polar_check.setEnabled(False)
        self.polar_check.blockSignals(False)
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
        # console_grid_remove dropped the now-stale corrected_data, so flag
        # q-Map Cal. for a re-run to get the correction reapplied on top of
        # the grid-removed image (same handling as BKG sub. & Norm.).
        self._mark_ok(self.qmap_button, False)
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
        self._render_curve(row_pixels[valid], column[valid], xlabel="pixel (beam-centered, z)")
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
        # Show the actual subtracted/normalized result on the 2D Image
        # Plot Viewer *before* asking -- the confirmation dialog's text
        # alone gave no way to judge the result (reported: "메시지만
        # 나오고 subtracted 또는 normalized image가 어떤지 나오지
        # 않아"). Not yet committed to session state, so a "No" below just
        # re-renders the actual (unchanged) data -- no un-commit needed.
        preview_array = cs.get_session().pending_normalize_preview
        if preview_array is not None:
            # Same reasoning as the commit branch below -- Contrast holding
            # onto the previous data's min/max would make this preview
            # look blank or blown out, defeating the point of showing it.
            self.cmin_spin.blockSignals(True)
            self.cmax_spin.blockSignals(True)
            self.cmin_spin.setValue(0.0)
            self.cmax_spin.setValue(0.0)
            self.cmin_spin.blockSignals(False)
            self.cmax_spin.blockSignals(False)
            self._render_pixel_image(data_override=preview_array)
        keep = QMessageBox.question(
            self,
            "Confirmation",
            "Preview computed. Keep this background-subtracted/normalized result?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if keep != QMessageBox.Yes:
            self._set_status("Normalization discarded.", True)
            self._render_pixel_image()
            return
        self._activate()
        committed = cs.console_normalize({**payload, "confirm": True})
        if not committed.get("ok"):
            self._set_status(committed.get("message", "Normalization failed."), False)
            return
        self.map_format_combo.blockSignals(True)
        self.map_format_combo.setCurrentText("Pixel")
        self.map_format_combo.blockSignals(False)
        # console_normalize just dropped the now-stale corrected_data (it
        # was built from the pre-normalization image) -- flag q-Map Cal.
        # as needing a re-run so its correction gets reapplied to the new
        # data, matching the "not done yet" indicator a fresh Load Data
        # already resets it to.
        self._mark_ok(self.qmap_button, False)
        # Background subtraction routinely changes the data's whole
        # intensity scale (raw counts vs. a subtracted/normalized
        # difference) -- Contrast holding onto whatever min/max applied to
        # the *previous* data would leave the image looking blank or
        # blown out. 0/0 is this app's "auto-contrast" sentinel (see
        # _render_image), so resetting to it here re-derives min/max from
        # the new data, same as a fresh Load Data does.
        self.cmin_spin.blockSignals(True)
        self.cmax_spin.blockSignals(True)
        self.cmin_spin.setValue(0.0)
        self.cmax_spin.setValue(0.0)
        self.cmin_spin.blockSignals(False)
        self.cmax_spin.blockSignals(False)
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
            "q_uncertainty": self.q_uncertain_check.isChecked() if self.q_uncertain_check is not None else False,
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
        # Signals are blocked above, so _on_map_format_changed (which would
        # otherwise handle this) never runs -- enable Polar here instead,
        # same condition it uses: a single unambiguous Q mapping (TrXS).
        self.polar_check.setEnabled(self._only_q_mapping() is not None)
        self._render_pixel_image()
        self._update_azi_q_label()
        self._set_status("q-Map Cal. complete.", True)

    # ------------------------------------------------------------------
    # Map format (Pixel / Qxy/Qz / Qy/Qz)
    # ------------------------------------------------------------------
    def _on_map_format_changed(self, mode: str) -> None:
        self._activate()
        session = cs.get_session()
        # Any format change lands on the Cartesian view -- Polar always
        # starts back off (its own transform is keyed to whichever mapping
        # was active when it was turned on, and Qxy/Qz vs Qy/Qz have
        # different horizontal axes, so a stale polar view would be wrong).
        self.polar_check.blockSignals(True)
        self.polar_check.setChecked(False)
        self.polar_check.blockSignals(False)
        if mode == "Pixel":
            # TrXS only ever has one Q mapping (Qy/Qz), so Polar can jump
            # straight there from Pixel with no ambiguity -- keep it
            # enabled rather than forcing an intermediate Qy/Qz click.
            # GIXS has two (Qxy/Qz, Qy/Qz), so Map format must be picked
            # explicitly first.
            self.polar_check.setEnabled(session.qmap is not None and self._only_q_mapping() is not None)
            if session.data is None:
                return
            self._render_pixel_image()
            self._set_status("Pixel view restored.", True)
            return
        if session.qmap is None:
            self.polar_check.setEnabled(False)
            self._set_status("Run q-Map Cal. first.", False)
            self.map_format_combo.blockSignals(True)
            self.map_format_combo.setCurrentText("Pixel")
            self.map_format_combo.blockSignals(False)
            return
        result = cs.console_qspace_image({"mapping": mode})
        if not result.get("ok"):
            self.polar_check.setEnabled(False)
            self._set_status(result.get("message", f"{mode} transform failed."), False)
            self.map_format_combo.blockSignals(True)
            self.map_format_combo.setCurrentText("Pixel")
            self.map_format_combo.blockSignals(False)
            return
        self.polar_check.setEnabled(True)
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
        # Plain text, not "q_{xy}"/"q_z" -- matplotlib's set_xlabel/
        # set_ylabel only render underscore-brace as a LaTeX subscript
        # inside mathtext ($...$), which this app doesn't use for any of
        # its other axis labels (matches "Q, 1/A" elsewhere).
        label = "Qxy, 1/A" if mode == "Qxy/Qz" else "Qy, 1/A"
        self._render_image(preview, extent, label, "Qz, 1/A")
        self.image_canvas.reset_view()
        self._set_status(f"{mode} image ready.", True)

    def _only_q_mapping(self) -> str | None:
        """The single non-Pixel Map format, when there's exactly one (TrXS:
        just Qy/Qz) -- lets Polar be toggled straight from Pixel view with
        no intermediate Cartesian step, since there's no ambiguity about
        which mapping to use. None when there's more than one (GIXS has
        both Qxy/Qz and Qy/Qz), where Map format must be picked first."""

        q_formats = [fmt for fmt in self.config.map_formats if fmt != "Pixel"]
        return q_formats[0] if len(q_formats) == 1 else None

    def _on_polar_toggled(self, checked: bool) -> None:
        self._activate()
        combo_mode = self.map_format_combo.currentText()
        mapping = combo_mode if combo_mode != "Pixel" else self._only_q_mapping()
        if mapping is None:
            if checked:
                self.polar_check.blockSignals(True)
                self.polar_check.setChecked(False)
                self.polar_check.blockSignals(False)
            return
        if not checked:
            # "Recover" whatever was showing before Polar -- literally just
            # re-running the same fetch the combo box itself would trigger
            # (Pixel view if Polar was toggled straight from Pixel; the
            # Cartesian view otherwise), since Polar never mutates session
            # state.
            self._on_map_format_changed(combo_mode)
            return
        self._axis_mode = "q"
        result = cs.console_polar_image({"mapping": mapping})
        if not result.get("ok"):
            self._set_status(result.get("message", "Polar transform failed."), False)
            self.polar_check.blockSignals(True)
            self.polar_check.setChecked(False)
            self.polar_check.blockSignals(False)
            return
        # Row 0 of the polar image is the smallest |Q| bin (see
        # polar_transform_q_grid's ascending r_axis); flip so the largest
        # |Q| ends up at row 0, matching every other view in this app where
        # the largest vertical-axis value renders at the top (origin="upper").
        preview = np.asarray(result["preview"], dtype=np.float64)[::-1, :]
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
        # aspect="auto" (not the "equal" every other view uses): angle
        # (degrees, spanning 0-360) and |Q| (1/A, spanning maybe 0-0.1)
        # have completely different natural scales, so forcing equal data
        # units per pixel would crush the plot into a sliver a few pixels
        # tall -- this is the actual bug the polar view shipped with.
        self._render_image(preview, extent, "Angle, deg", "q, 1/A", aspect="auto")
        self.image_canvas.reset_view()
        self._set_status(f"Polar ({mapping}) view ready.", True)

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
            aspect=self._current_aspect,
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
        #
        # Also clear any Position/Draw/Azimuthal-ring overlay -- a line or
        # circle drawn earlier otherwise stays on screen indefinitely
        # (nothing else ever clears it), and Update is the control the user
        # reaches for to get back to a plain image (reported: "라인을
        # 그렸다가, Update누르면 라인이 없는 퓨어이미지가 되게 해줘"). Must
        # also disarm Position mode, not just wipe the artists -- every live
        # display control (Y-Range/Z-Range/Colormap/...) re-renders through
        # _render_image, which redraws the line whenever _position_mode is
        # still "line" (see its own tail), so leaving it armed brought the
        # "cleared" line right back on the next Y/Z-Range edit (reported:
        # "update누르면 선이 없어지는데, Y-range, Z-range를 바꾸면 다시
        # 나오네"). Matches _on_load_finished's own reset-to-clean-state
        # pattern (clear_overlays + _position_mode = None together).
        self.image_canvas.clear_overlays()
        self._position_mode = None
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
    # 1D Result: X/Y scale (live) + Save
    # ------------------------------------------------------------------
    def _render_curve(
        self, x: np.ndarray, y: np.ndarray, *, xlabel: str = "",
        yerr: np.ndarray | None = None, xerr: np.ndarray | None = None,
    ) -> None:
        """Every 1D Result render (z-plot, Line, Azimuthal, Circular) goes
        through here so the ColorScale-style X/Y log toggles apply
        uniformly, not just to whichever curve happened to be plotted
        last."""

        self.curve_canvas.render(
            x,
            y,
            xlabel=xlabel,
            xlog=self.curve_xscale_combo.currentText() == "Log",
            ylog=self.curve_yscale_combo.currentText() == "Log",
            yerr=yerr,
            xerr=xerr,
        )

    def _apply_curve_scales(self) -> None:
        self.curve_canvas.set_scales(
            self.curve_xscale_combo.currentText() == "Log",
            self.curve_yscale_combo.currentText() == "Log",
        )

    def _on_curve_reset_zoom_clicked(self) -> None:
        self.curve_canvas.reset_view()

    def _on_curve_save_clicked(self) -> None:
        self._activate()
        result = cs.console_save_curve({})
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Save failed."), False)
            return
        curve_path = Path(result["curve_path"])
        image_path = curve_path.with_name(f"{curve_path.stem}.png")
        try:
            self.curve_canvas.save_figure(str(image_path))
        except Exception as exc:  # noqa: BLE001
            self._set_curve_status(f"Saved {curve_path}, but image export failed: {exc}", False)
            return
        self._set_curve_status(f"Saved {curve_path} and {image_path}.", True)

    # ------------------------------------------------------------------
    # Click-to-position (Line / Azimuthal)
    # ------------------------------------------------------------------
    def _arm_position_mode(self, mode: str) -> None:
        self._position_mode = mode
        if mode == "line":
            self._set_status("Click a point on the 2D image to set the line angle.", True)
        elif mode == "azimuthal":
            self._set_status("Click a point on the 2D image to set the azimuthal radius.", True)

    def _display_xy_to_pixel_offset(self, x: float, y: float) -> tuple[float, float]:
        """Convert a click's (x, y) in the *currently displayed* coordinate
        system into a true pixel offset from the beam center.

        Pixel mode's display already is pixel offsets (identity, matching
        `_current_beam_center()`'s subtraction in the caller). Q-space mode
        used to look up the actual detector pixel whose true (nonlinear)
        per-pixel Q was nearest the click -- exact for *that* map, but
        inconsistent with `_draw_q_space_line`'s straight-line overlay,
        which (like `console_line_1d`'s own Qxy/Qz cut) now treats the
        Q-space display as the uniform grid `image_transform_q_gi` actually
        built: `console_pixel_from_q_space` inverts that same linear
        mapping exactly instead, so a click and the line it sets up stay
        consistent with each other."""

        if self._axis_mode != "q":
            return (x, y)
        result = cs.console_pixel_from_q_space({"qx": x, "qy": y})
        if not result.get("ok"):
            return (x, y)
        pixel_dx = result["col"] - self.beam_y_spin.value()
        pixel_dz = result["row"] - self.beam_z_spin.value()
        return (pixel_dx, pixel_dz)

    def _on_image_clicked(self, x: float, y: float) -> None:
        if self._position_mode is None:
            return
        if self.image_canvas.toolbar.mode:
            return
        beam_y, beam_z = self._current_beam_center()
        if self._position_mode == "line":
            px, pz = self._display_xy_to_pixel_offset(x, y)
            raw_angle = (math.degrees(math.atan2(-(pz - beam_z), -(px - beam_y))) + 360.0) % 360.0
            # `_display_xy_to_pixel_offset` already resolves a Q-space
            # click into a genuine (unflipped) pixel offset -- the same
            # "no further correction needed" case as raw/uncentered pixel
            # mode, where raw_angle already IS the physical angle.
            # Applying `_mode_corrected_angle` on top of that (which
            # centered-pixel mode still needs, since its own identity
            # passthrough leaves the display's y-flip uncorrected) mirrored
            # the angle a *second* time, landing the drawn line 180deg away
            # from the clicked ring/feature (reported: "Qxy/Qz map에서
            # 똑같은 위치에 클릭하여 line을 그리면... 다른곳에 라인이
            # 그려져").
            angle = raw_angle if self._axis_mode == "q" else self._mode_corrected_angle(raw_angle)
            self.line_angle_spin.setValue(angle)
            self._set_status(f"Line position clicked at ({x:.1f}, {y:.1f}) -> angle {angle:.2f} deg.", True)
        elif self._position_mode == "azimuthal":
            # Same conversion the "line" branch above already applies --
            # in Q-space (Qxy/Qz) display mode, (x, y) are Q values, not
            # pixel coordinates, so hypot-ing them directly against the
            # pixel beam center put a Q-magnitude straight into a
            # pixel-radius field (reported: "Qxy/Qz 모드일때... 클릭하면
            # Q값이 아닌 pixel값에 Q값이 들어가서 이상한 곳에 포지션이 생겨").
            px, pz = self._display_xy_to_pixel_offset(x, y)
            radius = math.hypot(px - beam_y, pz - beam_z)
            self.azi_pixel_spin.setValue(radius)
            self._set_status(f"Azimuthal position clicked at ({x:.1f}, {y:.1f}) -> radius {radius:.1f} px.", True)

    # ------------------------------------------------------------------
    # 1D Line Process
    # ------------------------------------------------------------------
    def _update_pos_move_availability(self) -> None:
        """Pos move only ever applies to one axis at a time, and only for
        an axis-aligned cut: 0/180 moves along z (y stays 0), 90 moves
        along y (z stays 0), every other angle ignores both. Disable and
        zero whichever field doesn't apply so a leftover nonzero value
        can't sit in a field that's silently having no effect. No-op for
        TrXS, which has no Pos move fields at all (see
        has_pos_move_and_accumulation)."""

        if self.y_pos_move_spin is None or self.z_pos_move_spin is None:
            return

        angle = self.line_angle_spin.value()
        if angle in (0.0, 180.0):
            y_enabled, z_enabled = False, True
        elif angle == 90.0:
            y_enabled, z_enabled = True, False
        else:
            y_enabled, z_enabled = False, False

        self.y_pos_move_spin.setEnabled(y_enabled)
        if not y_enabled:
            self.y_pos_move_spin.setValue(0.0)
        self.z_pos_move_spin.setEnabled(z_enabled)
        if not z_enabled:
            self.z_pos_move_spin.setValue(0.0)

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
            "y_pos_move": self.y_pos_move_spin.value() if self.y_pos_move_spin is not None else 0.0,
            "z_pos_move": self.z_pos_move_spin.value() if self.z_pos_move_spin is not None else 0.0,
            "accumulation": self.accum_check.isChecked() if self.accum_check is not None else False,
            "accum_minus": self.accum_minus_spin.value() if self.accum_minus_spin is not None else -1.0,
            "accum_plus": self.accum_plus_spin.value() if self.accum_plus_spin is not None else 1.0,
            # Reported: "1D process해서 pixel map에서 cutting하는 것이
            # Qxy/Qz cutting에 반영되서 진행되는데, 그렇게 하지말고
            # 독립적으로 되면 좋겠어" -- Map format now genuinely selects
            # which image (raw pixel data, or the Qxy/Qz-remapped one)
            # the line/accumulation/cake cut actually runs against,
            # instead of the Map format combo only ever affecting the 2D
            # display.
            "mapping": self.map_format_combo.currentText(),
        }
        result = cs.console_line_1d(payload)
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "1D line plot failed."), False)
            return
        # `preview` is downsampled and JSON-sanitized (masked points become
        # None) for the web UI's HTTP transport -- neither limit applies
        # in-process, and asarray()-ing None values silently produces an
        # object-dtype array that crashes float() the moment the hover
        # crosshair lands on a masked point. `last_curve` is the real,
        # full-resolution float array (masked points are NaN, not None).
        curve = cs.get_session().last_curve
        q_std = curve.get("q_std")
        xerr = np.asarray(q_std) if q_std is not None else None
        self._render_curve(np.asarray(curve["q"]), np.asarray(curve["intensity"]), xlabel="q [1/A]", xerr=xerr)
        diag = result.get("coverage_diagnostics", {})
        status = (
            f"Line 1D ready. Coverage median {diag.get('coverage_median', 'n/a')}, "
            f"low bins {diag.get('low_bin_count', 0)}, drops {diag.get('drop_bin_count', 0)}."
        )
        warning = result.get("warning")
        if warning:
            status += f" Warning: {warning}"
        self._set_curve_status(status, True)

    # ------------------------------------------------------------------
    # Azimuthal Process
    # ------------------------------------------------------------------
    def _on_azi_draw_clicked(self) -> None:
        self.image_canvas.clear_overlays()
        center = self._current_beam_center()
        pixel = self.azi_pixel_spin.value()
        # Draw both ring boundaries (pixel + range_minus/plus), not just a
        # single circle at `pixel` -- a lone center-radius circle never
        # visually changes when Range(-/+) is edited, which is exactly why
        # it looked like Range had no effect even though console_azimuthal_1d
        # was already using it correctly (radius_range = pixel +/- range).
        r_minus = max(0.0, pixel + self.azi_range_minus_spin.value())
        r_plus = max(0.0, pixel + self.azi_range_plus_spin.value())
        if self._axis_mode == "q":
            # azi_pixel_spin/Range(-/+) are always real pixel counts, but in
            # Qxy/Qz display mode the image's own axes are scaled in Q
            # units -- a circle radius of e.g. 200 (pixels) drawn directly
            # on a plot whose visible span is a few 1/A landed far outside
            # the view, reading as "nothing drawn" (reported: "Qxy/Qz
            # map에서 Azimuthal process에 Draw가 되고 있지 않아"). Convert
            # each pixel radius to its equivalent Q-space radius first --
            # exact, since the uniform Q-space grid scales isotropically
            # (console_q_space_point's own d_q factor applies the same way
            # in every direction from the beam).
            r_minus = self._pixel_radius_to_q_radius(r_minus)
            r_plus = self._pixel_radius_to_q_radius(r_plus)
        self.image_canvas.draw_beam_circle(center, r_minus)
        self.image_canvas.draw_beam_circle(center, r_plus)

    def _pixel_radius_to_q_radius(self, radius_px: float) -> float:
        """The Q-space distance from the beam origin that a circle of
        `radius_px` real pixels maps to -- exact, since
        `console_q_space_point`'s uniform grid scales isotropically (any
        direction from the beam by the same pixel distance lands the same
        Q distance from the origin)."""

        result = cs.console_q_space_point({"col": self.beam_y_spin.value() + radius_px, "row": self.beam_z_spin.value()})
        if not result.get("ok"):
            return radius_px
        return math.hypot(result["qx"], result["qy"])

    def _update_azi_q_label(self) -> None:
        """Show the Q value at the current Azimuthal pixel radius, sampled
        along the beam-center row (matches MATLAB GIXS_gui.m's
        `CalcPos_Callback`: `param.Q(round(bpos(2)), round(pix_pos +
        bpos(1)))`) -- a representative Q for the ring, not a per-angle
        value, since the ring's whole point is to average over angle."""

        session = cs.get_session()
        q_map = session.qmap.get("q") if session.qmap is not None else None
        if q_map is None or session.data is None:
            self.azi_q_label.setText("0")
            return
        beam_y, beam_z = session.calibration["beam_pos"]
        n_rows, n_cols = q_map.shape
        # -1 for the 1-based pixel convention used throughout this codebase
        # (pixel_angle_map, build_radius_bin_map) vs. 0-based numpy indices.
        row_idx = int(np.clip(round(beam_z) - 1, 0, n_rows - 1))
        col_idx = int(np.clip(round(beam_y + self.azi_pixel_spin.value()) - 1, 0, n_cols - 1))
        q_value = float(q_map[row_idx, col_idx])
        self.azi_q_label.setText(f"{q_value:.4f}" if np.isfinite(q_value) else "n/a")

    def _on_azi_plot_clicked(self) -> None:
        self._activate()
        payload = {
            "pixel": self.azi_pixel_spin.value(),
            "range_minus": self.azi_range_minus_spin.value(),
            "range_plus": self.azi_range_plus_spin.value(),
            "mean_or_median": self.azi_meanmedian_combo.currentText(),
            "mapping": self.map_format_combo.currentText(),
        }
        result = cs.console_azimuthal_1d(payload)
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Azimuthal plot failed."), False)
            return
        curve = cs.get_session().last_curve
        self._render_curve(np.asarray(curve["angle"]), np.asarray(curve["intensity"]), xlabel="angle [deg]")
        diag = result.get("coverage_diagnostics", {})
        self._set_curve_status(
            f"Azimuthal profile ready. Coverage median {diag.get('coverage_median', 'n/a')}, "
            f"low bins {diag.get('low_bin_count', 0)}, drops {diag.get('drop_bin_count', 0)}.",
            True,
        )

    # ------------------------------------------------------------------
    # Circular Process (TrXS)
    # ------------------------------------------------------------------
    def _update_circ_sector_overlay(self) -> None:
        """Redraw the Ang. Avg sector's boundary rays live, in step with
        circ_angle_min_spin/circ_angle_max_spin -- no separate Draw button,
        since there's nothing to confirm first (mirrors how Contrast/
        Colormap/etc. already apply live elsewhere in this app)."""

        self._activate()
        if self.circ_angle_min_spin is None:
            return
        self.image_canvas.clear_overlays()
        angle_min = self.circ_angle_min_spin.value()
        angle_max = self.circ_angle_max_spin.value()
        # A 0-360deg (or any full-turn) sweep has no meaningful boundary to
        # show -- both rays would land on the same angle, which reads as a
        # stray line rather than "this is the whole detector".
        if (angle_max - angle_min) % 360.0 == 0.0:
            return
        if self._axis_mode != "pixel":
            beam_y = self.beam_y_spin.value()
            beam_z = self.beam_z_spin.value()
            self._draw_q_space_line(beam_y, beam_z, angle_min, bidirectional=False)
            self._draw_q_space_line(beam_y, beam_z, angle_max, bidirectional=False)
        else:
            center = self._current_beam_center()
            self.image_canvas.draw_beam_line(center, self._mode_corrected_angle(angle_min), bidirectional=False)
            self.image_canvas.draw_beam_line(center, self._mode_corrected_angle(angle_max), bidirectional=False)

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
        curve = cs.get_session().last_curve
        std = curve.get("std")
        yerr = np.asarray(std) if self.circ_std_check.isChecked() and std is not None else None
        self._render_curve(np.asarray(curve["q"]), np.asarray(curve["intensity"]), xlabel="q [1/A]", yerr=yerr)
        diag = result.get("coverage_diagnostics", {})
        mask_note = " (mask applied)" if result.get("mask_active") else ""
        self._set_curve_status(
            f"Circular average ready{mask_note}. Coverage median {diag.get('coverage_median', 'n/a')}, "
            f"low bins {diag.get('low_bin_count', 0)}, drops {diag.get('drop_bin_count', 0)}.",
            True,
        )

    def _on_circ_mask_draw_clicked(self) -> None:
        """Port of MATLAB's `get_modern_mask`: draw the mask polygon in its
        own large window instead of the small "2D Image Plot" quadrant --
        that quadrant is too small to click out a precise polygon in a
        4-way-split window."""

        self._activate()
        if self._axis_mode != "pixel":
            self._set_status("Switch Map format to Pixel before drawing a mask.", False)
            return
        session = cs.get_session()
        if session.data is None or self._current_image_data is None:
            self._set_status("Load a detector image first.", False)
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

        canvas.render(
            self._current_image_data,
            cmap=self.colormap_combo.currentText(),
            log_scale=self.color_scale_combo.currentText() == "Log",
            vmin=self.cmin_spin.value() if self.cmax_spin.value() > self.cmin_spin.value() else None,
            vmax=self.cmax_spin.value() if self.cmax_spin.value() > self.cmin_spin.value() else None,
            extent=self._current_image_extent,
            xlabel=self._current_xlabel,
            ylabel=self._current_ylabel,
            aspect=self._current_aspect,
        )

        def _on_complete(vertices: list[tuple[float, float]]) -> None:
            self._on_circ_mask_polygon_complete(vertices)
            dialog.close()

        canvas.start_polygon_selector(_on_complete)

        def _on_finished(_result: int) -> None:
            # Same shiboken-crash guard as the GIXS2/TrXS2 Play/Heatmap/
            # ROI/Mask dialogs: matplotlib's PolygonSelector queues its own
            # draw_idle() on every click (including the one that completes
            # the polygon), so stop it and defer the canvas's actual
            # deletion by one event-loop tick to let any already-queued
            # redraw run first.
            canvas.stop_polygon_selector()
            if self._mask_dialog is dialog:
                self._mask_dialog = None
            QTimer.singleShot(0, dialog.deleteLater)

        dialog.finished.connect(_on_finished)
        self._mask_dialog = dialog
        dialog.show()

    def _on_circ_mask_polygon_complete(self, vertices: list[tuple[float, float]]) -> None:
        """The drawn polygon marks the region to *exclude* -- everything
        outside it survives into Circular Average (console_set_circular_mask's
        `mask` is True = keep, so the polygon's inside is inverted below)."""

        self._activate()
        session = cs.get_session()
        if session.data is None or len(vertices) < 3:
            self._set_status("Mask needs at least 3 points.", False)
            return
        rows, cols = session.data.shape
        if self._pixel_centered:
            beam_y = self.beam_y_spin.value()
            beam_z = self.beam_z_spin.value()
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
        # The drawn polygon marks the region to EXCLUDE -- everything
        # outside it is what survives into Circular Average.
        keep = ~inside

        result = cs.console_set_circular_mask({"mask": keep})
        if not result.get("ok"):
            self._set_status(result.get("message", "Mask failed."), False)
            return
        kept = result["pixel_count"]
        total = result["total_pixels"]
        self.circ_mask_status_label.setText(f"Mask: {kept}/{total} px kept")
        self._set_status(f"Mask set ({kept}/{total} px kept).", True)

    def _on_circ_mask_clear_clicked(self) -> None:
        self._activate()
        if self._mask_dialog is not None:
            self._mask_dialog.close()
        cs.console_set_circular_mask({"clear": True})
        self.circ_mask_status_label.setText("Mask: none")
        self._set_status("Mask cleared.", True)
