"""QSS stylesheet for the desktop console.

Evolves the light, MATLAB-panel-shaped palette validated against the real
GUI earlier in this project (white panels, navy primary actions, pale-cyan
editable/calibration fields) into a deliberately instrument-flavored look:
two accents held to strict, separate jobs (navy ACCENT for actions you can
press; teal RING for structure you read -- section dividers, the ring
signature mark), and a monospace face reserved for every numeric readout
(Energy, SDD, pixel/degree, q values) so they read as instrument-grade
figures rather than generic form fields.
"""

from __future__ import annotations

WINDOW_BG = "#f4f5f8"
PANEL_BG = "#ffffff"
PANEL_ALT_BG = "#fafbfc"
BORDER = "#dadde3"
BORDER_STRONG = "#c3c8d1"
TEXT = "#171b26"
MUTED = "#5b6472"
ACCENT = "#1d5c9e"
ACCENT_HOVER = "#184c84"
ACCENT_PRESSED = "#123c69"
# Signature accent: reserved for structural elements you read, never for
# controls you press (that's ACCENT's job) -- section dividers and the
# concentric-ring mark (_ring_icon in console_widget.py) that stands in for
# the scattering rings this whole app exists to measure.
RING = "#0e7c86"
EDITABLE_BG = "#e8fbfb"
EDITABLE_BORDER = "#a9e4e4"
READONLY_BG = "#eef0f3"
SUCCESS_BG = "#e0f5e4"
SUCCESS_BORDER = "#8ed19f"
WARN = "#b3441f"
# Cascadia Mono ships with Windows 11/Windows Terminal and was designed with
# tabular figures for exactly this job -- aligned columns of numbers.
# Consolas is the pre-11 Windows fallback; generic monospace covers the rest.
MONO = '"Cascadia Mono", "Consolas", "JetBrains Mono", monospace'

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    color: {TEXT};
}}

QMainWindow, QWidget#rootBackground {{
    background: {WINDOW_BG};
}}

QWidget {{
    background: transparent;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    top: -1px;
    background: {WINDOW_BG};
}}

QTabBar::tab {{
    background: {PANEL_ALT_BG};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 18px;
    margin-right: 2px;
    color: {MUTED};
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {TEXT};
}}

QTabBar::tab:hover {{
    color: {TEXT};
}}

QGroupBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {TEXT};
    background: {PANEL_BG};
}}

QGroupBox#plotGroup {{
    background: {PANEL_ALT_BG};
}}

QLabel {{
    background: transparent;
}}

QLabel[role="divider"] {{
    color: {RING};
    font-weight: 700;
    font-size: 11px;
    padding: 0;
}}

QLabel[role="muted"] {{
    color: {MUTED};
    font-size: 11px;
}}

QLabel[role="status-ok"] {{
    color: #1f7a3d;
    font-size: 11px;
}}

QLabel[role="status-error"] {{
    color: {WARN};
    font-size: 11px;
}}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 20px;
    selection-background-color: {ACCENT};
}}

QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}

QLineEdit[readonly="true"] {{
    background: {READONLY_BG};
    color: {MUTED};
    font-family: {MONO};
}}

QLineEdit[cyan="true"], QDoubleSpinBox[cyan="true"] {{
    background: {EDITABLE_BG};
    border: 1px solid {EDITABLE_BORDER};
}}

/* Every field that holds a measured or computed number -- geometry spin
   boxes, IC/BC counters -- gets the monospace utility face instead of the
   UI sans, so figures line up and read as instrument output rather than
   ordinary form text. */
QDoubleSpinBox, QSpinBox {{
    font-family: {MONO};
}}

QLabel[role="numeric"] {{
    font-family: {MONO};
    font-weight: 600;
    color: {TEXT};
}}

/* Spin box up/down buttons were dropped entirely (ConsoleWidget sets
   QAbstractSpinBox.NoButtons) -- they rendered illegibly small at this
   field height even after fixing hit-testing and drawing an explicit
   arrow glyph. Typing a value or scrolling over the field still works
   without them, so no subcontrol styling is needed here anymore. */

QComboBox::drop-down {{
    border: none;
    width: 18px;
}}

/* The popup list is a separate top-level widget (QAbstractItemView), not a
   QComboBox subcontrol -- it doesn't inherit the combo box's own QSS rules
   and otherwise falls back to a dark default, so it needs its own explicit
   colors to match the rest of the light theme. */
QComboBox QAbstractItemView {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_STRONG};
    color: {TEXT};
    outline: none;
    selection-background-color: {ACCENT};
    selection-color: white;
}}

QCheckBox {{
    spacing: 6px;
}}

QPushButton {{
    background: {PANEL_ALT_BG};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background: #eef1f5;
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background: #e2e6ec;
}}

QPushButton:disabled {{
    color: #a7acb5;
    background: {PANEL_ALT_BG};
    border-color: {BORDER};
}}

QPushButton[variant="primary"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-weight: 700;
}}

QPushButton[variant="primary"]:hover {{
    background: {ACCENT_HOVER};
}}

QPushButton[variant="primary"]:pressed {{
    background: {ACCENT_PRESSED};
}}

QPushButton[state="ok"] {{
    background: {SUCCESS_BG};
    border: 1px solid {SUCCESS_BORDER};
    color: #1f5c30;
}}

QSplitter::handle {{
    background: {BORDER};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QProgressDialog, QMessageBox {{
    background: {PANEL_BG};
}}

QToolBar {{
    background: {PANEL_ALT_BG};
    border: none;
    spacing: 2px;
}}

QStatusBar {{
    background: {WINDOW_BG};
    color: {MUTED};
}}
"""
