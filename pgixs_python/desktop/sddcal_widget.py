"""SDD Calibration tab: port of MATLAB `SDDcal_gui.m` (+ `OneDplotpixel_gui.m`/
`OneDplotpixelR_gui.m`, inlined here as one panel instead of separate popups
-- see the module docstring in `pgixs_python.ui_backend.sddcal_session`).

Load a calibration-ring image (e.g. Silver behenate), mark/refine the beam
center, take a radial or line-cut intensity profile, Gaussian-fit a ring
peak, compute the sample-to-detector distance (SDD), and save a calibration
file GIXS/TrXS's own "Load Info" already reads. "SMART AUTO RUN" automates
the whole center-refinement + ring-search + SDD flow (see
`sddcal_session.sddcal_auto_run`); everything else here is the
manual/semi-manual path it's built on top of.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pgixs_python.desktop.console_widget import _COLORMAPS, _divider, _double_spin, _field, _muted, _row, _wrap_long_tokens
from pgixs_python.desktop.plot_canvas import CurveCanvas, ImageCanvas
from pgixs_python.ui_backend import sddcal_session as ss

_DETECTOR_LIST = ["Eiger4M", "MarCCD", "MX170HS", "PI-CCD", "MAR345"]
_STANDARD_LIST = ["SEBS", "Silver behenate", "Mesoporous Silica", "Sucrose"]
_ORDER_LIST = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]


class SDDcalWidget(QWidget):
    """Single-instance tab -- unlike GIXS/TrXS's `ConsoleWidget`, there is
    only ever one SDD Calibration tab, so this talks to
    `sddcal_session`'s module-level session directly with no per-instance
    session-swap dance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_dir = str(Path.home())
        self._center_mode = False
        self._range_state: dict[str, Any] | None = None
        self._last_range_bounds: dict[str, Any] | None = None
        self._current_image_data: np.ndarray | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])

        self._update_pixel_dependent_buttons(loaded=False)

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        group = QGroupBox("SDD Calibration")
        layout = QVBoxLayout(group)
        layout.setSpacing(7)

        self.dir_edit = QLineEdit(self._last_dir)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.dir_set_button = QPushButton("Set")
        self.dir_set_button.clicked.connect(self._on_set_directory_clicked)
        layout.addWidget(_field("Directory", self.dir_edit, self.dir_set_button))

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(_DETECTOR_LIST)
        layout.addWidget(_field("Detector", self.detector_combo))

        self.load_button = QPushButton("LOAD IMAGE DATA")
        self.load_button.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_button)

        layout.addWidget(_divider("Manual Calibration"))
        self.center_y_spin = _double_spin(-1.0e6, 1.0e6, 2, 0.0, 1.0)
        self.center_z_spin = _double_spin(-1.0e6, 1.0e6, 2, 0.0, 1.0)
        layout.addWidget(_field("Center [Y, Z]", self.center_y_spin, self.center_z_spin))

        self.set_center_button = QPushButton("Set C")
        self.set_center_button.setToolTip("Click a point on the 2D image to set the center there.")
        self.set_center_button.clicked.connect(self._on_set_center_clicked)
        self.update_center_button = QPushButton("Update Center")
        self.update_center_button.setToolTip("Apply the typed Center [Y, Z] values.")
        self.update_center_button.clicked.connect(self._on_update_center_clicked)
        layout.addWidget(_row(self.set_center_button, self.update_center_button))

        self.yplot_button = QPushButton("Y_Plot (Fit)")
        self.yplot_button.clicked.connect(lambda: self._on_line_plot_clicked("y"))
        self.zplot_button = QPushButton("Z_Plot (Fit)")
        self.zplot_button.clicked.connect(lambda: self._on_line_plot_clicked("z"))
        layout.addWidget(_row(self.yplot_button, self.zplot_button))

        self.radial_button = QPushButton("Rad.Ave (Fit)")
        self.radial_button.clicked.connect(self._on_radial_clicked)
        layout.addWidget(self.radial_button)

        layout.addWidget(_divider("Peak Fit"))
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(["Single Peak", "Symmetric Peaks (+/-)"])
        self.fit_mode_combo.setToolTip("Symmetric mode only applies to Y_Plot/Z_Plot line cuts, not Rad.Ave.")
        layout.addWidget(_field("Fit Mode", self.fit_mode_combo))

        self.select_range_button = QPushButton("1. Select Data Range")
        self.select_range_button.clicked.connect(self._on_select_range_clicked)
        layout.addWidget(self.select_range_button)

        self.fit_button = QPushButton("2. Gaussian Fit Peak(s)")
        self.fit_button.clicked.connect(self._on_fit_clicked)
        layout.addWidget(self.fit_button)

        self.fit_result_label = _muted("Offset: -")
        layout.addWidget(self.fit_result_label)

        self.apply_center_button = QPushButton("Apply to Center")
        self.apply_center_button.setToolTip("Apply the last Y_Plot/Z_Plot fit's offset onto the beam center.")
        self.apply_center_button.clicked.connect(self._on_apply_to_center_clicked)
        layout.addWidget(self.apply_center_button)

        # Widgets Smart Auto Run and manual Calc SDD both read from/write to.
        self.standard_combo = QComboBox()
        self.standard_combo.addItems(_STANDARD_LIST)
        self.standard_combo.setCurrentText("Silver behenate")
        self.order_combo = QComboBox()
        self.order_combo.addItems(_ORDER_LIST)
        self.energy_spin = _double_spin(0.1, 200.0, 4, 9.5, 0.1)
        self.pixel_radius_spin = _double_spin(0.0, 1.0e6, 2, 0.0, 1.0)

        layout.addWidget(_divider("Smart Auto Run"))
        self.auto_run_button = QPushButton("SMART AUTO RUN")
        self.auto_run_button.setToolTip(
            "Automatically refine the beam center, search for calibration rings,\n"
            "and compute SDD. This is a *local* refinement -- Center [Y, Z] should\n"
            "already be roughly right first (Mark / Update Center)."
        )
        self.auto_run_button.clicked.connect(self._on_auto_run_clicked)
        layout.addWidget(self.auto_run_button)

        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels(["Radius", "Order", "Coverage", "R2", "SDD", "Use"])
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.candidate_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.candidate_table.setMaximumHeight(160)
        layout.addWidget(self.candidate_table)

        layout.addWidget(_divider("Calc SDD"))
        layout.addWidget(_field("Standard / Order", self.standard_combo, self.order_combo))
        layout.addWidget(_field("Energy [keV] / Pixel", self.energy_spin, self.pixel_radius_spin))

        self.calc_sdd_button = QPushButton("Calc SDD")
        self.calc_sdd_button.clicked.connect(self._on_calc_sdd_clicked)
        layout.addWidget(self.calc_sdd_button)

        self.sdd_result_label = _muted("Result: - mm")
        layout.addWidget(self.sdd_result_label)

        layout.addWidget(_divider("Save"))
        self.save_button = QPushButton("Save Calibration")
        self.save_button.clicked.connect(self._on_save_clicked)
        layout.addWidget(self.save_button)

        layout.addStretch(1)

        self.status_label = _muted("")
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
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        plot_group = QGroupBox("2D Image Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.image_canvas = ImageCanvas()
        self.image_canvas.pixel_clicked.connect(self._on_image_clicked)
        plot_layout.addWidget(self.image_canvas)
        splitter.addWidget(plot_group)

        controller_group = QGroupBox("Image Controller")
        controller_layout = QVBoxLayout(controller_group)
        self.color_scale_combo = QComboBox()
        self.color_scale_combo.addItems(["Linear", "Log"])
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(_COLORMAPS)
        default_index = self.colormap_combo.findText("viridis")
        if default_index >= 0:
            self.colormap_combo.setCurrentIndex(default_index)
        self.color_scale_combo.currentTextChanged.connect(self._apply_display_settings)
        self.colormap_combo.currentTextChanged.connect(self._apply_display_settings)
        # 6 decimals, not 3 -- see console_widget.py's cmin_spin/cmax_spin
        # for why: QDoubleSpinBox.setValue() rounds the *stored* value to
        # `decimals`, and low-count calibration frames can have a contrast
        # range well under 1.0.
        self.cmin_spin = _double_spin(-1.0e12, 1.0e12, 6, 0.0, 1.0)
        self.cmax_spin = _double_spin(-1.0e12, 1.0e12, 6, 0.0, 1.0)
        for spin in (self.cmin_spin, self.cmax_spin):
            spin.valueChanged.connect(self._apply_display_settings)
        controller_layout.addWidget(
            _row(
                _field("ColorScale", self.color_scale_combo, label_width=70),
                _field("Colormap", self.colormap_combo, label_width=64),
            )
        )
        controller_layout.addWidget(_field("Contrast", self.cmin_spin, self.cmax_spin, label_width=54))
        splitter.addWidget(controller_group)

        curve_group = QGroupBox("1D Result")
        curve_layout = QVBoxLayout(curve_group)
        self.curve_canvas = CurveCanvas()
        self.curve_canvas.point_clicked.connect(self._on_curve_clicked)
        curve_layout.addWidget(self.curve_canvas)
        self.curve_status_label = _muted("")
        curve_layout.addWidget(self.curve_status_label)
        splitter.addWidget(curve_group)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([420, 90, 340])
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

    def _update_pixel_dependent_buttons(self, loaded: bool) -> None:
        for button in (
            self.set_center_button,
            self.update_center_button,
            self.yplot_button,
            self.zplot_button,
            self.radial_button,
            self.auto_run_button,
            self.calc_sdd_button,
            self.save_button,
        ):
            button.setEnabled(loaded)

    # ------------------------------------------------------------------
    # Directory ("Set" mirrors GIXS_gui.m's Directory field + Set button --
    # seeds where LOAD IMAGE DATA opens, matching ConsoleWidget's own
    # `pathset_Callback`-equivalent pattern.)
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
    # Load Image Data
    # ------------------------------------------------------------------
    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a calibration-ring image", self._last_dir, "All files (*)")
        if not path:
            return
        file_path = Path(path)
        self._set_last_dir(str(file_path.parent))
        payload = {"directory": str(file_path.parent), "filename": file_path.name, "detector": self.detector_combo.currentText()}
        result = ss.sddcal_load(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "Load failed."), False)
            return

        self._range_state = None
        self._last_range_bounds = None
        self.curve_canvas.clear()
        self.fit_result_label.setText("Offset: -")
        self.sdd_result_label.setText("Result: - mm")

        center_y, center_z = result["center"]
        self.center_y_spin.setValue(center_y)
        self.center_z_spin.setValue(center_z)
        self._render_image()
        self._update_pixel_dependent_buttons(loaded=True)
        self._set_status(f"Loaded {'x'.join(str(v) for v in result['shape'])} from {result['path']}.", True)

    def _render_image(self) -> None:
        session = ss.get_session()
        if session.data is None:
            return
        self._current_image_data = session.data
        rows, cols = session.data.shape
        cmin = self.cmin_spin.value()
        cmax = self.cmax_spin.value()
        vmin = cmin if cmax > cmin else None
        vmax = cmax if cmax > cmin else None
        used_vmin, used_vmax = self.image_canvas.render(
            self._current_image_data,
            cmap=self.colormap_combo.currentText(),
            log_scale=self.color_scale_combo.currentText() == "Log",
            vmin=vmin,
            vmax=vmax,
            extent=(1, cols, rows, 1),
            xlabel="Y, pixel",
            ylabel="Z, pixel",
        )
        if vmin is None:
            self.cmin_spin.blockSignals(True)
            self.cmax_spin.blockSignals(True)
            self.cmin_spin.setValue(used_vmin)
            self.cmax_spin.setValue(used_vmax)
            self.cmin_spin.blockSignals(False)
            self.cmax_spin.blockSignals(False)
        self.image_canvas.reset_view()
        self._draw_crosshair()

    def _apply_display_settings(self) -> None:
        if self._current_image_data is None:
            return
        cmin = self.cmin_spin.value()
        cmax = self.cmax_spin.value()
        vmin = cmin if cmax > cmin else None
        vmax = cmax if cmax > cmin else None
        rows, cols = self._current_image_data.shape
        used_vmin, used_vmax = self.image_canvas.render(
            self._current_image_data,
            cmap=self.colormap_combo.currentText(),
            log_scale=self.color_scale_combo.currentText() == "Log",
            vmin=vmin,
            vmax=vmax,
            extent=(1, cols, rows, 1),
            xlabel="Y, pixel",
            ylabel="Z, pixel",
            keep_limits=True,
        )
        if vmin is None:
            self.cmin_spin.blockSignals(True)
            self.cmax_spin.blockSignals(True)
            self.cmin_spin.setValue(used_vmin)
            self.cmax_spin.setValue(used_vmax)
            self.cmin_spin.blockSignals(False)
            self.cmax_spin.blockSignals(False)
        self._draw_crosshair()

    def _draw_crosshair(self) -> None:
        session = ss.get_session()
        if session.data is None:
            return
        center = (self.center_y_spin.value(), self.center_z_spin.value())
        self.image_canvas.draw_beam_line(center, 0.0)
        self.image_canvas.draw_beam_line(center, 90.0)

    # ------------------------------------------------------------------
    # Center: Set C (click the image) / Update Center (apply typed values)
    #
    # "Set C" matches MATLAB `Center_Callback`'s intent -- set the center by
    # pointing at the image -- simplified from MATLAB's draggable
    # `drawpoint`/`MovingROI` live-drag to a single click (see
    # sddcal_widget's module docstring / the SDD Calibration plan for why:
    # no draggable-artist infrastructure exists in this app yet, and
    # ImageCanvas.render() rebuilds its axes from scratch on every call,
    # which would fight any drag-session state). "Update Center" instead
    # matches MATLAB `Marking_Callback` -- apply whatever is currently
    # typed into Center [Y, Z].
    # ------------------------------------------------------------------
    def _on_set_center_clicked(self) -> None:
        self._center_mode = True
        self._set_status("Click a point on the 2D image to set the center.", True)

    def _on_image_clicked(self, x: float, y: float) -> None:
        if not self._center_mode:
            return
        self._center_mode = False
        self._apply_center(x, y)

    def _on_update_center_clicked(self) -> None:
        self._apply_center(self.center_y_spin.value(), self.center_z_spin.value())

    def _apply_center(self, y: float, z: float) -> None:
        result = ss.sddcal_set_center({"y": y, "z": z})
        if not result.get("ok"):
            self._set_status(result.get("message", "Set center failed."), False)
            return
        self.center_y_spin.setValue(result["center"][0])
        self.center_z_spin.setValue(result["center"][1])
        self.image_canvas.clear_overlays()
        self._draw_crosshair()
        self._set_status(f"Center set to ({result['center'][0]:.2f}, {result['center'][1]:.2f}).", True)

    # ------------------------------------------------------------------
    # Y_Plot / Z_Plot / Rad.Ave
    # ------------------------------------------------------------------
    def _on_line_plot_clicked(self, direction: str) -> None:
        result = ss.sddcal_line_cut({"direction": direction})
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Line cut failed."), False)
            return
        self._render_last_curve(xlabel=f"Distance from center ({direction}), pixel")
        self._set_curve_status(f"{direction.upper()}_Plot ready. Select a range and fit a peak.", True)

    def _on_radial_clicked(self) -> None:
        result = ss.sddcal_radial_profile({"use_median": False})
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Radial average failed."), False)
            return
        self._render_last_curve(xlabel="Radius, pixel")
        self._set_curve_status("Radial profile ready. Select a range and fit a peak.", True)

    def _render_last_curve(self, *, xlabel: str) -> None:
        curve = ss.get_session().last_curve
        self.curve_canvas.render(np.asarray(curve["x"]), np.asarray(curve["y"]), xlabel=xlabel, ylabel="Intensity, arb. units")
        self._range_state = None
        self._last_range_bounds = None
        self.fit_result_label.setText("Offset: -")

    # ------------------------------------------------------------------
    # Peak Fit: range selection + Gaussian fit
    # ------------------------------------------------------------------
    def _on_select_range_clicked(self) -> None:
        session = ss.get_session()
        if session.last_curve is None:
            self._set_curve_status("Run Y_Plot, Z_Plot, or Rad.Ave first.", False)
            return
        self.curve_canvas.clear_overlays()
        self._last_range_bounds = None
        symmetric = session.last_curve["kind"] != "radial" and self.fit_mode_combo.currentText() == "Symmetric Peaks (+/-)"
        if symmetric:
            self._range_state = {"mode": "symmetric", "stage": "left1", "bounds": {}}
            self._set_curve_status("[Minus] Click the left peak's start.", True)
        else:
            self._range_state = {"mode": "single", "stage": "p1", "bounds": {}}
            self._set_curve_status("[1/2] Click the peak's start.", True)

    def _on_curve_clicked(self, x: float, _y: float) -> None:
        if self._range_state is None:
            return
        state = self._range_state
        stage = state["stage"]
        bounds = state["bounds"]

        if stage == "p1":
            bounds["c1"] = x
            state["stage"] = "p2"
            self._set_curve_status(f"[2/2] Click the peak's end (start={x:.1f}).", True)
        elif stage == "p2":
            bounds["c2"] = x
            self.curve_canvas.mark_range(bounds["c1"], bounds["c2"], color="b")
            self._last_range_bounds = {"mode": "single", "c1": bounds["c1"], "c2": bounds["c2"]}
            self._range_state = None
            lo, hi = sorted((bounds["c1"], bounds["c2"]))
            self._set_curve_status(f"Range: {lo:.1f} to {hi:.1f} px. Ready to fit.", True)
        elif stage == "left1":
            bounds["cl1"] = x
            state["stage"] = "left2"
            self._set_curve_status(f"[Minus] Click the left peak's end (start={x:.1f}).", True)
        elif stage == "left2":
            bounds["cl2"] = x
            self.curve_canvas.mark_range(bounds["cl1"], bounds["cl2"], color="b")
            state["stage"] = "right1"
            self._set_curve_status("[Plus] Click the right peak's start.", True)
        elif stage == "right1":
            bounds["cr1"] = x
            state["stage"] = "right2"
            self._set_curve_status(f"[Plus] Click the right peak's end (start={x:.1f}).", True)
        elif stage == "right2":
            bounds["cr2"] = x
            self.curve_canvas.mark_range(bounds["cr1"], bounds["cr2"], color="r")
            self._last_range_bounds = {
                "mode": "symmetric",
                "cl1": bounds["cl1"],
                "cl2": bounds["cl2"],
                "cr1": bounds["cr1"],
                "cr2": bounds["cr2"],
            }
            self._range_state = None
            self._set_curve_status("Range selected. Ready to fit.", True)

    def _on_fit_clicked(self) -> None:
        if self._last_range_bounds is None:
            self._set_curve_status("Select Data Range first.", False)
            return
        payload = dict(self._last_range_bounds)
        mode = payload.pop("mode")
        result = ss.sddcal_fit_peak({"mode": mode, **payload})
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Fit failed."), False)
            return

        session = ss.get_session()
        kind = session.last_curve["kind"] if session.last_curve else None
        if mode == "single":
            self.curve_canvas.overlay_line(np.asarray(result["fit_x"]), np.asarray(result["fit_y"]), color="r")
            self.fit_result_label.setText(f"Offset: {result['center']:.2f} px (R2={result['r_square']:.3f})")
            if kind == "radial":
                self.pixel_radius_spin.setValue(result["center"])
        else:
            self.curve_canvas.overlay_line(np.asarray(result["left_fit_x"]), np.asarray(result["left_fit_y"]), color="c")
            self.curve_canvas.overlay_line(np.asarray(result["right_fit_x"]), np.asarray(result["right_fit_y"]), color="m")
            self.fit_result_label.setText(f"Center Offset: {result['center']:.2f} px (Radius: {result['radius']:.1f} px)")
            self.pixel_radius_spin.setValue(result["radius"])
        self._set_curve_status("Fit successful.", True)

    def _on_apply_to_center_clicked(self) -> None:
        result = ss.sddcal_apply_fit_to_center({})
        if not result.get("ok"):
            self._set_curve_status(result.get("message", "Apply to Center failed."), False)
            return
        self.center_y_spin.setValue(result["center"][0])
        self.center_z_spin.setValue(result["center"][1])
        self.image_canvas.clear_overlays()
        self._draw_crosshair()
        self._set_curve_status(f"Center updated to ({result['center'][0]:.2f}, {result['center'][1]:.2f}).", True)

    # ------------------------------------------------------------------
    # Smart Auto Run
    # ------------------------------------------------------------------
    def _on_auto_run_clicked(self) -> None:
        payload = {"standard_sample": self.standard_combo.currentText(), "energy_kev": self.energy_spin.value()}
        result = ss.sddcal_auto_run(payload)

        # Center may have moved (and the candidate table should reflect
        # what was actually tried) even on a failed run -- update both
        # unconditionally before handling ok/fail.
        if result.get("center"):
            self.center_y_spin.setValue(result["center"][0])
            self.center_z_spin.setValue(result["center"][1])
            self.image_canvas.clear_overlays()
            self._draw_crosshair()
        self._populate_candidate_table(result.get("candidates", []))

        if not result.get("ok"):
            self._set_status(result.get("message", "Smart Auto Run failed."), False)
            return

        self.pixel_radius_spin.setValue(result["pixel_radius"])
        order = result.get("order")
        if order and 1 <= order <= len(_ORDER_LIST):
            self.order_combo.setCurrentText(_ORDER_LIST[order - 1])
        self.sdd_result_label.setText(f"Result: {result['sdd_mm']:.2f} mm")

        profile = result.get("profile")
        if profile:
            self.curve_canvas.render(
                np.asarray(profile["x"], dtype=np.float64),
                np.asarray(profile["y"], dtype=np.float64),
                xlabel="Radius, pixel",
                ylabel="Intensity, arb. units",
            )
            if result.get("fit_x"):
                self.curve_canvas.overlay_line(np.asarray(result["fit_x"]), np.asarray(result["fit_y"]), color="r")
        self._range_state = None
        self._last_range_bounds = None
        self.fit_result_label.setText(f"Offset: {result['pixel_radius']:.2f} px (R2={result['r_square']:.3f})")

        message = (
            f"Smart Auto Run: SDD = {result['sdd_mm']:.2f} mm "
            f"({result['accepted_count']} rings accepted, spread {result['sdd_std']:.2f} mm)."
        )
        warnings = result.get("warnings") or []
        if warnings:
            QMessageBox.warning(self, "Smart Auto Run", "\n\n".join(warnings))
        self._set_status(message, True)
        self._set_curve_status("Radial profile + fit from Smart Auto Run.", True)

    def _populate_candidate_table(self, candidates: list[dict[str, Any]]) -> None:
        table = self.candidate_table
        table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            order = candidate.get("order")
            sdd = candidate.get("sdd_mm")
            values = [
                f"{candidate['rough_radius']:.1f}",
                str(order) if order is not None else "-",
                f"{candidate['coverage']:.2f}",
                f"{candidate['r_square']:.3f}" if np.isfinite(candidate["r_square"]) else "-",
                f"{sdd:.2f}" if sdd is not None and np.isfinite(sdd) else "-",
                "Yes" if candidate["accepted"] else "No",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, col, item)

    # ------------------------------------------------------------------
    # Calc SDD / Save
    # ------------------------------------------------------------------
    def _on_calc_sdd_clicked(self) -> None:
        order = _ORDER_LIST.index(self.order_combo.currentText()) + 1
        payload = {
            "pixel_radius": self.pixel_radius_spin.value(),
            "order": order,
            "standard_sample": self.standard_combo.currentText(),
            "energy_kev": self.energy_spin.value(),
        }
        result = ss.sddcal_calc_sdd(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "Calc SDD failed."), False)
            return
        self.sdd_result_label.setText(f"Result: {result['sdd_mm']:.2f} mm")
        self._set_status(f"SDD = {result['sdd_mm']:.2f} mm (wavelength {result['wavelength']:.4f} A).", True)

    def _on_save_clicked(self) -> None:
        result = ss.sddcal_save({})
        if not result.get("ok"):
            self._set_status(result.get("message", "Save failed."), False)
            return
        extras = [path for path in (result.get("mat_path"), result.get("std_path")) if path]
        note = f", {', '.join(extras)}" if extras else " (mat/std save failed)"
        self._set_status(f"Saved {result['json_path']}{note}.", True)
