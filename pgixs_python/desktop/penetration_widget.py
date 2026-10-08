"""Penetration Depth tab: port of MATLAB `Penetration_GUI.m` -- X-ray
evanescent-wave penetration depth vs. incidence angle at a fixed photon
energy, for a given chemical formula and mass density (Zhang Jiang's
`refrac` toolbox, see `pgixs_python.physics.refrac`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pgixs_python.desktop.console_widget import _divider, _double_spin, _field, _muted, _row, _split_row, _wrap_long_tokens
from pgixs_python.desktop.plot_canvas import CurveCanvas
from pgixs_python.ui_backend import penetration_session as ps


class PenetrationWidget(QWidget):
    """Single-instance tab, same session-ownership model as SDDcalWidget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_dir = str(Path.home())

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        group = QGroupBox("Penetration Depth Parameters")
        layout = QVBoxLayout(group)
        layout.setSpacing(7)

        self.chem_edit = QLineEdit("Si")
        layout.addWidget(_field("Chem. Formula", self.chem_edit))

        self.density_spin = _double_spin(0.001, 30.0, 4, 2.329, 0.01)
        layout.addWidget(_field("Density, g/cm3", self.density_spin))

        self.energy_spin = _double_spin(0.03, 30.0, 4, 11.3, 0.1)
        layout.addWidget(_field("Energy, keV", self.energy_spin))

        self.angle_min_spin = _double_spin(-10.0, 90.0, 4, 0.0, 0.05)
        self.angle_max_spin = _double_spin(-10.0, 90.0, 4, 1.0, 0.05)
        layout.addWidget(
            _split_row(
                _field("Angle Min", self.angle_min_spin, label_width=80),
                _field("Max", self.angle_max_spin, label_width=40),
            )
        )

        self.calc_button = QPushButton("RUN CALCULATION")
        self.calc_button.clicked.connect(self._on_calc_clicked)
        layout.addWidget(self.calc_button)

        layout.addWidget(_divider("Information"))
        self.delta_label = _muted("Delta: -")
        self.beta_label = _muted("Beta: -")
        self.critical_label = _muted("Critical Angle: -")
        layout.addWidget(self.delta_label)
        layout.addWidget(self.beta_label)
        layout.addWidget(self.critical_label)

        layout.addWidget(_divider("Save"))
        self.dir_edit = QLineEdit(self._last_dir)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.dir_set_button = QPushButton("Set")
        self.dir_set_button.clicked.connect(self._on_set_directory_clicked)
        layout.addWidget(_field("Directory", self.dir_edit, self.dir_set_button))

        self.table_button = QPushButton("Data Table")
        self.table_button.clicked.connect(self._on_table_clicked)
        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self._on_save_clicked)
        layout.addWidget(_row(self.table_button, self.save_button))

        layout.addStretch(1)

        self.status_label = _muted("")
        self.status_label.setMinimumHeight(3 * self.status_label.fontMetrics().lineSpacing())
        layout.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(group)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(420)
        return scroll

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_group = QGroupBox("Result Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.curve_canvas = CurveCanvas()
        plot_layout.addWidget(self.curve_canvas)
        layout.addWidget(plot_group, 1)

        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(["linear", "log"])
        self.y_scale_combo.currentTextChanged.connect(self._apply_scales)
        self.grid_check = QCheckBox("Grid On")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self._apply_scales)
        ctrl_layout.addWidget(_field("Y Scale", self.y_scale_combo, label_width=56))
        ctrl_layout.addWidget(self.grid_check)
        layout.addWidget(ctrl_group)
        return container

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------
    def _set_status(self, message: str, ok: bool) -> None:
        self.status_label.setText(_wrap_long_tokens(message))
        self.status_label.setProperty("role", "status-ok" if ok else "status-error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    def _set_last_dir(self, directory: str) -> None:
        self._last_dir = directory
        self.dir_edit.setText(directory)

    def _on_set_directory_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select save directory", self._last_dir)
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
    # Calculate / plot
    # ------------------------------------------------------------------
    def _on_calc_clicked(self) -> None:
        payload = {
            "chem_formula": self.chem_edit.text(),
            "mass_density": self.density_spin.value(),
            "energy_kev": self.energy_spin.value(),
            "angle_min": self.angle_min_spin.value(),
            "angle_max": self.angle_max_spin.value(),
        }
        result = ps.penetration_calculate(payload)
        if not result.get("ok"):
            self._set_status(result.get("message", "Calculation failed."), False)
            return

        self.delta_label.setText(f"Delta: {result['delta']:.4e}")
        self.beta_label.setText(f"Beta: {result['beta']:.4e}")
        self.critical_label.setText(f"Critical Angle: {result['critical_angle']:.4f} deg")

        self._render_result(result)
        self._set_status(f"Wavelength = {result['wavelength']:.4f} A.", True)

    def _render_result(self, result: dict[str, Any]) -> None:
        self.curve_canvas.render(
            np.asarray(result["x"], dtype=np.float64),
            np.asarray(result["y"], dtype=np.float64),
            xlabel="Incidence Angle, deg",
            ylabel="Penetration Depth, nm",
        )
        self.curve_canvas.draw_vline(result["critical_angle"], color="r")
        self._apply_scales()

    def _apply_scales(self) -> None:
        if ps.get_session().last_result is None:
            return
        self.curve_canvas.set_scales(False, self.y_scale_combo.currentText() == "log")
        self.curve_canvas.ax.grid(self.grid_check.isChecked(), alpha=0.25)
        self.curve_canvas.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Data Table / Save
    # ------------------------------------------------------------------
    def _on_table_clicked(self) -> None:
        session = ps.get_session()
        if session.last_result is None:
            self._set_status("Run a calculation first.", False)
            return
        r = session.last_result
        dialog = QDialog(self)
        dialog.setWindowTitle("Data Table")
        dialog.resize(320, 420)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(r["x"]), 2)
        table.setHorizontalHeaderLabels(["Incidence_Angle_deg", "Penetration_Depth_nm"])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, (x_val, y_val) in enumerate(zip(r["x"], r["y"])):
            table.setItem(row, 0, QTableWidgetItem(f"{x_val:.6g}"))
            table.setItem(row, 1, QTableWidgetItem(f"{y_val:.6g}"))
        layout.addWidget(table)
        dialog.exec()
        dialog.deleteLater()  # parented to this widget; would otherwise live until exit

    def _on_save_clicked(self) -> None:
        result = ps.penetration_save({"directory": self._last_dir})
        if not result.get("ok"):
            self._set_status(result.get("message", "Save failed."), False)
            return
        self._set_status(f"Saved {result['path']}.", True)
