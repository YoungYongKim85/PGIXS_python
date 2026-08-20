"""Smoke tests for the XTrans tab -- mirrors test_desktop_sddcal_smoke.py's conventions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.xtrans_widget import XTransWidget
from pgixs_python.ui_backend import xtrans_session as xts


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_xtrans_widget_constructs(qapp):
    widget = XTransWidget()
    assert widget.mode_combo.count() == 4
    assert widget.chem_edit.text() == "C8H8"


def test_xtrans_widget_directory_field_seeds_last_dir(qapp, tmp_path):
    widget = XTransWidget()
    widget._set_last_dir(str(tmp_path))
    assert widget.dir_edit.text() == str(tmp_path)
    assert widget._last_dir == str(tmp_path)


def test_xtrans_widget_run_calculation_updates_plot_and_status(qapp):
    xts.reset_session()
    widget = XTransWidget()
    widget.chem_edit.setText("Si")
    widget.density_spin.setValue(2.329)
    widget.energy_min_spin.setValue(8.0)
    widget.energy_max_spin.setValue(12.0)
    widget.n_points_spin.setValue(10)
    widget.mode_combo.setCurrentText("Critical Angle")

    widget._on_calc_clicked()

    assert "MW=" in widget.status_label.text()
    assert xts.get_session().last_result is not None
    assert len(widget.curve_canvas._data_x) == 10


def test_xtrans_widget_run_calculation_reports_invalid_formula(qapp):
    xts.reset_session()
    widget = XTransWidget()
    widget.chem_edit.setText("Xx2")

    widget._on_calc_clicked()

    assert widget.status_label.property("role") == "status-error"


def test_xtrans_widget_save_writes_a_file(qapp, tmp_path):
    xts.reset_session()
    widget = XTransWidget()
    widget._set_last_dir(str(tmp_path))
    widget.mode_combo.setCurrentText("Transmittance")
    widget._on_calc_clicked()

    widget._on_save_clicked()

    assert "Saved" in widget.status_label.text()
    assert any(tmp_path.glob("Transmittance_*.dat"))


def test_xtrans_widget_table_button_without_calculation_shows_error(qapp):
    xts.reset_session()
    widget = XTransWidget()

    widget._on_table_clicked()

    assert widget.status_label.property("role") == "status-error"
