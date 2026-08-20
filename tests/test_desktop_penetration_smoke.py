"""Smoke tests for the Penetration Depth tab -- mirrors
test_desktop_sddcal_smoke.py's conventions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.penetration_widget import PenetrationWidget
from pgixs_python.ui_backend import penetration_session as ps


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_penetration_widget_constructs(qapp):
    widget = PenetrationWidget()
    assert widget.chem_edit.text() == "Si"
    assert widget.density_spin.value() == pytest.approx(2.329)


def test_penetration_widget_run_calculation_updates_info_and_plot(qapp):
    ps.reset_session()
    widget = PenetrationWidget()
    widget.energy_spin.setValue(11.3)
    widget.angle_min_spin.setValue(0.0)
    widget.angle_max_spin.setValue(1.0)

    widget._on_calc_clicked()

    assert "Delta:" in widget.delta_label.text()
    assert "Beta:" in widget.beta_label.text()
    assert "Critical Angle:" in widget.critical_label.text()
    assert ps.get_session().last_result is not None
    assert len(widget.curve_canvas._data_x) == 200
    assert len(widget.curve_canvas._overlay_artists) == 1  # critical-angle vline


def test_penetration_widget_reports_invalid_formula(qapp):
    ps.reset_session()
    widget = PenetrationWidget()
    widget.chem_edit.setText("Xx2")

    widget._on_calc_clicked()

    assert widget.status_label.property("role") == "status-error"


def test_penetration_widget_save_writes_a_file(qapp, tmp_path):
    ps.reset_session()
    widget = PenetrationWidget()
    widget._set_last_dir(str(tmp_path))
    widget._on_calc_clicked()

    widget._on_save_clicked()

    assert "Saved" in widget.status_label.text()
    assert any(tmp_path.glob("PenetrationDepth_*.dat"))


def test_penetration_widget_table_button_without_calculation_shows_error(qapp):
    ps.reset_session()
    widget = PenetrationWidget()

    widget._on_table_clicked()

    assert widget.status_label.property("role") == "status-error"
