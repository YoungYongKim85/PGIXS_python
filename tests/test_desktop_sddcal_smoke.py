"""Smoke tests for the SDD Calibration tab: constructs the real Qt widget
tree under an offscreen platform plugin and asserts the wiring holds
together. Mirrors test_desktop_smoke.py's conventions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.sddcal_widget import SDDcalWidget
from pgixs_python.ui_backend import sddcal_session as ss


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_synthetic_ring_frame(path, shape=(200, 200), center=(100.5, 100.5), ring_radius=60.0, ring_sigma=3.0, seed=0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(seed)
    data = 10.0 + 500.0 * np.exp(-((r - ring_radius) ** 2) / (2.0 * ring_sigma**2))
    data = data + rng.uniform(0.0, 1.0, size=shape)
    data = data.astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])
    return data


def _write_synthetic_multi_ring_frame(path, shape=(400, 400), center=(200.5, 200.5), true_sdd=250.0, energy_kev=9.5, pixel_size_mm=0.075, seed=0):
    """Rings at the exact radii Silver behenate orders 1-4 would sit at for
    `true_sdd`/`energy_kev` -- same construction as the backend's own
    `_synthetic_multi_ring_frame` in test_ui_backend_sddcal_session.py."""

    from pgixs_python.calibration.standards import standard_q_value

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    wavelength = 12.398419843320026 / energy_kev
    data = np.full(shape, 10.0, dtype=np.float64)
    for order in range(1, 5):
        q = standard_q_value("Silver behenate", order)
        two_theta = 2.0 * np.arcsin(q * wavelength / (4.0 * np.pi))
        radius = true_sdd * np.tan(two_theta) / pixel_size_mm
        data += 500.0 * np.exp(-((r - radius) ** 2) / (2.0 * 2.0**2))
    rng = np.random.default_rng(seed)
    data = data + rng.uniform(0.0, 1.0, size=shape)
    data = data.astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])
    return data


def _load_test_frame(tmp_path, center=(100.5, 100.5), **kwargs):
    ss.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=center, **kwargs)
    widget = SDDcalWidget()
    result = ss.sddcal_load({"directory": str(tmp_path), "filename": frame_path.name, "detector": "TestDetector"})
    assert result["ok"], result.get("message")
    widget._update_pixel_dependent_buttons(loaded=True)
    widget._render_image()
    # Set the *true* ring center explicitly (not sddcal_load's own default,
    # which is just the geometric image center) so tests that fit a peak
    # against this synthetic ring see it at the expected radius.
    widget.center_y_spin.setValue(center[0])
    widget.center_z_spin.setValue(center[1])
    widget._on_update_center_clicked()
    return widget


def test_sddcal_widget_constructs(qapp):
    widget = SDDcalWidget()
    assert widget.detector_combo.count() == 5
    assert widget.standard_combo.currentText() == "Silver behenate"


def test_sddcal_widget_directory_field_seeds_last_dir(qapp, tmp_path):
    """Regression: "SDD Cal에도 Directory setting되게 해주겠어?" -- mirrors
    ConsoleWidget's Directory field + Set button, seeding where LOAD IMAGE
    DATA's file dialog opens."""

    widget = SDDcalWidget()
    assert widget.dir_edit.text() == widget._last_dir

    widget._set_last_dir(str(tmp_path))
    assert widget.dir_edit.text() == str(tmp_path)
    assert widget._last_dir == str(tmp_path)

    # Typing a valid directory and leaving the field commits it.
    other_dir = tmp_path / "sub"
    other_dir.mkdir()
    widget.dir_edit.setText(str(other_dir))
    widget._on_directory_edited()
    assert widget._last_dir == str(other_dir)

    # Typing a bogus path reverts the field to the last good directory.
    widget.dir_edit.setText(str(tmp_path / "does_not_exist"))
    widget._on_directory_edited()
    assert widget._last_dir == str(other_dir)
    assert widget.dir_edit.text() == str(other_dir)


def test_sddcal_widget_load_clicked_updates_directory_field(qapp, tmp_path, monkeypatch):
    """Picking a file via LOAD IMAGE DATA's dialog must also update the
    Directory field to match, same as ConsoleWidget's own LOAD DATA does."""

    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(100.5, 100.5))
    ss.reset_session()
    widget = SDDcalWidget()
    # "TestDetector" (not a real detector) so the synthetic 200x200 frame
    # doesn't hit a real detector's mask-shape mismatch and the load
    # actually succeeds -- same reasoning as the other synthetic-frame tests.
    widget.detector_combo.addItem("TestDetector")
    widget.detector_combo.setCurrentText("TestDetector")
    monkeypatch.setattr(
        "pgixs_python.desktop.sddcal_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(frame_path), "")),
    )

    widget._on_load_clicked()

    assert widget._last_dir == str(tmp_path)
    assert widget.dir_edit.text() == str(tmp_path)


def test_sddcal_widget_load_updates_center_and_enables_buttons(qapp, tmp_path):
    ss.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(100.5, 100.5))
    widget = SDDcalWidget()
    assert not widget.update_center_button.isEnabled()

    widget._last_dir = str(tmp_path)
    # "TestDetector" (not the combo's real-detector default) so the
    # synthetic 200x200 frame doesn't hit a real detector's mask-shape
    # mismatch -- unknown detector types fall back to "no mask".
    payload = {"directory": str(tmp_path), "filename": frame_path.name, "detector": "TestDetector"}
    result = ss.sddcal_load(payload)
    assert result["ok"]
    widget.center_y_spin.setValue(result["center"][0])
    widget.center_z_spin.setValue(result["center"][1])
    widget._render_image()
    widget._update_pixel_dependent_buttons(loaded=True)

    assert widget.update_center_button.isEnabled()
    assert widget.center_y_spin.value() == pytest.approx(100.0)
    assert widget.center_z_spin.value() == pytest.approx(100.0)
    assert len(widget.image_canvas._overlay_artists) == 2  # crosshair


def test_sddcal_widget_set_c_relocates_center_via_click(qapp, tmp_path):
    """Regression: "Set C 버튼을 통해 이미지에서 Center position을 설정하게 해야돼" --
    "Set C" arms click-to-relocate mode (matches MATLAB Center_Callback's
    intent, simplified from drag to a single click); "Update Center"
    applies whatever is currently typed in Center [Y, Z] instead (matches
    MATLAB Marking_Callback)."""

    widget = _load_test_frame(tmp_path, center=(100.5, 100.5))

    widget._on_set_center_clicked()
    assert widget._center_mode

    widget._on_image_clicked(120.0, 90.0)

    assert not widget._center_mode
    assert widget.center_y_spin.value() == pytest.approx(120.0)
    assert widget.center_z_spin.value() == pytest.approx(90.0)
    assert ss.get_session().center == (120.0, 90.0)


def test_sddcal_widget_image_click_is_ignored_when_not_armed(qapp, tmp_path):
    widget = _load_test_frame(tmp_path, center=(100.5, 100.5))

    widget._on_image_clicked(5.0, 5.0)

    # Center unchanged -- Mark was never clicked, so the click should be a no-op.
    assert widget.center_y_spin.value() == pytest.approx(100.5)
    assert widget.center_z_spin.value() == pytest.approx(100.5)


def test_sddcal_widget_radial_fit_and_calc_sdd_end_to_end(qapp, tmp_path):
    """Full click-driven workflow: Rad.Ave -> Select Data Range (2 clicks)
    -> Gaussian Fit -> Calc SDD, against a synthetic ring at a known radius."""

    widget = _load_test_frame(tmp_path, center=(100.5, 100.5), ring_radius=60.0)

    widget._on_radial_clicked()
    assert widget._range_state is None  # not yet armed

    widget._on_select_range_clicked()
    assert widget._range_state is not None
    assert widget._range_state["mode"] == "single"

    widget._on_curve_clicked(40.0, 0.0)
    assert widget._range_state["stage"] == "p2"
    widget._on_curve_clicked(80.0, 0.0)
    assert widget._range_state is None
    assert widget._last_range_bounds == {"mode": "single", "c1": 40.0, "c2": 80.0}
    assert len(widget.curve_canvas._overlay_artists) == 2  # range bound lines

    widget._on_fit_clicked()
    assert widget.pixel_radius_spin.value() == pytest.approx(60.0, abs=1.0)
    assert "Offset" in widget.fit_result_label.text()
    assert len(widget.curve_canvas._overlay_artists) == 3  # 2 bounds + fit curve

    widget.energy_spin.setValue(9.5)
    widget.standard_combo.setCurrentText("Silver behenate")
    widget.order_combo.setCurrentText("1st")
    widget._on_calc_sdd_clicked()

    assert "mm" in widget.sdd_result_label.text()
    assert ss.get_session().sdd_result is not None
    assert ss.get_session().sdd_result["sdd_mm"] > 0


def test_sddcal_widget_symmetric_fit_on_line_cut_and_apply_to_center(qapp, tmp_path):
    widget = _load_test_frame(tmp_path, center=(105.5, 100.5), ring_radius=60.0)
    widget.center_y_spin.setValue(100.5)  # deliberately off-center by 5px in y
    widget._on_update_center_clicked()

    widget._on_line_plot_clicked("y")
    widget.fit_mode_combo.setCurrentText("Symmetric Peaks (+/-)")
    widget._on_select_range_clicked()
    assert widget._range_state["mode"] == "symmetric"

    for x in (-80.0, -40.0, 40.0, 80.0):
        widget._on_curve_clicked(x, 0.0)
    assert widget._range_state is None
    assert widget._last_range_bounds["mode"] == "symmetric"

    widget._on_fit_clicked()
    assert widget.pixel_radius_spin.value() == pytest.approx(60.0, abs=1.0)
    assert "Center Offset" in widget.fit_result_label.text()

    widget._on_apply_to_center_clicked()
    assert widget.center_y_spin.value() == pytest.approx(105.5, abs=1.0)


def test_sddcal_widget_symmetric_fit_rejected_for_radial_curve(qapp, tmp_path):
    widget = _load_test_frame(tmp_path, center=(100.5, 100.5), ring_radius=60.0)

    widget._on_radial_clicked()
    widget.fit_mode_combo.setCurrentText("Symmetric Peaks (+/-)")
    widget._on_select_range_clicked()

    # Rad.Ave always forces single-peak mode, regardless of the combo.
    assert widget._range_state["mode"] == "single"


def test_sddcal_widget_auto_run_populates_table_and_result(qapp, tmp_path):
    """Regression: "매트랩의 프로그램 처럼 auto calculation process를 넣어야돼" --
    SMART AUTO RUN should refine the center, fill the candidate table, and
    set the SDD result label + curve, without any manual Select
    Range/Gaussian Fit/Calc SDD clicks."""

    ss.reset_session()
    true_center = (200.5, 200.5)
    true_sdd = 250.0
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_multi_ring_frame(frame_path, center=true_center, true_sdd=true_sdd)
    widget = SDDcalWidget()
    payload = {"directory": str(tmp_path), "filename": frame_path.name, "detector": "TestDetector"}
    result = ss.sddcal_load(payload)
    assert result["ok"], result.get("message")
    widget._render_image()
    widget._update_pixel_dependent_buttons(loaded=True)
    # Start from a nearby (not exact) guess -- Auto Run is a local refinement.
    widget.center_y_spin.setValue(true_center[0] + 4.0)
    widget.center_z_spin.setValue(true_center[1] - 3.0)
    widget._on_update_center_clicked()
    widget.energy_spin.setValue(9.5)
    widget.standard_combo.setCurrentText("Silver behenate")

    widget._on_auto_run_clicked()

    assert widget.center_y_spin.value() == pytest.approx(true_center[0], abs=2.0)
    assert widget.center_z_spin.value() == pytest.approx(true_center[1], abs=2.0)
    assert widget.candidate_table.rowCount() >= 2
    assert "mm" in widget.sdd_result_label.text()
    assert ss.get_session().sdd_result["sdd_mm"] == pytest.approx(true_sdd, rel=0.05)
    assert "Smart Auto Run" in widget.status_label.text()


def test_sddcal_widget_auto_run_requires_loaded_data(qapp):
    ss.reset_session()
    widget = SDDcalWidget()

    widget._on_auto_run_clicked()

    assert "Load a detector image first" in widget.status_label.text()
    assert widget.candidate_table.rowCount() == 0


def test_sddcal_widget_save_writes_calibration(qapp, tmp_path):
    widget = _load_test_frame(tmp_path, center=(100.5, 100.5), ring_radius=60.0)
    widget._on_radial_clicked()
    widget._on_select_range_clicked()
    widget._on_curve_clicked(40.0, 0.0)
    widget._on_curve_clicked(80.0, 0.0)
    widget._on_fit_clicked()
    widget._on_calc_sdd_clicked()

    widget._on_save_clicked()

    assert "Saved" in widget.status_label.text()
    assert (tmp_path / "frame_standard.json").exists()
    assert (tmp_path / "frame_standard.mat").exists()
    assert (tmp_path / "frame_standard.std").exists()
