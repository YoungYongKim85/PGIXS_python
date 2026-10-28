"""Smoke tests for the TrXS2 tab -- mirrors test_desktop_gixs2_smoke.py's
conventions, adapted for TrXS2's own flows: no incidence-angle q-Cal, a
Circular Average primary 1D mode (vs. GIXS2's plain line-cut default), and a
live Background Subtraction pipeline with point-click normalization that
GIXS2 has no equivalent of."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from pgixs_python.desktop.plot_canvas import ImageCanvas
from pgixs_python.desktop.trxs2_widget import _ROI_COLORS, TrXS2Widget, _finite_range
from pgixs_python.ui_backend import trxs2_session as ts


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_synthetic_series(tmp_path, n_frames=3, shape=(200, 200), center=(100.5, 100.5), radii=None, prefix="ABCD"):
    if radii is None:
        radii = [60.0] * n_frames
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(0)
    files = []
    for idx, radius in enumerate(radii):
        data = 10.0 + 500.0 * np.exp(-((r - radius) ** 2) / (2.0 * 3.0**2))
        data = (data + rng.uniform(0.0, 1.0, size=shape)).astype(np.float32)
        h5_file = tmp_path / f"{prefix}_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=data)
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([10.0 + idx], dtype=np.float32))
            base.create_dataset("Counter2", data=np.array([5.0 + idx], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([1000.0 + idx * 10.0], dtype=np.float64))
            base.create_dataset("Temp1", data=np.array([25.0 + idx], dtype=np.float32))
        files.append(h5_file)
    return files


def _write_background_series(tmp_path, shape=(200, 200), n_frames=1, level=2.0, prefix="BKG"):
    files = []
    for idx in range(n_frames):
        data = np.full(shape, level, dtype=np.float32)
        h5_file = tmp_path / f"{prefix}_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=data)
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([10.0], dtype=np.float32))
            base.create_dataset("Counter2", data=np.array([5.0], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([2000.0 + idx * 10.0], dtype=np.float64))
            base.create_dataset("Temp1", data=np.array([25.0], dtype=np.float32))
        files.append(h5_file)
    return files


def _load_series_into_widget(widget, tmp_path, **kwargs):
    files = _write_synthetic_series(tmp_path, **kwargs)
    widget._last_dir = str(tmp_path)
    result = ts.trxs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert result["ok"], result.get("message")
    widget._render_image()
    return files


def _qcal_sync(widget, monkeypatch):
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_qcal_clicked()


def _circular_exec_sync(widget, monkeypatch):
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_circular_exec_clicked()


def _line_exec_sync(widget, monkeypatch):
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_line_exec_clicked()


def _azimuthal_exec_sync(widget, monkeypatch):
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_azimuthal_exec_clicked()


def _auto_refresh_sync(widget, monkeypatch):
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_auto_refresh_clicked()


def _background_load_sync(widget, monkeypatch):
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_background_clicked()


def _simulate_roi_drawing(widget, monkeypatch, rectangles):
    def fake_exec(self):
        canvas = self.findChildren(ImageCanvas)[0]
        for x1, y1, x2, y2 in rectangles:
            selector = canvas._rectangle_selector
            selector.onselect(SimpleNamespace(xdata=x1, ydata=y1), SimpleNamespace(xdata=x2, ydata=y2))
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)


def test_finite_range_pads_apart_a_degenerate_range():
    assert _finite_range(np.array([3.0, 1.0, np.nan, 5.0])) == (1.0, 5.0)
    lo, hi = _finite_range(np.array([2.0, 2.0, np.nan]))
    assert lo < 2.0 < hi
    assert _finite_range(np.array([np.nan, np.inf])) == (0.0, 1.0)


def test_trxs2_widget_constructs(qapp):
    widget = TrXS2Widget()
    assert widget.detector_combo.count() == 4
    assert widget.expmode_combo.count() == 2
    assert widget.expmode_combo.currentText() == "Time"
    assert widget.tmode_combo.count() == 3
    assert not widget.parallel_check.isChecked()
    assert not widget.subtract_check.isChecked()


def test_trxs2_widget_no_incidence_angle_controls(qapp):
    """TrXS is normal-incidence transmission -- confirm the widget has no
    incidence-angle/Change-ai controls at all, unlike GIXS2Widget."""

    widget = TrXS2Widget()
    assert not hasattr(widget, "ai_deg_spin")
    assert not hasattr(widget, "change_ai_check")


def test_trxs2_widget_directory_field_seeds_last_dir(qapp, tmp_path):
    widget = TrXS2Widget()
    widget._set_last_dir(str(tmp_path))
    assert widget.dir_edit.text() == str(tmp_path)
    assert widget._last_dir == str(tmp_path)


def test_trxs2_widget_load_series_updates_image_and_status(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    files = _load_series_into_widget(widget, tmp_path, n_frames=3)

    widget._on_series_loaded({"ok": True, "n_files": len(files), "shape": [3, 200, 200], "warnings": []})

    assert "Loaded 3 files" in widget.status_label.text()
    assert widget._current_image_data is not None
    assert widget.set_table_button.property("state") == "ok"


def test_trxs2_widget_load_data_marks_button_ok_and_resets_downstream_buttons(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    widget._last_dir = str(tmp_path)
    widget.detector_combo.addItem("TestDetector")
    widget.detector_combo.setCurrentText("TestDetector")
    files = _write_synthetic_series(tmp_path, n_frames=1)

    widget._mark_ok(widget.qcal_button, True)
    widget._mark_ok(widget.grid_remove_button, True)
    widget._mark_ok(widget.set_table_button, True)

    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(files[0]), "")),
    )

    widget._on_load_clicked()

    assert widget.load_button.property("state") == "ok"
    assert widget.qcal_button.property("state") is None
    assert widget.grid_remove_button.property("state") is None
    assert widget.set_table_button.property("state") is None


def test_trxs2_widget_load_data_failure_marks_status_error(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    widget._last_dir = str(tmp_path)
    missing_file = tmp_path / "does_not_exist.h5"

    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(missing_file), "")),
    )

    widget._on_load_clicked()

    assert widget.load_button.property("state") is None
    assert widget.status_label.property("role") == "status-error"


def test_trxs2_widget_load_info_marks_button_ok_and_recenters_axis(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1, shape=(200, 200))
    assert not widget._pixel_centered

    calibration_path = tmp_path / "cal.json"
    from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json

    save_standard_calibration_json(
        calibration_path,
        StandardCalibration(
            beam_center=np.array([100.0, 100.0]), wavelength=1.0, pixel_size=0.075, sdd=300.0,
            detector="Eiger4M", energy_kev=9.5, standard_peak=np.array([100.0]), standard_q_value=0.1,
            detector_index="eiger4m", image_shape=np.array([200.0, 200.0]),
        ),
    )
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(calibration_path), "")),
    )

    widget._on_load_info_clicked()

    assert widget.load_info_button.property("state") == "ok"
    assert widget._pixel_centered
    assert "Loaded Info" in widget.status_label.text()


def test_trxs2_widget_qcal_circular_exec_and_background_subtraction_end_to_end(qapp, tmp_path, monkeypatch):
    """Core end-to-end flow: Load -> q-Cal -> Circular Exec -> Background
    load -> live Subtraction toggle actually changes the plotted curve, with
    no separate recompute step."""

    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True

    _qcal_sync(widget, monkeypatch)
    assert "q-Cal applied" in widget.status_label.text()
    assert widget.qcal_button.property("state") == "ok"

    _circular_exec_sync(widget, monkeypatch)
    assert "Circular series" in widget.status_label.text()
    assert session.last_1d is not None
    assert len(widget.oned_canvas._data_x) > 0

    bkg_files = _write_background_series(tmp_path, shape=(200, 200), level=2.0)
    widget._last_dir = str(tmp_path)
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(bkg_files[0]), "")),
    )
    _background_load_sync(widget, monkeypatch)
    assert widget.background_button.property("state") == "ok"
    assert session.background is not None

    # Re-run Circular Exec so bkg_1d gets cut through the same radius map.
    _circular_exec_sync(widget, monkeypatch)
    assert session.bkg_1d is not None

    pre_subtract_y = widget.oned_canvas.ax.get_lines()[0].get_ydata().copy()
    widget.subtract_check.setChecked(True)
    post_subtract_y = widget.oned_canvas.ax.get_lines()[0].get_ydata().copy()

    assert not np.allclose(pre_subtract_y, post_subtract_y)

    widget.subtract_check.setChecked(False)
    back_y = widget.oned_canvas.ax.get_lines()[0].get_ydata().copy()
    np.testing.assert_allclose(back_y, pre_subtract_y)


def test_trxs2_widget_normalize_1d_combo_applies_live(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _circular_exec_sync(widget, monkeypatch)

    # The last frame: IC is relative to the first frame's counter (as in
    # MATLAB), so frame 1 itself is unchanged by normalizing.
    none_y = widget.oned_canvas.ax.get_lines()[2].get_ydata().copy()
    widget.normalize_1d_combo.setCurrentText("IC")
    ic_y = widget.oned_canvas.ax.get_lines()[2].get_ydata().copy()
    assert not np.allclose(ic_y, none_y)

    widget.normalize_1d_combo.setCurrentText("BC")
    bc_y = widget.oned_canvas.ax.get_lines()[2].get_ydata().copy()
    assert not np.allclose(bc_y, ic_y)

    widget.normalize_1d_combo.setCurrentText("None")
    np.testing.assert_allclose(widget.oned_canvas.ax.get_lines()[2].get_ydata(), none_y)


def test_trxs2_widget_line_exec_clears_std_data(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)

    widget.circ_std_check.setChecked(True)
    _circular_exec_sync(widget, monkeypatch)
    assert session.last_1d["std"] is not None

    widget.line_angle_spin.setValue(90.0)
    _line_exec_sync(widget, monkeypatch)

    assert "1D Line series" in widget.status_label.text()
    assert session.last_1d["std"] is None


def test_trxs2_widget_azimuthal_exec_and_subtraction(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)

    widget.azi_pixel_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    _azimuthal_exec_sync(widget, monkeypatch)

    assert "Azimuthal series" in widget.status_label.text()
    assert session.last_azimuthal is not None
    assert len(widget.azi_canvas._data_x) > 0

    # Azimuthal's own Deg Min-Max default (0-360, not GIXS2's 0-180).
    assert widget.azi_degmin_spin.value() == pytest.approx(0.0)
    assert widget.azi_degmax_spin.value() == pytest.approx(360.0)


def test_trxs2_widget_roi_exec_ignores_subtraction(qapp, tmp_path, monkeypatch):
    """Confirms ROI never subtracts background -- matches MATLAB's own
    ROIdef2_Callback/Saveget4_Callback, which have no background-related
    code at all."""

    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)
    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0)])

    widget._on_draw_roi_clicked()

    assert "ROI series" in widget.status_label.text()
    session = ts.get_session()
    assert session.last_roi is not None
    unsubtracted = session.last_roi.copy()

    widget.subtract_check.setChecked(True)
    np.testing.assert_array_equal(session.last_roi, unsubtracted)


def test_trxs2_widget_play_requires_data_loaded(qapp):
    ts.reset_session()
    widget = TrXS2Widget()
    widget._on_play_clicked()
    assert widget.status_label.property("role") == "status-error"


def test_trxs2_widget_play_opens_frame_viewer_with_locked_contrast(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=4)
    widget.image_cmin_spin.setValue(1.0)
    widget.image_cmax_spin.setValue(100.0)

    widget._on_play_clicked()

    dialog = widget._play_dialog
    assert dialog is not None
    from PySide6.QtWidgets import QSlider

    slider = dialog.findChildren(QSlider)[0]
    assert slider.minimum() == 1
    assert slider.maximum() == 4

    canvas = dialog.findChildren(ImageCanvas)[0]
    assert canvas._image.norm.vmin == pytest.approx(1.0)
    assert canvas._image.norm.vmax == pytest.approx(100.0)

    slider.setValue(3)
    from PySide6.QtWidgets import QLabel

    label = dialog.findChild(QLabel, "play_frame_label")
    assert "3 / 4" in label.text()

    dialog.close()


def test_trxs2_widget_heatmap_buttons_require_exec_first(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

    widget._on_1d_heatmap_clicked()
    assert widget.status_label.property("role") == "status-error"
    assert widget._heatmap_dialog is None

    widget._on_azi_heatmap_clicked()
    assert widget.status_label.property("role") == "status-error"
    assert widget._heatmap_dialog is None


def test_trxs2_widget_1d_heatmap_opens_after_circular_exec(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _circular_exec_sync(widget, monkeypatch)

    widget._on_1d_heatmap_clicked()

    assert widget._heatmap_dialog is not None
    widget._heatmap_dialog.close()


def test_trxs2_widget_heatmap_save_image_writes_a_png(qapp, tmp_path, monkeypatch):
    """Regression: "heatmap의 image를 세이브하게 만들어줘" -- the Heatmap
    dialog only had matplotlib's own small toolbar save icon; add an
    explicit "Save Image" button matching this app's usual save-button
    convention, wired to a real file save dialog + ImageCanvas.save_figure."""

    from PySide6.QtWidgets import QFileDialog, QPushButton

    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _circular_exec_sync(widget, monkeypatch)

    widget._on_1d_heatmap_clicked()
    dialog = widget._heatmap_dialog
    assert dialog is not None

    save_button = next(
        b for b in dialog.findChildren(QPushButton) if b.text() == "Save Image"
    )
    out_path = tmp_path / "heatmap_export.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))

    save_button.click()

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    dialog.close()


def test_trxs2_widget_draw_roi_converts_centered_drag_to_raw_pixel(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1, shape=(200, 200))

    calibration_path = tmp_path / "cal.json"
    from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json

    save_standard_calibration_json(
        calibration_path,
        StandardCalibration(
            beam_center=np.array([100.0, 100.0]), wavelength=1.0, pixel_size=0.075, sdd=300.0,
            detector="Eiger4M", energy_kev=9.5, standard_peak=np.array([100.0]), standard_q_value=0.1,
            detector_index="eiger4m", image_shape=np.array([200.0, 200.0]),
        ),
    )
    monkeypatch.setattr(
        "pgixs_python.desktop.trxs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(calibration_path), "")),
    )
    widget._on_load_info_clicked()
    assert widget._pixel_centered

    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(-10.0, -10.0, 10.0, 10.0)])

    widget._on_draw_roi_clicked()

    session = ts.get_session()
    assert session.roi_list == [(90, 110, 90, 110)]


def test_trxs2_widget_circular_overlay_hidden_before_calibration(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

    assert len(widget.image_canvas._overlay_artists) == 0

    widget.circ_angle_min_spin.setValue(30.0)
    assert len(widget.image_canvas._overlay_artists) == 0


def test_trxs2_widget_circular_overlay_draws_and_updates_live(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    ts.get_session().corrected = True

    widget.circ_angle_min_spin.setValue(30.0)

    assert len(widget.image_canvas._overlay_artists) == 2

    # A full 0-360 sweep has nothing meaningful to draw.
    widget.circ_angle_min_spin.setValue(0.0)
    widget.circ_angle_max_spin.setValue(360.0)
    assert len(widget.image_canvas._overlay_artists) == 0


def test_trxs2_widget_line_overlay_draws_single_line(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    ts.get_session().corrected = True

    widget.line_angle_spin.setValue(45.0)

    assert len(widget.image_canvas._overlay_artists) == 1


def test_trxs2_widget_azimuthal_position_button_click_sets_pix_and_arms_overlay(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=1, center=center, radii=[60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    session.corrected = True

    widget._arm_position_mode("azimuthal")
    assert widget._position_mode == "azimuthal"
    assert len(widget.image_canvas._overlay_artists) == 0

    widget._on_image_clicked(140.5, 100.5)  # 40 px right of the beam center

    assert widget.azi_pixel_spin.value() == pytest.approx(40.0, abs=0.01)
    assert widget._active_overlay == "azimuthal"
    assert len(widget.image_canvas._overlay_artists) == 2

    widget.image_canvas.toolbar.mode = "pan/zoom"
    widget._on_image_clicked(999.0, 999.0)
    assert widget.azi_pixel_spin.value() == pytest.approx(40.0, abs=0.01)


def test_trxs2_widget_line_position_button_click_sets_angle_and_arms_overlay(qapp, tmp_path):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=1, center=center, radii=[60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    session.corrected = True

    widget._arm_position_mode("line")
    assert len(widget.image_canvas._overlay_artists) == 0

    widget._on_image_clicked(60.5, 140.5)  # below-left of beam center

    assert widget._active_overlay == "line"
    assert len(widget.image_canvas._overlay_artists) == 1


def test_trxs2_widget_overlay_exclusivity_across_modes(qapp, tmp_path):
    """Only one process-cut overlay is ever shown at a time (circular, line,
    or azimuthal) -- switching modes replaces, never stacks."""

    from matplotlib.patches import Circle

    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    ts.get_session().corrected = True

    widget.line_angle_spin.setValue(45.0)
    assert widget._active_overlay == "line"
    assert len(widget.image_canvas._overlay_artists) == 1

    widget.azi_pixel_spin.setValue(60.0)
    assert widget._active_overlay == "azimuthal"
    artists = widget.image_canvas._overlay_artists
    assert len(artists) == 2
    assert all(isinstance(a, Circle) for a in artists)

    widget.circ_angle_min_spin.setValue(30.0)
    assert widget._active_overlay == "circular"
    assert not any(isinstance(a, Circle) for a in widget.image_canvas._overlay_artists)


def test_trxs2_widget_circular_mask_draw_and_clear(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1, shape=(60, 60))

    # Draw Mask now opens its own dedicated window (a fresh ImageCanvas,
    # not the small "2D Image Plot" quadrant) -- patch the class method so
    # it intercepts whichever ImageCanvas instance the dialog creates.
    captured = {}
    monkeypatch.setattr(
        ImageCanvas, "start_polygon_selector", lambda self, cb: captured.setdefault("cb", cb)
    )
    widget._on_circ_mask_draw_clicked()
    assert "cb" in captured
    assert widget._mask_dialog is not None
    assert isinstance(widget._mask_dialog, QDialog)

    vertices = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    captured["cb"](vertices)

    assert "px kept" in widget.circ_mask_status_label.text()
    session = ts.get_session()
    assert session.circular_mask is not None
    assert not session.circular_mask.all()

    widget._on_circ_mask_clear_clicked()
    assert widget.circ_mask_status_label.text() == "Mask: none"
    assert ts.get_session().circular_mask is None


def test_trxs2_widget_auto_refresh_extends_circular_series(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _circular_exec_sync(widget, monkeypatch)
    assert session.last_1d["intensity"].shape[0] == 2

    _write_synthetic_series(tmp_path, n_frames=3, center=center, radii=[60.0] * 3)

    _auto_refresh_sync(widget, monkeypatch)

    assert "Auto Refresh" in widget.status_label.text()
    assert session.last_1d["intensity"].shape[0] == 3
    assert widget.oned_canvas._data_x.shape[0] > 0


def test_trxs2_widget_auto_refresh_no_new_files(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    _auto_refresh_sync(widget, monkeypatch)

    assert "No new files" in widget.status_label.text()
    assert widget._progress_dialog is None
    assert widget.auto_refresh_button.isEnabled()


def test_trxs2_widget_save_1d_azi_integ_write_matching_png(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _circular_exec_sync(widget, monkeypatch)
    widget.azi_pixel_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    _azimuthal_exec_sync(widget, monkeypatch)
    widget._last_dir = str(tmp_path)

    widget.save_1d_button.click()
    csv_1d = tmp_path / "ABCD" / "ABCD_1D.csv"
    png_1d = tmp_path / "ABCD" / "ABCD_1D.png"
    assert csv_1d.exists()
    assert png_1d.exists()

    widget.save_azi_button.click()
    csv_azi = tmp_path / "ABCD" / "ABCD_Azi.csv"
    png_azi = tmp_path / "ABCD" / "ABCD_Azi.png"
    assert csv_azi.exists()
    assert png_azi.exists()

    widget.save_integ_button.click()
    integ_result = ts.trxs2_save_integration({"directory": str(tmp_path)})
    assert Path(integ_result["path"]).exists()

    widget.save_xaxis_button.click()
    xaxis_csv = tmp_path / "ABCD" / "ABCD_XAxis.csv"
    # Directory listing, not Path.exists(): exists() ignores case on Windows
    # and macOS, so it would accept a wrong-cased name here and only fail on
    # Linux (as the GIXS-series equivalent of this assertion did).
    assert xaxis_csv.name in {p.name for p in xaxis_csv.parent.iterdir()}


def test_trxs2_widget_save_1d_interleaves_std_columns(qapp, tmp_path, monkeypatch):
    """Regression: "Save 1D STD를 따로 세이브 하기보다는... Save 1D에서
    Column을 q, 1st data, 1st std, 2nd data, 2nd std 식으로 세이브 되면
    좋을 것 같아" -- Save 1D alone (no separate Save 1D STD button) writes
    interleaved data/std columns once Circular Exec's Std checkbox ran."""

    ts.reset_session()
    widget = TrXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)

    widget.circ_std_check.setChecked(True)
    _circular_exec_sync(widget, monkeypatch)

    widget._last_dir = str(tmp_path)
    widget.save_1d_button.click()
    csv_1d = tmp_path / "ABCD" / "ABCD_1D.csv"
    assert csv_1d.exists()
    header = csv_1d.read_text().splitlines()[0].split(",")
    # Column labels come from the series' own X-axis (Temperature here,
    # already set by rendering) rather than the "frame1"/"frame2" fallback
    # -- each data column is still immediately followed by its own _STD one.
    assert header[0] == "Q_1/A"
    assert len(header) == 1 + 2 * session.last_1d["rendered"].shape[0]
    for i, label in enumerate(header[1:]):
        if i % 2 == 0:
            assert not label.endswith("_STD")
        else:
            assert label == f"{header[1 + i - 1]}_STD"
    assert not (tmp_path / "ABCD" / "ABCD_1D_STD.csv").exists()


def test_trxs2_widget_four_quadrants_have_equal_stretch(qapp):
    ts.reset_session()
    widget = TrXS2Widget()
    widget.resize(1400, 900)
    widget.show()
    outer, top, bottom = widget._quadrant_splitters

    for splitter in (outer, top, bottom):
        assert splitter.count() == 2
        sizes = splitter.sizes()
        assert sizes[0] == sizes[1] > 0


def test_trxs2_widget_set_table_lists_files_and_imports_only_checked(qapp, tmp_path, monkeypatch):
    ts.reset_session()
    files = _write_synthetic_series(tmp_path, n_frames=3, prefix="ABCD")
    widget = TrXS2Widget()
    widget._last_dir = str(tmp_path)
    widget.filename_edit.setText("ABCD")

    captured = {}

    def fake_exec(self):
        from PySide6.QtWidgets import QListWidget

        list_widget = self.findChildren(QListWidget)[0]
        assert list_widget.count() == 3
        list_widget.item(1).setCheckState(Qt.Unchecked)
        captured["names"] = [list_widget.item(i).text() for i in range(3)]
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    started = {}
    monkeypatch.setattr(widget, "_start_series_load", lambda filenames: started.setdefault("filenames", filenames))

    widget._on_set_table_clicked()

    assert set(captured["names"]) == {f.name for f in files}
    assert started["filenames"] == [captured["names"][0], captured["names"][2]]
