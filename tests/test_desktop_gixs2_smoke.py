"""Smoke tests for the GIXS2 tab -- mirrors test_desktop_sddcal_smoke.py's
conventions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from pgixs_python.desktop.gixs2_widget import GIXS2Widget
from pgixs_python.ui_backend import gixs2_session as gs


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


def _load_series_into_widget(widget, tmp_path, **kwargs):
    files = _write_synthetic_series(tmp_path, **kwargs)
    widget._last_dir = str(tmp_path)
    result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert result["ok"], result.get("message")
    widget._render_image()
    return files


def _exec_1d_sync(widget):
    """`_on_exec_1d_clicked` now runs `gixs2_linecut_exec` on a background
    QThread (with a progress dialog) instead of calling it inline -- for
    tests that only care about the end state, replay the same payload
    synchronously rather than pumping the Qt event loop to wait for a
    real thread. `test_gixs2_widget_exec_1d_runs_in_background` below
    covers the actual threading/progress-dialog behavior."""

    payload = {
        "angle_deg": widget.phi_spin.value(),
        "cakecut": widget.cakecut_check.isChecked(),
        "range_minus": widget.cakecut_minus_spin.value(),
        "range_plus": widget.cakecut_plus_spin.value(),
        "y_pos_move": widget.y_pos_move_spin.value(),
        "z_pos_move": widget.z_pos_move_spin.value(),
        "accumulation": widget.accum_check.isChecked(),
        "accum_minus": widget.accum_minus_spin.value(),
        "accum_plus": widget.accum_plus_spin.value(),
    }
    result = gs.gixs2_linecut_exec(payload)
    widget._on_exec_1d_finished(result)
    return result


def test_gixs2_widget_constructs(qapp):
    widget = GIXS2Widget()
    assert widget.detector_combo.count() == 4
    assert widget.insitu_combo.currentText() == "Time"
    assert not widget.parallel_check.isChecked()


def test_gixs2_widget_start_series_load_passes_parallel_flag(qapp, tmp_path, monkeypatch):
    widget = GIXS2Widget()
    widget._last_dir = str(tmp_path)
    widget.parallel_check.setChecked(True)

    captured = {}
    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.run_in_background",
        lambda fn, payload, callback: captured.setdefault("payload", payload),
    )

    widget._start_series_load(["a.h5", "b.h5"])

    assert captured["payload"]["parallel"] is True


def test_gixs2_widget_directory_field_seeds_last_dir(qapp, tmp_path):
    widget = GIXS2Widget()
    widget._set_last_dir(str(tmp_path))
    assert widget.dir_edit.text() == str(tmp_path)
    assert widget._last_dir == str(tmp_path)


def test_gixs2_widget_insitu_mode_updates_sub_mode_items(qapp):
    widget = GIXS2Widget()
    widget.insitu_combo.setCurrentText("Temperature")
    assert widget.sub_mode_combo.count() == 4
    widget.insitu_combo.setCurrentText("Time")
    assert widget.sub_mode_combo.count() == 1


def test_gixs2_widget_load_series_updates_image_and_status(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    files = _load_series_into_widget(widget, tmp_path, n_frames=3)

    widget._on_series_loaded({"ok": True, "n_files": len(files), "shape": [3, 200, 200], "warnings": []})

    assert "Loaded 3 files" in widget.status_label.text()
    assert widget._current_image_data is not None
    assert widget.set_table_button.property("state") == "ok"


def test_gixs2_widget_load_data_marks_button_ok_and_resets_downstream_buttons(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    widget._last_dir = str(tmp_path)
    # A named real detector's mask shape won't match this synthetic
    # 200x200 frame -- "TestDetector" hits the unknown-detector fallback
    # (no mask applied at all), same convention test_desktop_sddcal_smoke.py uses.
    widget.detector_combo.addItem("TestDetector")
    widget.detector_combo.setCurrentText("TestDetector")
    files = _write_synthetic_series(tmp_path, n_frames=1)

    # Pretend q-Cal/Grid Remove/Set-Table already succeeded once, from a
    # previous dataset -- a fresh LOAD DATA must reset all three, matching
    # MATLAB's Openimage2_Callback resetting GUI.btnQcal/bGrid.
    widget._mark_ok(widget.qcal_button, True)
    widget._mark_ok(widget.grid_remove_button, True)
    widget._mark_ok(widget.set_table_button, True)

    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(files[0]), "")),
    )

    widget._on_load_clicked()

    assert widget.load_button.property("state") == "ok"
    assert widget.qcal_button.property("state") is None
    assert widget.grid_remove_button.property("state") is None
    assert widget.set_table_button.property("state") is None


def test_gixs2_widget_load_data_failure_marks_button_not_ok(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    widget._last_dir = str(tmp_path)
    missing_file = tmp_path / "does_not_exist.h5"

    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(missing_file), "")),
    )

    widget._on_load_clicked()

    assert widget.load_button.property("state") is None
    assert widget.status_label.property("role") == "status-error"


def test_gixs2_widget_load_info_dialog_restricted_to_json(qapp, monkeypatch):
    widget = GIXS2Widget()
    captured = {}

    def fake_get_open_file_name(*args, **kwargs):
        captured["filter"] = args[3] if len(args) > 3 else kwargs.get("filter")
        return "", ""

    monkeypatch.setattr("pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName", fake_get_open_file_name)
    widget._on_load_info_clicked()

    assert captured["filter"] == "Calibration (*.json)"


def test_gixs2_widget_load_info_marks_button_ok_on_success(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

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
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(calibration_path), "")),
    )

    widget._on_load_info_clicked()

    assert widget.load_info_button.property("state") == "ok"
    assert "Loaded Info" in widget.status_label.text()


def test_gixs2_widget_load_info_recenters_pixel_axis(qapp, tmp_path, monkeypatch):
    """Port of GIXS_gui.m Openinfo_Callback's beam-centering, requested for
    GIXS-series too: after Load Info, the 2D image's pixel axis re-centers
    on the beam position instead of reading raw detector row/column
    indices, matching console_widget.py's single-frame "GIXS" behavior."""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1, shape=(200, 200))
    assert not widget._pixel_centered
    raw_extent = list(widget.image_canvas._image.get_extent())
    assert raw_extent == [1, 200, 200, 1]

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
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(calibration_path), "")),
    )

    widget._on_load_info_clicked()

    assert widget._pixel_centered
    beam_y, beam_z = gs.get_session().calibration["beam_pos"]
    centered_extent = list(widget.image_canvas._image.get_extent())
    assert centered_extent == pytest.approx([1 - beam_y, 200 - beam_y, beam_z - 200, beam_z - 1])

    # A fresh LOAD DATA invalidates it again, back to raw pixel indices.
    # "TestDetector" hits the unknown-detector fallback (no mask applied)
    # so this synthetic 200x200 frame loads successfully, same convention
    # test_gixs2_widget_load_data_marks_button_ok_and_resets_downstream_buttons uses.
    widget.detector_combo.addItem("TestDetector")
    widget.detector_combo.setCurrentText("TestDetector")
    files = _write_synthetic_series(tmp_path, n_frames=1, shape=(200, 200))
    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(files[0]), "")),
    )
    widget._on_load_clicked()
    assert not widget._pixel_centered


def test_gixs2_widget_roi_click_converts_centered_click_to_raw_pixel(qapp, tmp_path, monkeypatch):
    """ROI rectangles are stored/consumed as raw array indices
    (`process_roi_stack`) -- once Load Info has recentered the display,
    a click's displayed (x, y) must be converted back to raw pixel
    row/col before building the ROI tuple, or ROI selection silently
    grabs the wrong region of the detector."""

    gs.reset_session()
    widget = GIXS2Widget()
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
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(calibration_path), "")),
    )
    widget._on_load_info_clicked()
    assert widget._pixel_centered

    widget.n_roi_combo.setCurrentText("1")
    widget._on_draw_roi_clicked()
    # Beam center is (100, 100) -> displayed (0, 0) is raw pixel (100, 100).
    # Click displayed (-10, -10) then (10, 10): raw col = x + 100, raw row = 100 - y.
    widget._on_image_clicked(-10.0, -10.0)
    widget._on_image_clicked(10.0, 10.0)

    assert not widget._roi_click_mode
    session = gs.get_session()
    assert session.roi_list == [(90, 110, 90, 110)]


def test_gixs2_widget_load_info_failure_does_not_mark_button_ok(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    missing_path = tmp_path / "missing.json"

    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(missing_path), "")),
    )

    widget._on_load_info_clicked()

    assert widget.load_info_button.property("state") is None
    assert widget.status_label.property("role") == "status-error"


def test_gixs2_widget_qcal_and_exec_1d_end_to_end(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})

    widget._on_qcal_clicked()
    assert "q-Cal applied" in widget.status_label.text()

    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    assert "1D series" in widget.status_label.text()
    assert session.last_1d is not None
    assert len(widget.oned_canvas._data_x) > 0


def test_gixs2_widget_exec_1d_runs_in_background_with_progress_dialog(qapp, tmp_path, monkeypatch):
    """Exec Cut runs the per-frame 1D-cut loop on a background QThread
    (same pattern as series loading) and shows a progress dialog while it
    runs, rather than freezing the UI for however long a real series takes."""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

    captured = {}

    def fake_run_in_background(fn, payload, callback):
        captured["fn"] = fn
        captured["payload"] = payload

    monkeypatch.setattr("pgixs_python.desktop.gixs2_widget.run_in_background", fake_run_in_background)

    widget._on_exec_1d_clicked()

    assert captured["fn"] is gs.gixs2_linecut_exec
    assert captured["payload"]["angle_deg"] == widget.phi_spin.value()
    assert not widget.exec_1d_button.isEnabled()
    assert widget._progress_dialog is not None


def test_gixs2_widget_1d_overlay_hidden_before_calibration(qapp, tmp_path):
    """Before Load Info or q-Cal has actually run, `calibration["beam_pos"]`
    is still the session dataclass's own placeholder default -- drawing a
    "beam center" line through it right after a fresh LOAD DATA would show
    a confident-looking marker with no relation to this frame's real beam
    position, so the overlay must stay hidden until real calibration
    exists."""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

    assert len(widget.image_canvas._overlay_artists) == 0

    widget.phi_spin.setValue(45.0)
    assert len(widget.image_canvas._overlay_artists) == 0


def _mark_calibrated(widget) -> None:
    """Simulate Load Info/q-Cal having run (without needing real
    calibration math) for overlay tests that only care about the
    geometry-drawing logic once calibration exists."""

    gs.get_session().corrected = True
    widget._draw_1d_overlay()


def test_gixs2_widget_1d_overlay_draws_plain_line_and_updates_live(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    _mark_calibrated(widget)

    assert len(widget.image_canvas._overlay_artists) == 1

    widget.phi_spin.setValue(45.0)
    assert len(widget.image_canvas._overlay_artists) == 1


def test_gixs2_widget_1d_overlay_draws_cakecut_wedge(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    _mark_calibrated(widget)

    widget.cakecut_check.setChecked(True)

    assert len(widget.image_canvas._overlay_artists) == 2
    assert not widget.accum_check.isChecked()  # mutually exclusive


def test_gixs2_widget_1d_overlay_draws_accumulation_band(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    _mark_calibrated(widget)

    widget.accum_check.setChecked(True)

    assert len(widget.image_canvas._overlay_artists) == 2
    assert not widget.cakecut_check.isChecked()  # mutually exclusive


def test_gixs2_widget_1d_plot_uses_series_colormap_with_colorbar(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    widget._on_qcal_clicked()

    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    # 3 frames -> 3 plotted lines (excluding the hover marker), plus a
    # colorbar -- the series X-axis (here: "Time", from this series'
    # baselined timestamps) is recomputed live on every render, matching
    # MATLAB's own inline `getExperimentXAxis` call, no separate Set
    # X-Axis click needed first.
    assert widget.oned_canvas._colorbar is not None
    assert widget.oned_canvas._colorbar.ax.get_ylabel() == "Time"
    plotted_lines = [ln for ln in widget.oned_canvas.ax.get_lines() if ln is not widget.oned_canvas._hover_marker]
    assert len(plotted_lines) == 3


def test_gixs2_widget_1d_plot_colorbar_uses_x_axis_label_after_set_x_axis(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    widget._on_qcal_clicked()
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    widget._on_set_x_axis_clicked()

    assert widget.oned_canvas._colorbar.ax.get_ylabel() == session.x_axis_label


def test_gixs2_widget_1d_scale_combo_toggles_log_axes(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    widget._on_qcal_clicked()
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    assert widget.oned_canvas.ax.get_xscale() == "linear"
    assert widget.oned_canvas.ax.get_yscale() == "linear"

    widget.oned_scale_combo.setCurrentText("Logxy")

    assert widget.oned_canvas.ax.get_xscale() == "log"
    assert widget.oned_canvas.ax.get_yscale() == "log"


def test_gixs2_widget_azimuthal_plot_uses_series_colormap_with_colorbar(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])

    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    widget._on_exec_azi_clicked()

    # Same live X-axis recompute as the 1D Plot Viewer -- see
    # test_gixs2_widget_1d_plot_uses_series_colormap_with_colorbar.
    assert widget.azi_canvas._colorbar is not None
    assert widget.azi_canvas._colorbar.ax.get_ylabel() == "Time"
    plotted_lines = [ln for ln in widget.azi_canvas.ax.get_lines() if ln is not widget.azi_canvas._hover_marker]
    assert len(plotted_lines) == 3


def test_gixs2_widget_series_color_values_falls_back_to_frame_index_without_timestamps(qapp, tmp_path):
    """Guards the fallback path itself (no series loaded yet -- or more
    generally no timestamps available -- so a live X-axis can't be
    computed): color-code by a plain 1..n_frames index instead."""

    gs.reset_session()
    widget = GIXS2Widget()
    color_values, label = widget._series_color_values(4)

    assert label == "Frame, #"
    np.testing.assert_array_equal(color_values, [1.0, 2.0, 3.0, 4.0])


def test_gixs2_widget_azimuthal_plot_colorbar_uses_x_axis_label_after_set_x_axis(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    widget._on_exec_azi_clicked()

    widget._on_set_x_axis_clicked()

    assert widget.azi_canvas._colorbar.ax.get_ylabel() == session.x_axis_label


def test_gixs2_widget_set_x_axis_refreshes_both_1d_and_azimuthal_colorbars(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    widget._on_qcal_clicked()
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)
    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    widget._on_exec_azi_clicked()

    widget._on_set_x_axis_clicked()

    assert widget.oned_canvas._colorbar.ax.get_ylabel() == session.x_axis_label
    assert widget.azi_canvas._colorbar.ax.get_ylabel() == session.x_axis_label


def test_gixs2_widget_contrast_defaults_to_auto_and_reflects_computed_range(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    # 0/0 is the auto-contrast sentinel -- after a render, the fields
    # reflect back whatever ImageCanvas actually auto-computed instead of
    # staying at 0/0.
    assert widget.image_cmax_spin.value() > widget.image_cmin_spin.value()
    used_vmin = widget.image_canvas._image.norm.vmin
    used_vmax = widget.image_canvas._image.norm.vmax
    assert widget.image_cmin_spin.value() == pytest.approx(used_vmin)
    assert widget.image_cmax_spin.value() == pytest.approx(used_vmax)


def test_gixs2_widget_contrast_manual_override_is_applied(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    widget.image_cmin_spin.setValue(5.0)
    widget.image_cmax_spin.setValue(50.0)

    assert widget.image_canvas._image.norm.vmin == pytest.approx(5.0)
    assert widget.image_canvas._image.norm.vmax == pytest.approx(50.0)


def test_gixs2_widget_play_requires_data_loaded(qapp):
    gs.reset_session()
    widget = GIXS2Widget()
    widget._on_play_clicked()
    assert widget.status_label.property("role") == "status-error"


def test_gixs2_widget_play_opens_frame_viewer_with_locked_contrast(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
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

    from pgixs_python.desktop.plot_canvas import ImageCanvas

    canvas = dialog.findChildren(ImageCanvas)[0]
    assert canvas._image.norm.vmin == pytest.approx(1.0)
    assert canvas._image.norm.vmax == pytest.approx(100.0)

    slider.setValue(3)
    from PySide6.QtWidgets import QLabel

    label = dialog.findChild(QLabel, "play_frame_label")
    assert "3 / 4" in label.text()

    dialog.close()


def test_gixs2_widget_change_ai_checkbox_wires_into_qcal(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})

    widget.change_ai_check.setChecked(True)
    widget.ai_step_spin.setValue(0.05)
    widget._on_qcal_clicked()

    assert "Change ai" in widget.status_label.text()
    assert session.change_ai is True
    assert session.q_stack is not None
    assert session.qmap is None


def test_gixs2_widget_roi_click_flow(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)
    widget.n_roi_combo.setCurrentText("1")

    widget._on_draw_roi_clicked()
    assert widget._roi_click_mode

    widget._on_image_clicked(10.0, 10.0)
    assert widget._roi_first_corner is not None
    widget._on_image_clicked(20.0, 20.0)

    assert not widget._roi_click_mode
    assert "ROI series" in widget.status_label.text()
    assert gs.get_session().last_roi is not None


def test_gixs2_widget_set_x_axis_and_integration_plot(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    widget._on_qcal_clicked()
    _exec_1d_sync(widget)

    widget._on_set_x_axis_clicked()

    assert "X-Axis set" in widget.status_label.text()
    assert len(widget.integ_canvas._data_x) == 3


def test_gixs2_widget_auto_refresh_no_new_files(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    widget._on_auto_refresh_clicked()

    assert "No new files" in widget.status_label.text()


def test_gixs2_widget_ai_pix_deg_sync(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    gs.get_session().calibration.update({"sdd_mm": 300.0, "pixel_mm": 0.075})

    widget.ai_pix_spin.setValue(100.0)
    widget._on_ai_pix_changed()

    assert widget.ai_deg_spin.value() != 0.1  # moved from its default


def test_gixs2_widget_set_table_lists_files_and_imports_only_checked(qapp, tmp_path, monkeypatch):
    """Set/Table must list every file sharing the Filename prefix (each
    checked by default) and, on Import, load only whichever are still
    checked -- unchecking one file here must exclude it from the loaded
    series."""

    gs.reset_session()
    files = _write_synthetic_series(tmp_path, n_frames=3, prefix="ABCD")
    widget = GIXS2Widget()
    widget._last_dir = str(tmp_path)
    widget.filename_edit.setText("ABCD")

    captured = {}

    def fake_exec(self):
        from PySide6.QtWidgets import QListWidget

        list_widget = self.findChildren(QListWidget)[0]
        assert list_widget.count() == 3
        assert all(list_widget.item(i).checkState() == Qt.Checked for i in range(3))
        list_widget.item(1).setCheckState(Qt.Unchecked)  # uncheck the middle file
        captured["names"] = [list_widget.item(i).text() for i in range(3)]
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    started = {}
    monkeypatch.setattr(widget, "_start_series_load", lambda filenames: started.setdefault("filenames", filenames))

    widget._on_set_table_clicked()

    assert set(captured["names"]) == {f.name for f in files}
    assert started["filenames"] == [captured["names"][0], captured["names"][2]]


def test_gixs2_widget_set_table_select_all_and_deselect_all(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    _write_synthetic_series(tmp_path, n_frames=3, prefix="ABCD")
    widget = GIXS2Widget()
    widget._last_dir = str(tmp_path)
    widget.filename_edit.setText("ABCD")

    def fake_exec(self):
        from PySide6.QtWidgets import QListWidget

        list_widget = self.findChildren(QListWidget)[0]
        widget._set_all_checked(list_widget, False)
        assert all(list_widget.item(i).checkState() == Qt.Unchecked for i in range(3))
        widget._set_all_checked(list_widget, True)
        assert all(list_widget.item(i).checkState() == Qt.Checked for i in range(3))
        return QDialog.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    widget._on_set_table_clicked()
