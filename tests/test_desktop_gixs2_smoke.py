"""Smoke tests for the GIXS2 tab -- mirrors test_desktop_sddcal_smoke.py's
conventions."""

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

from pgixs_python.desktop.gixs2_widget import _ROI_COLORS, GIXS2Widget, _finite_range
from pgixs_python.desktop.plot_canvas import ImageCanvas
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
        "parallel": widget.exec_1d_parallel_check.isChecked(),
    }
    result = gs.gixs2_linecut_exec(payload)
    widget._on_exec_1d_finished(result)
    return result


def _exec_azi_sync(widget):
    """Same rationale as `_exec_1d_sync` -- `_on_exec_azi_clicked` now runs
    `gixs2_azimuthal_exec` on a background QThread too (with its own
    progress dialog). `test_gixs2_widget_exec_azi_runs_in_background_with_progress_dialog`
    below covers the actual threading/progress-dialog behavior."""

    payload = {
        "pixel": widget.azi_pix_spin.value(),
        "range_minus": widget.azi_minus_spin.value(),
        "range_plus": widget.azi_plus_spin.value(),
        "mean_or_median": widget.azi_mean_median_combo.currentText(),
        "reflect_missing_wedge": widget.azi_reflect_missing_wedge_check.isChecked(),
    }
    result = gs.gixs2_azimuthal_exec(payload)
    widget._on_exec_azi_finished(result)
    return result


def _auto_refresh_sync(widget, monkeypatch, payload=None):
    """`_on_auto_refresh_clicked` now runs `gixs2_auto_refresh` on a
    background QThread too (with a progress dialog, same as Exec Cut/Exec
    Azi) -- replay it synchronously for tests that only care about the end
    state. `test_gixs2_widget_auto_refresh_shows_progress_dialog` below
    covers the actual threading/progress-dialog behavior."""

    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_auto_refresh_clicked()


def _qcal_sync(widget, monkeypatch):
    """`_on_qcal_clicked` now runs `gixs2_qcal` on a background QThread too
    (with a progress dialog, same as Exec Cut/Exec Azi/Auto Refresh) --
    replay it synchronously for tests that only care about the end state.
    `test_gixs2_widget_qcal_shows_progress_dialog` below covers the actual
    threading/progress-dialog behavior."""

    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.run_in_background",
        lambda fn, call_payload, callback: callback(fn(call_payload)),
    )
    widget._on_qcal_clicked()


def _simulate_roi_drawing(widget, monkeypatch, rectangles):
    """Drive "Draw ROI"'s separate dialog through its sequential
    RectangleSelector flow (one drag per ROI, in display coordinates)
    without a real Qt event loop or mouse simulation -- `rectangles` is
    `[(x1, y1, x2, y2), ...]`, one tuple per ROI in order. Mirrors this
    file's existing QDialog.exec monkeypatch convention (see
    test_gixs2_widget_set_table_lists_files_and_imports_only_checked)."""

    def fake_exec(self):
        canvas = self.findChildren(ImageCanvas)[0]
        for x1, y1, x2, y2 in rectangles:
            selector = canvas._rectangle_selector
            selector.onselect(SimpleNamespace(xdata=x1, ydata=y1), SimpleNamespace(xdata=x2, ydata=y2))
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)


def test_finite_range_pads_apart_a_degenerate_range():
    """Port of MATLAB's own `finiteRange` -- seeds a Heatmap dialog's X/Y
    Range fields with the real data bounds (ignoring NaN/Inf), padding a
    zero-width range apart so min < max always holds for the caller's
    `xlim`/`ylim`. No `qapp` needed -- pure numpy, no Qt involved."""

    assert _finite_range(np.array([3.0, 1.0, np.nan, 5.0])) == (1.0, 5.0)

    lo, hi = _finite_range(np.array([2.0, 2.0, np.nan]))
    assert lo < 2.0 < hi

    assert _finite_range(np.array([np.nan, np.inf])) == (0.0, 1.0)


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
    # Regression: a white line through the beam center appeared right
    # after Load Info re-centered the axis, before the user ever asked to
    # see a 1D Plot Process cut -- q-Cal (which also satisfies the
    # calibration gate) must not auto-show it either; only actually
    # touching a 1D control should.
    assert len(widget.image_canvas._overlay_artists) == 0
    session = gs.get_session()
    session.calibration.update({"beam_pos": (beam_y, beam_z), "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    assert session.corrected
    assert len(widget.image_canvas._overlay_artists) == 0
    widget.phi_spin.setValue(45.0)
    assert len(widget.image_canvas._overlay_artists) == 1

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
    assert widget._active_overlay is None
    assert len(widget.image_canvas._overlay_artists) == 0


def test_gixs2_widget_draw_roi_converts_centered_drag_to_raw_pixel(qapp, tmp_path, monkeypatch):
    """ROI rectangles are stored/consumed as raw array indices
    (`process_roi_stack`) -- once Load Info has recentered the display, a
    drag's displayed (x, y) must be converted back to raw pixel row/col
    before building the ROI tuple, or ROI selection silently grabs the
    wrong region of the detector."""

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
    # Beam center is (100, 100) -> displayed (0, 0) is raw pixel (100, 100).
    # Drag displayed (-10, -10) to (10, 10): raw col = x + 100, raw row = 100 - y.
    _simulate_roi_drawing(widget, monkeypatch, [(-10.0, -10.0, 10.0, 10.0)])

    widget._on_draw_roi_clicked()

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


def test_gixs2_widget_qcal_and_exec_1d_end_to_end(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True

    _qcal_sync(widget, monkeypatch)
    assert "q-Cal applied" in widget.status_label.text()

    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    assert "1D series" in widget.status_label.text()
    assert session.last_1d is not None
    assert len(widget.oned_canvas._data_x) > 0


def test_gixs2_widget_exec_1d_and_azi_run_in_background_with_progress_dialog(qapp, tmp_path, monkeypatch):
    """Exec Cut and Exec Azi each run their per-frame loop on a background
    QThread (same pattern as series loading) and show a progress dialog
    while they run, rather than freezing the UI for however long a real
    series takes -- Exec Azi previously ran `gixs2_azimuthal_exec` inline
    with no progress feedback at all. Reuses one widget for both checks
    (rather than a `GIXS2Widget()` each) -- see
    test_gixs2_widget_exec_1d_shows_progress_dialog_for_every_geometry_mode's
    own docstring on why this suite favors reuse."""

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

    widget._stop_progress_dialog()
    widget.exec_1d_button.setEnabled(True)
    captured.clear()

    widget._on_exec_azi_clicked()

    assert captured["fn"] is gs.gixs2_azimuthal_exec
    assert captured["payload"]["pixel"] == widget.azi_pix_spin.value()
    assert captured["payload"]["reflect_missing_wedge"] is False
    assert not widget.exec_azi_button.isEnabled()
    assert widget._progress_dialog is not None

    widget._stop_progress_dialog()
    widget.exec_azi_button.setEnabled(True)
    captured.clear()

    widget.azi_reflect_missing_wedge_check.setChecked(True)
    widget._on_exec_azi_clicked()

    assert captured["payload"]["reflect_missing_wedge"] is True

    widget._stop_progress_dialog()


def test_gixs2_widget_exec_1d_shows_progress_dialog_for_every_geometry_mode(qapp, tmp_path, monkeypatch):
    """The background thread + progress dialog wraps Exec Cut itself, not
    any one geometry branch -- confirm it shows up whether the active mode
    is a plain angle cut, Cakecut, or Accumulation. Reuses one widget
    across all three modes (rather than one `GIXS2Widget()` per mode) --
    each instance owns 4 matplotlib canvases/toolbars, and this suite
    already constructs plenty of those elsewhere."""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

    captured = {}

    def fake_run_in_background(fn, payload, callback):
        captured["payload"] = payload

    monkeypatch.setattr("pgixs_python.desktop.gixs2_widget.run_in_background", fake_run_in_background)

    for mode in ("angle", "cakecut", "accum"):
        widget.cakecut_check.setChecked(mode == "cakecut")
        widget.accum_check.setChecked(mode == "accum")
        widget._stop_progress_dialog()
        widget.exec_1d_button.setEnabled(True)
        captured.clear()

        widget._on_exec_1d_clicked()

        assert captured["payload"]["cakecut"] == (mode == "cakecut")
        assert captured["payload"]["accumulation"] == (mode == "accum")
        assert widget._progress_dialog is not None

    widget._stop_progress_dialog()


def test_gixs2_widget_exec_1d_parallel_checkbox_wires_into_payload(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    widget.exec_1d_parallel_check.setChecked(True)

    captured = {}
    monkeypatch.setattr(
        "pgixs_python.desktop.gixs2_widget.run_in_background",
        lambda fn, payload, callback: captured.setdefault("payload", payload),
    )

    widget._on_exec_1d_clicked()

    assert captured["payload"]["parallel"] is True


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
    calibration math) *and* the user having engaged with the 1D Plot
    Process, for overlay tests that only care about the geometry-drawing
    logic once both gates are satisfied.
    `test_gixs2_widget_load_info_recenters_pixel_axis` covers the second
    gate (hidden until the user touches a 1D control) on its own."""

    gs.get_session().corrected = True
    widget._arm_1d_overlay()


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


def test_gixs2_widget_1d_plot_uses_series_colormap_with_colorbar(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)

    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    # 3 frames -> 3 plotted lines (excluding the hover marker), plus a
    # colorbar -- the series X-axis (here: "Time", from this series'
    # baselined timestamps) is recomputed live on every render, matching
    # MATLAB's own inline `getExperimentXAxis` call, no separate Set
    # X-Axis click needed first.
    assert widget.oned_canvas._colorbar is not None
    assert widget.oned_canvas._colorbar.ax.get_ylabel() == "Time, s"
    plotted_lines = [ln for ln in widget.oned_canvas.ax.get_lines() if ln is not widget.oned_canvas._hover_marker]
    assert len(plotted_lines) == 3

    # The Colormap combo (reported: "1D plot Viewer의 그래프와... colormap을
    # 바꿔 플랏 색상을 바꿀 수 있으면 좋을것 같아") defaults to viridis and
    # actually changes the plotted line colors when switched.
    assert widget.oned_colormap_combo.currentText() == "viridis"
    viridis_color = widget.oned_canvas.ax.get_lines()[0].get_color()

    widget.oned_colormap_combo.setCurrentText("magma")

    magma_color = widget.oned_canvas.ax.get_lines()[0].get_color()
    assert magma_color != viridis_color


def test_gixs2_widget_1d_plot_colorbar_uses_x_axis_label_after_set_x_axis(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    widget._on_set_x_axis_clicked()

    assert widget.oned_canvas._colorbar.ax.get_ylabel() == session.x_axis_label


def test_gixs2_widget_1d_scale_combo_toggles_log_axes(qapp, tmp_path, monkeypatch):
    """Also covers the newly-added Series Integration Viewer Scale combo
    (both its "sum" render path, exercised here via Exec Cut, and its
    "roi" render_multi path, exercised via Draw ROI) -- reuses this one
    widget rather than a `GIXS2Widget()` per scenario (see
    test_gixs2_widget_exec_1d_shows_progress_dialog_for_every_geometry_mode's
    own docstring on why)."""

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)

    assert widget.oned_canvas.ax.get_xscale() == "linear"
    assert widget.oned_canvas.ax.get_yscale() == "linear"
    assert widget.integ_canvas.ax.get_xscale() == "linear"
    assert widget.integ_canvas.ax.get_yscale() == "linear"

    widget.oned_scale_combo.setCurrentText("Logxy")

    assert widget.oned_canvas.ax.get_xscale() == "log"
    assert widget.oned_canvas.ax.get_yscale() == "log"
    # The 1D Plot Viewer's own Scale combo doesn't touch the Series
    # Integration Viewer -- that has its own, independent Scale combo.
    assert widget.integ_canvas.ax.get_xscale() == "linear"
    assert widget.integ_canvas.ax.get_yscale() == "linear"

    widget.integ_scale_combo.setCurrentText("Logxy")

    assert widget.integ_canvas.ax.get_xscale() == "log"
    assert widget.integ_canvas.ax.get_yscale() == "log"

    # And the render_multi (ROI) path honors it too.
    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0)])
    widget._on_draw_roi_clicked()

    assert widget.integ_canvas.ax.get_xscale() == "log"
    assert widget.integ_canvas.ax.get_yscale() == "log"

    widget.integ_scale_combo.setCurrentText("Linear")

    assert widget.integ_canvas.ax.get_xscale() == "linear"
    assert widget.integ_canvas.ax.get_yscale() == "linear"


def test_gixs2_widget_azimuthal_plot_uses_series_colormap_with_colorbar(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])

    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    _exec_azi_sync(widget)

    # Same live X-axis recompute as the 1D Plot Viewer -- see
    # test_gixs2_widget_1d_plot_uses_series_colormap_with_colorbar.
    assert widget.azi_canvas._colorbar is not None
    assert widget.azi_canvas._colorbar.ax.get_ylabel() == "Time, s"
    plotted_lines = [ln for ln in widget.azi_canvas.ax.get_lines() if ln is not widget.azi_canvas._hover_marker]
    assert len(plotted_lines) == 3

    # Same Colormap combo as the 1D Plot Viewer -- see
    # test_gixs2_widget_1d_plot_uses_series_colormap_with_colorbar.
    assert widget.azi_colormap_combo.currentText() == "viridis"
    viridis_color = widget.azi_canvas.ax.get_lines()[0].get_color()

    widget.azi_colormap_combo.setCurrentText("magma")

    magma_color = widget.azi_canvas.ax.get_lines()[0].get_color()
    assert magma_color != viridis_color


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
    _exec_azi_sync(widget)

    widget._on_set_x_axis_clicked()

    assert widget.azi_canvas._colorbar.ax.get_ylabel() == session.x_axis_label


def test_gixs2_widget_set_x_axis_refreshes_both_1d_and_azimuthal_colorbars(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)
    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    _exec_azi_sync(widget)

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


def test_gixs2_widget_change_ai_checkbox_wires_into_qcal(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True

    widget.change_ai_check.setChecked(True)
    widget.ai_step_spin.setValue(0.05)
    _qcal_sync(widget, monkeypatch)

    assert "Change ai" in widget.status_label.text()
    assert session.change_ai is True
    assert session.q_stack is not None
    assert session.qmap is None


def test_gixs2_widget_draw_roi_flow(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)
    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0)])

    widget._on_draw_roi_clicked()

    assert "ROI series" in widget.status_label.text()
    assert gs.get_session().last_roi is not None


def test_gixs2_widget_draw_roi_requires_data_loaded(qapp):
    gs.reset_session()
    widget = GIXS2Widget()

    widget._on_draw_roi_clicked()

    assert widget.status_label.property("role") == "status-error"


def test_gixs2_widget_draw_roi_cancelled_does_not_run_exec(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    widget.n_roi_combo.setCurrentText("2")
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Rejected)

    widget._on_draw_roi_clicked()

    assert "cancel" in widget.status_label.text().lower()
    assert gs.get_session().last_roi is None


def test_gixs2_widget_draw_roi_draws_matching_colored_box_and_curve(qapp, tmp_path, monkeypatch):
    """Port of MATLAB's `ROIdef2_Callback`/`drawROIOverlay`/`PlotROI_Callback`:
    each ROI gets its own fixed color, used both for its rectangle overlay
    on the 2D image and its curve in the Series Integration Viewer, so a
    box and its curve are visually linked. Also covers the ROI Normalize
    combo actually applying live (see below), reusing this same widget
    rather than constructing another."""

    import matplotlib.colors
    from matplotlib.patches import Rectangle

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=3)
    assert gs.gixs2_set_x_axis({})["ok"]
    widget.n_roi_combo.setCurrentText("2")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0), (30.0, 30.0, 40.0, 40.0)])

    widget._on_draw_roi_clicked()

    assert "ROI series" in widget.status_label.text()
    assert len(gs.get_session().roi_list) == 2

    boxes = [a for a in widget.image_canvas._overlay_artists if isinstance(a, Rectangle)]
    assert len(boxes) == 2
    assert boxes[0].get_edgecolor()[:3] == pytest.approx(matplotlib.colors.to_rgb(_ROI_COLORS[0]))
    assert boxes[1].get_edgecolor()[:3] == pytest.approx(matplotlib.colors.to_rgb(_ROI_COLORS[1]))

    lines = [ln for ln in widget.integ_canvas.ax.get_lines() if ln is not widget.integ_canvas._hover_marker]
    assert len(lines) == 2
    assert lines[0].get_color() == _ROI_COLORS[0]
    assert lines[1].get_color() == _ROI_COLORS[1]
    assert widget.integ_canvas.ax.get_legend() is not None

    # The ROI Normalize combo previously existed with no effect at all --
    # confirm switching it to IC/BC now actually changes the Series
    # Integration Viewer's curves (render_multi rebuilds the axes on every
    # call, so re-fetch the lines after each change rather than reusing
    # the stale `lines` references above).
    def first_curve_y():
        current = [ln for ln in widget.integ_canvas.ax.get_lines() if ln is not widget.integ_canvas._hover_marker]
        return current[0].get_ydata().copy()

    none_y = first_curve_y()

    widget.normalize_roi_combo.setCurrentText("IC")
    ic_y = first_curve_y()
    assert not np.allclose(ic_y, none_y)

    widget.normalize_roi_combo.setCurrentText("BC")
    bc_y = first_curve_y()
    assert not np.allclose(bc_y, ic_y)

    widget.normalize_roi_combo.setCurrentText("None")
    np.testing.assert_allclose(first_curve_y(), none_y)


def test_gixs2_widget_draw_roi_overlay_survives_image_rerender(qapp, tmp_path, monkeypatch):
    """`_render_image` fully rebuilds the figure/axes on every call (see
    ImageCanvas.render's own docstring), which would otherwise silently
    drop the ROI boxes -- confirm they get redrawn."""

    from matplotlib.patches import Rectangle

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0)])
    widget._on_draw_roi_clicked()

    widget._current_image_data = None
    widget._render_image()

    boxes = [a for a in widget.image_canvas._overlay_artists if isinstance(a, Rectangle)]
    assert len(boxes) == 1


def test_gixs2_widget_set_x_axis_and_integration_plot(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _exec_1d_sync(widget)

    widget._on_set_x_axis_clicked()

    assert "X-Axis set" in widget.status_label.text()
    assert len(widget.integ_canvas._data_x) == 3
    # Series Integration Viewer's x-axis label comes straight from
    # session.x_axis_label -- carries the unit (here "s") too, same as
    # the 1D/Azimuthal Plot Viewers' colorbars.
    assert widget.integ_canvas.ax.get_xlabel() == "Time, s"
    # Reported: "Series Integration Viewer에서는 각 포인트에 마커가 있으면
    # 보기 더 편할 것 같아" -- the sum-of-1D/Azimuthal curve (this test's
    # own auto_met == "linecut" path) now plots a marker at every point,
    # not just a bare line. The ROI path already had this via render_multi.
    (curve,) = [ln for ln in widget.integ_canvas.ax.get_lines() if ln is not widget.integ_canvas._hover_marker]
    assert curve.get_marker() == "o"


def test_gixs2_widget_save_1d_azi_integ_write_matching_png(qapp, tmp_path, monkeypatch):
    """Port of MATLAB's own `savePanelSnapshot` (called from Save 1D/Save
    Azi/Save Integ.'s callbacks): each data export now gets a same-named
    PNG of the panel's current plot alongside it, same convention as the
    single-frame console's own Save button (see
    test_curve_save_writes_csv_and_reports_full_paths in
    test_desktop_smoke.py). Reuses one widget across all three saves
    (rather than one per) -- this suite already constructs plenty of
    GIXS2Widget instances elsewhere."""

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    _exec_1d_sync(widget)
    _exec_azi_sync(widget)
    widget._on_set_x_axis_clicked()
    widget._last_dir = str(tmp_path)

    widget.save_1d_button.click()
    csv_1d = tmp_path / "ABCD" / "ABCD_1D.csv"
    png_1d = tmp_path / "ABCD" / "ABCD_1D.png"
    assert csv_1d.exists()
    assert png_1d.exists()
    # _wrap_long_tokens inserts zero-width spaces into long paths so they
    # wrap in the status label -- strip them before substring-checking.
    status_text = widget.status_label.text().replace(chr(0x200B), "")
    assert str(csv_1d) in status_text
    assert str(png_1d) in status_text

    widget.save_azi_button.click()
    csv_azi = tmp_path / "ABCD" / "ABCD_Azi.csv"
    png_azi = tmp_path / "ABCD" / "ABCD_Azi.png"
    assert csv_azi.exists()
    assert png_azi.exists()

    widget.save_integ_button.click()
    result = gs.gixs2_save_integration({"directory": str(tmp_path)})
    integ_png = Path(result["path"]).with_suffix(".png")
    assert Path(result["path"]).exists()
    assert integ_png.exists()

    # Save X-Axis has no single associated plot (MATLAB's own
    # SavegetX_Callback doesn't call savePanelSnapshot either) -- no PNG
    # alongside it, and no `canvas` argument in its own button wiring.
    widget.save_xaxis_button.click()
    xaxis_csv = tmp_path / "ABCD" / "ABCD_XAxis.csv"
    # Compared against the directory listing rather than Path.exists(): on a
    # case-insensitive filesystem exists() happily matches a name whose case
    # is wrong, which hid a misspelling here ("ABCD_Xaxis.csv") until the
    # suite first ran on Linux.
    assert xaxis_csv.name in {p.name for p in xaxis_csv.parent.iterdir()}
    assert not xaxis_csv.with_suffix(".png").exists()


def test_gixs2_widget_auto_refresh_no_new_files(qapp, tmp_path, monkeypatch):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    _auto_refresh_sync(widget, monkeypatch)

    assert "No new files" in widget.status_label.text()
    assert widget._progress_dialog is None
    assert widget.auto_refresh_button.isEnabled()


def test_gixs2_widget_auto_refresh_extends_roi_only_series_integration_plot(qapp, tmp_path, monkeypatch):
    """Port of MATLAB `Autoplot_Callback`'s `switch Auto.Met` dispatch
    (case AUTO_ROI -> autoUpdateROI -> PlotROI_Callback): a ROI-only
    workflow (Draw ROI, never Exec Cut/Exec Azi) has neither `last_1d` nor
    `last_azimuthal` set, so `_on_auto_refresh_clicked` must re-render the
    Series Integration Viewer directly rather than relying on `_render_1d`/
    `_render_azimuthal`'s own internal `_render_integration()` call, which
    would silently no-op here and leave the plot stuck on the pre-refresh
    frame count."""

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    assert gs.gixs2_set_x_axis({})["ok"]
    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0)])
    widget._on_draw_roi_clicked()
    assert gs.get_session().last_1d is None
    assert gs.get_session().last_azimuthal is None
    assert widget.integ_canvas._data_x.shape[0] == 2

    _write_synthetic_series(tmp_path, n_frames=3, center=center, radii=[60.0] * 3)

    _auto_refresh_sync(widget, monkeypatch)

    assert gs.get_session().last_roi.shape[0] == 3
    assert widget.integ_canvas._data_x.shape[0] == 3


def test_gixs2_widget_auto_refresh_shows_progress_dialog(qapp, tmp_path, monkeypatch):
    """Auto Refresh can load+correct+reprocess many new files -- it now
    runs `gixs2_auto_refresh` on a background QThread with a progress
    dialog (same pattern as series loading/Exec Cut/Exec Azi) instead of
    freezing the UI with no feedback while it works."""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    captured = {}

    def fake_run_in_background(fn, payload, callback):
        captured["fn"] = fn
        captured["payload"] = payload

    monkeypatch.setattr("pgixs_python.desktop.gixs2_widget.run_in_background", fake_run_in_background)

    widget._on_auto_refresh_clicked()

    assert captured["fn"] is gs.gixs2_auto_refresh
    assert captured["payload"]["directory"] == widget._last_dir
    assert not widget.auto_refresh_button.isEnabled()
    assert widget._progress_dialog is not None

    widget._stop_progress_dialog()
    widget.auto_refresh_button.setEnabled(True)


def test_gixs2_widget_qcal_shows_progress_dialog(qapp, tmp_path, monkeypatch):
    """q-Cal's "Change ai" branch recomputes a whole new q-map per frame
    (unlike its plain branch's single shared map) -- it now runs
    `gixs2_qcal` on a background QThread with a progress dialog (same
    pattern as series loading/Exec Cut/Exec Azi/Auto Refresh) instead of
    freezing the UI for however long that per-frame loop takes. Reported:
    "Change ai가 check가 되어 step이 있는 경우 q-map을 하나하나
    계산해야돼... process box가 있으면 좋을 것 같아.\""""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=2)

    captured = {}

    def fake_run_in_background(fn, payload, callback):
        captured["fn"] = fn
        captured["payload"] = payload

    monkeypatch.setattr("pgixs_python.desktop.gixs2_widget.run_in_background", fake_run_in_background)

    widget.change_ai_check.setChecked(True)
    widget._on_qcal_clicked()

    assert captured["fn"] is gs.gixs2_qcal
    assert captured["payload"]["change_ai"] is True
    assert not widget.qcal_button.isEnabled()
    assert widget._progress_dialog is not None

    widget._stop_progress_dialog()
    widget.qcal_button.setEnabled(True)


def test_gixs2_widget_four_quadrants_have_equal_stretch(qapp):
    """The 2D Image/1D Plot/Azimuthal/Series Integration viewers previously
    had no explicit sizes or stretch factors on their splitters, so their
    own size hints (a wider 2D Image row with its Contrast/Play controls,
    a wider Series Integration Viewer) left them visibly uneven, and that
    unevenness shifted with whatever window size the app happened to open
    at (reported: "이 크기가 맞았으면 좋겠어... 모니터에 따라 크기가 너무
    다른데"). All three splitters should now start at an equal 1:1 split
    with equal stretch factors, so the four quadrants stay a true grid
    across window sizes."""

    gs.reset_session()
    widget = GIXS2Widget()
    # A never-shown splitter falls back to distributing space by each
    # section's own minimum-size hint (the 2D Image row's wider Scale/
    # Colormap/Contrast/Play controls vs. the 1D Plot row's shorter one,
    # e.g.) instead of honoring the equal split -- give it a real,
    # realistic window size and an actual show() (needed for QSplitter to
    # run its own layout pass at all) first, matching how the app opens.
    widget.resize(1400, 900)
    widget.show()
    outer, top, bottom = widget._quadrant_splitters

    # QSplitter has no public getter for stretch factor -- `sizes()` (which
    # setStretchFactor's equal-split behavior on resize depends on having
    # started equal) is the directly observable half of the fix.
    for splitter in (outer, top, bottom):
        assert splitter.count() == 2
        sizes = splitter.sizes()
        assert sizes[0] == sizes[1] > 0


def test_gixs2_widget_ai_pix_deg_sync(qapp, tmp_path):
    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    gs.get_session().calibration.update({"sdd_mm": 300.0, "pixel_mm": 0.075})
    gs.get_session().calibration_loaded = True

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


def test_gixs2_widget_azimuthal_position_button_click_sets_pix_and_arms_overlay(qapp, tmp_path):
    """Port of console_widget.py's own Position button for the Azimuthal
    `pix` field: arm the click-on-image mode, then a click on the 2D Image
    Plot Viewer sets `azi_pix_spin` to the clicked point's radius from the
    beam center. `azi_pix_spin` is itself wired to `_arm_azi_overlay` (see
    the Azimuthal Process UI construction), so the click also makes the
    ring appear live."""

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=1, center=center, radii=[60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    session.corrected = True

    widget._arm_position_mode("azimuthal")
    assert widget._position_mode == "azimuthal"
    assert len(widget.image_canvas._overlay_artists) == 0  # position mode alone draws nothing

    widget._on_image_clicked(140.5, 100.5)  # 40 px right of the beam center

    assert widget.azi_pix_spin.value() == pytest.approx(40.0, abs=0.01)
    assert widget._active_overlay == "azimuthal"
    assert len(widget.image_canvas._overlay_artists) == 2

    # Ignored while the toolbar's own pan/zoom mode is active, matching
    # console_widget.py's own guard.
    widget.image_canvas.toolbar.mode = "pan/zoom"
    widget._on_image_clicked(999.0, 999.0)
    assert widget.azi_pix_spin.value() == pytest.approx(40.0, abs=0.01)


def test_gixs2_widget_1d_position_button_click_sets_angle_and_arms_overlay(qapp, tmp_path):
    """Port of console_widget.py's own Position button for the 1D Plot
    Process angle field: arm the click-on-image mode, then a click on the
    2D Image Plot Viewer sets `phi_spin` to the clicked point's angle from
    the beam center. `phi_spin` is itself wired to `_arm_1d_overlay` (see
    the 1D Plot Process UI construction), so the click also makes the line
    appear live.

    The clicked point (60.5, 140.5), below-left of the beam center
    (100.5, 100.5), computes to a raw angle of 315 deg -- exercises the
    fold-back-into-(-180,180] branch (phi_spin's own range, narrower than
    console's line_angle_spin) down to the equivalent -45 deg."""

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=1, center=center, radii=[60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    session.corrected = True

    widget._arm_position_mode("line")
    assert widget._position_mode == "line"
    assert len(widget.image_canvas._overlay_artists) == 0  # position mode alone draws nothing

    widget._on_image_clicked(60.5, 140.5)

    assert widget.phi_spin.value() == pytest.approx(-45.0, abs=0.01)
    assert widget._active_overlay == "1d"
    assert len(widget.image_canvas._overlay_artists) == 1

    # Ignored while the toolbar's own pan/zoom mode is active, matching
    # console_widget.py's own guard.
    widget.image_canvas.toolbar.mode = "pan/zoom"
    widget._on_image_clicked(999.0, 999.0)
    assert widget.phi_spin.value() == pytest.approx(-45.0, abs=0.01)


def test_gixs2_widget_azimuthal_overlay_hidden_until_armed_then_tracks_range_live(qapp, tmp_path):
    """Port of console_widget.py's own `_on_azi_draw_clicked`: two circles
    at pix+Range(-) and pix+Range(+), hidden until the user actually
    engages the Azimuthal Process controls or clicks Draw (same "armed"
    guard used for the 1D Plot Process overlay -- see `_active_overlay`'s
    own docstring), and tracking further pix/Range edits live without
    accumulating extra circles."""

    from matplotlib.patches import Circle

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=1, center=center, radii=[60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True

    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    # Armed, but not calibrated yet -- same two-gate guard as `_arm_1d_overlay`.
    assert len(widget.image_canvas._overlay_artists) == 0

    session.corrected = True
    widget._arm_azi_overlay()

    circles = [a for a in widget.image_canvas._overlay_artists if isinstance(a, Circle)]
    assert len(circles) == 2
    assert sorted(c.radius for c in circles) == pytest.approx([55.0, 65.0])

    widget.azi_plus_spin.setValue(10.0)
    circles = [a for a in widget.image_canvas._overlay_artists if isinstance(a, Circle)]
    assert len(circles) == 2
    assert sorted(c.radius for c in circles) == pytest.approx([55.0, 70.0])

    # Survives a full image re-render (ImageCanvas.render rebuilds the
    # whole figure -- see _redraw_overlays' own docstring).
    widget._render_image()
    circles = [a for a in widget.image_canvas._overlay_artists if isinstance(a, Circle)]
    assert len(circles) == 2
    assert sorted(c.radius for c in circles) == pytest.approx([55.0, 70.0])


def test_gixs2_widget_overlay_exclusivity_and_display_controls_clear_process_overlay(qapp, tmp_path, monkeypatch):
    """Two related regressions on one widget (reused across both checks --
    see test_gixs2_widget_exec_1d_shows_progress_dialog_for_every_geometry_mode's
    own docstring on why this suite favors reuse over one `GIXS2Widget()`
    per scenario):

    1. Doing Azimuthal Process work used to leave whatever 1D Plot Process
       line/wedge/band was already on screen still drawn underneath the new
       ring instead of replacing it. console_widget.py's own Line/Azimuthal
       Draw buttons each clear the image before drawing their own geometry
       -- only one process-cut overlay is ever shown at a time; GIXS2 now
       matches that (see `_active_overlay`'s own docstring).

    2. Scale/Colormap/Contrast are pure display changes on the 2D Image
       Plot Viewer, not a new cut/process/ROI request -- confirm they clear
       everything drawn on the image (1D/Azimuthal process overlay, ROI
       boxes alike) back to the plain image, and that redrawing ROI/
       touching a process control again brings each back.
    """

    from matplotlib.patches import Circle, Rectangle

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)
    gs.get_session().corrected = True

    widget.phi_spin.setValue(45.0)
    assert widget._active_overlay == "1d"
    assert len(widget.image_canvas._overlay_artists) == 1

    widget.azi_pix_spin.setValue(60.0)
    assert widget._active_overlay == "azimuthal"
    artists = widget.image_canvas._overlay_artists
    assert len(artists) == 2
    assert all(isinstance(a, Circle) for a in artists)  # the 1D line is gone

    widget.phi_spin.setValue(90.0)
    assert widget._active_overlay == "1d"
    assert len(widget.image_canvas._overlay_artists) == 1
    assert not any(isinstance(a, Circle) for a in widget.image_canvas._overlay_artists)

    widget.n_roi_combo.setCurrentText("1")
    _simulate_roi_drawing(widget, monkeypatch, [(10.0, 10.0, 20.0, 20.0)])
    widget._on_draw_roi_clicked()

    widget._arm_1d_overlay()
    assert len(widget.image_canvas._overlay_artists) == 3  # 1D line + ROI rect + ROI label

    widget.image_colormap_combo.setCurrentText("magma")

    # Reported: ROI boxes should clear on a display-only change too, not
    # just the 1D/Azimuthal process overlay -- back to the plain image.
    assert len(widget.image_canvas._overlay_artists) == 0

    # Any subsequent full redraw (touching a process control, Draw ROI,
    # Load Info, ...) brings everything currently defined back at once --
    # `session.roi_list` was never cleared, only its on-screen drawing.
    widget._arm_1d_overlay()
    assert len(widget.image_canvas._overlay_artists) == 3
    boxes = [a for a in widget.image_canvas._overlay_artists if isinstance(a, Rectangle)]
    assert len(boxes) == 1


def test_gixs2_widget_normalize_combo_applies_live_to_1d_and_azimuthal_curves(qapp, tmp_path, monkeypatch):
    """The Normalize combo previously only affected Save output
    (gixs2_session.py's private `_normalized_matrix`, only called from
    gixs2_save_1d/gixs2_save_azimuthal) -- confirm switching it to IC/BC
    now actually changes the on-screen curve too, via the newly-public
    `gixs2_normalized_matrix`."""

    gs.reset_session()
    widget = GIXS2Widget()
    center = (100.5, 100.5)
    _load_series_into_widget(widget, tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    _qcal_sync(widget, monkeypatch)
    widget.phi_spin.setValue(90.0)
    _exec_1d_sync(widget)
    widget.azi_pix_spin.setValue(60.0)
    widget.azi_minus_spin.setValue(-5.0)
    widget.azi_plus_spin.setValue(5.0)
    _exec_azi_sync(widget)

    def first_curve_y(canvas):
        lines = [ln for ln in canvas.ax.get_lines() if ln is not canvas._hover_marker]
        return lines[0].get_ydata().copy()

    none_1d = first_curve_y(widget.oned_canvas)
    none_azi = first_curve_y(widget.azi_canvas)

    widget.normalize_1d_combo.setCurrentText("IC")
    ic_1d = first_curve_y(widget.oned_canvas)
    assert not np.allclose(ic_1d, none_1d)

    widget.normalize_1d_combo.setCurrentText("BC")
    bc_1d = first_curve_y(widget.oned_canvas)
    assert not np.allclose(bc_1d, ic_1d)

    widget.normalize_azi_combo.setCurrentText("IC")
    ic_azi = first_curve_y(widget.azi_canvas)
    assert not np.allclose(ic_azi, none_azi)

    widget.normalize_azi_combo.setCurrentText("None")
    back_azi = first_curve_y(widget.azi_canvas)
    np.testing.assert_allclose(back_azi, none_azi)


def test_gixs2_widget_heatmap_buttons_require_exec_first(qapp, tmp_path):
    """Port of MATLAB's HeatmapQ_gui/HeatmapAzi_gui, opened here via a
    "Heatmap" button on the 1D/Azimuthal Plot Viewer's own control row --
    both require Exec Cut/Exec Azi to have produced a curve first, same as
    every other Save/Heatmap-style action in this tab."""

    gs.reset_session()
    widget = GIXS2Widget()
    _load_series_into_widget(widget, tmp_path, n_frames=1)

    widget._on_1d_heatmap_clicked()
    assert widget.status_label.property("role") == "status-error"
    assert widget._heatmap_dialog is None

    widget._on_azi_heatmap_clicked()
    assert widget.status_label.property("role") == "status-error"
    assert widget._heatmap_dialog is None


# `_open_heatmap_dialog`'s actual rendering (pcolormesh shape, labels, live
# Z Scale/Colormap updates) is covered directly against `ImageCanvas.
# render_mesh` in test_desktop_smoke.py
# (test_image_canvas_render_mesh_handles_non_uniform_x_axis) instead of via
# a full GIXS2Widget + QDialog here -- that path was verified manually
# (screenshot) end to end, but a second real matplotlib canvas on top of
# this file's existing widget count reliably tipped the whole suite into
# the native Windows access-violation crash from cumulative Qt/matplotlib
# widget construction (see git history).
