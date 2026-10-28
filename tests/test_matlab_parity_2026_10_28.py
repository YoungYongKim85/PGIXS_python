"""Regression tests for the 2026-10-28 MATLAB parity check.

- Heatmaps reorder frames with the sorted X axis (Order / Temp.Sort).
- Integration and Save Integ. sum only the Q / angle Min-Max window.
- IC/BC are taken relative to the first frame's counter (MATLAB RefIC/RefBC).
- The console incidence angle uses |pixel| (MATLAB CalIncidence callbacks).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pgixs_python.ui_backend import gixs2_session as gs
from pgixs_python.ui_backend.series_processing import integrate_frames


def test_integrate_frames_sums_only_inside_the_range():
    matrix = np.array([[1.0, 2.0, 4.0, np.nan], [10.0, 20.0, 40.0, 80.0]])
    axis = np.array([0.1, 0.2, 0.3, 0.4])

    np.testing.assert_allclose(integrate_frames(matrix, axis), [7.0, 150.0])
    np.testing.assert_allclose(integrate_frames(matrix, axis, (0.15, 0.35)), [6.0, 60.0])
    # 0/0 (the widgets' "not set" value) or a reversed window sums everything.
    np.testing.assert_allclose(integrate_frames(matrix, axis, (0.0, 0.0)), [7.0, 150.0])
    np.testing.assert_allclose(integrate_frames(matrix, axis, (0.3, 0.2)), [7.0, 150.0])


def _series(tmp_path):
    h5py = pytest.importorskip("h5py")
    files = []
    for idx in range(3):
        path = tmp_path / f"S_{idx + 1:04d}.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("/entry/data/data", data=np.full((40, 40), 10.0 + idx, dtype=np.float32))
            params = h5.create_group("/entry/instrument/user/params")
            params.create_dataset("Counter1", data=np.array([2.0 * (idx + 1)], dtype=np.float32))
            params.create_dataset("Counter2", data=np.array([1.0], dtype=np.float32))
            params.create_dataset("TimeStamp", data=np.array([1000.0 + 60.0 * idx]))
            params.create_dataset("Temp1", data=np.array([25.0], dtype=np.float32))
        files.append(path.name)
    gs.reset_session()
    result = gs.gixs2_load_series({"directory": str(tmp_path), "filenames": files, "insitu_mode": "Time"})
    assert result["ok"], result.get("message")
    return gs.get_session()


def test_gixs2_save_integration_honours_the_range_and_first_frame_counter(tmp_path):
    session = _series(tmp_path)
    session.last_1d = {
        "x_axis": np.array([0.1, 0.2, 0.3]),
        "intensity": np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [1.0, 2.0, 4.0]], dtype=np.float32),
    }
    session.auto_met = "1d"
    assert gs.gixs2_set_x_axis({})["ok"]

    result = gs.gixs2_save_integration({"directory": str(tmp_path), "normalize": "IC", "range": [0.15, 0.35]})
    assert result["ok"], result.get("message")
    saved = np.genfromtxt(result["path"], delimiter=",", skip_header=1)
    # IC = 2, 4, 6 -> relative 1, 2, 3; only Q 0.2 and 0.3 are summed (6).
    np.testing.assert_allclose(saved[:, -1], [6.0, 3.0, 2.0], rtol=1e-6)


def test_gixs2_time_axis_default_is_matlabs_exp_time(tmp_path):
    _series(tmp_path)
    seconds = gs.gixs2_set_x_axis({})
    np.testing.assert_allclose(seconds["x_axis"], [1.0, 61.0, 121.0])
    minutes = gs.gixs2_set_x_axis({"time_offset_seconds": 60.0, "time_unit": "min"})
    np.testing.assert_allclose(minutes["x_axis"], [1.0, 2.0, 3.0])


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_gixs2_heatmap_reorders_frames_like_the_curves(qapp, tmp_path, monkeypatch):
    from pgixs_python.desktop.gixs2_widget import GIXS2Widget

    session = _series(tmp_path)
    session.last_1d = {
        "x_axis": np.array([0.1, 0.2]),
        "intensity": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    }
    widget = GIXS2Widget()
    # Stand-in for a sorted X axis that is not file order.
    monkeypatch.setattr(gs, "order_frames", lambda values, axis=0: np.asarray(values)[::-1])
    shown = {}
    monkeypatch.setattr(widget, "_open_heatmap_dialog", lambda *args: shown.setdefault("matrix", args[3]))

    widget._on_1d_heatmap_clicked()

    np.testing.assert_allclose(shown["matrix"][:, 0], [3.0, 2.0, 1.0])


def test_console_incidence_angle_uses_the_pixel_distance(qapp):
    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    console = ConsoleWidget(gixs_config())
    console.sdd_spin.setValue(300.0)
    console._pixel_mm = 0.075
    console.ai_pix_spin.setValue(-40.0)
    console._incidence_from_pixel()
    below = console.ai_deg_spin.value()
    console.ai_pix_spin.setValue(40.0)
    console._incidence_from_pixel()

    assert below > 0
    assert below == pytest.approx(console.ai_deg_spin.value())
