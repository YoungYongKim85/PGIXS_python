"""Norm(Pt): scale every frame's 1D curve to 1 at a picked Q (MATLAB
GIXS2 `NormalFactor_Callback`, TrXS2 `applyNormalization`)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pgixs_python.ui_backend import gixs2_session as gs
from pgixs_python.ui_backend import trxs2_session as ts
from pgixs_python.ui_backend.series_processing import normalize_at_point

X_AXIS = np.array([0.1, 0.2, 0.3])
CURVES = np.array([[2.0, 4.0, 8.0], [5.0, np.nan, 20.0], [3.0, 0.0, 9.0]], dtype=np.float32)


def test_normalize_at_point_uses_the_nearest_bin_and_skips_bad_factors():
    out = normalize_at_point(CURVES, X_AXIS, 0.19)
    np.testing.assert_allclose(out[0], [0.5, 1.0, 2.0])
    # NaN or 0 at the point: the frame is left unscaled, like sanitizePGIXSScaleFactors.
    np.testing.assert_allclose(out[1], [5.0, np.nan, 20.0])
    np.testing.assert_allclose(out[2], [3.0, 0.0, 9.0])


def test_gixs2_norm_point_reaches_save_and_integration_and_turns_off(tmp_path):
    gs.reset_session()
    session = gs.get_session()
    session.last_1d = {"x_axis": X_AXIS, "intensity": CURVES[:1].copy()}
    session.x_axis_values = np.array([1.0])
    session.x_axis_label = "Frame"
    session.auto_met = "1d"

    picked = gs.gixs2_set_norm_point({"x": 0.29})
    assert picked["x"] == pytest.approx(0.3) and picked["unscaled"] == 0
    np.testing.assert_allclose(gs.gixs2_1d_matrix("None"), [[0.25, 0.5, 1.0]])
    saved = gs.gixs2_save_integration({"directory": str(tmp_path)})
    assert saved["ok"], saved.get("message")
    np.testing.assert_allclose(np.genfromtxt(saved["path"], delimiter=",", skip_header=1)[-1], 1.75)

    assert gs.gixs2_set_norm_point({"x": None})["ok"]
    np.testing.assert_allclose(gs.gixs2_1d_matrix("None"), CURVES[:1])
    assert gs.gixs2_set_norm_point({"x": "abc"})["ok"] is False


def test_trxs2_norm_point_scales_frames_and_background_before_subtracting():
    ts.reset_session()
    session = ts.get_session()
    session.last_1d = {"x_axis": X_AXIS, "intensity": CURVES[:1].copy(), "std": None}
    session.bkg_1d = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    session.subtract_background = True

    result = ts.trxs2_set_norm_point({"x": 0.3})
    assert result["ok"] and result["x"] == pytest.approx(0.3)
    # 2,4,8 / 8 - 1,1,2 / 2 = 0.25-0.5, 0.5-0.5, 1-1
    np.testing.assert_allclose(session.last_1d["rendered"], [[-0.25, 0.0, 0.0]])

    ts.trxs2_set_norm_point({"x": None})
    np.testing.assert_allclose(session.last_1d["rendered"], [[1.0, 3.0, 6.0]])


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_gixs2_widget_norm_point_button_picks_on_the_1d_plot(qapp):
    from pgixs_python.desktop.gixs2_widget import GIXS2Widget

    gs.reset_session()
    gs.get_session().last_1d = {"x_axis": X_AXIS, "intensity": CURVES[:1].copy()}
    widget = GIXS2Widget()

    widget.oned_normpt_button.setChecked(True)
    widget.oned_canvas.point_clicked.emit(0.2, 1.0)

    lines = [ln for ln in widget.oned_canvas.ax.get_lines() if ln is not widget.oned_canvas._hover_marker]
    np.testing.assert_allclose(lines[0].get_ydata(), [0.5, 1.0, 2.0])
    # One pick per press: a second click does not move the point.
    widget.oned_canvas.point_clicked.emit(0.1, 1.0)
    assert gs.get_session().last_1d["norm_point"] == pytest.approx(0.2)

    widget.oned_normpt_button.setChecked(False)
    lines = [ln for ln in widget.oned_canvas.ax.get_lines() if ln is not widget.oned_canvas._hover_marker]
    np.testing.assert_allclose(lines[0].get_ydata(), [2.0, 4.0, 8.0])


def test_trxs2_widget_norm_point_button_picks_on_the_1d_plot(qapp):
    from pgixs_python.desktop.trxs2_widget import TrXS2Widget

    ts.reset_session()
    session = ts.get_session()
    session.last_1d = {"x_axis": X_AXIS, "intensity": CURVES[:1].copy(), "std": None}
    ts.trxs2_render_1d({})
    widget = TrXS2Widget()

    widget.oned_normpt_button.setChecked(True)
    widget.oned_canvas.point_clicked.emit(0.1, 1.0)
    np.testing.assert_allclose(session.last_1d["rendered"], [[1.0, 2.0, 4.0]])

    widget.oned_normpt_button.setChecked(False)
    np.testing.assert_allclose(session.last_1d["rendered"], [[2.0, 4.0, 8.0]])


def test_norm_point_says_when_frames_have_no_value_there():
    gs.reset_session()
    gs.get_session().last_1d = {"x_axis": X_AXIS, "intensity": CURVES.copy()}
    result = gs.gixs2_set_norm_point({"x": 0.2})
    # Frame 2 is NaN and frame 3 is 0 at Q = 0.2.
    assert result["unscaled"] == 2
    assert "2 of 3 frame(s)" in result["message"]
