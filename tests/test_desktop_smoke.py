"""Smoke tests for the desktop app: constructs the real Qt widget tree under
an offscreen platform plugin (no real display needed) and asserts the wiring
holds together. This catches import/attribute-wiring errors; it cannot
verify layout, spacing, or colors -- that needs a human looking at a real
window."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config, trxs_config
from pgixs_python.ui_backend import console_session as cs


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs_with_two_tabs(qapp):
    from pgixs_python.desktop.main_window import MainWindow

    window = MainWindow()
    assert window.tabs.count() == 2
    assert window.tabs.tabText(0) == "Console (GIXS)"
    assert window.tabs.tabText(1) == "Console (TrXS)"


def test_gixs_console_has_incidence_and_footprint_widgets(qapp):
    console = ConsoleWidget(gixs_config())
    assert console.ai_pix_spin is not None
    assert console.ai_deg_spin is not None
    assert console.footprint_spin is not None
    assert console.cakecut_check is not None
    assert console.circ_button is None


def test_trxs_console_omits_incidence_but_has_circular(qapp):
    console = ConsoleWidget(trxs_config())
    assert console.ai_pix_spin is None
    assert console.footprint_spin is None
    assert console.cakecut_check is None
    assert console.circ_button is not None


def test_gixs_and_trxs_consoles_keep_independent_sessions(qapp):
    gixs = ConsoleWidget(gixs_config())
    trxs = ConsoleWidget(trxs_config())

    gixs._activate()
    assert cs.get_session() is gixs._session
    assert cs.get_session().mode == "gixs"

    trxs._activate()
    assert cs.get_session() is trxs._session
    assert cs.get_session().mode == "trxs"

    gixs._activate()
    assert cs.get_session() is gixs._session


def test_incidence_pixel_degree_round_trip_matches_gixs_gui_formula(qapp):
    import math

    console = ConsoleWidget(gixs_config())
    console._pixel_mm = 0.075
    console.sdd_spin.setValue(3000.0)
    console.ai_pix_spin.setValue(139.63)

    console._incidence_from_pixel()

    expected_deg = math.degrees(math.atan((139.63 * 0.075) / 3000.0)) / 2.0
    # ai_deg_spin is configured with 4 decimal places, so QDoubleSpinBox
    # rounds the stored value -- match that precision, not exact equality.
    assert console.ai_deg_spin.value() == pytest.approx(expected_deg, abs=5e-5)


def test_directory_field_seeds_load_dialogs_and_stays_in_sync(qapp, tmp_path):
    console = ConsoleWidget(gixs_config())

    console._set_last_dir(str(tmp_path))

    assert console._last_dir == str(tmp_path)
    assert console.dir_edit.text() == str(tmp_path)


def test_editing_directory_field_to_invalid_path_reverts(qapp, tmp_path):
    console = ConsoleWidget(gixs_config())
    console._set_last_dir(str(tmp_path))

    console.dir_edit.setText(str(tmp_path / "does-not-exist"))
    console._on_directory_edited()

    assert console._last_dir == str(tmp_path)
    assert console.dir_edit.text() == str(tmp_path)


def test_editing_directory_field_to_valid_path_updates_last_dir(qapp, tmp_path):
    console = ConsoleWidget(gixs_config())
    other_dir = tmp_path / "series"
    other_dir.mkdir()

    console.dir_edit.setText(str(other_dir))
    console._on_directory_edited()

    assert console._last_dir == str(other_dir)


def test_run_in_background_anchors_worker_lifetime_to_thread(qapp):
    """Regression for the exact bug reported live: LOAD DATA got stuck
    forever on "Loading ..." with no error and no image. Root cause: the
    QObject worker running console_load() on the background thread had no
    Qt parent, and nothing else held a Python reference to it once
    run_in_background() returned -- only the QThread was kept alive by the
    caller (e.g. ConsoleWidget._load_thread). The local `worker` variable
    going out of scope made it eligible for garbage collection before
    `run()` ever executed, silently dropping the whole load.

    This checks the fix's actual mechanism directly (the worker is
    anchored as an attribute on the thread object, so keeping the thread
    alive keeps the worker alive too) rather than racing GC against a real
    background thread, which is flaky to assert on deterministically."""

    import gc

    from pgixs_python.desktop.background_tasks import ConsoleCallWorker, run_in_background

    thread = run_in_background(lambda payload: {"ok": True}, {}, lambda result: None)
    try:
        worker = thread._worker
        assert isinstance(worker, ConsoleCallWorker)

        gc.collect()  # would have collected a bare local-variable worker

        assert thread._worker is worker
    finally:
        thread.quit()
        thread.wait(2000)


def test_image_canvas_centers_wide_aspect_equal_plot_with_colorbar(qapp):
    """Regression: `figure.colorbar(im, ax=self.ax, ...)` on an
    aspect='equal' axes makes matplotlib's constrained_layout silently set
    ax.set_anchor((1.0, 0.5)) -- right-aligning the whole image+colorbar
    pair inside a wide panel instead of centering it (reported live: the
    ring image rendered flush against the right edge with a huge empty gap
    on the left). The fix derives the colorbar axes from make_axes_locatable
    instead, which doesn't trigger that re-anchoring. This renders into a
    wide, short canvas (matching the real "2D Image Plot" panel's
    proportions) and checks the drawn content's bounding box is
    horizontally centered, not pinned to one edge."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import ImageCanvas

    canvas = ImageCanvas()
    canvas.resize(1050, 350)
    canvas.show()
    qapp.processEvents()

    rng = np.random.default_rng(0)
    data = rng.uniform(50, 200, size=(400, 380)).astype(np.float32)
    canvas.render(data, cmap="viridis", extent=(1, 380, 400, 1), xlabel="Y, pixel", ylabel="Z, pixel")
    qapp.processEvents()
    qapp.processEvents()

    image = canvas.canvas.grab().toImage()
    width, height = image.width(), image.height()

    def is_background(x: int, y: int) -> bool:
        color = image.pixelColor(x, y)
        return color.red() > 250 and color.green() > 250 and color.blue() > 250

    columns_with_content = [
        x for x in range(0, width, 2) if any(not is_background(x, y) for y in range(0, height, 8))
    ]
    assert columns_with_content, "nothing was drawn onto the canvas"

    left, right = min(columns_with_content), max(columns_with_content)
    content_center = (left + right) / 2
    canvas_center = width / 2

    # Generous tolerance (asymmetric tick-label widths on the y-axis vs.
    # colorbar sides are expected) -- this only needs to catch the ~600px
    # right-alignment regression, not enforce pixel-perfect centering.
    assert abs(content_center - canvas_center) < width * 0.15, (
        f"content center {content_center} is not centered in canvas width {width} "
        f"(left margin {left}, right margin {width - right})"
    )


def test_load_info_recenters_pixel_axis_on_beam_position(qapp, tmp_path):
    """Regression, verified against GIXS_gui.m's Openinfo_Callback: once
    Load Info has set the beam position, the pixel-mode axis re-centers on
    it -- HD = col - beamY, VD = beamZ - row -- instead of showing raw
    1-based pixel indices, and this stays sticky (matches param.Expinfo)
    across subsequent renders."""

    import h5py
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    frame_path = tmp_path / "frame.h5"
    rows, cols = 20, 24
    data = np.random.default_rng(0).uniform(10, 100, size=(rows, cols)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])

    console = ConsoleWidget(gixs_config())
    console._activate()
    payload = {
        "directory": str(tmp_path),
        "filename": frame_path.name,
        "mode": "gixs",
        "detector": "TestDetector",
        "multishot": False,
    }
    from pgixs_python.ui_backend import console_session as cs

    result = cs.console_load(payload)
    assert result["ok"], result.get("message")
    console._on_load_finished(result)

    # Before Load Info: raw 1-based pixel axis.
    assert console._pixel_centered is False
    assert console.yrange_min_spin.value() == pytest.approx(1)
    assert console.yrange_max_spin.value() == pytest.approx(cs.get_session().data.shape[1])

    beam_y, beam_z = 10.0, 15.0
    # Set on the session's calibration (what a real Load Info does), not
    # just the widget -- otherwise the next console_load()'s calibration
    # response pulls the old default and clobbers the widget again.
    cs.get_session().calibration["beam_pos"] = (beam_y, beam_z)
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    console._pixel_centered = True
    console._render_pixel_image()

    real_rows, real_cols = cs.get_session().data.shape
    expected_y_min = 1 - beam_y
    expected_y_max = real_cols - beam_y
    expected_v_row1 = beam_z - 1  # row 1's VD value (top of the display)
    expected_v_row_last = beam_z - real_rows  # last row's VD value (bottom)

    assert console.yrange_min_spin.value() == pytest.approx(min(expected_y_min, expected_y_max))
    assert console.yrange_max_spin.value() == pytest.approx(max(expected_y_min, expected_y_max))
    assert console.zrange_min_spin.value() == pytest.approx(min(expected_v_row1, expected_v_row_last))
    assert console.zrange_max_spin.value() == pytest.approx(max(expected_v_row1, expected_v_row_last))

    # extent = (left, right, bottom, top); top must be row 1's value (beam_z
    # - 1) since origin='upper' always draws row 1 at the geometric top.
    left, right, bottom, top = console._current_image_extent
    assert top == pytest.approx(expected_v_row1)
    assert bottom == pytest.approx(expected_v_row_last)
    assert left == pytest.approx(expected_y_min)
    assert right == pytest.approx(expected_y_max)

    # Beam center reads as the origin once the axis is centered.
    assert console._current_beam_center() == (0.0, 0.0)

    # Sticky: a fresh Load Data must NOT reset _pixel_centered.
    result2 = cs.console_load(payload)
    console._on_load_finished(result2)
    assert console._pixel_centered is True
    assert console.yrange_min_spin.value() == pytest.approx(min(expected_y_min, expected_y_max))


def test_zplot_marks_its_column_on_the_2d_image(qapp, tmp_path):
    """z-plot reads down a single detector column (offset from the beam by
    the "ai [deg] / y-pos" value); the user asked to see which column that
    was on the 2D image itself. Checks the vertical marker line drawn by
    _run_zplot lands at the right display x-coordinate, both in raw pixel
    mode and after Load Info has beam-centered the axis."""

    import h5py
    import numpy as np
    from matplotlib.lines import Line2D

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
    from pgixs_python.ui_backend import console_session as cs

    frame_path = tmp_path / "frame.h5"
    rows, cols = 20, 24
    data = np.random.default_rng(0).uniform(10, 100, size=(rows, cols)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])

    console = ConsoleWidget(gixs_config())
    console._activate()
    payload = {
        "directory": str(tmp_path),
        "filename": frame_path.name,
        "mode": "gixs",
        "detector": "TestDetector",
        "multishot": False,
    }
    result = cs.console_load(payload)
    assert result["ok"], result.get("message")
    console._on_load_finished(result)

    def marker_x() -> float:
        lines = [a for a in console.image_canvas._overlay_artists if isinstance(a, Line2D)]
        assert len(lines) == 1, "expected exactly one column-marker overlay"
        xdata = lines[0].get_xdata()
        assert xdata[0] == xdata[1]
        return float(xdata[0])

    # Raw pixel mode: beam_y=10, offset=5 -> target column 15, and since
    # it's not beam-centered the display x is the raw column itself.
    console.beam_y_spin.setValue(10.0)
    console.ai_ypix_spin.setValue(5.0)
    console._run_zplot()
    assert console._pixel_centered is False
    assert marker_x() == pytest.approx(15.0)

    # Beam-centered mode: display x = target_col - beam_y = offset itself.
    cs.get_session().calibration["beam_pos"] = (10.0, console.beam_z_spin.value())
    console._pixel_centered = True
    console._render_pixel_image()
    console.ai_ypix_spin.setValue(3.0)
    console._run_zplot()
    assert marker_x() == pytest.approx(3.0)


def test_curve_canvas_hover_marks_nearest_point(qapp):
    """Regression for the 1D Result hover-crosshair request: moving the
    mouse over the curve should snap a '+' marker (and coordinate readout)
    to the nearest plotted data point, and hide again once the mouse leaves
    the axes -- mirrors _on_click's existing event-object-driven test style
    rather than simulating a real Qt mouse-move pixel by pixel."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import CurveCanvas

    class _FakeEvent:
        def __init__(self, inaxes, xdata):
            self.inaxes = inaxes
            self.xdata = xdata

    canvas = CurveCanvas()
    x = np.linspace(0.0, 10.0, 11)
    y = x**2
    canvas.render(x, y)

    # Nothing shown before the mouse moves over the axes.
    assert canvas._hover_marker.get_visible() is False
    assert canvas._hover_text.get_visible() is False

    canvas._on_motion(_FakeEvent(canvas.ax, 5.2))
    assert canvas._hover_marker.get_visible() is True
    xdata, ydata = canvas._hover_marker.get_data()
    assert xdata[0] == pytest.approx(5.0)  # nearest sample to 5.2
    assert ydata[0] == pytest.approx(25.0)
    assert "25" in canvas._hover_text.get_text()

    # Leaving the axes hides both the marker and the coordinate readout.
    canvas._on_motion(_FakeEvent(None, None))
    assert canvas._hover_marker.get_visible() is False
    assert canvas._hover_text.get_visible() is False

    # A fresh render() (ax.clear()) must not leave stale hover artists
    # pointing at data that no longer exists.
    canvas.render(x, y * 2)
    canvas._on_motion(_FakeEvent(canvas.ax, 5.2))
    ydata = canvas._hover_marker.get_data()[1]
    assert ydata[0] == pytest.approx(50.0)
