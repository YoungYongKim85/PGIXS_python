"""Smoke tests for the desktop app: constructs the real Qt widget tree under
an offscreen platform plugin (no real display needed) and asserts the wiring
holds together. This catches import/attribute-wiring errors; it cannot
verify layout, spacing, or colors -- that needs a human looking at a real
window."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox

from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config, trxs_config
from pgixs_python.ui_backend import console_session as cs


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs_with_seven_tabs(qapp):
    """SDD Calibration comes first -- it's the first thing done in a real
    experiment, and GIXS/TrXS both need its output before Q-space analysis
    works at all. GIXS-series sits right after GIXS (same instrument/
    geometry); TrXS-series sits right after TrXS for the same reason (see
    the TrXS2 implementation plan); XTrans/Penetration Depth are standalone
    calculators after that."""

    from pgixs_python.desktop.main_window import MainWindow

    window = MainWindow()
    assert window.tabs.count() == 7
    assert window.tabs.tabText(0) == "SDD Calibration"
    assert window.tabs.tabText(1) == "GIXS"
    assert window.tabs.tabText(2) == "GIXS-series"
    assert window.tabs.tabText(3) == "TrXS"
    assert window.tabs.tabText(4) == "TrXS-series"
    assert window.tabs.tabText(5) == "XTrans"
    assert window.tabs.tabText(6) == "Penetration Depth"


def test_main_window_uses_the_intrologo_icon(qapp):
    from pgixs_python.desktop.main_window import MainWindow

    window = MainWindow()

    assert not window.windowIcon().isNull()


def test_gixs_console_has_incidence_and_footprint_widgets(qapp):
    console = ConsoleWidget(gixs_config())
    assert console.ai_pix_spin is not None
    assert console.ai_deg_spin is not None
    assert console.footprint_spin is not None
    assert console.cakecut_check is not None
    assert console.circ_button is None
    # ColorScale/Colormap/Map format used to size to their own natural
    # content (Colormap needing much more room than ColorScale's "Linear"/
    # "Log") sitting right next to each other on the Image Controller's own
    # row -- _size_combo (console_widget.py) gives all three the same
    # content-length budget so they line up. Checking the policy/budget
    # directly (rather than the rendered .width(), which needs a real
    # show()'d layout pass to reflect the scroll area's own width
    # constraint -- and .show() is what triggers this suite's native
    # access-violation crash, see git history) still proves the fix is
    # wired up, without paying that cost. Reuses this test's own console
    # instance rather than constructing another.
    for combo in (console.color_scale_combo, console.colormap_combo, console.map_format_combo):
        assert combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon
        assert combo.minimumContentsLength() == 12
    # Same follow-up for the left panel's own "choice" combos (Mean/Median,
    # Qxy/Qz<->Qy/Qz, Normalization None/IC/BC), which each used to size to
    # their own natural content too. Normalization gets a smaller budget
    # (8, not 10) purely to leave the left panel a little more breathing
    # room -- it's a standalone row, not sitting next to the other two.
    for combo, chars in (
        (console.azi_meanmedian_combo, 10),
        (console.mapping_combo, 10),
        (console.normal_combo, 8),
    ):
        assert combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon
        assert combo.minimumContentsLength() == chars


def test_trxs_console_omits_incidence_but_has_circular(qapp):
    console = ConsoleWidget(trxs_config())
    assert console.ai_pix_spin is None
    assert console.footprint_spin is None
    assert console.cakecut_check is None
    assert console.circ_button is not None
    assert console.mapping_combo is None  # Qxy/Qz vs Qy/Qz is a GIXS-only distinction
    # Same left-panel combo width fix as GIXS's own (see
    # test_gixs_console_has_incidence_and_footprint_widgets, including why
    # this checks the policy/budget rather than .width()) applies here too.
    assert console.azi_meanmedian_combo.minimumContentsLength() == 10
    assert console.normal_combo.minimumContentsLength() == 8


def test_circular_process_is_positioned_above_1d_plot_process(qapp):
    """Regression: "Circular Process를 1D plot process 위로 오게 배치해줘" --
    the left panel's section dividers, top to bottom, must have Circular
    Process before 1D Plot Process (Azimuthal Process's relative order
    wasn't part of the request, so it's left where it already was, after
    1D Plot Process)."""

    from PySide6.QtWidgets import QLabel

    console = ConsoleWidget(trxs_config())
    dividers = [label.text() for label in console.findChildren(QLabel) if label.property("role") == "divider"]
    assert dividers.index("Circular Process") < dividers.index("1D Plot Process")


def test_contrast_auto_resets_after_background_subtraction(qapp, tmp_path, monkeypatch):
    """Regression: "백그라운드를 Subtraction하면 Contrast가 자동으로 변경되게 해줘" --
    Contrast used to hold onto whatever min/max applied to the raw data,
    so after subtracting a background (a completely different intensity
    scale) the image looked wrong until the user manually reset it."""

    import h5py
    import numpy as np
    from PySide6.QtWidgets import QMessageBox

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    console.cmin_spin.setValue(500.0)
    console.cmax_spin.setValue(600.0)

    bkg_path = tmp_path / "bkg.h5"
    bkg_data = np.random.default_rng(99).uniform(1, 5, size=(20, 24)).astype(np.float32)
    with h5py.File(bkg_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=bkg_data[None, :, :])
    bkg_result = cs.console_background_load({"directory": str(tmp_path), "filename": bkg_path.name, "multishot": False})
    assert bkg_result["ok"], bkg_result.get("message")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    console._on_subtract_clicked()

    assert (console.cmin_spin.value(), console.cmax_spin.value()) != (500.0, 600.0)


def test_contrast_spin_boxes_keep_precision_for_small_subtracted_values(qapp, tmp_path, monkeypatch):
    """Regression: "background subtraction 후에 이미지의 contrast가 소수점으로
    되는데, 그게 Contrast 텍스트 박스와 colormap에 제대로 반영이 안돼" --
    QDoubleSpinBox.setValue() *rounds* to its `decimals` setting, not just
    display-formats, so the old 3-decimal cmin/cmax spins silently
    collapsed small background-subtracted contrast values (routinely well
    under 1.0) toward 0.000, sometimes making cmin == cmax -- which
    _apply_display_settings then treats as "auto" and throws away the
    real computed range on the very next live-control change."""

    import h5py
    import numpy as np
    from PySide6.QtWidgets import QMessageBox

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)

    # Raw data clustered tightly around 10.0 and a background right at
    # 10.0 -- the subtracted result's whole range sits well under 0.001,
    # exactly the regime the 3-decimal spin boxes used to round away.
    # _load_test_frame's own frame is irrelevant here -- overwrite it with a
    # fresh console_load carrying the tight-cluster values instead.
    frame_path = tmp_path / "frame.h5"
    raw_data = (10.0 + np.random.default_rng(1).uniform(0.0001, 0.0003, size=(20, 24))).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=raw_data[None, :, :])
    result = cs.console_load(
        {"directory": str(tmp_path), "filename": frame_path.name, "mode": "gixs", "detector": "TestDetector", "multishot": False}
    )
    assert result["ok"], result.get("message")
    console._on_load_finished(result)

    bkg_path = tmp_path / "bkg.h5"
    bkg_data = np.full((20, 24), 10.0, dtype=np.float32)
    with h5py.File(bkg_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=bkg_data[None, :, :])
    bkg_result = cs.console_background_load({"directory": str(tmp_path), "filename": bkg_path.name, "multishot": False})
    assert bkg_result["ok"], bkg_result.get("message")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    console._on_subtract_clicked()

    cmin, cmax = console.cmin_spin.value(), console.cmax_spin.value()
    assert cmax > cmin  # distinct, not both rounded down to the same value
    assert cmax - cmin < 0.01  # actually representing the real, tiny range

    # And a subsequent live-control change (e.g. Colormap) must keep using
    # that real range, not silently fall back to "auto" because cmin/cmax
    # read back as equal.
    console._apply_display_settings()
    assert (console.cmin_spin.value(), console.cmax_spin.value()) == (cmin, cmax)


def test_subtract_after_qcal_actually_updates_the_displayed_image(qapp, tmp_path, monkeypatch):
    """Regression: "TrXS에서 백그라운드 데이터를 넣어서 BKGsub. & Norm을
    누르면 바뀐 이미지를 유지하겠냐고 나오는데, 사실 이미지가 바뀌지
    않고 메시지 박스만 나와" -- if q-Map Cal. already ran, its cached
    corrected_data (built from the pre-normalization image) took priority
    over the freshly normalized data in _render_pixel_image's own
    fallback, so the confirmation dialog worked but the visible image
    never changed. console_normalize now drops that stale copy on
    confirm; the widget must also flag q-Map Cal. as needing a re-run so
    the user isn't left thinking the (uncorrected) image on screen already
    has the detector correction applied."""

    import h5py
    import numpy as np
    from PySide6.QtWidgets import QMessageBox

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    qcal_result = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 12.0, "beam_z": 10.0})
    assert qcal_result["ok"], qcal_result.get("message")
    console._mark_ok(console.qmap_button, True)
    assert cs.get_session().corrected_data is not None

    bkg_path = tmp_path / "bkg.h5"
    bkg_data = np.random.default_rng(99).uniform(1, 5, size=(20, 24)).astype(np.float32)
    with h5py.File(bkg_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=bkg_data[None, :, :])
    bkg_result = cs.console_background_load({"directory": str(tmp_path), "filename": bkg_path.name, "multishot": False})
    assert bkg_result["ok"], bkg_result.get("message")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    console._on_subtract_clicked()

    assert cs.get_session().corrected_data is None
    assert console.qmap_button.property("state") != "ok"
    # The rendered image is now genuinely the new (normalized) data, not
    # the stale pre-normalization corrected copy.
    np.testing.assert_array_equal(console._current_image_data, cs.get_session().data)


def test_subtract_shows_the_preview_before_asking_and_reverts_on_discard(qapp, tmp_path, monkeypatch):
    """Regression: "이러한 메시지만 나오고 subtracted 또는 normalized
    image가 어떤지 나오지 않아" -- the confirmation dialog's text alone
    gave no way to judge the background-subtracted/normalized result
    before deciding Yes/No. The 2D Image Plot Viewer must already show
    the real preview by the time the dialog appears, and discarding
    ("No") must put the actual (unchanged) data back on screen -- not
    leave the discarded preview showing."""

    import h5py
    import numpy as np
    from PySide6.QtWidgets import QMessageBox

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    original_data = cs.get_session().data.copy()

    bkg_path = tmp_path / "bkg.h5"
    bkg_data = np.random.default_rng(99).uniform(1, 5, size=(20, 24)).astype(np.float32)
    with h5py.File(bkg_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=bkg_data[None, :, :])
    bkg_result = cs.console_background_load({"directory": str(tmp_path), "filename": bkg_path.name, "multishot": False})
    assert bkg_result["ok"], bkg_result.get("message")

    seen_at_dialog_time = {}

    def fake_question(*_args, **_kwargs):
        # By the time the dialog would appear, the canvas must already be
        # showing the real preview -- capture it here rather than after
        # _on_subtract_clicked returns, so this can't pass by accident
        # just because the later commit path also renders something.
        seen_at_dialog_time["image"] = console._current_image_data.copy()
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    console._on_subtract_clicked()

    expected_preview = original_data - bkg_data  # method="None", plus_data=1.0 default
    np.testing.assert_allclose(seen_at_dialog_time["image"], expected_preview)
    # Discarded -- session state (and the canvas) must be back to the
    # real, unchanged data, not left showing the rejected preview.
    assert not cs.get_session().normalized
    np.testing.assert_array_equal(cs.get_session().data, original_data)
    np.testing.assert_array_equal(console._current_image_data, original_data)


def test_trxs_circular_mask_draw_start_and_reject_outside_pixel_mode(qapp, tmp_path, monkeypatch):
    """Draw Mask opens its own dedicated window (port of MATLAB's
    `get_modern_mask`) rather than drawing inline on the small "2D Image
    Plot" quadrant -- patch the class method so it intercepts whichever
    ImageCanvas instance the dialog creates, not `console.image_canvas`."""

    from pgixs_python.desktop.plot_canvas import ImageCanvas

    console = _load_test_frame_trxs(tmp_path)
    captured = {}
    monkeypatch.setattr(
        ImageCanvas, "start_polygon_selector", lambda self, cb: captured.setdefault("cb", cb)
    )

    console._on_circ_mask_draw_clicked()
    assert "cb" in captured
    assert console._mask_dialog is not None

    console._mask_dialog.close()
    console._mask_dialog = None
    captured.clear()
    console._axis_mode = "q"
    console._on_circ_mask_draw_clicked()
    assert "cb" not in captured
    assert console._mask_dialog is None


def test_trxs_circular_mask_polygon_complete_sets_and_clears_mask(qapp, tmp_path):
    """Regression: "circular process에서 mask를 만들 수 있도록 해줘" -- drawing a
    polygon on the 2D image must set console_session's circular_mask and
    update the status label; Clear Mask must remove it again. Also locks
    in the polarity the user explicitly specified afterward ("선택한 영역을
    마스킹하여, 나머지영역만 살리는 역할"): the drawn polygon's *inside* is
    excluded, and everything *outside* it survives -- not the other way
    around."""

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    assert console.circ_mask_status_label.text() == "Mask: none"

    # _load_test_frame_trxs's synthetic frame is 20 rows x 24 cols; cover
    # roughly the left half in uncentered Pixel-mode display coordinates
    # (1-based col/row, identity mapping -- see _on_circ_mask_polygon_complete).
    vertices = [(0.5, 0.5), (12.0, 0.5), (12.0, 20.5), (0.5, 20.5)]
    console._on_circ_mask_polygon_complete(vertices)

    mask = cs.get_session().circular_mask
    assert mask is not None
    assert mask.shape == (20, 24)
    assert 0 < mask.sum() < mask.size
    assert console.circ_mask_status_label.text() != "Mask: none"

    # Column 5 (1-based) sits inside the drawn polygon (x in [0.5, 12.0])
    # -> must be excluded (False). Column 20 sits outside it -> must be
    # kept (True). Array indices are 0-based, so col 5 -> index 4, col 20
    # -> index 19; row is irrelevant here so just check row 0.
    assert not mask[0, 4]
    assert mask[0, 19]

    console._on_circ_mask_clear_clicked()
    assert cs.get_session().circular_mask is None
    assert console.circ_mask_status_label.text() == "Mask: none"


def test_circular_plot_draws_error_bars_only_when_std_is_checked(qapp, tmp_path):
    """Regression: "TrXS에서 Circular Process에서... Std가 켜고 프로세스
    했을때, 1D Result에서 Y축 errorbar plot이 되면 좋을 것 같아" -- the
    Std checkbox already asked the backend to compute a per-bin std, but
    nothing ever rendered it. With Std checked, the 1D Result curve must
    be drawn via Axes.errorbar (a real ErrorbarContainer, not just a plain
    Line2D); with Std unchecked, it must stay a plain line exactly as
    before (no error bars where the user didn't ask for them)."""

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    beam_y, beam_z = 12.0, 10.0
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    qcal_result = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": beam_y, "beam_z": beam_z})
    assert qcal_result["ok"], qcal_result.get("message")
    console.circ_angle_min_spin.setValue(0.0)
    console.circ_angle_max_spin.setValue(360.0)
    console.circ_min_coverage_spin.setValue(0.0)

    console.circ_std_check.setChecked(False)
    console._on_circular_clicked()
    assert len(console.curve_canvas.ax.containers) == 0

    console.circ_std_check.setChecked(True)
    console._on_circular_clicked()
    assert len(console.curve_canvas.ax.containers) == 1


def test_circ_sector_overlay_draws_live_as_angle_spins_change(qapp, tmp_path):
    """Regression: "Draw 버튼이 아닌 이건 바로 라인이 그어지고 진행이 되면 좋을 것
    같아" -- editing Ang. Avg min/max must redraw the sector boundary rays
    immediately, with no separate Draw button to click first (mirrors how
    Contrast/Colormap/etc. already apply live elsewhere in this app)."""

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    assert not hasattr(console, "circ_draw_button")

    console.circ_angle_min_spin.setValue(90.0)
    console.circ_angle_max_spin.setValue(180.0)

    assert len(console.image_canvas._overlay_artists) == 2


def test_circ_sector_overlay_skips_a_full_360_sweep(qapp, tmp_path):
    """A full-circle Ang. Avg range has no meaningful boundary -- both rays
    would land on the same angle, so nothing should draw instead of a
    misleading stray line."""

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]

    assert console.circ_angle_min_spin.value() == 0.0
    assert console.circ_angle_max_spin.value() == 360.0
    console._update_circ_sector_overlay()

    assert len(console.image_canvas._overlay_artists) == 0


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


def test_image_canvas_render_mesh_and_surface_modes(qapp):
    """`render_mesh`/`render_surface` back GIXS2's Heatmap dialogs (port of
    MATLAB's HeatmapQ_gui/HeatmapAzi_gui's Heatmap/Surf/Waterfall "Plot
    Type" selector). `render_mesh` uses `pcolormesh` instead of `render`'s
    own `imshow`+`extent`, specifically because `imshow` assumes evenly-
    spaced columns: a series X-axis (Time/Temperature) routinely isn't.
    Reuses one `ImageCanvas` across all three modes (rather than one per
    mode) and deliberately skips show()/processEvents() throughout (unlike
    its sibling test_image_canvas_centers_wide_aspect_equal_plot_with_colorbar
    above) -- checking the Axes/artist data directly needs neither, and
    this suite is already right at the edge of a native Windows access-
    violation crash from cumulative Qt/matplotlib widget construction
    (see git history)."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import ImageCanvas

    canvas = ImageCanvas()
    x = np.array([0.0, 1.0, 2.0, 20.0])  # deliberately non-uniform gaps
    y = np.array([0.1, 0.2, 0.3])
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    z[1, 2] = np.nan  # exercise the masked/NaN-handling path too

    vmin, vmax = canvas.render_mesh(x, y, z, xlabel="Time, s", ylabel="Q, 1/A", log_scale=False)

    # `pcolormesh` (shading="nearest") reports per-cell *edges*, one more
    # than the number of centers given -- the last edge-to-edge gap here
    # must dwarf the first two, reflecting the big real gap between x=2
    # and x=20 rather than an imshow-style evenly-spaced approximation.
    edges = sorted(set(canvas._image.get_coordinates()[..., 0].ravel()))
    gaps = np.diff(edges)
    assert gaps[-1] > 10 * gaps[0]
    assert canvas.ax.get_xlabel() == "Time, s"
    assert canvas.ax.get_ylabel() == "Q, 1/A"
    assert vmin < vmax
    assert canvas._colorbar is not None

    # X/Y Scale and X/Y Range (the plot's own axes) are independent of
    # log_scale/vmin/vmax (the *color* scale) -- reported: "heatmap에서
    # x축과 y축도 linear/log scale로 바꿀 수 있게 해주고, range도 설정이
    # 가능하도록 textbox를 추가해줘".
    canvas.render_mesh(
        x, y, z, xlabel="Time, s", ylabel="Q, 1/A",
        xscale="log", yscale="linear", xlim=(1.0, 20.0), ylim=(0.15, 0.25),
    )
    assert canvas.ax.get_xscale() == "log"
    assert canvas.ax.get_yscale() == "linear"
    assert canvas.ax.get_xlim() == (1.0, 20.0)
    assert canvas.ax.get_ylim() == (0.15, 0.25)

    # `Axes3D.set_xscale`/`set_yscale("log")` doesn't actually work --
    # it only relabels ticks without transforming the 3D projection,
    # corrupting the geometry into an off-center sliver (reported:
    # "surf와 waterfall의 이미지가 가운데 있지 않고... log스케일에서
    # 이미지에 문제가 생겨"). render_surface instead log10-transforms the
    # plotted *coordinates* and leaves the axes' own scale "linear", so
    # get_xscale/get_yscale stay "linear" and get_xlim/get_ylim report the
    # transformed range, with a FuncFormatter relabeling ticks back to the
    # real value. `x` has a leading 0.0 (see above) -- log of <=0 is
    # undefined, so the X axis silently falls back to linear (both its own
    # geometry and the caller's raw xlim) whenever that 0.0 point is still
    # within the requested X Range (here (-1.0, 20.0), which keeps it) --
    # while Y (all positive) transforms successfully.
    vmin, vmax = canvas.render_surface(
        x, y, z, mode="surf", xlabel="Time, s", ylabel="Q, 1/A", zlabel="Intensity",
        xscale="log", yscale="log", xlim=(-1.0, 20.0), ylim=(0.15, 0.25),
    )
    assert canvas.ax.name == "3d"
    assert canvas.ax.get_zlabel() == "Intensity"
    assert canvas.ax.get_xscale() == "linear"
    assert canvas.ax.get_yscale() == "linear"
    assert canvas.axis_log_fallback == ["X"]
    assert canvas.ax.get_xlim() == (-1.0, 20.0)
    assert canvas.ax.get_ylim() == pytest.approx((np.log10(0.15), np.log10(0.25)))
    assert canvas.ax.yaxis.get_major_formatter()(np.log10(0.25), 0) == "0.25"
    assert vmin < vmax
    assert canvas._colorbar is not None
    assert canvas._rotation_label is not None
    assert canvas._rotation_label.get_text() == "Elev 30°, Azim -60°"

    # Regression: X/Y Range used to only narrow the 3D axes' own viewport
    # (`Axes3D.set_xlim`/`set_ylim`) without touching the plotted geometry,
    # so a Range narrower than the full data still built (and drew)
    # `plot_surface`/the waterfall curtains at their full original size --
    # the surface visibly poked out past the narrowed box walls (reported:
    # "Waterfall/Surf에서 range를 바꾸니까 axis 밖으로 삐져나와"). The
    # source arrays must be trimmed to the requested range *before* any
    # geometry is built from them. Excluding x=0.0 from the requested range
    # this time (unlike the xlim=(-1.0, 20.0) case just above, which
    # deliberately kept it) means the X log-transform, attempted only
    # against the now-trimmed (all-positive) data, succeeds instead of
    # falling back -- a direct consequence of trimming happening first.
    canvas.render_surface(
        x, y, z, mode="surf", xlabel="Time, s", ylabel="Q, 1/A", zlabel="Intensity",
        xscale="log", yscale="log", xlim=(1.0, 20.0), ylim=(0.15, 0.25),
    )
    assert canvas.axis_log_fallback == []
    assert canvas.ax.get_xlim() == pytest.approx((0.0, np.log10(20.0)))
    assert canvas.ax.get_ylim() == pytest.approx((np.log10(0.15), np.log10(0.25)))

    # Regression check for the reported "가운데 있지 않고" bug:
    # `figure.colorbar(..., ax=self.ax, shrink=..., pad=...)` (matplotlib's
    # simple form for a 3D axes) reanchors the parent axes toward its own
    # right edge to make room, not its center, visibly pushing the whole
    # 3D box to one side. render_surface uses an explicit `cax` instead
    # (see its own comment) specifically so `self.ax` keeps the centered
    # position set at the top of the method -- assert its x-midpoint is
    # still close to that rect's own center (0.02 + 0.90/2 = 0.47), not
    # dragged toward the colorbar on the right.
    ax_x0, _, ax_x1, _ = canvas.ax.get_position().extents
    assert (ax_x0 + ax_x1) / 2 == pytest.approx(0.47, abs=0.05)

    # All-positive data on both axes -- both transform successfully this
    # time, and the fallback list comes back empty.
    y_positive = np.array([0.1, 0.2, 0.3])
    x_positive = np.array([1.0, 2.0, 5.0, 20.0])
    canvas.render_surface(
        x_positive, y_positive, z, mode="surf", xlabel="Time, s", ylabel="Q, 1/A",
        xscale="log", yscale="log", xlim=(1.0, 20.0), ylim=(0.15, 0.25),
    )
    assert canvas.axis_log_fallback == []
    assert canvas.ax.get_xlim() == pytest.approx((0.0, np.log10(20.0)))
    assert canvas.ax.get_ylim() == pytest.approx((np.log10(0.15), np.log10(0.25)))
    assert canvas.ax.xaxis.get_major_formatter()(np.log10(20.0), 0) == "20"

    # Z Scale ("log_scale") drives the Z axis' own geometry too, not just
    # color (matching MATLAB's combined ZScale+ColorScale control) --
    # reported: "z-scale을 로그로 하니 intensity만 log scale로 되고
    # surf나 waterfall 이미지에서 z축이 log-scale로 바뀌지 않고 있어."
    # `z` has a 0.0 entry (np.arange starts at 0) -- that one cell masks
    # out (log of 0 undefined) without blocking the whole Z axis, unlike a
    # single non-positive X/Y *coordinate*.
    canvas.render_surface(
        x_positive, y_positive, z, mode="surf", xlabel="Time, s", ylabel="Q, 1/A",
        log_scale=True, vmin=1.0, vmax=10.0,
    )
    assert canvas.axis_log_fallback == []
    # Surf's Z limits auto-fit to the plotted (log10-transformed) data
    # range, same as the pre-existing linear-mode behavior -- unlike
    # Waterfall below, Surf never clips its Z geometry to vmin/vmax. `z`'s
    # positive values span 1..11, so its log10 range is (0, log10(11)).
    assert canvas.ax.get_zlim() == pytest.approx((0.0, np.log10(11.0)), abs=0.05)
    assert canvas.ax.zaxis.get_major_formatter()(1.0, 0) == "10"
    # Color still comes from `norm` over the *raw* intensity, not the
    # log10-transformed plotted height -- a `LogNorm` over an already-
    # logged array would double-log it.
    assert type(canvas._image.norm).__name__ == "LogNorm"
    assert (canvas._image.norm.vmin, canvas._image.norm.vmax) == (1.0, 10.0)

    canvas.render_surface(
        x_positive, y_positive, z, mode="waterfall", xlabel="Time, s", ylabel="Q, 1/A",
        log_scale=True, vmin=1.0, vmax=10.0,
    )
    assert canvas.ax.get_zlim() == pytest.approx((0.0, 1.0))
    assert canvas.ax.zaxis.get_major_formatter()(1.0, 0) == "10"

    # An entirely non-positive Z array can't be log-transformed at all --
    # falls back to linear (raw vmin/vmax as the Z limits, no formatter),
    # same shape of fallback X/Y already have.
    z_nonpositive = -np.ones((3, 4), dtype=np.float32)
    canvas.render_surface(
        x_positive, y_positive, z_nonpositive, mode="surf", xlabel="Time, s", ylabel="Q, 1/A",
        log_scale=True, vmin=1.0, vmax=10.0,
    )
    assert canvas.axis_log_fallback == ["Z"]

    vmin, vmax = canvas.render_surface(x, y, z, mode="waterfall", xlabel="Time, s", ylabel="Q, 1/A")
    assert canvas.ax.name == "3d"
    assert len(canvas.ax.collections) == 1  # one PolyCollection, one curtain per x column (frame)
    assert vmin < vmax
    assert canvas._colorbar is not None
    # Default coloring is by intensity -- no explicit colorbar label set.
    assert canvas._colorbar.ax.get_ylabel() == ""

    # color_by="index" colors each curtain by its own frame position
    # (0..n-1) on a separate scale, independent of the vmin/vmax just
    # returned for the z-axis/intensity range -- the z limits stay
    # intensity-based either way (see render_surface's own docstring).
    index_vmin, index_vmax = canvas.render_surface(
        x, y, z, mode="waterfall", xlabel="Time, s", ylabel="Q, 1/A",
        color_by="index", colorbar_label="Frame # (0-3)",
    )
    assert (index_vmin, index_vmax) == (vmin, vmax)
    assert canvas._colorbar.ax.get_ylabel() == "Frame # (0-3)"

    # An explicit xlim/ylim overrides waterfall's own internal
    # set_xlim/set_ylim (auto-fit to the full x/y data range).
    canvas.render_surface(
        x, y, z, mode="waterfall", xlabel="Time, s", ylabel="Q, 1/A", xlim=(1.0, 20.0), ylim=(0.15, 0.25),
    )
    assert canvas.ax.get_xlim() == (1.0, 20.0)
    assert canvas.ax.get_ylim() == (0.15, 0.25)

    # And switching back to the plain 2D mesh after three 3D renders works.
    canvas.render_mesh(x, y, z, xlabel="Time, s", ylabel="Q, 1/A")
    assert canvas.ax.name == "rectilinear"


def test_reset_view_restores_full_extent_after_zoom(qapp):
    """Regression: Reset Zoom (ImageCanvas.reset_view -> toolbar.home()) was
    a silent no-op. We never route through NavigationToolbar2's own pan/zoom
    handlers (all axis changes go through ax.set_xlim/set_ylim directly via
    set_axis_limits), which are what normally populate its nav stack -- left
    alone the stack stays empty forever, and home() does nothing if there's
    nothing on the stack to go home to. render() now seeds that stack with
    the fresh full-extent view every time it draws a complete new dataset,
    so Home has a real baseline to restore."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import ImageCanvas

    canvas = ImageCanvas()
    data = np.random.default_rng(0).uniform(1, 100, size=(50, 60))
    canvas.render(data, extent=(1, 60, 50, 1))
    full_xlim, full_ylim = canvas.get_axis_limits()

    canvas.set_axis_limits((5.0, 20.0), (10.0, 30.0))
    assert canvas.get_axis_limits() != (full_xlim, full_ylim)

    canvas.reset_view()
    assert canvas.get_axis_limits() == (full_xlim, full_ylim)


def test_colorscale_and_colormap_apply_immediately(qapp, tmp_path):
    """ColorScale and Colormap take effect the moment you pick one, same as
    every other Image Controller display setting (see
    test_display_spins_apply_live and test_update_resets_range_to_full_extent
    below) -- none of them wait for a separate Update click anymore."""

    import h5py
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
    from pgixs_python.ui_backend import console_session as cs

    frame_path = tmp_path / "frame.h5"
    data = np.random.default_rng(0).uniform(10, 100, size=(20, 24)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])

    console = ConsoleWidget(gixs_config())
    console._activate()
    result = cs.console_load(
        {
            "directory": str(tmp_path),
            "filename": frame_path.name,
            "mode": "gixs",
            "detector": "TestDetector",
            "multishot": False,
        }
    )
    assert result["ok"], result.get("message")
    console._on_load_finished(result)

    assert type(console.image_canvas._image.norm).__name__ == "Normalize"
    console.color_scale_combo.setCurrentText("Log")
    assert type(console.image_canvas._image.norm).__name__ == "LogNorm"

    assert console.image_canvas._image.get_cmap().name == "viridis"
    console.colormap_combo.setCurrentText("magma")
    assert console.image_canvas._image.get_cmap().name == "magma"


def _load_test_frame(tmp_path):
    import h5py
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
    from pgixs_python.ui_backend import console_session as cs

    frame_path = tmp_path / "frame.h5"
    data = np.random.default_rng(0).uniform(10, 100, size=(20, 24)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])

    console = ConsoleWidget(gixs_config())
    console._activate()
    result = cs.console_load(
        {
            "directory": str(tmp_path),
            "filename": frame_path.name,
            "mode": "gixs",
            "detector": "TestDetector",
            "multishot": False,
        }
    )
    assert result["ok"], result.get("message")
    console._on_load_finished(result)
    cs.get_session().calibration_loaded = True
    return console


def _load_test_frame_trxs(tmp_path):
    """TrXS counterpart of _load_test_frame -- same synthetic frame, but
    through trxs_config() (no Cakecut, has Circular) so widget-level tests
    can confirm a GIXS-side fix actually reaches the shared ConsoleWidget/
    console_session.py code TrXS runs through too, not just assume it via
    the shared architecture."""

    import h5py
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, trxs_config
    from pgixs_python.ui_backend import console_session as cs

    frame_path = tmp_path / "frame.h5"
    data = np.random.default_rng(0).uniform(10, 100, size=(20, 24)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])

    console = ConsoleWidget(trxs_config())
    console._activate()
    result = cs.console_load(
        {
            "directory": str(tmp_path),
            "filename": frame_path.name,
            "mode": "trxs",
            "detector": "TestDetector",
            "multishot": False,
        }
    )
    assert result["ok"], result.get("message")
    console._on_load_finished(result)
    cs.get_session().calibration_loaded = True
    return console


def test_azi_q_label_updates_after_qmap_cal_and_pixel_change(qapp, tmp_path):
    """Regression: "azimuthal process에서 pixel position이 정해져도 q값이 얼마인지
    안나오네" -- the "Q [1/A]" readout in Azimuthal Process was built but
    never wired to anything, so it always showed the initial placeholder
    "0" no matter what pixel position or qmap was set."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    # Small enough that beam_y + pixel stays inside the 24-column test
    # frame both before and after the +3 step below -- the default 202px
    # pixel value clips to the last column on this tiny synthetic frame,
    # which would make the "before" and "after" Q reads identical for a
    # reason unrelated to the wiring under test.
    console.azi_pixel_spin.setValue(5.0)
    assert console.azi_q_label.text() == "0"

    console._on_qmap_clicked()
    first_text = console.azi_q_label.text()
    assert first_text not in ("0", "n/a")

    console.azi_pixel_spin.setValue(console.azi_pixel_spin.value() + 3.0)
    second_text = console.azi_q_label.text()
    assert second_text != first_text


def test_line_plot_draws_x_axis_error_bars_only_when_q_uncertainty_is_on(qapp, tmp_path):
    """Regression: "GIXS에서 FP를 입력한 후 q uncertainity에 마크하여
    q-Map Cal.을 하면... 1D process에도 column이 생성되게 해줘. 그래서 1D
    result에 x축 errorbar가 있으면 좋겠어" -- with the "q uncertainty"
    checkbox off (or FP left at 0), q-Map Cal. must behave exactly as
    before (plain line, no error bars); with it on and a real footprint
    set, 1d Plot must draw the resulting q_std as a horizontal (x-axis)
    error bar -- a real Axes.errorbar, not just a plain line."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)

    console.footprint_spin.setValue(0.0)
    console.q_uncertain_check.setChecked(False)
    console._on_qmap_clicked()
    console._on_line_plot_clicked()
    assert len(console.curve_canvas.ax.containers) == 0

    console.footprint_spin.setValue(5.0)
    console.q_uncertain_check.setChecked(True)
    console._on_qmap_clicked()
    console._on_line_plot_clicked()
    assert len(console.curve_canvas.ax.containers) == 1


def test_azi_draw_shows_range_boundaries_not_a_single_circle(qapp, tmp_path):
    """Regression: "range를 정해져도 반영이 안되는것 같아" -- Draw only ever plotted
    one circle at the raw pixel value, so editing Range(-/+) never visibly
    changed the overlay even though console_azimuthal_1d's actual
    computation already used it correctly. It should draw both ring
    boundaries (pixel + range_minus, pixel + range_plus)."""

    console = _load_test_frame(tmp_path)
    console.azi_pixel_spin.setValue(10.0)
    console.azi_range_minus_spin.setValue(-2.0)
    console.azi_range_plus_spin.setValue(4.0)

    console._on_azi_draw_clicked()

    circles = console.image_canvas._overlay_artists
    assert len(circles) == 2
    radii = sorted(c.get_radius() for c in circles)
    assert radii == pytest.approx([8.0, 14.0])


def test_display_spins_apply_live(qapp, tmp_path):
    """Contrast/Y-Range/Z-Range/Font Sz used to require an explicit Update
    click; now every one of them applies as soon as it changes (spin boxes
    have setKeyboardTracking(False), so this fires on step/scroll/Enter/
    focus-loss, not on every keystroke)."""

    console = _load_test_frame(tmp_path)

    console.cmin_spin.setValue(15.0)
    console.cmax_spin.setValue(80.0)
    assert (console.image_canvas._image.norm.vmin, console.image_canvas._image.norm.vmax) == (15.0, 80.0)

    console.font_sz_spin.setValue(20)
    assert console.image_canvas.ax.xaxis.label.get_fontsize() == 20.0

    console.yrange_min_spin.setValue(5.0)
    console.yrange_max_spin.setValue(15.0)
    assert console.image_canvas.ax.get_xlim() == (5.0, 15.0)

    console.zrange_min_spin.setValue(3.0)
    console.zrange_max_spin.setValue(12.0)
    # Raw (uncentered) pixel mode reads top-to-bottom as small-to-large, so
    # the axis-reversed ylim is (max, min).
    assert console.image_canvas.ax.get_ylim() == (12.0, 3.0)


def test_update_resets_range_to_full_extent_keeps_other_settings(qapp, tmp_path):
    """Regression for the redefined Update button: it no longer re-applies
    ColorScale/Colormap/Contrast/Font (those are live now, see
    test_display_spins_apply_live) -- its only job is putting Y-Range/
    Z-Range back to the full extent of the current data, leaving every
    other display setting exactly as the user left it."""

    console = _load_test_frame(tmp_path)
    full_xlim = console.image_canvas.ax.get_xlim()
    full_ylim = console.image_canvas.ax.get_ylim()

    console.color_scale_combo.setCurrentText("Log")
    console.colormap_combo.setCurrentText("magma")
    console.cmin_spin.setValue(15.0)
    console.cmax_spin.setValue(80.0)
    console.font_sz_spin.setValue(20)
    console.yrange_min_spin.setValue(5.0)
    console.yrange_max_spin.setValue(15.0)
    console.zrange_min_spin.setValue(3.0)
    console.zrange_max_spin.setValue(12.0)
    assert console.image_canvas.ax.get_xlim() != full_xlim

    console._on_update_clicked()

    assert console.image_canvas.ax.get_xlim() == full_xlim
    assert console.image_canvas.ax.get_ylim() == full_ylim
    assert (console.yrange_min_spin.value(), console.yrange_max_spin.value()) == full_xlim
    assert (console.zrange_min_spin.value(), console.zrange_max_spin.value()) == (
        min(full_ylim),
        max(full_ylim),
    )
    # Everything else stays as the user set it.
    assert type(console.image_canvas._image.norm).__name__ == "LogNorm"
    assert console.image_canvas._image.get_cmap().name == "magma"
    assert (console.image_canvas._image.norm.vmin, console.image_canvas._image.norm.vmax) == (15.0, 80.0)
    assert console.image_canvas.ax.xaxis.label.get_fontsize() == 20.0


def test_update_clears_a_previously_drawn_line_overlay(qapp, tmp_path):
    """Reported live: "라인을 그렸다가, Update누르면 라인이 없는 퓨어이미지가
    되게 해줘" -- a line/wedge/circle drawn via Position+Draw (or Azimuthal
    Draw) had no way to go away short of switching Map format or reloading.
    Update is the button the user already reaches for to get back to a
    plain view (see test_update_resets_range_to_full_extent_keeps_other_settings),
    so it now also wipes any overlay currently on the image and disarms
    Position mode (see test_update_stays_clear_after_a_live_range_edit for
    why disarming, not just wiping the artists, is required)."""

    console = _load_test_frame(tmp_path)
    console._arm_position_mode("line")
    console._draw_line_overlay()
    assert len(console.image_canvas._overlay_artists) > 0

    console._on_update_clicked()

    assert len(console.image_canvas._overlay_artists) == 0
    assert console._position_mode is None


def test_update_stays_clear_after_a_live_range_edit(qapp, tmp_path):
    """Regression: Update alone made the line disappear, but every other
    display control (Y-Range/Z-Range/Colormap/...) re-renders live through
    _apply_display_settings -> _render_image, which redraws the line
    whenever _position_mode is still "line" -- so the "cleared" line came
    right back the moment Y-Range or Z-Range was edited afterward
    (reported: "update누르면 선이 없어지는데, Y-range, Z-range를 바꾸면
    다시 나오네"). Update must disarm Position mode, not just wipe the
    artists once, so a later live-display edit stays clear too."""

    console = _load_test_frame(tmp_path)
    console._arm_position_mode("line")
    console._draw_line_overlay()
    console._on_update_clicked()
    assert len(console.image_canvas._overlay_artists) == 0

    console.yrange_min_spin.setValue(2.0)
    console.yrange_max_spin.setValue(18.0)

    assert len(console.image_canvas._overlay_artists) == 0


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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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


def test_curve_canvas_hover_survives_none_in_data(qapp):
    """Regression for a crash reported live: hovering over the 1D Result
    curve raised `TypeError: float() argument must be a string or a real
    number, not 'NoneType'` inside _on_motion. Root cause was upstream (see
    test_line_plot_uses_full_resolution_curve_not_json_preview below), but
    render()/_data_x/_data_y should be robust on their own to whatever a
    caller hands them -- JSON-style data with None standing in for
    NaN/masked points is a completely ordinary shape to receive."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import CurveCanvas

    class _FakeEvent:
        def __init__(self, inaxes, xdata):
            self.inaxes = inaxes
            self.xdata = xdata

    canvas = CurveCanvas()
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, None, 9.0, None, 25.0]  # matches downsample_1d_for_json's shape
    canvas.render(x, y)

    assert canvas._data_y.dtype == np.float64
    assert np.isnan(canvas._data_y[1])

    for value in x:
        canvas._on_motion(_FakeEvent(canvas.ax, value))  # must not raise


def test_curve_canvas_scroll_zooms_in_and_out_around_cursor(qapp):
    """Regression: the 1D Result plot had no zoom interaction at all --
    scroll-wheel over the curve must shrink/grow the view around the
    cursor position, mirroring ImageCanvas's existing scroll-zoom."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import CurveCanvas

    class _FakeScrollEvent:
        def __init__(self, inaxes, xdata, ydata, button):
            self.inaxes = inaxes
            self.xdata = xdata
            self.ydata = ydata
            self.button = button

    canvas = CurveCanvas()
    x = np.linspace(0.0, 10.0, 11)
    y = x**2
    canvas.render(x, y)

    xlim0, ylim0 = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    canvas._on_scroll(_FakeScrollEvent(canvas.ax, 5.0, 25.0, "up"))
    xlim1, ylim1 = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    assert (xlim1[1] - xlim1[0]) < (xlim0[1] - xlim0[0])
    assert (ylim1[1] - ylim1[0]) < (ylim0[1] - ylim0[0])

    canvas._on_scroll(_FakeScrollEvent(canvas.ax, 5.0, 25.0, "down"))
    xlim2, ylim2 = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    assert (xlim2[1] - xlim2[0]) > (xlim1[1] - xlim1[0])
    assert (ylim2[1] - ylim2[0]) > (ylim1[1] - ylim1[0])


def test_curve_canvas_scroll_zoom_stays_positive_on_log_axis(qapp):
    """Naive linear interpolation of a log axis's limits can walk a bound
    through zero/negative, which matplotlib rejects -- _zoomed_limits must
    do the interpolation in log10 space instead when the axis is log."""

    import numpy as np

    from pgixs_python.desktop.plot_canvas import CurveCanvas

    class _FakeScrollEvent:
        def __init__(self, inaxes, xdata, ydata, button):
            self.inaxes = inaxes
            self.xdata = xdata
            self.ydata = ydata
            self.button = button

    canvas = CurveCanvas()
    x = np.linspace(1.0, 10.0, 10)
    y = x**2
    canvas.render(x, y, xlog=True, ylog=True)

    for _ in range(5):
        canvas._on_scroll(_FakeScrollEvent(canvas.ax, 2.0, 4.0, "up"))
    xlim, ylim = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    assert xlim[0] > 0 and xlim[1] > 0
    assert ylim[0] > 0 and ylim[1] > 0


def test_curve_reset_zoom_button_restores_full_view(qapp):
    import numpy as np

    console = ConsoleWidget(gixs_config())
    x = np.linspace(0.0, 10.0, 11)
    y = x**2
    console.curve_canvas.render(x, y)
    full_xlim, full_ylim = console.curve_canvas.ax.get_xlim(), console.curve_canvas.ax.get_ylim()

    console.curve_canvas.ax.set_xlim(2.0, 4.0)
    console.curve_canvas.ax.set_ylim(2.0, 20.0)
    assert console.curve_canvas.ax.get_xlim() != full_xlim

    console._on_curve_reset_zoom_clicked()
    assert console.curve_canvas.ax.get_xlim() == pytest.approx(full_xlim)
    assert console.curve_canvas.ax.get_ylim() == pytest.approx(full_ylim)


def test_line_plot_uses_full_resolution_curve_not_json_preview(qapp, tmp_path):
    """Regression for the same live crash: _on_line_plot_clicked (and its
    Azimuthal/Circular siblings) were rendering `result["preview"]` --
    downsampled *and* JSON-sanitized (masked points become None) for the
    web UI's HTTP transport, a limit that doesn't apply in-process. They
    should render `session.last_curve` instead: the real, full-resolution
    float array where masked points are NaN, not None."""

    import h5py
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
    from pgixs_python.ui_backend import console_session as cs

    frame_path = tmp_path / "frame.h5"
    data = np.random.default_rng(0).uniform(10, 100, size=(20, 24)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])

    console = ConsoleWidget(gixs_config())
    console._activate()
    result = cs.console_load(
        {
            "directory": str(tmp_path),
            "filename": frame_path.name,
            "mode": "gixs",
            "detector": "TestDetector",
            "multishot": False,
        }
    )
    assert result["ok"], result.get("message")
    console._on_load_finished(result)
    cs.get_session().calibration_loaded = True

    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0})
    assert qmap_result["ok"], qmap_result.get("message")

    # Force console_line_1d's *preview* (only) to contain a None -- if the
    # widget were still rendering from it, this would resurface the crash;
    # last_curve is left as the session naturally computed it.
    original_console_line_1d = cs.console_line_1d

    def _patched(payload):
        response = original_console_line_1d(payload)
        if response.get("ok"):
            response["preview"]["intensity"][0] = None
        return response

    cs.console_line_1d = _patched
    try:
        console._on_line_plot_clicked()
    finally:
        cs.console_line_1d = original_console_line_1d

    assert console.curve_canvas._data_y.dtype == np.float64
    assert None not in console.curve_canvas._data_y.tolist()


def test_line_plot_map_format_selects_an_independent_qspace_cut(qapp, tmp_path):
    """Regression: "1D process해서 pixel map에서 cutting하는 것이 Qxy/Qz
    cutting에 반영되서 진행되는데, 그렇게 하지말고 독립적으로 되면
    좋겠어" -- Map format used to only affect the 2D image display; Line
    1D Process always cut the raw pixel data regardless of it.
    _on_line_plot_clicked must now forward the combo's own selection into
    console_line_1d's payload, and switching it to "Qxy/Qz" must produce
    a genuinely different curve from "Pixel" (see
    test_console_line_1d_qxyqz_mapping_runs_an_independent_cakecut in
    test_ui_backend_console_session.py for the backend-level version of
    this same check)."""

    import numpy as np

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})
    assert qmap_result["ok"], qmap_result.get("message")

    console.map_format_combo.blockSignals(True)
    console.map_format_combo.setCurrentText("Pixel")
    console.map_format_combo.blockSignals(False)
    console._on_line_plot_clicked()
    assert cs.get_session().last_curve["mapping"] == "Pixel"
    pixel_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    console.map_format_combo.blockSignals(True)
    console.map_format_combo.setCurrentText("Qxy/Qz")
    console.map_format_combo.blockSignals(False)
    console._on_line_plot_clicked()
    assert cs.get_session().last_curve["mapping"] == "Qxy/Qz"
    qspace_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert qspace_intensity.shape != pixel_intensity.shape or not np.allclose(
        qspace_intensity, pixel_intensity, equal_nan=True
    )


def test_status_message_with_long_path_does_not_widen_panel(qapp):
    """Regression reported live: clicking Save Img visibly resized the left
    panel. Root cause: its status message includes full Windows paths
    (backslashes, no spaces), and Qt's word-wrap only breaks at whitespace
    -- an unbreakable ~800px-wide token forced the label, and with it the
    whole scroll area, wider instead of wrapping. _set_status now runs
    every message through _wrap_long_tokens, which inserts a zero-width
    space after each backslash so long paths can still wrap."""

    from PySide6.QtWidgets import QScrollArea

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    # No show()/processEvents() needed -- minimumSizeHint() is computed from
    # the already-built layout tree regardless of visibility, and skipping
    # them avoids piling up yet another shown top-level widget in a test
    # session that already constructs 190+ of them against one shared
    # QApplication.
    console = ConsoleWidget(gixs_config())
    scroll = console.findChildren(QScrollArea)[0]
    group = scroll.widget()
    baseline_width = group.minimumSizeHint().width()

    long_path_message = (
        r"Saved D:\Claude_Code\PGIXS_python\testdata\GrazingIncidence\HZO\HDH_155_0001_2Dinfo.h5 "
        r"and D:\Claude_Code\PGIXS_python\testdata\GrazingIncidence\HZO\HDH_155_0001_2d.png."
    )
    console._set_status(long_path_message, True)

    assert group.minimumSizeHint().width() == baseline_width
    # The message itself must still be fully recoverable -- only zero-width
    # break points were inserted, nothing was truncated or elided.
    assert console.status_label.text().replace(chr(0x200B), "") == long_path_message


def _load_and_plot_line(tmp_path):
    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    # _load_test_frame's 20x24 synthetic frame keeps the default beam_pos
    # (1000, 1800), which sits nowhere near the array -- cake_average then
    # finds zero pixels along any line through it. Move the beam inside the
    # frame so console_line_1d actually has geometry to average over.
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})
    assert qmap_result["ok"], qmap_result.get("message")
    console._on_line_plot_clicked()
    assert cs.get_session().last_curve is not None, console.curve_status_label.text()
    return console


def test_curve_scales_apply_live_and_persist_across_new_plots(qapp, tmp_path):
    """1D Result's X/Y scale controls: log toggles apply the moment they're
    picked (same live-control pattern as Image Controller's ColorScale),
    are independent of each other, and stay in effect when a *different*
    1D plot (e.g. switching from Line to Azimuthal) is rendered next --
    _render_curve reads the combos on every render, not just at toggle
    time."""

    console = _load_and_plot_line(tmp_path)
    assert (console.curve_canvas.ax.get_xscale(), console.curve_canvas.ax.get_yscale()) == ("linear", "linear")

    console.curve_xscale_combo.setCurrentText("Log")
    console.curve_yscale_combo.setCurrentText("Log")
    assert (console.curve_canvas.ax.get_xscale(), console.curve_canvas.ax.get_yscale()) == ("log", "log")

    console.curve_yscale_combo.setCurrentText("Linear")
    assert (console.curve_canvas.ax.get_xscale(), console.curve_canvas.ax.get_yscale()) == ("log", "linear")

    console._on_azi_plot_clicked()
    assert console.curve_canvas.ax.get_xscale() == "log"


def test_curve_save_writes_csv_and_reports_full_paths(qapp, tmp_path):
    """Regression for "이 1D profile도 세이브가 되어야돼": the 1D Result panel's
    own Save button (independent of Image Controller's Save Img, which
    requires q-Map Cal.'s 2D-info bundle) exports the current curve as a
    CSV via the new console_save_curve backend function, plus a PNG
    snapshot of the plot."""

    console = _load_and_plot_line(tmp_path)

    console._on_curve_save_clicked()

    # Saves land in their own <stem>/ subfolder now (see
    # test_console_save_creates_stem_named_subfolder in
    # test_ui_backend_console_session.py), and a "line" curve's filename
    # stamps its reference angle -- line_angle_spin defaults to 90.0.
    csv_path = tmp_path / "frame" / "frame_1D_line_ang90deg.csv"
    png_path = tmp_path / "frame" / "frame_1D_line_ang90deg.png"
    assert csv_path.exists()
    assert png_path.exists()
    status_text = console.curve_status_label.text().replace(chr(0x200B), "")
    assert str(csv_path) in status_text
    assert str(png_path) in status_text


def test_curve_save_before_any_plot_fails_clearly(qapp, tmp_path):
    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    console = ConsoleWidget(gixs_config())
    console._activate()

    console._on_curve_save_clicked()

    status_text = console.curve_status_label.text().replace(chr(0x200B), "")
    assert "1D plot" in status_text


def test_cakecut_and_accumulation_checkboxes_are_mutually_exclusive(qapp):
    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    console = ConsoleWidget(gixs_config())

    console.cakecut_check.setChecked(True)
    assert console.cakecut_check.isChecked() is True
    assert console.accum_check.isChecked() is False

    console.accum_check.setChecked(True)
    assert console.accum_check.isChecked() is True
    assert console.cakecut_check.isChecked() is False


def test_pos_move_fields_enable_only_the_relevant_axis(qapp):
    """Regression for "0또는180에서는 y는 0이고... 그 외의 모든 angle cut에서는
    y는 0, z는 0": Pos move only ever applies to one axis, and only for an
    axis-aligned cut -- angle 0/180 moves along z only, angle 90 moves
    along y only, every other angle ignores both. The irrelevant field(s)
    must be disabled *and* reset to 0 so a leftover value can't sit there
    silently having no effect."""

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    console = ConsoleWidget(gixs_config())

    # Default angle is 90 -> y usable, z locked at 0.
    assert console.y_pos_move_spin.isEnabled() is True
    assert console.z_pos_move_spin.isEnabled() is False
    assert console.z_pos_move_spin.value() == 0.0

    console.z_pos_move_spin.setValue(7.0)  # would be a no-op while disabled in the UI; set directly for the test
    console.line_angle_spin.setValue(0.0)
    assert console.y_pos_move_spin.isEnabled() is False
    assert console.y_pos_move_spin.value() == 0.0
    assert console.z_pos_move_spin.isEnabled() is True

    console.line_angle_spin.setValue(180.0)
    assert console.y_pos_move_spin.isEnabled() is False
    assert console.z_pos_move_spin.isEnabled() is True

    console.y_pos_move_spin.setValue(9.0)
    console.line_angle_spin.setValue(45.0)
    assert console.y_pos_move_spin.isEnabled() is False
    assert console.y_pos_move_spin.value() == 0.0
    assert console.z_pos_move_spin.isEnabled() is False
    assert console.z_pos_move_spin.value() == 0.0


def test_display_xy_to_pixel_offset_pixel_mode_is_identity(qapp):
    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config

    console = ConsoleWidget(gixs_config())
    assert console._axis_mode == "pixel"

    assert console._display_xy_to_pixel_offset(12.5, -7.25) == (12.5, -7.25)


def test_display_xy_to_pixel_offset_q_mode_round_trips_through_real_qmap(qapp, tmp_path):
    """Regression for the "Position" click giving a wrong angle in Qxy/Qz
    mode: qr is a magnitude/projection, not a linear pixel coordinate, so
    the old atan2-on-display-xy approach was only valid in Pixel mode.
    _display_xy_to_pixel_offset now looks up the actual nearest detector
    pixel in the real per-pixel Q maps -- feeding a pixel's own Q value
    back in must recover that same pixel's offset from the beam."""

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    beam_y, beam_z = 12.0, 10.0
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": beam_y, "beam_z": beam_z})
    assert qmap_result["ok"], qmap_result.get("message")

    console._axis_mode = "q"
    console.map_format_combo.blockSignals(True)
    console.map_format_combo.setCurrentText("Qxy/Qz")
    console.map_format_combo.blockSignals(False)

    session = cs.get_session()
    row, col = 3, 15  # an arbitrary pixel away from the beam and the array edges
    qr_value = float(session.qmap["qr"][row, col])
    qz_value = float(session.qmap["qz"][row, col])

    px, pz = console._display_xy_to_pixel_offset(qr_value, qz_value)

    assert px == pytest.approx((col + 1) - beam_y)
    assert pz == pytest.approx((row + 1) - beam_z)


def test_azimuthal_position_click_in_q_mode_sets_a_pixel_radius_not_a_q_value(qapp, tmp_path):
    """Regression: "Qxy/Qz 모드일때, Azimuthal 프로세스에서 포지션을 정해서
    클릭하면 Q값이 아닌 pixel값에 Q값이 들어가서 이상한 곳에 포지션이 생겨" --
    _on_image_clicked's azimuthal branch hypot'd the click's raw display
    (x, y) straight against the pixel beam center, which in Q-space mode
    are Q values (Angstrom^-1), not pixel coordinates -- azi_pixel_spin
    ended up holding a Q magnitude instead of a real pixel radius."""

    import math

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    beam_y, beam_z = 12.0, 10.0
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": beam_y, "beam_z": beam_z})
    assert qmap_result["ok"], qmap_result.get("message")

    console._axis_mode = "q"
    console.map_format_combo.blockSignals(True)
    console.map_format_combo.setCurrentText("Qxy/Qz")
    console.map_format_combo.blockSignals(False)

    session = cs.get_session()
    row, col = 3, 15  # an arbitrary pixel away from the beam and the array edges
    qr_value = float(session.qmap["qr"][row, col])
    qz_value = float(session.qmap["qz"][row, col])
    expected_radius = math.hypot((col + 1) - beam_y, (row + 1) - beam_z)

    console._arm_position_mode("azimuthal")
    console._on_image_clicked(qr_value, qz_value)

    # azi_pixel_spin only keeps 1 decimal place, so compare with that
    # precision in mind rather than an exact/relative match.
    assert console.azi_pixel_spin.value() == pytest.approx(expected_radius, abs=0.05)
    # The old bug would have set this to hypot(qr_value, qz_value) instead --
    # a wildly different, tiny Q-space magnitude, not a pixel count.
    assert console.azi_pixel_spin.value() != pytest.approx(math.hypot(qr_value, qz_value), rel=1e-2)


def test_azi_draw_in_q_mode_draws_circles_scaled_to_q_not_raw_pixel_radii(qapp, tmp_path):
    """Regression: "Qxy/Qz map에서 Azimuthal process에 Draw가 되고 있지
    않아" -- azi_pixel_spin/Range(-/+) are always real pixel counts, but
    _on_azi_draw_clicked drew a Circle patch of that same radius directly
    on the Q-space-scaled axes in Qxy/Qz mode, landing far outside the
    visible (a few 1/A wide) plot -- it read as nothing being drawn at
    all. Circle radii must be converted to their Q-space equivalent first."""

    import math

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    beam_y, beam_z = 12.0, 10.0
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": beam_y, "beam_z": beam_z})
    assert qmap_result["ok"], qmap_result.get("message")

    console._axis_mode = "q"
    console.map_format_combo.blockSignals(True)
    console.map_format_combo.setCurrentText("Qxy/Qz")
    console.map_format_combo.blockSignals(False)

    console.azi_pixel_spin.setValue(10.0)
    console.azi_range_minus_spin.setValue(-2.0)
    console.azi_range_plus_spin.setValue(4.0)

    console._on_azi_draw_clicked()

    circles = console.image_canvas._overlay_artists
    assert len(circles) == 2
    radii = sorted(c.get_radius() for c in circles)

    expected = sorted(
        console._pixel_radius_to_q_radius(r) for r in (10.0 - 2.0, 10.0 + 4.0)
    )
    assert radii[0] == pytest.approx(expected[0], rel=1e-4)
    assert radii[1] == pytest.approx(expected[1], rel=1e-4)
    # The old bug would have drawn radii of 8/14 (raw pixel counts) --
    # confirm these are genuinely different, much smaller Q-space values.
    assert radii[1] < 1.0


def test_line_overlay_reflects_pos_move_offset(qapp, tmp_path):
    """Regression reported live: '커팅은 되는데 라인을 그릴 때는 반영이 안되네' --
    _on_line_plot_clicked's actual cut already honored Pos move, but
    _draw_line_overlay always drew straight through the beam center,
    ignoring y/z-pos move entirely. Check the drawn line's position in
    both raw and beam-centered pixel display modes, where the two modes'
    opposite y-axis conventions (raw: display y is the row index directly;
    centered: display y is VD = beam_z - row, inverted) require opposite
    signs for the same underlying row shift."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)

    # Raw (uncentered) pixel mode: display y is the row index directly, so
    # a +z_pos_move cut-row shift of beam_z-z_pos_move shows up as -z_pos_move.
    console.line_angle_spin.setValue(0.0)
    console.z_pos_move_spin.setValue(4.0)
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()
    line = console.image_canvas._overlay_artists[-1]
    assert line.get_ydata()[0] == pytest.approx(10.0 - 4.0)

    # Beam-centered pixel mode: display y is VD (inverted relative to raw
    # row), so the same z_pos_move shows up as +z_pos_move instead.
    console._pixel_centered = True
    console._render_pixel_image()
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()
    line = console.image_canvas._overlay_artists[-1]
    assert line.get_ydata()[0] == pytest.approx(4.0)

    # y_pos_move at angle 90 (centered mode): display x is HD, same
    # direction as raw column, so no sign flip needed for this axis.
    console.line_angle_spin.setValue(90.0)
    console.y_pos_move_spin.setValue(3.0)
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()
    line = console.image_canvas._overlay_artists[-1]
    assert line.get_xdata()[0] == pytest.approx(3.0)


def test_accumulation_overlay_draws_both_band_edges(qapp, tmp_path):
    """Regression reported live: 'Accumlation cut이 제대로 되는지 표시가 안나와' --
    the overlay drew a single line through the (correctly offset) center,
    giving no indication of the band's actual width. Accumulation is a
    perpendicular pixel band, not a line, so its overlay should show both
    edges (Pos move's offset plus the band's own minus/plus range) instead
    of one centerline -- matching GIXS_gui.m's drawLineOverlay, which draws
    two boundary segments for this mode."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(20.0)
    console.beam_z_spin.setValue(25.0)
    console.line_angle_spin.setValue(90.0)
    console.accum_check.setChecked(True)
    console.y_pos_move_spin.setValue(6.0)
    console.accum_minus_spin.setValue(-2.0)
    console.accum_plus_spin.setValue(3.0)

    console.image_canvas.clear_overlays()
    console._draw_line_overlay()

    lines = console.image_canvas._overlay_artists
    assert len(lines) == 2
    x_positions = sorted(line.get_xdata()[0] for line in lines)
    assert x_positions[0] == pytest.approx(20.0 + 6.0 - 2.0)
    assert x_positions[1] == pytest.approx(20.0 + 6.0 + 3.0)


def _click_angle_deg(x: float, y: float, beam_y: float, beam_z: float) -> float:
    """The same angle formula _on_image_clicked's "line" branch uses --
    ground truth for "what angle does this on-screen point represent",
    independent of whatever formula the overlay-drawing code uses to go
    the other way. Any overlay geometry test should check against this,
    not against the drawing code's own math (self-referential checks can't
    catch a sign bug shared by both the assertion and the code under
    test -- see test_line_and_wedge_overlays_point_at_the_clicked_angle)."""

    import math

    return (math.degrees(math.atan2(-(y - beam_z), -(x - beam_y))) + 360.0) % 360.0


def test_cakecut_overlay_draws_wedge_boundaries_not_pos_move_offset(qapp, tmp_path):
    """Regression reported live: 'Cakecut이 제대로 되는지 표시가 안나와'. Cakecut
    always emanates from the true beam center (ignoring Pos move, matching
    console_line_1d/GIXS_gui.m) and is an angular wedge, not a line -- its
    overlay should show both boundary angles (line_angle +/- the cakecut
    range) instead of a single line at the reference angle."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(20.0)
    console.beam_z_spin.setValue(25.0)
    console.line_angle_spin.setValue(0.0)
    console.cakecut_check.setChecked(True)
    console.cakecut_minus_spin.setValue(-10.0)
    console.cakecut_plus_spin.setValue(15.0)
    # Cakecut ignores Pos move -- set it anyway to prove the wedge lines
    # still pass through the true beam center, not an offset one.
    console.z_pos_move_spin.setValue(4.0)

    console.image_canvas.clear_overlays()
    console._draw_line_overlay()

    lines = console.image_canvas._overlay_artists
    assert len(lines) == 2
    expected_angles = [0.0 + (-10.0), 0.0 + 15.0]  # draw order: minus boundary, then plus
    for line, expected_angle in zip(lines, expected_angles):
        xdata, ydata = line.get_xdata(), line.get_ydata()
        # Each boundary is a ray starting exactly at the true beam center
        # (20, 25), not a full line through it -- the region beyond center
        # in the opposite direction was never part of the selected wedge,
        # and drawing it there looked like an X/bowtie instead of a wedge.
        assert xdata[0] == pytest.approx(20.0)
        assert ydata[0] == pytest.approx(25.0)
        reported = _click_angle_deg(xdata[-1], ydata[-1], 20.0, 25.0)
        assert reported == pytest.approx(expected_angle % 360.0, abs=1e-2)


def test_line_and_wedge_overlays_point_at_the_clicked_angle(qapp, tmp_path):
    """Regression reported live: 'line이 반대로 그려져' -- a Cakecut wedge
    boundary at a non-cardinal angle (e.g. 80 deg, from line_angle=90 +
    cakecut_minus=-10) was drawn mirrored across the vertical axis. Root
    cause: draw_beam_line/​_draw_q_space_line used dx=cos(angle) where the
    correct direction (matching _on_image_clicked's angle formula) is
    dx=-cos(angle) -- invisible for a *bidirectional* line at exactly the
    cardinal 0/90/180/270 angles this app used everywhere until Cakecut's
    wedge boundaries needed a one-directional ray at an arbitrary angle.
    Round-trip check across several non-cardinal angles: draw a plain ray
    at angle theta, then feed its far endpoint through the same formula a
    click on that point would use, and confirm it reports back theta."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)

    for angle in (30.0, 45.0, 80.0, 100.0, 140.0, 200.0, 315.0):
        console.image_canvas.clear_overlays()
        console.image_canvas.draw_beam_line((12.0, 10.0), angle, bidirectional=False)
        (line,) = console.image_canvas._overlay_artists
        xdata, ydata = line.get_xdata(), line.get_ydata()
        reported = _click_angle_deg(xdata[-1], ydata[-1], 12.0, 10.0)
        assert reported == pytest.approx(angle % 360.0, abs=1e-2), f"angle={angle}"


def test_line_overlay_and_click_agree_in_beam_centered_pixel_mode_too(qapp, tmp_path):
    """Regression reported live: after fixing the ray-direction sign bug in
    raw pixel mode, the Cakecut wedge came out upside-down specifically
    once beam-centered mode was active (post Load Info). Root cause:
    beam-centered mode's display-y is VD = beam_z - row, which increases
    going *up* the screen -- the opposite of raw mode's display-y (= row
    directly, increases going *down*). Neither _on_image_clicked's click
    formula nor draw_beam_line's direction accounted for that flip, so
    "angle 90 = up" (correct in raw mode) became "angle 90 = down" once
    centered. _mode_corrected_angle now mirrors the angle across the
    horizontal axis specifically in centered pixel mode to compensate.

    True end-to-end round trip: for several target angles (including
    Cakecut's actual non-cardinal wedge boundaries), draw the overlay via
    _draw_line_overlay, then feed its far endpoint back through
    _on_image_clicked (as if the user clicked there) and confirm
    line_angle_spin lands back on the original target angle."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(100.0)
    console.beam_z_spin.setValue(100.0)
    console._pixel_centered = True
    console._render_pixel_image()
    assert console._axis_mode == "pixel" and console._pixel_centered

    # Physical-direction sanity check first: clicking "up" (larger VD, per
    # centered mode's flipped convention) must still report 90, matching
    # the "90: up" legend -- same requirement as raw mode.
    console._arm_position_mode("line")
    console._on_image_clicked(0.0, 10.0)
    assert console.line_angle_spin.value() == pytest.approx(90.0)

    for target_angle in (30.0, 90.0, 140.0, 200.0):
        console.cakecut_check.setChecked(False)
        console.line_angle_spin.setValue(target_angle)
        console.image_canvas.clear_overlays()
        console._draw_line_overlay()
        (line,) = console.image_canvas._overlay_artists
        xdata, ydata = line.get_xdata(), line.get_ydata()

        console._arm_position_mode("line")
        console._on_image_clicked(xdata[1], ydata[1])
        assert console.line_angle_spin.value() == pytest.approx(target_angle % 360.0, abs=1e-2), f"angle={target_angle}"

    # And the actual reported scenario: Cakecut's non-cardinal wedge
    # boundaries (line_angle=90, -10/+50 -> 80 and 140 deg).
    console.line_angle_spin.setValue(90.0)
    console.cakecut_check.setChecked(True)
    console.cakecut_minus_spin.setValue(-10.0)
    console.cakecut_plus_spin.setValue(50.0)
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()
    lines = console.image_canvas._overlay_artists
    for line, expected_angle in zip(lines, [80.0, 140.0]):
        xdata, ydata = line.get_xdata(), line.get_ydata()
        console._arm_position_mode("line")
        console._on_image_clicked(xdata[1], ydata[1])
        assert console.line_angle_spin.value() == pytest.approx(expected_angle, abs=1e-2)


def _load_test_frame_in_q_mode(tmp_path, beam_y=12.0, beam_z=10.0, mapping="Qxy/Qz"):
    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": beam_y, "beam_z": beam_z})
    assert qmap_result["ok"], qmap_result.get("message")
    # Unblocked (not blockSignals + a manual _axis_mode assignment) --
    # _draw_q_space_line now draws its straight line via draw_beam_line,
    # which extends the ray to the axes' *current* view limits, so those
    # limits actually need to be the real Q-space ones from rendering the
    # Qxy/Qz (or Qy/Qz) background image, not whatever the previous
    # pixel-space image happened to leave them at.
    console.map_format_combo.setCurrentText(mapping)
    return console


def test_polar_check_enabled_only_once_a_q_map_view_is_active(qapp, tmp_path):
    console = _load_test_frame(tmp_path)
    assert not console.polar_check.isEnabled()
    assert not console.polar_check.isChecked()

    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    from pgixs_python.ui_backend import console_session as cs

    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]

    console.map_format_combo.setCurrentText("Qxy/Qz")
    assert console.polar_check.isEnabled()

    console.map_format_combo.setCurrentText("Pixel")
    assert not console.polar_check.isEnabled()
    assert not console.polar_check.isChecked()


def test_polar_toggle_switches_view_and_recovers_cartesian_on_uncheck(qapp, tmp_path):
    """The actual feature request: from Qxy/Qz, be able to switch to a
    polar (angle vs |Q|) view and switch back. Checking Polar must change
    the displayed data/axis labels; unchecking must recover the exact
    Cartesian view (same data console_qspace_image would produce)."""

    import numpy as np

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    console.map_format_combo.setCurrentText("Qxy/Qz")
    cartesian_data = console._current_image_data.copy()
    cartesian_xlabel = console._current_xlabel

    console.polar_check.setChecked(True)
    assert console._current_xlabel == "Angle, deg"
    # Regression: "polar coordinate로 보면 Q로 나와. q로 맞춰줘" -- Cartesian
    # views label q_xy/q_y/q_z in lowercase; Polar's radial axis must match
    # that convention (it was shipping as uppercase "Q, 1/A").
    assert console._current_ylabel == "q, 1/A"
    assert console._current_image_data.shape != cartesian_data.shape or not np.allclose(
        console._current_image_data, cartesian_data, equal_nan=True
    )

    console.polar_check.setChecked(False)
    assert console._current_xlabel == cartesian_xlabel
    np.testing.assert_allclose(console._current_image_data, cartesian_data, equal_nan=True)


def test_polar_view_uses_auto_aspect_not_equal(qapp, tmp_path):
    """Regression: "polar view를 보면 그림이 이렇게 찌그러져" (the polar plot renders
    squashed to a sliver). Root cause was aspect="equal" being hardcoded --
    angle (0-360deg) and |Q| (maybe 0-0.1 1/A) have wildly different
    natural scales, so forcing equal data-units-per-pixel crushes the
    image into a thin strip. Polar must render with aspect="auto"; every
    other view (Pixel, Cartesian Qxy/Qz, Qy/Qz) must keep aspect="equal"."""

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    console.map_format_combo.setCurrentText("Qxy/Qz")
    assert console.image_canvas.ax.get_aspect() == 1.0

    console.polar_check.setChecked(True)
    assert console.image_canvas.ax.get_aspect() == "auto"

    console.polar_check.setChecked(False)
    assert console.image_canvas.ax.get_aspect() == 1.0


def test_polar_check_resets_when_map_format_changes(qapp, tmp_path):
    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    console.map_format_combo.setCurrentText("Qxy/Qz")
    console.polar_check.setChecked(True)
    assert console.polar_check.isChecked()

    console.map_format_combo.setCurrentText("Qy/Qz")
    assert not console.polar_check.isChecked()
    assert console.polar_check.isEnabled()


# ----------------------------------------------------------------------
# TrXS parity: ConsoleWidget/console_session.py are shared between GIXS
# and TrXS (parameterized by ConsoleConfig/_SESSION.mode), so every fix
# above already reaches TrXS through the same code -- these tests confirm
# that at the widget level instead of just assuming it from the shared
# architecture, and lock in the one genuine behavioral difference
# (TrXS has no Cakecut, and sweeps the full 360deg where GIXS is
# restricted to 180deg).
# ----------------------------------------------------------------------
def test_trxs_has_no_cakecut_pos_move_or_accumulation_widgets(qapp, tmp_path):
    """Regression: "1D plot process에서 pos move는 필요없어... Accumulation도
    없어도돼" -- TrXS_gui.m's line cut (Anglecut1d) always goes through the
    beam center with no offset, so neither Cakecut, Pos move, nor
    Accumulation exist for TrXS (see has_pos_move_and_accumulation)."""

    console = _load_test_frame_trxs(tmp_path)
    assert console.cakecut_check is None
    assert console.cakecut_minus_spin is None
    assert console.cakecut_plus_spin is None
    assert console.y_pos_move_spin is None
    assert console.z_pos_move_spin is None
    assert console.accum_check is None
    assert console.accum_minus_spin is None
    assert console.accum_plus_spin is None
    # The redundant, never-wired Qxy/Qz-vs-Qy/Qz "q uncertainty" row is
    # GIXS-only too -- TrXS's map_formats never even offers Qxy/Qz.
    assert console.q_uncertain_check is None
    assert console.mapping_combo is None


def test_trxs_azi_q_label_and_draw_range_boundaries_work(qapp, tmp_path):
    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    console.azi_pixel_spin.setValue(5.0)
    assert console.azi_q_label.text() == "0"

    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    console._on_qmap_clicked()
    assert console.azi_q_label.text() not in ("0", "n/a")

    console.azi_range_minus_spin.setValue(-2.0)
    console.azi_range_plus_spin.setValue(4.0)
    console._on_azi_draw_clicked()
    circles = console.image_canvas._overlay_artists
    assert len(circles) == 2
    radii = sorted(c.get_radius() for c in circles)
    assert radii == pytest.approx([3.0, 9.0])


def test_trxs_curve_zoom_and_reset_work(qapp):
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, trxs_config

    console = ConsoleWidget(trxs_config())
    console._render_curve(np.array([0.0, 1.0, 2.0]), np.array([1.0, 5.0, 3.0]))
    full_xlim = console.curve_canvas.ax.get_xlim()

    class _FakeScrollEvent:
        def __init__(self, inaxes, xdata, ydata, button):
            self.inaxes = inaxes
            self.xdata = xdata
            self.ydata = ydata
            self.button = button

    console.curve_canvas._on_scroll(_FakeScrollEvent(console.curve_canvas.ax, 1.0, 3.0, "up"))
    zoomed_xlim = console.curve_canvas.ax.get_xlim()
    assert (zoomed_xlim[1] - zoomed_xlim[0]) < (full_xlim[1] - full_xlim[0])

    console._on_curve_reset_zoom_clicked()
    assert console.curve_canvas.ax.get_xlim() == pytest.approx(full_xlim)


def test_trxs_line_plot_always_cuts_through_beam_center(qapp, tmp_path):
    """Regression: "여기선 모두 [0,0]을 중심으로 cut 할꺼야" -- with no Pos move
    field to offset it, a TrXS line cut/overlay must always go through the
    beam center. _on_line_plot_clicked's payload must send zeroed
    y_pos_move/z_pos_move/accumulation regardless of anything else, and
    the overlay helpers must resolve to the plain beam center."""

    import numpy as np

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]

    assert console._pos_move_display_offset() == (12.0, 10.0)
    assert console._pos_move_pixel_point() == (12.0, 10.0)

    console.line_angle_spin.setValue(90.0)
    console._on_line_plot_clicked()
    assert cs.get_session().last_curve is not None

    console.image_canvas.clear_overlays()
    console._draw_line_overlay()
    # A plain (bidirectional) line is two vertices, -length and +length
    # from the center along the angle -- their average is the center,
    # since neither endpoint is the center itself.
    (line,) = console.image_canvas._overlay_artists
    xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    assert float(np.mean(xdata)) == pytest.approx(12.0)
    assert float(np.mean(ydata)) == pytest.approx(10.0)


def test_trxs_azimuthal_sweeps_full_360_not_restricted_like_gixs(qapp, tmp_path):
    import numpy as np

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]

    console.azi_pixel_spin.setValue(5.0)
    console.azi_range_minus_spin.setValue(-2.0)
    console.azi_range_plus_spin.setValue(2.0)
    console._on_azi_plot_clicked()
    angle = np.asarray(cs.get_session().last_curve["angle"])
    assert angle.max() > 180.0


def test_trxs_polar_toggle_sweeps_full_360_not_restricted_like_gixs(qapp, tmp_path):
    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]

    console.map_format_combo.setCurrentText("Qy/Qz")
    assert console.polar_check.isEnabled()
    console.polar_check.setChecked(True)
    assert console._current_xlabel == "Angle, deg"
    assert console.image_canvas.ax.get_aspect() == "auto"

    polar_result = cs.console_polar_image({"mapping": "Qy/Qz"})
    assert polar_result["ok"], polar_result.get("message")
    assert polar_result["horizontal_range"]["max"] > 180.0

    console.polar_check.setChecked(False)
    assert console._current_xlabel == "Qy, 1/A"
    assert console.image_canvas.ax.get_aspect() == 1.0


def test_trxs_polar_toggles_directly_from_pixel_view(qapp, tmp_path):
    """Regression: "Transmission에서는 polar coordinate 바꿀때, pixel 맵을 바로
    바꿀수있도록 해줘" -- TrXS only ever has one Q mapping (Qy/Qz), so Polar
    should be reachable straight from Pixel view once q-Map Cal. has run,
    with no need to first click over to Qy/Qz. Unchecking must recover
    Pixel (not Qy/Qz), since that's what was actually showing before."""

    from pgixs_python.ui_backend import console_session as cs

    console = _load_test_frame_trxs(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)
    assert console.map_format_combo.currentText() == "Pixel"
    assert not console.polar_check.isEnabled()

    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    console._on_qmap_clicked()
    assert console.map_format_combo.currentText() == "Pixel"
    assert console.polar_check.isEnabled()

    console.polar_check.setChecked(True)
    assert console._current_xlabel == "Angle, deg"
    assert console._current_ylabel == "q, 1/A"
    assert console._axis_mode == "q"

    console.polar_check.setChecked(False)
    assert console.map_format_combo.currentText() == "Pixel"
    assert console._current_xlabel == "Y, pixel"
    assert console._axis_mode == "pixel"
    assert console.polar_check.isEnabled()


def test_gixs_polar_still_requires_an_explicit_map_format_pick(qapp, tmp_path):
    """GIXS has two Q mappings (Qxy/Qz, Qy/Qz) -- unlike TrXS, Polar must
    stay disabled from Pixel view even after q-Map Cal., since there's no
    single unambiguous mapping to jump to."""

    console = _load_test_frame(tmp_path)
    console.beam_y_spin.setValue(12.0)
    console.beam_z_spin.setValue(10.0)

    from pgixs_python.ui_backend import console_session as cs

    assert cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": 12.0, "beam_z": 10.0})["ok"]
    console._on_qmap_clicked()
    assert console.map_format_combo.currentText() == "Pixel"
    assert not console.polar_check.isEnabled()


def test_line_overlay_in_q_mode_draws_a_real_line_not_a_single_point(qapp, tmp_path):
    """Regression for "Qxy/Qz와 같은 맵에서는 ... 표시가 안 되는거야?": an
    overlay drawn while Map format was Qxy/Qz or Qy/Qz must not collapse
    to a single point (or the beam origin regardless of the cut's actual
    angle) -- must span a real, non-degenerate range on both axes.

    The overlay is a straight line in Q-space (`_draw_q_space_line` via
    `draw_beam_line`, `console_q_space_point`'s own uniform-grid Q
    coordinate for its one anchor point) -- a brief detour through
    sampling the true per-pixel Q maps point by point made it a "real
    curve" instead, but that curved what `console_line_1d`'s own Qxy/Qz
    cut treats as a straight cut through the same uniform grid (see
    console_session.py's `_q_space_cut_inputs`), reported: "Qxy/Qz
    맵에서 90도 line cut을... 해보면 일직선 라인이 아니라 휘어져 나와...
    그냥 일직선 라인이 되면 좋을 것 같아.\""""

    import numpy as np

    console = _load_test_frame_in_q_mode(tmp_path)

    console.line_angle_spin.setValue(45.0)
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()

    (line,) = console.image_canvas._overlay_artists
    xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    # draw_beam_line plots exactly two points (start, end of the segment)
    # -- confirms this is a genuine straight line, not the old multi-point
    # sampled-and-curved path.
    assert xdata.shape == (2,)
    finite = np.isfinite(xdata) & np.isfinite(ydata)
    assert finite.sum() > 1
    assert np.nanmax(xdata[finite]) - np.nanmin(xdata[finite]) > 0
    assert np.nanmax(ydata[finite]) - np.nanmin(ydata[finite]) > 0


def test_accumulation_overlay_in_q_mode_shows_two_distinct_edges(qapp, tmp_path):
    """Before this fix, _pos_move_display_offset ignored band_edge entirely
    in Q mode and returned the same beam-origin point for both edges --
    Accumulation's band would visually collapse to a single point/line,
    exactly the "표시가 안 나와" symptom reported. The two edges must now be
    genuinely different paths."""

    import numpy as np

    console = _load_test_frame_in_q_mode(tmp_path)

    console.line_angle_spin.setValue(90.0)
    console.accum_check.setChecked(True)
    console.accum_minus_spin.setValue(-2.0)
    console.accum_plus_spin.setValue(2.0)
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()

    edge_minus, edge_plus = console.image_canvas._overlay_artists
    x1 = np.asarray(edge_minus.get_xdata())
    x2 = np.asarray(edge_plus.get_xdata())
    finite = np.isfinite(x1) & np.isfinite(x2)
    assert finite.sum() > 1
    assert not np.allclose(x1[finite], x2[finite])


def test_cakecut_overlay_in_q_mode_shows_two_distinct_wedge_lines(qapp, tmp_path):
    """Regression reported live: '웨지 오버레이가 기대한 모양과 다릅니다' -- an X/bowtie
    instead of a clean V. Root cause: each boundary was drawn as a full
    bidirectional line through the beam origin (both angle_deg and its
    180-opposite), so two crossing full lines read as a confusing X
    rather than a wedge. Each boundary is now a ray starting at the
    beam's Q-space origin (~(0, 0)) and extending only outward in its
    actual direction, straight (see
    test_line_overlay_in_q_mode_draws_a_real_line_not_a_single_point)."""

    import h5py
    import numpy as np

    from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
    from pgixs_python.ui_backend import console_session as cs

    frame_path = tmp_path / "frame.h5"
    data = np.random.default_rng(0).uniform(10, 100, size=(200, 240)).astype(np.float32)
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])
    console = ConsoleWidget(gixs_config())
    console._activate()
    result = cs.console_load(
        {"directory": str(tmp_path), "filename": frame_path.name, "mode": "gixs", "detector": "TestDetector", "multishot": False}
    )
    assert result["ok"], result.get("message")
    console._on_load_finished(result)
    cs.get_session().calibration_loaded = True
    beam_y, beam_z = 120.0, 100.0
    console.beam_y_spin.setValue(beam_y)
    console.beam_z_spin.setValue(beam_z)
    qmap_result = cs.console_qmap({"incident_angle_deg": 0.0, "beam_y": beam_y, "beam_z": beam_z})
    assert qmap_result["ok"], qmap_result.get("message")
    # Unblocked -- draw_beam_line extends its ray to the axes' *current*
    # view limits, so those need to actually be the real Q-space ones
    # from rendering the Qxy/Qz background first (see
    # _load_test_frame_in_q_mode's own comment).
    console.map_format_combo.setCurrentText("Qxy/Qz")

    console.line_angle_spin.setValue(0.0)
    console.cakecut_check.setChecked(True)
    console.cakecut_minus_spin.setValue(-10.0)
    console.cakecut_plus_spin.setValue(15.0)
    console.image_canvas.clear_overlays()
    console._draw_line_overlay()

    line_minus, line_plus = console.image_canvas._overlay_artists
    expected_angles = [0.0 + (-10.0), 0.0 + 15.0]  # draw order: minus boundary, then plus
    for line, expected_angle in zip((line_minus, line_plus), expected_angles):
        xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        # Exactly two points (start, end) -- a genuine straight segment.
        assert xdata.shape == (2,)
        assert xdata[0] == pytest.approx(0.0, abs=1e-2)
        assert ydata[0] == pytest.approx(0.0, abs=1e-2)
        # Ground-truth check independent of _draw_q_space_line's own math:
        # take the ray's far endpoint, convert it back to a pixel offset
        # via the already-tested reverse lookup
        # (_display_xy_to_pixel_offset), and confirm that offset's angle
        # (by the same formula _on_image_clicked uses) matches the wedge
        # boundary this ray was supposed to represent.
        px, pz = console._display_xy_to_pixel_offset(xdata[1], ydata[1])
        reported = _click_angle_deg(px, pz, 0.0, 0.0)
        assert reported == pytest.approx(expected_angle % 360.0, abs=1e-2)

    y1 = np.asarray(line_minus.get_ydata())
    y2 = np.asarray(line_plus.get_ydata())
    finite = np.isfinite(y1) & np.isfinite(y2)
    assert finite.sum() > 1
    assert not np.allclose(y1[finite], y2[finite])


def test_line_overlay_and_click_agree_in_q_mode_too(qapp, tmp_path):
    """Regression reported live, with a screenshot of the "1D process"
    Position click working correctly in Pixel mode but landing the drawn
    line on the wrong side of the rings once Map format was Qxy/Qz: "Qxy/Qz
    map에서 똑같은 위치에 클릭하여 line을 그리면... 다른곳에 라인이
    그려져." Root cause: `_on_image_clicked`'s "line" branch always mirrored
    the click angle a second time via `_mode_corrected_angle`, but in Q
    mode `_display_xy_to_pixel_offset` (unlike centered-pixel mode's plain
    identity passthrough) already fully resolves the display's y-flip into
    a genuine unflipped pixel offset -- stacking the mirror on top of that
    doubled it, landing every clicked line 180deg from where it should be.

    Same true end-to-end round trip as
    test_line_overlay_and_click_agree_in_beam_centered_pixel_mode_too: draw
    the overlay for a target angle via _draw_line_overlay, then feed its
    far endpoint back through _on_image_clicked (as if the user clicked
    there) and confirm line_angle_spin recovers the original target
    angle -- this is exactly "click the same feature the line already
    passes through and expect the line to stay there."""

    console = _load_test_frame_in_q_mode(tmp_path)

    for target_angle in (30.0, 90.0, 140.0, 200.0):
        console.cakecut_check.setChecked(False)
        console.line_angle_spin.setValue(target_angle)
        console.image_canvas.clear_overlays()
        console._draw_line_overlay()
        (line,) = console.image_canvas._overlay_artists
        xdata, ydata = line.get_xdata(), line.get_ydata()

        console._arm_position_mode("line")
        console._on_image_clicked(xdata[1], ydata[1])
        assert console.line_angle_spin.value() == pytest.approx(target_angle % 360.0, abs=1e-2), f"angle={target_angle}"
