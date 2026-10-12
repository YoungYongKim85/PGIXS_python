import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config
from pgixs_python.ui_backend import console_session as cs


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_single_shot_load_is_window_modal_until_the_worker_finishes(qapp, tmp_path, monkeypatch):
    """A single-shot load used to disable only LOAD DATA, so Load Bkg. could
    start a second worker and drop the first, still-running QThread."""

    import threading
    import time

    from PySide6.QtCore import Qt

    from pgixs_python.desktop import console_widget

    release = threading.Event()

    def slow_load(_payload):
        release.wait(5)
        return {"ok": False, "message": "stub"}

    monkeypatch.setattr(cs, "console_load", slow_load)
    monkeypatch.setattr(
        console_widget.QFileDialog, "getOpenFileName", lambda *a, **k: (str(tmp_path / "frame.h5"), "")
    )
    console = ConsoleWidget(gixs_config())
    console._on_load_clicked()

    dialog = console._progress_dialog
    assert dialog is not None
    assert dialog.windowModality() == Qt.WindowModal

    release.set()
    deadline = time.monotonic() + 5
    while console._progress_dialog is not None and time.monotonic() < deadline:
        qapp.processEvents()
    assert console._progress_dialog is None
    console._load_thread.wait(2000)


def _console_with_frame(tmp_path, value, name="frame.h5"):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(tmp_path / name, "w") as handle:
        handle.create_dataset("entry/data/data", data=np.full((1, 20, 24), value, dtype=np.float32))
    return cs.console_load(
        {"directory": str(tmp_path), "filename": name, "mode": "gixs", "detector": "TestDetector", "multishot": False}
    )


def test_switching_tabs_installs_the_shown_consoles_session(qapp):
    from pgixs_python.desktop.main_window import MainWindow

    window = MainWindow()
    window.tabs.setCurrentWidget(window.trxs_console)
    assert cs.get_session() is window.trxs_console._session

    window.tabs.setCurrentWidget(window.gixs_console)
    assert cs.get_session() is window.gixs_console._session


def test_a_new_image_does_not_inherit_the_previous_contrast(qapp, tmp_path):
    console = ConsoleWidget(gixs_config())
    console._activate()
    console._on_load_finished(_console_with_frame(tmp_path, 5000.0, "bright.h5"))
    bright_range = (console.cmin_spin.value(), console.cmax_spin.value())

    console._on_load_finished(_console_with_frame(tmp_path, 3.0, "weak.h5"))

    assert (console.cmin_spin.value(), console.cmax_spin.value()) != bright_range
    assert console.cmax_spin.value() < 100.0  # auto-contrast from the weak frame, not the bright one


def test_changing_the_incidence_pixel_marks_qmap_stale(qapp):
    console = ConsoleWidget(gixs_config())
    console._session.qmap = {}
    console._mark_ok(console.qmap_button, True)

    console.ai_pix_spin.setValue(console.ai_pix_spin.value() + 10.0)
    console._incidence_from_pixel()

    assert console.qmap_button.property("state") is None


def test_loading_a_new_series_closes_the_play_window(qapp, tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from pgixs_python.desktop.gixs2_widget import GIXS2Widget
    from pgixs_python.ui_backend import gixs2_session as gs

    gs.reset_session()
    names = []
    for i in range(3):
        names.append(f"S_{i + 1:04d}.h5")
        with h5py.File(tmp_path / names[-1], "w") as handle:
            handle.create_dataset("/entry/data/data", data=np.full((20, 24), float(i + 1), dtype=np.float32))
    load = lambda files: gs.gixs2_load_series({"directory": str(tmp_path), "filenames": files})  # noqa: E731
    widget = GIXS2Widget()
    widget._on_series_loaded(load(names))
    widget._on_play_clicked()
    assert widget._play_dialog is not None

    widget._on_series_loaded(load(names[:1]))

    assert widget._play_dialog is None


def test_load_data_keeps_the_calibration_fields_the_user_typed(qapp, tmp_path):
    console = ConsoleWidget(gixs_config())
    console._activate()
    console._on_load_finished(_console_with_frame(tmp_path, 10.0, "first.h5"))
    console.energy_spin.setValue(12.34)
    console.sdd_spin.setValue(456.0)

    console._on_load_finished(_console_with_frame(tmp_path, 10.0, "second.h5"))

    assert console.energy_spin.value() == pytest.approx(12.34)
    assert console.sdd_spin.value() == pytest.approx(456.0)


@pytest.mark.parametrize("tab", ["gixs2", "trxs2"])
def test_watch_auto_refreshes_new_frames_by_itself(qapp, tmp_path, tab):
    import os
    import time

    h5py = pytest.importorskip("h5py")
    import numpy as np

    if tab == "gixs2":
        from pgixs_python.desktop.gixs2_widget import GIXS2Widget as Widget
        from pgixs_python.ui_backend import gixs2_session as session_module
    else:
        from pgixs_python.desktop.trxs2_widget import TrXS2Widget as Widget
        from pgixs_python.ui_backend import trxs2_session as session_module
    session_module.reset_session()
    names = []
    for i in range(3):
        names.append(f"S_{i + 1:04d}.h5")
        with h5py.File(tmp_path / names[-1], "w") as handle:
            handle.create_dataset("/entry/data/data", data=np.full((20, 24), float(i + 1), dtype=np.float32))
    old = time.time() - 60
    for name in names:
        os.utime(tmp_path / name, (old, old))
    load = getattr(session_module, f"{tab}_load_series")
    widget = Widget()
    widget._last_dir = str(tmp_path)

    # No imported series yet: the watcher switches itself off.
    widget.watch_check.setChecked(True)
    widget._on_watch_tick()
    assert not widget.watch_check.isChecked()

    widget._on_series_loaded(load({"directory": str(tmp_path), "filenames": names[:2]}))
    widget.watch_check.setChecked(True)
    widget._on_watch_tick()

    deadline = time.monotonic() + 10
    while not widget.auto_refresh_button.isEnabled() and time.monotonic() < deadline:
        qapp.processEvents()
    assert session_module.get_session().stack.shape[0] == 3
    widget.watch_check.setChecked(False)
    widget._auto_refresh_thread.wait(2000)
