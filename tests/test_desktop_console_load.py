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
