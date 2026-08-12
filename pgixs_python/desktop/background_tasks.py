"""QThread background workers for long-running `console_session` calls.

Multishot load averages an arbitrary-length file series synchronously
inside `console_load`/`console_background_load` -- those functions already
publish per-file progress to the shared session (`load_progress`, read via
`console_load_progress()`) exactly the way the web console's polling relied
on. Here that same polling happens on a `QTimer` on the main thread while
the call itself runs on a `QThread`, driving a real `QProgressDialog`
(MATLAB's `uiprogressdlg`) instead of an HTTP round trip.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class ConsoleCallWorker(QObject):
    """Runs a single `console_session` function call on a background thread."""

    finished = Signal(dict)

    def __init__(self, fn: Callable[[dict[str, Any]], dict[str, Any]], payload: dict[str, Any]) -> None:
        super().__init__()
        self._fn = fn
        self._payload = payload

    def run(self) -> None:
        try:
            result = self._fn(self._payload)
        except Exception as exc:  # noqa: BLE001 - surface as a normal failure payload.
            result = {"ok": False, "message": str(exc)}
        self.finished.emit(result)


def run_in_background(
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    payload: dict[str, Any],
    on_finished: Callable[[dict[str, Any]], None],
) -> QThread:
    """Start `fn(payload)` on a QThread, calling `on_finished(result)` back on
    the main thread when done. Returns the QThread -- the caller must keep a
    reference alive until finished, or Qt may garbage-collect it mid-run."""

    thread = QThread()
    worker = ConsoleCallWorker(fn, payload)
    worker.moveToThread(thread)
    # `worker` has no Qt parent, so nothing keeps a Python reference to it
    # once this function returns except this attribute -- without it, the
    # caller only holds `thread`, and the worker (the QObject actually
    # running `console_load` on the background thread) can be garbage
    # collected before `run()` ever fires. That silently drops the whole
    # load: `started` has nothing left to invoke, `finished` never emits,
    # and the UI is stuck on "Loading..." forever with no error.
    thread._worker = worker  # type: ignore[attr-defined]
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
