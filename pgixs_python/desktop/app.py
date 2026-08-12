"""Entry point: `python -m pgixs_python.desktop.app`."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.main_window import MainWindow
from pgixs_python.desktop.theme import STYLESHEET


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PGIXS Python")
    # Custom QSS on top of Windows' native "windowsvista"/"windows11" style
    # is a known source of subcontrol quirks (spin box up/down button hit
    # regions in particular). Fusion is Qt's style built to support full
    # stylesheet overrides predictably, so force it before applying ours.
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
