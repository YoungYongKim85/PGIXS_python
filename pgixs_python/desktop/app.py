"""Entry point: `python -m pgixs_python.desktop.app`."""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pgixs_python.desktop.main_window import _ICON_PATH, MainWindow
from pgixs_python.desktop.theme import STYLESHEET, build_light_palette


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PGIXS Python")
    # Windows' taskbar icon follows the QApplication-level icon (a window's
    # own setWindowIcon alone only reliably updates its title bar) -- set
    # both from the same file so the taskbar and every window agree.
    if _ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(_ICON_PATH)))
    # Custom QSS on top of Windows' native "windowsvista"/"windows11" style
    # is a known source of subcontrol quirks (spin box up/down button hit
    # regions in particular). Fusion is Qt's style built to support full
    # stylesheet overrides predictably, so force it before applying ours.
    app.setStyle("Fusion")
    # A handful of Fusion's compound subcontrols (QTabWidget's pane,
    # QSplitter handles, the strip beyond a QTabBar's last tab) paint from
    # the raw QPalette instead of any QSS "background" rule -- setting the
    # palette explicitly is what actually keeps them light (see
    # theme.build_light_palette's docstring).
    app.setPalette(build_light_palette())
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    # The nested nested QSplitter/QGroupBox layout only reaches its final
    # on-screen size a tick after show() -- the *first* paint (still an
    # in-progress layout pass) can leave a few boundary regions never
    # marked dirty again once the layout settles, so they keep showing the
    # window surface's pre-paint black instead of the real background.
    # window.grab() (an off-screen re-render at the final size) never
    # shows this; only the live window does. Forcing one more full repaint
    # after the event loop is idle papers over it.
    QTimer.singleShot(0, window.update)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
