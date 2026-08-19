"""Main window: a tab per console mode, mirroring MATLAB's separate GIXS/TrXS windows."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config, trxs_config
from pgixs_python.desktop.sddcal_widget import SDDcalWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PGIXS Python")
        self.resize(1480, 940)

        self.tabs = QTabWidget()
        # SDD calibration comes first -- it's the first thing done in a real
        # experiment (GIXS/TrXS both need its output before Q-space analysis
        # works at all).
        self.sddcal = SDDcalWidget()
        self.gixs_console = ConsoleWidget(gixs_config())
        self.trxs_console = ConsoleWidget(trxs_config())
        self.tabs.addTab(self.sddcal, "SDD Calibration")
        self.tabs.addTab(self.gixs_console, "Console (GIXS)")
        self.tabs.addTab(self.trxs_console, "Console (TrXS)")
        self.setCentralWidget(self.tabs)

        # Both consoles share one console_session module-level slot (each
        # owns its own ConsoleSession and swaps it in before every call --
        # see ConsoleWidget._activate()). A background load on one tab must
        # finish before the other tab can install its own session, so lock
        # tab switching for the duration of any in-flight load.
        self.gixs_console.busy_changed.connect(self._set_tabs_locked)
        self.trxs_console.busy_changed.connect(self._set_tabs_locked)

        status_bar = QStatusBar()
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)

    def _set_tabs_locked(self, locked: bool) -> None:
        self.tabs.tabBar().setEnabled(not locked)
