"""Main window: a tab per console mode, mirroring MATLAB's separate GIXS/TrXS windows."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from pgixs_python.desktop.console_widget import ConsoleWidget, gixs_config, trxs_config
from pgixs_python.desktop.gixs2_widget import GIXS2Widget
from pgixs_python.desktop.penetration_widget import PenetrationWidget
from pgixs_python.desktop.sddcal_widget import SDDcalWidget
from pgixs_python.desktop.trxs2_widget import TrXS2Widget
from pgixs_python.desktop.xtrans_widget import XTransWidget

# pgixs_python/desktop/main_window.py -> pgixs_python/Intrologo.png
_ICON_PATH = Path(__file__).resolve().parent.parent / "Intrologo.png"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PGIXS Python")
        if _ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_ICON_PATH)))
        self.resize(1480, 940)

        self.tabs = QTabWidget()
        # Fusion's QTabBar paints the strip beyond the last tab (up to the
        # widget's full width) using the raw, unstyled palette base rather
        # than any QSS background rule -- document mode drops that base
        # panel entirely, so the empty strip picks up the window background
        # instead of rendering solid black.
        self.tabs.setDocumentMode(True)
        # Belt-and-braces for the same root cause: some regions of a
        # QTabWidget/QStatusBar's own rect are never touched by any paint
        # call under Fusion (not even the raw, unstyled palette fill) and
        # are left showing the window backing store's black default.
        # autoFillBackground forces the *whole* widget rect to be cleared
        # to its palette Window color first, before anything else paints.
        self.tabs.setAutoFillBackground(True)
        # SDD calibration comes first -- it's the first thing done in a real
        # experiment (GIXS/TrXS both need its output before Q-space analysis
        # works at all).
        self.sddcal = SDDcalWidget()
        self.gixs_console = ConsoleWidget(gixs_config())
        self.trxs_console = ConsoleWidget(trxs_config())
        # X-ray property calculators -- standalone tools in MATLAB
        # (XTrans_GUI.m / Penetration_GUI.m).
        self.xtrans = XTransWidget()
        self.penetration = PenetrationWidget()
        # GIXS-series: the series-analysis counterpart to GIXS, Phase 1
        # (see the implementation plan) -- static-geometry q-Cal only,
        # "Change ai"/Playback/Heatmaps/whole-stack save deferred. Placed
        # right after GIXS since it's the same instrument/geometry, not
        # with the standalone calculator tabs.
        self.gixs2 = GIXS2Widget()
        # TrXS-series: the series-analysis counterpart to TrXS, Phase 1 (see
        # the implementation plan) -- Circular Average/1D Line Ang/Azimuthal/
        # ROI, Background Subtraction, point-click normalization. Placed
        # right after "TrXS" per the user's explicit wording, not grouped
        # with GIXS2/GIXS.
        self.trxs2 = TrXS2Widget()
        self.tabs.addTab(self.sddcal, "SDD Calibration")
        self.tabs.addTab(self.gixs_console, "GIXS")
        self.tabs.addTab(self.gixs2, "GIXS-series")
        self.tabs.addTab(self.trxs_console, "TrXS")
        self.tabs.addTab(self.trxs2, "TrXS-series")
        self.tabs.addTab(self.xtrans, "XTrans")
        self.tabs.addTab(self.penetration, "Penetration Depth")
        self.setCentralWidget(self.tabs)

        # Both consoles share one console_session module-level slot (each
        # owns its own ConsoleSession and swaps it in before every call --
        # see ConsoleWidget._activate()). A background load on one tab must
        # finish before the other tab can install its own session, so lock
        # tab switching for the duration of any in-flight load.
        self.gixs_console.busy_changed.connect(self._set_tabs_locked)
        self.trxs_console.busy_changed.connect(self._set_tabs_locked)

        status_bar = QStatusBar()
        status_bar.setAutoFillBackground(True)
        status_bar.setSizeGripEnabled(False)
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)

    def _set_tabs_locked(self, locked: bool) -> None:
        self.tabs.tabBar().setEnabled(not locked)
