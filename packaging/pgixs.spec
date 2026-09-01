# PyInstaller spec for the PGIXS Python desktop app.
#
# Build (from the repo root, with the project's venv active and
# pyinstaller installed):
#   pyinstaller packaging/pgixs.spec --noconfirm
#
# PyInstaller never cross-compiles -- run this once per target OS
# (Windows/macOS/Linux) to get that OS's own build. The GitHub Actions
# workflow at .github/workflows/build.yml does exactly that across all
# three automatically.
#
# Only QtCore/QtGui/QtWidgets are ever imported (see console_widget.py
# and friends) -- PyInstaller's own PySide6 hook already excludes
# unimported Qt modules (WebEngine, Qml, Designer, ...) from analysis, so
# no explicit --exclude-module list is needed for those.

import sys
from pathlib import Path

block_cipher = None
repo_root = Path(SPECPATH).resolve().parent

# QIcon(str(_ICON_PATH)) in main_window.py/app.py loads this file at
# runtime (the in-app window/taskbar icon) -- separate from the
# packaging/icon.* files below, which are the *frozen executable's own*
# icon (what Explorer/Finder/the Dock show before the app even launches).
intrologo = repo_root / "pgixs_python" / "Intrologo.png"

if sys.platform == "win32":
    app_icon = str(repo_root / "packaging" / "icon.ico")
elif sys.platform == "darwin":
    app_icon = str(repo_root / "packaging" / "icon.icns")
else:
    app_icon = None  # Linux has no single embedded-exe-icon convention.

a = Analysis(
    [str(repo_root / "packaging" / "entry_point.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=[(str(intrologo), "pgixs_python")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PGIXS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=app_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PGIXS",
)

# macOS also gets a proper .app bundle wrapping the COLLECT output, so
# Finder/Dock/Spotlight treat it as one double-clickable application
# instead of a loose folder of files.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PGIXS.app",
        icon=app_icon,
        bundle_identifier="org.pal3c.pgixs",
        info_plist={
            "CFBundleName": "PGIXS",
            "CFBundleDisplayName": "PGIXS",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
