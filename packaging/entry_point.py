"""PyInstaller's Analysis script -- thin wrapper so the frozen app's own
entry point isn't `python -m pgixs_python.desktop.app` (PyInstaller needs
a real .py file to analyze, not a module path)."""

from pgixs_python.desktop.app import main

if __name__ == "__main__":
    raise SystemExit(main())
