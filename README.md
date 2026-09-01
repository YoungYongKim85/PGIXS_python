# PGIXS Python Port

This folder is the Python port workspace for PGIXS Tab1 only.

## Desktop app

The primary interface is a PySide6 desktop app (SDD Calibration / GIXS /
GIXS-series / TrXS / TrXS-series / XTrans / Penetration Depth tabs).

**End users** (no Python install needed): download the build for your OS
from this repo's [Releases](../../releases) page (built by
`.github/workflows/build.yml` whenever a `vX.Y.Z` tag is pushed):

- **Windows**: unzip `PGIXS-windows.zip`, run `PGIXS.exe`.
- **macOS**: unzip `PGIXS-macos.zip`, drag `PGIXS.app` to Applications. The
  build isn't code-signed/notarized, so the first launch needs
  right-click -> Open (or `xattr -d com.apple.quarantine PGIXS.app`) to get
  past Gatekeeper's "unidentified developer" warning.
- **Linux**: `tar xzf PGIXS-linux.tar.gz && ./PGIXS/PGIXS`.

**From source** (development):

```powershell
python -m pip install -r requirements.txt
python -m pgixs_python.desktop.app
```

**Building a standalone package yourself**, e.g. to test a change before
tagging a release (PyInstaller never cross-compiles -- this produces a
build for whichever OS you run it on):

```powershell
python -m pip install pyinstaller
pyinstaller packaging/pgixs.spec --noconfirm
```

The result is `packaging/dist/PGIXS` (`PGIXS.app` on macOS).

Use `docs/python_tab1_port_plan.md` as the active work tracker before each
implementation step.
Use `docs/cli_core_stabilization_checklist.md` as the UI readiness checklist.

Scope:

- Raw detector image loading for `.h5`, `.tif/.tiff`, `.mat`, text-like files.
- GIXS and TrXS Tab1 processing.
- SDD calibration file reading and application.
- Q-map generation.
- Detector corrections and NaN-preserving masking.
- 1D radial, angular, sector, line, and cake-cut style averaging.
- Python-native H5 2D-info output with metadata/diagnostics.
- H5-only 2D-info export for metadata extensibility.

Out of scope for this folder:

- Tab3 crystal indexing.
- HKL/Solve/Fine-tune GUIs.
- Web UI implementation.

## First Milestone

Build a tested core library that can run from the command line:

```powershell
python -m pgixs_python.workflows.trxs_process --help
python -m pgixs_python.workflows.gixs_process --help
```

The first implementation target is IO and contract-compatible data structures;
scientific algorithms should be ported one helper at a time from the MATLAB
files listed in `docs/matlab_to_python_tab1_map.md`.

Current workflow scaffold can load H5 file/folder inputs and report stack
diagnostics:

```powershell
python -m pgixs_python.workflows.trxs_process --input path\to\h5_folder
python -m pgixs_python.workflows.gixs_process --input path\to\frame.h5
```

Current config-driven examples:

```powershell
python -m pgixs_python.workflows.gixs_process --config docs\example_gixs_workflow_config.json
python -m pgixs_python.workflows.trxs_process --config docs\example_trxs_workflow_config.json
python -m pgixs_python.workflows.export_1d_from_h5 --input validation_outputs\sample_GIXS_2Dinfo.h5 --out validation_outputs
```

Batch execution:

```powershell
python -m pgixs_python.workflows.batch_process `
  --batch docs\example_batch_workflow.json `
  --manifest validation_outputs\batch_manifest.jsonl `
  --quiet `
  --log-dir validation_outputs\batch_logs `
  --summary-json validation_outputs\batch_summary.json
```

Regenerate JSON config schemas:

```powershell
python -m pgixs_python.workflows.config_schema --workflow all --out docs
```

## Environment

Use Python 3.10 for the project runtime.

```powershell
cd C:\Users\owner\Documents\PGIXS_matlab업그레이드\python_tab1
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install CUDA Torch only if/when a later analysis module needs it:

```powershell
python -m pip install -r requirements-torch-cu113.txt
```

Minimal smoke test without pytest:

```powershell
python tools\check_environment.py
python tests\run_smoke.py
python tests\run_unit_harness.py
```

Environment check plus quick validation:

```powershell
python tools\check_environment.py --run-tests
```

Run the local web UI scaffold:

```powershell
python -m pgixs_python.ui_backend.local_server
```

Then open `http://127.0.0.1:8765/`.

Pytest test suite:

```powershell
python -m pytest tests
```
