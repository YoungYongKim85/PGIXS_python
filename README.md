# PGIXS Python Port

This folder is the Python port workspace for PGIXS Tab1 only.

Use `docs/python_tab1_port_plan.md` as the active work tracker before each
implementation step.

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
python tests\run_smoke.py
python tests\run_unit_harness.py
```

Pytest test suite:

```powershell
python -m pytest tests
```
