# PGIXS Python CLI Quickstart

This guide covers the current Tab1 Python command-line workflow.

## Environment

Use Python 3.10.

```powershell
cd C:\Users\owner\Documents\PGIXS_matlab업그레이드\python_tab1
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install CUDA Torch only if a later module actually needs it.

```powershell
python -m pip install -r requirements-torch-cu113.txt
```

## Single Workflow

```powershell
python -m pgixs_python.workflows.gixs_process --config docs\example_gixs_workflow_config.json
python -m pgixs_python.workflows.trxs_process --config docs\example_trxs_workflow_config.json
```

Both workflows save Python-native `*_2Dinfo.h5` outputs when `out` and
`standard` are provided.

## Export 1D From H5 2D-info

```powershell
python -m pgixs_python.workflows.export_1d_from_h5 `
  --input validation_outputs\kinktest_0001_GIXS_2Dinfo.h5 `
  --out validation_outputs `
  --export-format csv `
  --q-bins 300 `
  --q-scale log `
  --export-std `
  --export-diagnostics `
  --report-json
```

## Batch Workflow

```powershell
python -m pgixs_python.workflows.batch_process `
  --batch docs\example_batch_workflow.json `
  --manifest validation_outputs\batch_manifest.jsonl `
  --quiet `
  --log-dir validation_outputs\batch_logs `
  --summary-json validation_outputs\batch_summary.json
```

## Config Schemas

Regenerate schema files after adding/removing CLI options.

```powershell
python -m pgixs_python.workflows.config_schema --workflow all --out docs
```

Schema files live in `docs/*_schema.json` and are intended to become the
future desktop/web UI control contract.

## Fast Checks

```powershell
python tests\run_smoke.py
python tests\run_unit_harness.py
python -m compileall pgixs_python tests
```
