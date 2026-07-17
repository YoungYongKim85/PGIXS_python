# PGIXS Python CLI Core Stabilization Checklist

This checklist tracks what should be stable before moving from the Tab1 Python
CLI core into desktop/web UI design.

## Status Legend

- `[x]` Implemented and smoke-tested in this repository.
- `[~]` Implemented, but needs more real-data or MATLAB comparison validation.
- `[ ]` Not implemented or not yet ready for UI use.

## Core Workflow

- `[x]` TrXS CLI workflow exists: `pgixs_python.workflows.trxs_process`.
- `[x]` GIXS CLI workflow exists: `pgixs_python.workflows.gixs_process`.
- `[x]` JSON config files can drive TrXS/GIXS workflows.
- `[x]` Unknown config keys stop early with clear errors.
- `[x]` Common config value mistakes stop early with clear errors.
- `[x]` JSON Schemas exist for TrXS, GIXS, 1D export, and batch configs.
- `[x]` Batch runner can execute multiple workflow jobs from one JSON file.
- `[x]` Batch runner supports quiet mode, per-job logs, summary JSON, and a
  shared manifest path.
- `[x]` Batch runner supports `--dry-run` config validation before processing.

## Input IO

- `[x]` H5 detector frame loading supports 2D and common 3D EIGER layouts.
- `[x]` H5 `frame_index` can be selected from CLI/config.
- `[x]` H5 load failures retry, then fall back to NaN frames when possible.
- `[x]` H5 metadata sidecars capture counters, timestamps, and scan values.
- `[x]` Folder input can be loaded as one averaged series.
- `[~]` `.mat`, `.tif/.tiff`, and text image loading helpers exist, but need
  broader real-data fixture validation.

## Calibration And Q Maps

- `[x]` Standard calibration can be read from JSON and legacy MAT files.
- `[x]` TrXS Q-map generation is wired into the workflow.
- `[x]` GIXS Q-map generation is wired into the workflow.
- `[~]` Q-map numerical parity against MATLAB needs selected real-data
  comparison cases.
- `[~]` SDD calibration helper exists, but automatic SDD workflows still need
  broader detector-distance edge-case validation.

## Processing

- `[x]` IC/BC/None normalization is available.
- `[x]` Background subtraction is available.
- `[x]` Negative clamp behavior is configurable.
- `[x]` Eiger4M grid removal is available and detector-gated.
- `[x]` Detector mask helpers are available.
- `[~]` Detector correction is implemented, but needs real-data scientific
  validation before UI defaults are finalized.

## Outputs

- `[x]` Python-native `*_2Dinfo.h5` output is H5-only.
- `[x]` 2D-info H5 round-trip helpers exist.
- `[x]` 1D export supports `.csv` and `.dat`.
- `[x]` Optional 1D figure export is available when matplotlib is installed.
- `[x]` Workflow JSON reports record inputs, outputs, options, calibration,
  retry/metadata status, processing diagnostics, and 1D coverage diagnostics.
- `[x]` JSONL manifests can collect compact job results for UI/batch tracking.
- `[x]` Batch summary JSON can collect high-level job status and log paths.

## Detector-Gap Diagnostics

- `[x]` q-bin averaging omits NaN values and records valid counts.
- `[x]` q-bin coverage is exported optionally as a column.
- `[x]` coverage summary is included in reports.
- `[x]` low-coverage and abrupt coverage-drop bins can be detected.
- `[~]` Gap-related kink behavior needs final comparison against MATLAB output
  and representative real detector-gap examples.

## Cross-Platform Readiness

- `[x]` Runtime paths use `pathlib` internally where practical.
- `[x]` CLI examples are documented for Windows PowerShell.
- `[x]` Linux shell examples are documented for core CLI workflows.
- `[~]` Python 3.10 environment should be tested on a clean Windows machine.
- `[~]` Python 3.10 environment should be tested on a clean Linux machine.

## Test And Validation

- `[x]` Minimal smoke test exists: `tests/run_smoke.py`.
- `[x]` No-pytest unit harness exists: `tests/run_unit_harness.py`.
- `[x]` Compile check is documented: `python -m compileall pgixs_python tests`.
- `[~]` Full pytest suite should be run in the real Python 3.10 environment.
- `[~]` Real H5 fixture regression set should be expanded.
- `[x]` MATLAB-vs-Python numeric comparison template exists:
  `docs/matlab_python_validation_template.md`.
- `[~]` Filled MATLAB-vs-Python validation cases still need to be captured for
  representative real datasets.

## UI Gate

Move into UI design after these items are acceptable:

- `[ ]` User confirms CLI output structure is sufficient for routine Tab1
  processing.
- `[ ]` User confirms config schema fields map cleanly to expected UI controls.
- `[ ]` At least one GIXS and one TrXS real dataset have been compared against
  MATLAB outputs.
- `[ ]` Batch result manifest/summary format is acceptable for UI result lists.
- `[ ]` Initial desktop/local-web UI direction is chosen.

## Recommended Next Stabilization Tasks

1. Run the current CLI on a clean Python 3.10 environment with all pinned
   dependencies installed.
2. Start UI workflow interview once the user accepts the CLI contract.
