# PGIXS Python Port Plan

This file is the working tracker for the Python Tab1 port. Use it before each
implementation step so the work stays focused on Tab1 raw-data processing.

Finished items should be removed or moved to the progress log section instead
of left as stale checklist noise.

## Scope

In scope:

- Tab1 GIXS raw processing.
- Tab1 TrXS raw processing.
- SDD calibration read/apply support.
- Detector image loading for `.h5`, `.tif/.tiff`, `.mat`, and text-like data.
- Q-map generation.
- Normalization, background handling, detector mask, and correction helpers.
- Radial/circular/angular/sector/cake-cut style 1D outputs.
- Python-native `*_2Dinfo.h5` output with metadata/diagnostics.
- Paper/report figure and table exports.

Out of scope for this folder:

- Tab3 indexing.
- HKL/Solve/Fine-tune logic.
- MATLAB GUI work.
- Web UI implementation.

## Guardrails

- Keep Python runtime on Python 3.10 for the real project environment.
- Use `pathlib.Path` for Windows/Linux path compatibility.
- Preserve detector gaps and failed frames as `NaN`, never silent zero.
- H5 frame loading should retry once, then mark failed frames as `NaN`.
- Average functions must omit `NaN` and return valid-count/coverage diagnostics
  where useful.
- Use H5 as the Python-native 2D-info output contract.
- Do not start Tab3 or web implementation from this folder.

## Phase 1. Environment

- Create Python 3.10 virtual environment.
- Install `requirements.txt`.
- Install `requirements-torch-cu113.txt` only if a later module genuinely needs
  Torch.
- Run:

```powershell
python tests\run_smoke.py
python -m pytest tests
```

Current scaffold smoke status:

- Bundled Codex Python 3.12 smoke passed for the minimal scaffold.
- Real project Python 3.10 environment is still needed.

## Phase 2. IO Implementation

Target files:

- `pgixs_python/io/image_loader.py`
- `pgixs_python/io/h5_loader.py`
- `pgixs_python/io/h5_metadata.py`
- `pgixs_python/io/mat_loader.py`

Tasks:

- Port `.h5` frame loading and EIGER orientation behavior.
- Port H5 counters/metadata reading.
- Implement retry-once behavior for frame loading.
- Port `.mat` image selection behavior from `loadPGIXSImageFile.m` as a
  secondary input path.
- Port `.tif/.tiff` loading.
- Add tests for each supported format using existing `test/` fixtures when
  practical.

## Phase 3. Calibration And Standards

Target files:

- `pgixs_python/calibration/sdd.py`
- `pgixs_python/calibration/standards.py`
- `pgixs_python/calibration/calibration_io.py`

Tasks:

- Port standard q/order values.
- Read MATLAB legacy standard `.mat` files.
- Read/write Python-friendly JSON calibration.
- Preserve SDD candidate diagnostics fields.
- Preserve short-SDD AgBH behavior: do not assume hidden 1st peak is visible.

## Phase 4. Q-Map Generation

Target files:

- `pgixs_python/qmap/trxs.py`
- `pgixs_python/qmap/gixs.py`

Tasks:

- Numerically align TrXS Q-map with `Qmaping_tr.m`.
- Port and test GIXS Q-map from `Qmaping_gi.m`.
- Confirm output shapes and axis orientation.
- Add small synthetic tests for beam-center and sign conventions.

## Phase 5. Processing Core

Target files:

- `pgixs_python/processing/normalization.py`
- `pgixs_python/processing/corrections.py`
- `pgixs_python/processing/grid_remove.py`
- `pgixs_python/processing/masking.py`

Tasks:

- Port normalization modes.
- Port background subtraction behavior.
- Port solid angle, polarization, and efficiency corrections.
- Keep detector mask/gap as `NaN`.
- Limit grid remove to Eiger4M-compatible images.

## Phase 6. 1D Averages And Cuts

Target files:

- `pgixs_python/averaging/qbin.py`
- `pgixs_python/averaging/radial.py`
- `pgixs_python/averaging/angular.py`
- `pgixs_python/averaging/cake.py`

Tasks:

- Port q-bin average with `NaN` omission.
- Port pixel-radius radial/circular average.
- Port TrXS angular/sector average.
- Port GIXS cake cut, including angle wrap-around.
- Return count/coverage sidecars where detector gaps affect interpretation.
- Add detector-gap tests to prevent deep-bump regressions.

## Phase 7. Export And Reports

Target files:

- `pgixs_python/io/two_d_info.py`
- `pgixs_python/export/data_export.py`
- `pgixs_python/export/figures.py`
- `pgixs_python/export/report_json.py`

Tasks:

- Save Python-native `*_2Dinfo.h5`.
- Save `.mat`, `.csv`, `.dat`.
- Save `.png/.jpg` figures.
- Save JSON metadata/diagnostics report.
- Preserve retry/metadata sidecar behavior.

## Phase 8. Workflow CLI

Target files:

- `pgixs_python/workflows/trxs_process.py`
- `pgixs_python/workflows/gixs_process.py`

Tasks:

- Implement TrXS command-line workflow.
- Implement GIXS command-line workflow.
- Support input file/series, standard file, output directory, and basic options.
- Emit progress and failure summaries.

Example target commands:

```powershell
python -m pgixs_python.workflows.trxs_process --input sample.h5 --standard standard.mat --out out_trxs
python -m pgixs_python.workflows.gixs_process --input sample.h5 --standard standard.mat --out out_gixs
```

## Phase 9. Real Data Validation

Tasks:

- Run existing `test/` fixture smoke tests.
- Compare selected Python outputs against MATLAB outputs when MATLAB becomes
  available again.
- Confirm detector-gap average behavior.
- Confirm saved `*_2Dinfo.h5` round-trips in Python.

## Phase 10. UI Design Planning Gate

This phase starts only after the Python Tab1 core workflow is stable enough
for routine real-data processing.

Tasks:

- Review the finished CLI/config workflow together before UI implementation.
- Interview target users and decide which workflows must be first-class UI
  actions.
- Discuss desktop UI vs local web UI tradeoffs before choosing a framework.
- Define the main screen layout, required panels, plot interactions, progress
  messages, error-reporting behavior, and file-management model.
- Decide how config JSON maps to UI controls so CLI, desktop UI, and later web
  UI stay consistent.
- Create UI mockups/wireframes and review them together before writing UI code.

## Progress Log

- Created initial `python_tab1/` package scaffold.
- Added MATLAB-to-Python Tab1 mapping document.
- Added minimal SDD ring calculation.
- Added minimal NaN-omitting q-bin average.
- Added CLI scaffolds for GIXS and TrXS workflows.
- Added minimal smoke runner.
- Switched Python-native 2D-info storage plan from MAT-first to H5-first for
  metadata extensibility.
- Added H5-native `TwoDInfo` save/load API scaffold.
- Added H5 detector frame loader scaffold with MATLAB-compatible transpose,
  saturation cleanup, mask application, and retry-to-NaN fallback.
- Added H5 metadata reader scaffold for IC/BC counters, timestamp, and
  temperature/angle scan values.
- Added H5 series loader scaffold that builds a `(frame, row, col)` stack and
  records retry/metadata failed-frame sidecars.
- Wired H5 file/folder stack loading into TrXS and GIXS workflow scaffolds.
- Added secondary `.mat`, `.tif/.tiff`, and numeric text image loaders to the
  image loading dispatcher.
- Added standard q-value lookup and SDD standard calibration JSON/MAT loading.
- Added vectorized TrXS and GIXS Q-map calculations with MATLAB sign/shape
  conventions.
- Wired standard calibration loading and Q-map summary generation into TrXS
  and GIXS workflow commands.
- Added workflow `*_2Dinfo.h5` assembly and save path for calibrated H5 inputs.
- Expanded q-bin averaging with median/Huber robust intensity, std proxy,
  valid-count, geometry-count, and coverage diagnostics.
- Added pixel-radius bin map and radius average with Q-axis and coverage
  diagnostics.
- Added angular/sector radius average with coverage drop diagnostics.
- Added GIXS-style cake average with relative-angle sector masking and coverage
  diagnostics.
- Expanded normalization/background subtraction helpers with IC/BC handling,
  counter fallbacks, failure reports, and negative clamp behavior.
- Added detector mask helpers, Eiger4M-only grid removal, and MATLAB-style
  detector correction factor functions with NaN-preserving application.
- Wired optional normalization, background subtraction, detector mask,
  Eiger4M grid removal, and detector correction into TrXS/GIXS workflow CLIs.
- Validated real H5 workflow execution for `test/kink_test` GIXS and
  `test/standard/8.95keVSAXS` TrXS JSON-standard cases; both saved and
  round-tripped Python-native `*_2Dinfo.h5` outputs.
- Replaced workflow stack `nanmean` with an all-NaN-safe mean to keep detector
  gaps as NaN without runtime warnings.
- Added 1D data export helpers for `.csv/.dat` with default q/intensity/std
  columns and optional count/coverage diagnostics, plus optional matplotlib
  figure export.
- Wired q-binned 1D export into TrXS/GIXS workflow CLIs with configurable
  q-range, bin count, linear/log binning, std, coverage diagnostics, and
  optional figure output.
- Added workflow JSON report export for input files, output files, processing
  options, calibration summary, and retry/metadata/processing diagnostics.
- Added optional JSON workflow config support so CLI, future desktop UI, and
  future web UI can share the same processing option schema.
- Added a post-processing CLI for exporting q-binned 1D curves and JSON
  reports directly from existing Python-native `*_2Dinfo.h5` files.
- Added JSON config key validation so misspelled workflow/UI option names stop
  early instead of being silently ignored.
- Added JSON config value validation for common type/range mistakes such as
  invalid q-bin counts, q-range order, booleans, and option choices.
- Added explicit H5 `frame_index` selection for multi-frame detector datasets
  and recorded the selected frame in workflow reports.
- Added compact 1D coverage diagnostics to workflow/export JSON reports so
  detector-gap affected q-bins can be flagged by the future UI.
- Added generated JSON Schema files for TrXS, GIXS, and 1D export configs so
  future desktop/web UI controls can share the CLI config contract.
- Added optional JSONL workflow manifest output for batch/UI result tracking
  across repeated TrXS/GIXS runs.
- Added a JSON-driven batch workflow runner that can execute TrXS/GIXS/1D
  export jobs sequentially and share a manifest path across jobs.
- Added quiet batch execution with per-job log files and structured batch
  summary JSON for future UI/automation use.
- Added a generated JSON Schema for batch workflow files.
- Updated Python 3.10 dependency pins and added a CLI quickstart covering
  single workflows, H5 1D export, batch execution, and schema regeneration.
- Added a lightweight no-pytest unit harness for quick validation on fresh
  desktop installs.
- Added a CLI core stabilization checklist to track UI readiness risks and
  remaining real-data validation work.
- Added Linux shell examples to the CLI quickstart and marked the Linux CLI
  documentation item complete in the stabilization checklist.
- Added a MATLAB-vs-Python validation template and comparison-log CSV scaffold
  for recording real-data numeric parity checks.
- Added batch `--dry-run` validation so batch files and workflow configs can be
  checked before loading detector data or writing processing outputs.

## Next Step

Continue Phase 8/9 workflow hardening and real-data validation:

1. Keep validating real EIGER H5 fixtures against TrXS/GIXS workflow outputs.
2. Add remaining batch-processing conveniences needed before UI work.
3. Compare selected Python `*_2Dinfo.h5` and 1D exports against MATLAB outputs
   once MATLAB is available.
4. Use the generated config schemas as the UI-control contract during the UI
   design review.
