# PGIXS Python Tab1 UI Design Interview

This document is the working checklist for designing the first Python UI for
Tab1 raw-data processing. The goal is to keep the UI discussion tied to the
already implemented CLI/config contract instead of drifting into a separate
program.

## Current Assumption

- Scope: Tab1 only. Tab3/indexing is out of scope for this Python UI phase.
- Primary users: beamline/lab users running analysis on personal computers.
- First target: local desktop or local web UI that can later share logic with a
  hosted web version.
- Data contract: workflow JSON configs, `*_2Dinfo.h5`, JSON reports, JSONL
  manifests, and 1D `.csv/.dat` exports.
- Scientific priority: preserve detector-gap NaNs, coverage diagnostics, and
  MATLAB-vs-Python validation visibility.

## Decisions To Make

| Topic | Options | Current Decision | Notes |
| --- | --- | --- | --- |
| First UI type | Desktop app / local web app / both staged | undecided | Local web may transfer well to future hosted web. |
| Main workflow | Wizard / single-page control panel / batch-first | undecided | Must stay efficient for repeated analysis. |
| Dataset loading | File picker / folder picker / recent-project list | undecided | Needs Windows and Linux paths. |
| Config handling | Save/load JSON / project profiles / presets | undecided | Should reuse generated JSON schemas. |
| Result browsing | Per-run report list / manifest table / project tree | undecided | Batch summary JSON already exists. |
| Plotting | Static matplotlib / interactive browser plots | undecided | Depends on UI type. |
| Error reporting | Inline panel / popup / log drawer / report file | undecided | Should expose callback/log details clearly. |

## Interview Questions

1. Should the first UI feel closer to the MATLAB GUI, or should it be redesigned
   around a cleaner workflow even if the layout changes?
2. Is the first Python UI mainly for single-file/folder processing, or should
   batch processing be visible from the beginning?
3. Which controls must always be visible on the main screen?
4. Which controls can live in an advanced panel?
5. Should GIXS and TrXS be separate tabs, separate pages, or one mode selector?
6. Should the user see q-map/2D image/1D curve in one screen, or switch between
   views?
7. How should detector-gap coverage warnings appear so they are noticeable but
   not annoying?
8. What should happen after processing finishes: auto-open 2D view, auto-open
   1D view, or show a run summary first?
9. Which output files should be shown as clickable results?
10. Which settings should be saved as reusable presets?
11. Should the UI prevent processing when clean-environment checks fail, or only
    warn the user?
12. Should MATLAB-vs-Python comparison be a separate validation tool page?

## Proposed First UI Workflow

1. Select mode: GIXS or TrXS.
2. Select input H5 file/folder.
3. Select standard/calibration file.
4. Optional: background, normalization, detector mask, grid removal, detector
   correction.
5. Set q/1D export options.
6. Run dry-run/config validation.
7. Run processing.
8. Show result summary with output paths, warnings, q coverage diagnostics, and
   log/report links.
9. Open 2D image and 1D curve views.
10. Save/reuse workflow config.

## UI Readiness Mapping

| UI Need | Existing Backend Contract |
| --- | --- |
| Form fields | `docs/*_workflow_config_schema.json` |
| Dry-run validation | `pgixs_python.workflows.batch_process --dry-run` |
| Run processing | `trxs_process`, `gixs_process` |
| 2D-info output | `*_2Dinfo.h5` |
| 1D export | `export_1d_from_h5` |
| Run report | `*_report.json` |
| Batch/result list | JSONL manifest and batch summary JSON |
| Environment check | `tools/check_environment.py` |
| MATLAB comparison | `tools/compare_1d_exports.py` |

## First UI Acceptance Criteria

- User can process one GIXS H5 input from the UI and save H5 2D-info.
- User can process one TrXS H5 input from the UI and save H5 2D-info.
- User can export 1D data from the generated H5 output.
- User can see warnings for low q-bin coverage or abrupt coverage drops.
- User can open the JSON report/log from the UI.
- User can save and reload the workflow config.
- UI paths work naturally on Windows and Linux.

## Open Notes

- Final UI framework is not chosen yet.
- Actual MATLAB-vs-Python validation data still needs to be filled before
  declaring the scientific backend ready for routine use.
- Clean Windows/Linux Python 3.10 environment checks still need to be run by the
  user or on target machines.

