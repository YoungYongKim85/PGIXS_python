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
| First UI type | Desktop app / local web app / both staged | local web first | Best match for future hosted web; can still be packaged locally. |
| Main workflow | Wizard / single-page control panel / batch-first | batch-first workbench | User prefers Batch processing as the first screen. |
| Dataset loading | File picker / folder picker / recent-project list | file/folder picker plus recent projects | Needs Windows and Linux paths. |
| Config handling | Save/load JSON / project profiles / presets | JSON configs plus presets | Reuse generated JSON schemas. |
| Result browsing | Per-run report list / manifest table / project tree | run-history table | Batch summary JSON already exists. |
| Plotting | Static matplotlib / interactive browser plots | interactive browser plots first | Better for q/coverage inspection and later web deployment. |
| Error reporting | Inline panel / popup / log drawer / report file | inline status plus log drawer | Should expose callback/log details clearly. |

## Recommended UI Direction

Build the first Python UI as a local web application that runs on the user's
computer and opens in a browser. The backend should call the existing Python
workflow functions directly, not shell out to MATLAB or duplicate processing
logic. This keeps the UI portable across Windows and Linux and leaves a clean
path to a future hosted web version when a domain/server exists.

Suggested stack for the first prototype:

- Backend: Python service wrapping existing `pgixs_python.workflows` modules.
- UI: browser-based workbench with GIXS/TrXS mode selector.
- Config contract: generated JSON schemas in `docs/*_workflow_config_schema.json`.
- Result contract: `*_2Dinfo.h5`, JSON reports, JSONL manifests, and 1D
  exports.
- Plotting: browser-side interactive plots for 2D image, 1D q curve, and
  coverage diagnostics.
- Packaging later: desktop launcher can start the local service and open the
  browser automatically.

## Interview Questions

| Question | Recommended starting answer |
| --- | --- |
| Should the first UI feel closer to MATLAB or be redesigned? | Decided: redesign around a cleaner workflow, but keep familiar terms from MATLAB. |
| Single-file/folder or batch from the beginning? | Decided: Batch processing should be the first screen. Single GIXS/TrXS runs become one-job batch entries. |
| Which controls must always be visible? | Mode, input, standard, output folder, normalization, q/export options, run button, status. |
| Which controls can be advanced? | Detector correction, masking, grid removal details, frame index, metadata mode, retry count. |
| GIXS/TrXS tabs or mode selector? | Mode selector at the top; shared layout underneath. |
| One screen or separate views? | One workbench with tabs for 2D image, 1D curve, coverage, report/log. |
| How should detector-gap warnings appear? | Non-blocking warning badge plus coverage tab and report details. |
| What happens after processing finishes? | Show run summary first, then make 2D/1D views immediately available. |
| Which outputs should be clickable? | H5 2D-info, 1D export, figure, JSON report, logs, manifest/summary. |
| Which settings become presets? | Detector/calibration defaults, q export settings, normalization/background conventions. |
| Environment check behavior? | Warn but do not block, unless a required package is missing. |
| MATLAB-vs-Python comparison page? | Yes, but as a validation/admin tool rather than the default workflow. |

## Proposed First UI Workflow

1. Start local UI service and open the browser workbench.
2. Start from the Batch Queue screen.
3. Add one or more jobs: GIXS, TrXS, or Export 1D.
4. For each job, select input H5 file/folder, output folder, and standard/calibration file as needed.
5. Set routine processing options: normalization, background, q export.
6. Expand advanced panel only when needed: masks, grid removal, detector
   correction, frame index, retry count, metadata mode.
7. Run config validation/dry-run.
8. Run processing.
9. Show run summary with output paths, warnings, q coverage diagnostics, and
   log/report links.
10. Open 2D image, 1D curve, coverage, and report/log tabs.
11. Save or reload workflow config/preset.

## Suggested Screen Layout

- Left sidebar: batch config, shared manifest/log paths, job defaults, preset/config controls.
- Main top toolbar: validate batch, run batch, add job, save batch config, open output folder.
- Main content tabs: batch queue, run summary, 2D image, 1D curve, coverage
  diagnostics, run report.
- Right inspector: selected q/intensity values, warnings, metadata, file paths.
- Bottom log drawer: warnings, errors, dry-run messages, backend traceback links.

The UI should avoid hiding scientific warnings in popups. Warnings should stay
visible in the run summary until the next run starts.

## Manual Policy

The UI must be developed together with a detailed Korean user manual:
`docs/user_manual_tab1_ko.md`.

Manual updates should be part of each user-facing UI change. The manual should
explain what the feature does, when to use it, expected inputs/outputs, warning
meaning, and troubleshooting steps.

## UI Readiness Mapping

| UI Need | Existing Backend Contract |
| --- | --- |
| Form fields | `docs/*_workflow_config_schema.json` |
| UI backend service | `pgixs_python.ui_backend.service` |
| UI discovery/status | `pgixs_python.ui_backend.discovery` |
| Local web scaffold | `pgixs_python.ui_backend.local_server` |
| Korean user manual | `docs/user_manual_tab1_ko.md` |
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

## Prototype Milestones

1. Backend service wrapper exists for validating and running workflow requests:
   `pgixs_python.ui_backend.service`.
2. Backend discovery/status helpers exist for schema-driven UI startup:
   `pgixs_python.ui_backend.discovery`.
3. Local web scaffold exists with startup/status/workflow API endpoints:
   `python -m pgixs_python.ui_backend.local_server`.
4. Minimal browser workbench shell exists with workflow controls, schema field
   preview, run summary, inspector, and log regions.
5. Batch-first workbench shell exists with batch config controls and Batch
   Queue as the first tab.
6. Validate Batch button is wired to `/api/validate` with batch dry-run
   validation.
7. Run Batch button is wired to `/api/run` for real batch processing.
8. Batch summary JSON is rendered as a Run Summary result table.
9. 2D-info H5 compact preview is wired to `/api/preview/2d`.
10. 2D-info compact preview is rendered as a browser heatmap.
11. 1D export CSV/DAT preview is wired to `/api/preview/1d` and rendered as an SVG curve.
12. 1D coverage diagnostics warn about low-coverage and abrupt-drop bins.
13. Dedicated coverage preview is wired to `/api/preview/coverage` and rendered as an SVG plot.
14. Add preset/config save-load.
15. Add MATLAB-vs-Python comparison page.
16. Package local launcher for Windows and Linux.

## Open Notes

- Final UI framework is not chosen yet.
- Actual MATLAB-vs-Python validation data still needs to be filled before
  declaring the scientific backend ready for routine use.
- Clean Windows/Linux Python 3.10 environment checks still need to be run by the
  user or on target machines.
