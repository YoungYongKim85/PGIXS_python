# GIXS/TrXS Console UI Design

This document maps the real MATLAB `GIXS_gui.m` / `TrXS_gui.m` consoles
(the Tab1 windows, not the Tab1 launcher `PGIXS.m`) to the Python web
console shipped in `pgixs_python.ui_backend`. It exists so future changes
to either side can be checked against the other, the same way
`docs/matlab_to_python_tab1_map.md` tracks the core processing helpers.

The web console replaced two earlier redesigns (a dark "instrument
console" theme, then a batch-queue-first workbench) that were rejected for
inventing UI structure the real MATLAB tool does not have. This version
instead mirrors `GIXS_gui.m` / `TrXS_gui.m` panel-for-panel and
control-for-control.

## Scope

Phase 1 shipped the GIXS console. Phase 2 (this pass) adds the TrXS
console using the same pattern, matching `TrXS_gui.m`
(`C:\Users\owner\Documents\PGIXS_matlab업그레이드\gui\tab1\TrXS\TrXS_gui.m`).
Both consoles share one backend module
(`pgixs_python.ui_backend.console_session`) and one frontend JS factory
(`createConsole(cfg)` in `local_server.py`) — see "Shared implementation"
below.

The existing Batch Queue workbench (job table, validate/run, 2D/1D/coverage
preview from already-written files) is kept as a secondary "Batch" tab —
MATLAB has no equivalent, but it is genuine Python-added value, not
invented UI language competing with the consoles.

## Panel mapping

| MATLAB | Web console |
| --- | --- |
| `uifigure` `Color [0.96 0.96 0.96]` (GIXS 1400x850, TrXS 1200x850) | `:root` light palette: `--bg:#f2f2f2`, `--panel:#fafafa`, `--panel-2:#fff` |
| `uipanel('Title','Single Data Analysis', ...)` (GIXS is `Scrollable,'on'`, TrXS is not) | `<fieldset class="console-panel">` left column |
| `uipanel('Title','2D Image Plot'/'2D Plot Viewer','BackgroundColor','w')` | `<fieldset class="console-panel plot">` with `<canvas class="console-canvas">` |
| `uipanel('Title','Image Controller')` | `<fieldset class="console-panel controller">` |
| `uibutton('LOAD DATA', BackgroundColor [0.1 0.3 0.5])` | `.console-load-btn.primary`, `--accent:#1a4d80` |
| `uieditfield` with `BackgroundColor [0.8 1 1]` (editable calibration) | `input.cyan-field`, `--editable-bg:#ccffff` |
| `uieditfield` with `BackgroundColor [0.9 0.9 0.9]`, `Editable off` | `input.readonly-field`, `--readonly-bg:#e5e5e5` |
| `'--- [ 1D Plot Process ] ---'` / `'--- [ Azimuthal Process ] ---'` labels | `.c-divider` elements, same text |
| Footer `'Designed By Y.Y.Kim v2.0'`, `FontColor [0.5 0.5 0.5]` | `.console-footer`, `--text-dim:#808080` |

## Shared implementation

Backend: one in-process `ConsoleSession` (module-level singleton in
`console_session.py`) with a `mode` field ("gixs" or "trxs"), set once by
`console_load`'s `mode` payload key and held fixed until the next load —
exactly like MATLAB, where GIXS and TrXS are separate console windows and
a window never switches geometry mid-session. `console_qmap` branches on
`session.mode` to call `qmap.gixs.qmap_gixs` (needs `incident_angle_deg`)
or `qmap.trxs.qmap_trxs` (no incidence angle); `console_save` branches the
same way to write `measurement_mode: "GIXS"/"TrXS"`, `incident_angle` only
for GIXS, and `qr_map` (GIXS) vs `qx_map` (TrXS). `console_line_1d`,
`console_azimuthal_1d`, and `console_save`'s file-writing are otherwise
mode-agnostic and fully shared.

Frontend: `createConsole(cfg)` in `local_server.py` is a factory that
takes `{mode, prefix, hasIncidence, hasFootprint, hasCakecut, hasCircular}`
and wires one full console (load/qmap/line/azimuthal/save/canvas/curve),
looking up every element as `` `${cfg.prefix}-${name}` ``. It's invoked
twice:

```js
createConsole({mode: "gixs", prefix: "c", hasIncidence: true,  hasFootprint: true,  hasCakecut: true,  hasCircular: false});
createConsole({mode: "trxs", prefix: "t", hasIncidence: false, hasFootprint: false, hasCakecut: false, hasCircular: true});
```

This is why GIXS fields are `c-*` ids and TrXS fields are `t-*` ids in the
HTML — the two consoles are separate DOM subtrees on the same page
(`#workspace-console-gixs` / `#workspace-console-trxs`, toggled by the
`.workspace-tab[data-workspace]` nav), not templated per-mode strings.

**Known limitation:** because there is one shared server-side
`ConsoleSession`, loading data in one console's tab replaces the other
console's loaded state (there is no per-tab session). This mirrors "only
one thing loaded at a time," which matches a single MATLAB window's
behavior, but does not match having both a `GIXS_gui` and a `TrXS_gui`
MATLAB window open simultaneously with independent state. A visible
consequence: switching tabs after loading in the other one leaves a stale
canvas/curve until you reload or rerun in the newly active tab.

## Control mapping and backend wiring

Routes live in `pgixs_python.ui_backend.local_server` under
`/api/console/*`, dispatching to `console_session` functions.

| MATLAB control / callback | Web field(s) (GIXS `c-*` / TrXS `t-*`) | Endpoint |
| --- | --- | --- |
| `pathset` (Directory) + `Set` | `-dir`, `-browse` | `GET /api/console/list-files` (populates the `-file` dropdown, mirroring what `uigetfile` shows when it opens in that folder) |
| Detector/Multishot + `Openimage_Callback` (`uigetfile` pick) | `-detector`, `-multishot`, `-file`, `-load` | `POST /api/console/load` (body: `directory`, `filename`, `mode`, `detector`, `multishot`) |
| IC/BC labels | `-ic`, `-bc` | (part of load response) |
| Energy/SDD/Center(/Footprint for GIXS) + `Openinfo_Callback` | `-energy`, `-sdd`, `-beam-y`, `-beam-z`, (`c-footprint`), `-info-path` + `-load-info` | `POST /api/console/load-info` |
| `GridRemove_Callback` | `-grid-remove` | `POST /api/console/grid-remove` |
| GIXS-only `CalIncidence_Callback` / `CalIncidencePixel_Callback` | `c-ai-pix`, `c-ai-deg` | client-side trig only, no round trip |
| Background + `Bkgload_Callback` | `-bkg-dir` + `-bkg-browse`, `-bkg-file`, `-load-bkg` | `POST /api/console/background` (same `directory`/`filename`/`multishot` contract as load) |
| `Subtraction_Callback` (with `uiconfirm`) | `-normal-method`, `-plus-data`, `-subtract` | `POST /api/console/normalize` twice (`confirm:false` preview, then `confirm:true`/discard) matching MATLAB's preview-then-confirm flow; the "keep?" prompt is `window.confirm(...)` |
| `Correction_Callback` (q-Map Cal., `processGIXS2D`/`processTrXS2D`) | `-qmap-cal`, `-q-range` | `POST /api/console/qmap` |
| GIXS `PointFindLine_Callback` / GIXS `PointFindAzi_Callback` / TrXS `PointFind_Callback` (`drawpoint`, single click only — see descope) | canvas click while `-line-position` (GIXS only) / `-azi-position` armed | client-side pixel math only |
| `Drawline_Callback` | `-line-draw` | client-side canvas overlay only |
| `Lineplot_Callback` | `-line-plot` | `POST /api/console/1d/line` |
| TrXS-only `CircularAve_Callback` (Huber M-estimator q-bin average, optional angle range, min-coverage nulling) | `t-min-coverage`, `t-median`, `t-std`, `t-ang-min`, `t-ang-max`, `t-circular-plot` | `POST /api/console/1d/circular` → `console_circular_average`, reusing `averaging.angular.sector_average` |
| `DrawPlot_Callback` (azimuthal) | `-azi-draw` | client-side ring overlay only |
| `AziPlot_Callback` | `-azi-plot` | `POST /api/console/1d/azimuthal` |
| ColorScale/Colormap/Contrast/Map format/Update | `-color-scale`, `-colormap`, `-cmin`/`-cmax`, `-map-format`, `-update` | client-side canvas re-render only |
| `Savefile_Callback` | `-save` | `POST /api/console/save` (writes `*_2Dinfo.h5` + `*_1D.csv`, then triggers a browser PNG download of the canvas) |

## Deliberate simplifications versus MATLAB

These are scoped-out on purpose, not oversights:

- **Line cut geometry.** MATLAB's GIXS dispatches among four functions
  (`Anglecut1d_gi`, `Anglecut1d_giplus`, `CakeCut_gi`, `Accumpix_gi`)
  depending on the Accum-Pixel/Cakecut checkboxes; TrXS always uses the
  single `Anglecut1d`. The web console always uses
  `averaging.cake.cake_average` (an angular wedge around the reference
  angle — the cakecut geometry) for both modes. This matches TrXS exactly
  and stands in for GIXS's more elaborate dispatch. GIXS's Accum-Pixel
  checkbox is visually present but not wired to a distinct algorithm.
- **Detector correction.** `q-Map Cal.` applies solid-angle x polarization
  (`processing.corrections.detector_correction_factor`) but not kapton or
  silicon attenuation, since those need externally supplied attenuation
  tables not wired up anywhere in this pass.
- **File picker.** `Directory` + `Browse` populates a `<select>` from
  `GET /api/console/list-files` instead of a native OS file dialog (a
  browser sandboxed to a local server can't open one). `Multishot`
  behavior itself is a faithful port: picking `HDH_155_0001.h5` with
  Multishot on strips the trailing `_NNNN` (`_normalize_series_prefix`,
  porting `normalizePGIXSSeriesPrefix.m`) and averages every same-prefix,
  same-extension file in that directory (`_list_series_files`, porting
  `listPGIXSSeriesFiles.m`) — not "every file in the folder."
- **Calibration file picker.** `-info-path` is a plain text path field,
  like every other path field in this app, instead of a native file
  dialog (the app is a local server + browser, so there is no direct
  filesystem picker without a lot of extra plumbing).
- **Popup plot windows.** MATLAB's `OneDplotQlinegi_gui`/`OneDplotQline_gui`,
  `OneDplotAzi_gi_gui`/`OneDplotAzi_gui`, and TrXS's `OneDplotQrad_gui`
  open separate figure windows. The web console renders the result inline
  in a "1D Result" panel on the same page.
- **Colormaps.** MATLAB offers 14 colormap options. The web console ships
  three (`heat`, `gray`, `hot`) — a representative subset, not a port of
  every LUT.
- **`z-plot` peak picking (GIXS-only).** `aiZplot_Callback` opens a
  separate figure and blocks on `ginput(1)`. Not implemented; the
  incidence-angle pixel/degree fields can still be set directly or via
  the Position-click-on-canvas flow.
- **TrXS live-drag ROI and `roipoly` masking.** `PointFind_Callback`'s
  `MovingROI`/`ROIMoved` listeners (continuous drag updates) and the
  optional polygon mask in `CircularAve_Callback`/`DrawPlot_Callback` are
  not implemented. Position is set by a single click; circular average
  and azimuthal draw always use the full unmasked geometry.
- **Azimuthal Q field is display-only in both consoles.** MATLAB's
  `CalcPos_Callback`/`CalcQpos_Callback` live pixel-to-Q conversion for
  the azimuthal section's `-azi-q` field isn't wired on either console
  yet (it always shows `0`); this is a pre-existing Phase 1 gap, not
  TrXS-specific.

## Verification fixtures

`tests/test_ui_backend_console_session.py` exercises the full golden path
against small synthetic H5 frames (`h5py.File` written to `tmp_path`,
matching the pattern in `tests/test_h5_loader.py`) for both modes: load,
q-Map Cal., 1D line plot, 1D azimuthal plot (plus TrXS-only circular
average), save-and-round-trip through `load_two_d_info_h5` (checking
`measurement_mode`, `incident_angle`, and `qx_map`/`qr_map` presence per
mode). Manual browser walkthroughs against real detector-shaped
(`PI-CCD`/`PI_CCD`, 1042x1042) synthetic frames confirmed the same path
end to end for both consoles, including canvas rendering, workspace-tab
switching among all three tabs (Console GIXS / Console TrXS / Batch), and
file writes to disk.
