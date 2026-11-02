# MATLAB cross-check

Runs the same real detector frame through **PGIXS_matlab** and **PGIXS_python**
with byte-identical calibration numbers, and reports where the two 1D curves
agree and where they don't.

This is a validation harness, not a test: it needs a MATLAB install, a checkout
of `PGIXS_matlab`, and real data, so it can't run in CI. Use it after touching
anything in the reduction path — Q maps, detector corrections, masking, binning,
the averaging primitives — to confirm the change didn't move the physics.

## Running it

Both sides read one JSON config so neither can quietly use a different beam
centre or SDD than the other. From the repo root:

```bash
matlab -batch "restoredefaultpath; addpath('tools/matlab_crosscheck'); run_matlab('tools/matlab_crosscheck/config.example.json')"
```

```bash
python tools/matlab_crosscheck/run_python.py tools/matlab_crosscheck/config.example.json
python tools/matlab_crosscheck/compare.py    tools/matlab_crosscheck/config.example.json --plot compare.png
```

`restoredefaultpath` is not optional. MATLAB persists its path across `-batch`
runs, so a directory added by an earlier session stays on it and can shadow the
repo — that has already caused one comparison to silently measure the wrong
code. `run_matlab.m` additionally drops `PGIXS_matlab/dist/` from `genpath`,
since that gitignored packaged copy of `functions/` would otherwise come first.

Two configs ship as examples: `config.example.json` (GIXS line cuts and
azimuthal) and `config.trxs.example.json` (TrXS circular average). Copy one and
edit the paths — `output_dir` defaults under `validation_outputs/`, which is
gitignored.

### Config fields

| field | meaning |
| --- | --- |
| `matlab_repo` | Path to the `PGIXS_matlab` checkout; relative paths are taken from the current folder (the examples assume it sits next to this repo). |
| `output_dir` | Where both runners write their CSVs. |
| `mode` | `"gixs"` or `"trxs"`. |
| `frame` | One detector image. |
| `beam_center` | `[y, z]`, **1-based**, as stored in a `.mat` calibration. |
| `sdd_mm`, `wavelength_a` | From the same calibration. |
| `incident_angle_deg` | GIXS only. |
| `reductions` | Any of `line` (`angles_deg`), `azimuthal` (`pixel`, `range_minus`, `range_plus`), `circular` (`angle_min`, `angle_max`, `min_coverage`). |

Take the calibration numbers from the **`.mat`** file. A `.json` written by a
MATLAB build older than PGIXS_matlab commit `89c4c85` stores the beam centre one
pixel low on both axes, which is enough to shift every q axis.

## Reading the report

Judge a run by these two lines:

- **`peak positions`** — the real check. Peak q depends on beam centre, SDD,
  energy and the detector corrections all being right, so the two sides must
  agree to well under one bin. Observed on `PTCDIC13_0001.h5`: ≤ 6e-4 1/Å.
- **`signed median diff`** — catches a systematic scale or offset error. Should
  sit near zero. Observed: −0.49% / +2.86% / −0.58% (line 90°, line 0°,
  azimuthal); +0.07% for the TrXS circular average.

**`|rel| median` is not an error measure.** It is expected to be nonzero, and on
low-count data it can be tens of percent, because the two programs do not use
the same estimator or the same sampling geometry. The known, deliberate
differences:

| reduction | MATLAB | Python |
| --- | --- | --- |
| GIXS 1D line cut | one bilinearly interpolated ray | ±1° wedge, averaged per q bin |
| GIXS azimuthal | polar grid, 0.2°, nearest pixel | real annulus pixels, 1° bins |

Consequences you will see in the report and should not chase:

- **Python's curve starts later.** A ±1° wedge holds less than one pixel below
  r ≈ 26 px, so there are no bins under q ≈ 0.046 on this geometry. MATLAB's ray
  interpolates and always returns a value, including inside the beamstop.
- **The horizontal (0°) line cut scatters more than the vertical one.** The
  wedge half-width grows with radius (±4.7 px at q = 0.5, ±14 px at q = 1.5), and
  in GIXS the horizontal direction crosses the fast-varying Yoneda/horizon
  region. Median 17.7% vs 6.0%. Its signed median (+2.9%) is the largest of the
  set for the same reason: the wedge takes in the brighter pixels just above the
  horizon, which a single ray along it does not sample.
- **The TrXS circular average matches closely** (median 0.07%, max 0.2%):
  both sides take the arithmetic mean of the same pixels in the same bins.
  Python used a Huber M-estimator here before v0.3.14, which put up to 220x
  differences into the bins at the beamstop edge.
- **q axes differ by ~1e-5 relative.** The two programs use slightly different
  values of hc — MATLAB 12.3985167 keV·Å (from `h` and `c` in
  `processGIXS2D.m`), Python 12.3984198 (`calibration/sdd.py`). 7.8 ppm. Both
  runners convert wavelength→energy separately so this stays visible rather than
  being cancelled out by sharing a wavelength.

A result that should worry you: peak positions moving by more than a bin, a
signed median that is no longer near zero, a NaN count that changes on only one
side, or a bin count that changes at all.

## Keeping it honest

`run_matlab.m` reproduces GUI callbacks that cannot be called from a script —
`GIXS_gui.m`'s `DrawPlot_Callback` is inlined, and `TrXS_gui.m`'s
`CircularAve_Callback` is copied. `Anglecut1d_gi` is called directly. If those
callbacks change on the MATLAB side, update the copies here; silent drift would
make Python look wrong when it isn't.

`run_python.py` deliberately goes through `console_session`'s public entry
points (`console_load` → `console_qmap` → `console_line_1d` /
`console_azimuthal_1d` / `console_circular_average`) rather than calling the
averaging primitives, so it exercises the same path the app does.
