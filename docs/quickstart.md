# Quickstart

A walk through the whole reduction chain — calibration, reciprocal-space
mapping, one-dimensional cuts, and an *in-situ* series — on data you can
generate in one command. Each step states the number you should get back, so
you can tell whether the software is working before you trust it with a
measurement.

Allow about twenty minutes.

> A far more detailed manual, covering every control on every tab, is in
> [`desktop_manual_ko.md`](desktop_manual_ko.md) (Korean).

## 1. Install

Python 3.10 is required (the pinned PySide6/NumPy versions do not build on
newer interpreters).

```bash
git clone https://github.com/YoungYongKim85/PGIXS_python.git
cd PGIXS_python
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the install:

```bash
python -m pytest tests --ignore=tests/test_desktop_smoke.py -q
python -m pytest tests/test_desktop_smoke.py -q
```

Both groups should pass. (They are run separately because Qt widget teardown
trips an access violation when the desktop tests share a process with the
rest; this is a known interaction, not a failure of either group.)

If you would rather not install Python at all, the
[Releases](https://github.com/YoungYongKim85/PGIXS_python/releases) page has
standalone builds for Windows, macOS and Linux. Everything below applies
unchanged.

## 2. Generate the example data

```bash
python examples/make_example_data.py
```

This writes about 15 MB into `examples/data/`:

| File | What it is |
| --- | --- |
| `calibration/agbeh_0001.h5` | Silver behenate powder rings |
| `gixs_series/example_gi_000{1,2,3}.h5` | Three grazing-incidence frames, with a temperature ramp |

These are **simulated** patterns, not measurements, written in the real
Eiger4M geometry (2167 × 2070 pixels, 75 µm pitch) and the beamline's own HDF5
layout, so they load through exactly the same code path as real data —
including the detector mask, which is what puts the module gaps in the images.

The point of simulating them is that the answers are known. The frames are
built at **SDD = 220.0 mm**, **λ = 1.1979 Å**, beam centre **(900.0, 1600.0)**,
incidence angle **0.12°**, so the numbers the software returns can be checked
against the numbers that went in.

## 3. Start the app

```bash
python -m pgixs_python.desktop.app
```

Seven tabs open: **SDD Calibration**, **GIXS**, **GIXS-series**, **TrXS**,
**TrXS-series**, **XTrans**, **Penetration Depth**. The workflow below uses the
first three.

## 4. Calibrate the sample–detector distance

On the **SDD Calibration** tab:

1. Set **Detector** to `Eiger4M`, then **LOAD IMAGE DATA** and pick
   `examples/data/calibration/agbeh_0001.h5`.
   Concentric rings appear. Set the colour scale to `Log` if they are hard to
   see.
2. Enter the photon energy: **10.3502 keV** (this is λ = 1.1979 Å).
3. Leave **Standard** at `Silver behenate`.
4. Tick **Set C** and click roughly in the middle of the rings. It does not
   have to be accurate — the next step refines it. Anywhere within ~10 pixels
   is fine.
5. Click **SMART AUTO RUN**.

Expected result:

| | Returned | True value |
| --- | --- | --- |
| Beam centre | 900.00, 1600.00 | 900.0, 1600.0 |
| SDD | **220.02 mm** | 220.0 mm |
| Rings accepted | 7 | — |
| Spread across rings | 0.024 mm (0.011%) | — |

Note which orders it used. The first-order ring sits at radius 58.8 px, inside
the beamstop shadow, so it is **not visible** — the visible rings are orders 2
through 8. Smart Auto Run works this out from the self-consistency of the SDD
each ring implies, rather than assuming the innermost visible ring is first
order. Assuming otherwise would give an SDD wrong by roughly a factor of two.

Now click **Save Calibration**. This writes `agbeh_0001_standard.json`
alongside `.mat` and `.std` copies for the MATLAB version. All three store the
beam centre as **1-based** pixel coordinates.

## 5. Reduce a single frame

On the **GIXS** tab:

1. Set **Detector** to `Eiger4M`, then **LOAD DATA** and pick
   `examples/data/gixs_series/example_gi_0001.h5`.
2. Click **Load Info** and select the `agbeh_0001_standard.json` you just
   saved. Energy, SDD, pixel size and beam centre fill in.
3. Set the **incidence angle** to `0.12` deg.
4. Click **q-Map Cal.**

q-Map Cal. does two things: it maps every pixel to (q<sub>xy</sub>, q<sub>z</sub>)
using the incidence angle, and it divides the image by a correction factor of
`solid angle × polarisation × Kapton-window transmission`, plus a silicon
sensor absorption term for the Eiger4M. Nothing downstream will run until this
has been done.

Now take a vertical cut. Set the cut angle to **90°**, click **Line** to place
it, then **1d Plot**:

| | Expected |
| --- | --- |
| Peak position | **q ≈ 0.407 Å⁻¹** (built at 0.409) |
| Curve gaps | Two, at q ≈ 0.87–0.95 and 1.79–1.85 Å⁻¹, where the cut crosses detector module gaps (plus a few bins lost at the detector edge, q > 2.58) |

The gaps are real and deliberate. Pixels bordering a module gap can carry
biased counts, so a 3-pixel shoulder around every gap is excluded from all 1D
reductions — those bins are dropped rather than averaged from whatever
survives.

Set the angle to **0°** and click **1d Plot** again for the in-plane
direction: the peak should land at **q ≈ 0.38 Å⁻¹**.

## 6. Reduce the series

The **GIXS-series** tab does the same reduction across a whole frame sequence
and plots the result against an experimental axis.

1. Set **Exp.Mode** to `Temperature`, then **LOAD DATA** and pick
   `example_gi_0001.h5`.
2. Click **Set / Table**, tick all three files, and **Import**.
3. **Load Info** and **q-Cal.** exactly as before (incidence angle `0.12`).
4. Set the cut angle to **90°** and click **Exec Cut**.
5. Under **Options**, set **Tmode** to `Temperature1`.

You should see three curves, coloured by temperature, with the axis reading
**25, 55, 85 °C**. The out-of-plane peak at q ≈ 0.41 Å⁻¹ grows strongly across
the series while the in-plane peak at q ≈ 0.38 Å⁻¹ does not — the intended
"ordering on heating" story of the simulated data:

| Frame | Temperature | Peak intensity at q ≈ 0.41 |
| --- | --- | --- |
| 1 | 25 °C | 2630 |
| 2 | 55 °C | 6190 |
| 3 | 85 °C | 9720 |

Worth trying from here:

- **Play** opens a frame viewer to scrub through the series.
- **Heatmap** plots the whole q-versus-temperature surface in one image.
- **Exec Azi** in the Azimuthal Process block (try `Pixel` = 230, range ±3)
  gives intensity versus azimuthal angle on a ring, which is how orientation
  is measured. Ticking **Reflect missing wedge (Qxy/Qz)** re-runs it against a uniform reciprocal-space grid
  so that coverage gaps show up as gaps — the values change, on purpose;
  section 10 of the manual explains why the two modes are not comparable in
  absolute intensity.
- **Normalize** divides each frame by its own ion-chamber (IC) or beam-counter
  (BC) reading. The example data carries both.

## 7. Using your own data

Nothing above is specific to the example. Point the same steps at your own
frames; the only requirements are that the detector is one PGIXS knows
(`Eiger4M`, `MarCCD`, `MX170HS`, `PI-CCD`, and `MAR345` on the calibration
tab) and that the files are HDF5 in the beamline layout, or TIFF.

If you have a calibration written by the **MATLAB** version, load the `.mat`
rather than the `.json`. A `.json` written by a MATLAB build older than
PGIXS_matlab commit `89c4c85` stores the beam centre one pixel low on both
axes, which is enough to shift every q axis.

## 8. Where to go next

- [`desktop_manual_ko.md`](desktop_manual_ko.md) — the full manual: every
  control, the formulas behind each calculation, the save-file formats, and a
  per-tab walkthrough. Korean.
- [`../tools/matlab_crosscheck/README.md`](../tools/matlab_crosscheck/README.md) —
  reduce the same frame with the MATLAB and Python versions and compare them.
  Read this before concluding that a difference between the two is a bug.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — reporting a result that looks
  wrong, and what a change to the reduction path needs.
