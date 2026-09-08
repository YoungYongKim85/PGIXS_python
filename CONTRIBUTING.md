# Contributing to PGIXS

Thanks for taking the time. PGIXS is maintained for users of the 3C beamline at
Pohang Accelerator Laboratory, so bug reports from real measurements are as
valuable as code.

## Getting help, or reporting something wrong

Open an issue at
<https://github.com/YoungYongKim85/PGIXS_python/issues>. There is no separate
support forum or mailing list — questions are welcome as issues too.

For a **result that looks wrong**, the following makes it possible to
reproduce:

- which tab and which button (e.g. "GIXS-series → Exec Azi"),
- the detector, and the calibration you loaded (`.mat`/`.std`/`.json`),
- the numbers you typed into q-Cal (energy, SDD, beam centre, incidence angle),
- what you expected versus what you got — a screenshot of the plot is fine,
- one frame that reproduces it, if you are able to share the data.

For a **crash**, paste the whole traceback. If you are running a packaged
build rather than the source, say which release.

## Changes to the reduction path

Anything that touches Q mapping, detector corrections, masking, binning or the
averaging primitives changes numbers that people have already published from.
Such a change needs:

1. **A test that fails without it.** The suite has roughly 630 tests; find the
   nearest one and follow its shape. Tests state the physical behaviour they
   are protecting, and reference the MATLAB routine or the user report behind
   it where there is one.
2. **A cross-check against the MATLAB version**, if you have MATLAB and real
   data. See [`tools/matlab_crosscheck/README.md`](tools/matlab_crosscheck/README.md).
   Peak positions and the signed median difference are what must not move; the
   relative-difference spread is expected to be nonzero.
3. **A note in `packaging/release_notes.md`** saying what changes numerically,
   scoped to where it actually applies.

The two implementations are deliberately not bit-identical, and the known
differences are catalogued in section 10 of the user manual. Please do not
"align" one to the other without discussing it in an issue first — in several
cases the Python behaviour is the better one.

## Running the tests

```bash
python -m pytest tests
```

The desktop tests need a Qt platform. On a headless machine:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests
```

## Style

Match the surrounding code rather than any external style guide. Comments in
this codebase explain *why* a piece of physics or a workaround is there —
particularly where behaviour was chosen to match (or deliberately diverge
from) the MATLAB version. Keep that habit; it is what makes the reduction path
reviewable.

## Code of conduct

Be decent to each other. Harassment, personal attacks and discriminatory
language are not acceptable in issues, pull requests or any other project
space. Report a problem to kimyy@postech.ac.kr.
