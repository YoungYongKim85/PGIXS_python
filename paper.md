---
title: 'PGIXS: an integrated data-reduction application for grazing-incidence and transmission X-ray scattering'
tags:
  - Python
  - MATLAB
  - X-ray scattering
  - GIXS
  - GIWAXS
  - SAXS
  - synchrotron
  - data reduction
authors:
  - name: Young Yong Kim
    orcid: 0000-0001-6102-9518
    corresponding: true
    affiliation: 1
affiliations:
  - name: Beamline Division, Pohang Accelerator Laboratory, POSTECH, Pohang, Gyeongbuk 37673, Republic of Korea
    index: 1
date: 8 September 2026   # TODO: update to the submission date
bibliography: paper.bib
---

# Summary

PGIXS is a desktop application that takes raw area-detector frames from
grazing-incidence and transmission X-ray scattering experiments through the
complete reduction chain a beamline user needs: calibrating the
sample-to-detector distance from a standard, mapping detector pixels into
reciprocal space, applying detector corrections, extracting one-dimensional
and azimuthal profiles, and analysing whole frame series from *in-situ*
measurements. It was written for users of the 3C beamline at the Pohang
Accelerator Laboratory (PAL) and is in routine use there.

The software exists as two implementations. The original is written in MATLAB;
a second, independent implementation in Python covers the same reduction
workflow, requires no commercial licence, and is distributed as a
self-contained application for Windows, macOS and Linux. Because the two were
developed against each other rather than by translation, they also serve as a
mutual check: a cross-validation harness reduces the same frame with both and
reports where they agree and where they do not.

# Statement of need

A grazing-incidence scattering measurement does not yield a physical quantity
until a chain of corrections and transformations has been applied to it, and
the chain is longer than for powder diffraction. The sample-to-detector
distance must be calibrated from a standard; detector pixels must be mapped
into reciprocal space using the incidence angle, which makes the mapping
asymmetric about the beam centre; intensities must be corrected for solid
angle, polarisation, window transmission and, for silicon sensors, incomplete
absorption; and the resulting cuts must account for module gaps, masked
regions and the uneven sampling those produce.

Excellent tools exist for parts of this. `pyFAI` [@Ashiotis2015] provides fast,
well-tested azimuthal integration and calibration and underpins many beamline
pipelines. `Nika` [@Ilavsky2012] and `DPDAK` [@Benecke2014] offer broad
two-dimensional reduction environments. `GIXSGUI` [@Jiang2015] addresses
grazing-incidence geometry specifically, and `xrayutilities` [@Kriegner2013]
supplies the reciprocal-space transformations as a library. What these leave
to the user is the assembly: at a beamline, calibration, mapping, correction,
cutting and series analysis have to run as one workflow, on a laptop, while
the measurement is still going, by someone whose expertise is in the sample
rather than in Python packaging.

PGIXS provides that assembly for a synchrotron beamline's own users. Each
stage passes its output to the next within a single application, so a
calibration measured at the start of a beamtime is carried through every
subsequent reduction without file bookkeeping. Two capabilities in particular
are aimed at how the beamline is actually used: *in-situ* series analysis,
where a sequence of frames is reduced against a time or temperature axis and
displayed as stacked curves, heatmaps or region-of-interest traces with live
background subtraction and normalisation; and integrated calculation of X-ray
transmittance and penetration depth, which users need in order to choose an
incidence angle before they measure.

The second implementation matters for access. MATLAB is available at the
beamline but not on every user's own machine, and a licence requirement puts
reduction out of reach once the user goes home with their data. The Python
version removes that barrier while keeping the same workflow, so a user can
continue an analysis started at the beamline without changing tools or
re-learning an interface.

# Implementation and validation

Both implementations share the same physics: reciprocal-space mapping for
grazing-incidence and transmission geometry, detector corrections applied as a
per-pixel factor, and reductions that track how completely each output bin was
sampled so that bins falling across module gaps can be identified rather than
silently averaged from fewer pixels. The Python implementation carries
approximately 630 automated tests covering the reduction and calibration
routines; the MATLAB implementation has its own core test suite.

Because the two were written independently, agreement between them is
evidence rather than tautology. A cross-check harness reduces the same
detector frame with both and compares the results. On real beamline data, peak
positions agree to better than $5\times10^{-4}\ \mathrm{\AA^{-1}}$ and signed
median intensity differences are below one percent. The remaining differences
are not errors but deliberate choices of estimator and sampling geometry --
for example, a robust Huber location estimator against an arithmetic mean for
$q$-binned averages -- and these are measured, documented, and reported by the
harness rather than hidden.

# Acknowledgements

I thank the users of the 3C beamline at PAL, whose measurements, bug reports
and requests shaped this software and supplied the data used to validate it.

<!-- TODO: funding sources and grant numbers, once confirmed. If PAL/POSTECH
     require a specific acknowledgement wording for beamline-developed
     software, use theirs verbatim. -->

Portions of the Python implementation, its test suite and this manuscript were
written with the assistance of a large language model (Anthropic Claude). The
physics, the algorithmic choices, and the validation against the MATLAB
implementation and against real beamline data were directed and verified by
the author, who takes full responsibility for the content of the software and
of this paper.

# References
