# MATLAB To Python Tab1 Port Map

This map intentionally excludes Tab3 indexing.

## IO

| MATLAB source | Python target | Notes |
| --- | --- | --- |
| `functions/IO/loadPGIXSImageFile.m` | `pgixs_tab1.io.image_loader` | Dispatch `.h5`, `.tif`, `.mat`, `.csv`, `.dat`, `.txt`, SPE later. |
| `functions/IO/loadPGIXSH5Frame.m` | `pgixs_tab1.io.h5_loader` | Keep EIGER orientation rules and stack-first-frame behavior. |
| `functions/IO/readPGIXSH5Metadata.m` | `pgixs_tab1.io.h5_metadata` | Preserve IC/BC/temperature/time metadata conventions. |
| `functions/IO/buildPGIXS2DInfo.m` | `pgixs_tab1.io.two_d_info` | Python-native bridge should be H5. |
| `functions/IO/savePGIXS2DInfoFile.m` | `pgixs_tab1.io.two_d_info` | MATLAB-readable `.mat` is compatibility export only. |
| `functions/IO/savePGIXSDataFile.m` | `pgixs_tab1.export.data_export` | Preserve MAT/CSV/DAT output. |

## Geometry And Processing

| MATLAB source | Python target | Notes |
| --- | --- | --- |
| `functions/Qmap/Qmaping_tr.m` | `pgixs_tab1.qmap.trxs` | Transmission Q maps. |
| `functions/Qmap/Qmaping_gi.m` | `pgixs_tab1.qmap.gixs` | Grazing-incidence Q maps. |
| `functions/Processing/processTrXS2D.m` | `pgixs_tab1.workflows.trxs_process` | Workflow wrapper around Q-map/correction helpers. |
| `functions/Processing/processGIXS2D.m` | `pgixs_tab1.workflows.gixs_process` | Workflow wrapper around Q-map/correction helpers. |
| `functions/Processing/processPGIXSNormalization.m` | `pgixs_tab1.processing.normalization` | Failed data remains NaN, not zero. |
| `functions/Processing/applyPGIXSDetectorCorrections.m` | `pgixs_tab1.processing.corrections` | Solid angle, polarization, efficiency. |
| `functions/Correction/Gridremover.m` | `pgixs_tab1.processing.grid_remove` | Eiger4M only. |

## 1D And Sector Averages

| MATLAB source | Python target | Notes |
| --- | --- | --- |
| `functions/Processing/processPGIXSRadiusBin1D.m` | `pgixs_tab1.averaging.radial` | Pixel-radius binning and NaN omission. |
| `functions/Processing/processPGIXSQBin1D.m` | `pgixs_tab1.averaging.qbin` | NaN omission to avoid detector-gap intensity dips. |
| `functions/Anglecut/Anglecut1d.m` | `pgixs_tab1.averaging.angular` | TrXS line/angular average. |
| `functions/Anglecut/CakeCut_gi*.m` | `pgixs_tab1.averaging.cake` | GIXS cake cut with wrap-around angle ranges. |

## Calibration

| MATLAB source | Python target | Notes |
| --- | --- | --- |
| `functions/Calibration/getPGIXSStandardSampleQ.m` | `pgixs_tab1.calibration.standards` | Standard q/order values. |
| `functions/Calibration/computePGIXSSDDFromRing.m` | `pgixs_tab1.calibration.sdd` | SDD from ring radius. |
| `functions/Calibration/analyzePGIXSSDDCandidates.m` | `pgixs_tab1.calibration.sdd` | Candidate table diagnostics. |

## Required Contracts

- Retry once for H5 frame load; failed frames become NaN.
- Detector gap and masked pixels remain NaN and are omitted from averages.
- Python-native 2D-info output should be H5 for metadata extensibility.
- MATLAB-readable 2D-info MAT export should remain available during transition.
- Windows and Linux paths must be handled through `pathlib.Path`.
