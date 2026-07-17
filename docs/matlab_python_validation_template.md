# MATLAB-vs-Python Validation Template

Use this template when comparing the Python Tab1 CLI output against the
existing MATLAB Tab1 workflow. Save one filled copy per dataset or per
processing condition.

## Dataset

- Dataset ID:
- Measurement type: GIXS / TrXS
- Raw input file or folder:
- Standard/calibration file:
- Background file:
- Normalization: None / IC / BC
- Grid removal: off / Eiger4M
- Mask/correction files:
- MATLAB version:
- Python commit:
- Date:

## Workflow Commands

MATLAB:

```text
Describe the GUI steps, selected options, and saved output filenames.
```

Python:

```text
python -m pgixs_python.workflows.gixs_process --config path/to/config.json
python -m pgixs_python.workflows.export_1d_from_h5 --input path/to/output_2Dinfo.h5 --out path/to/export
```

## 2D-Info Checks

| Check | MATLAB value | Python value | Tolerance | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| image shape | | | exact | | |
| qx/qy/qz shape | | | exact | | |
| q min/max | | | dataset-specific | | |
| intensity min/max | | | dataset-specific | | |
| NaN pixel count | | | dataset-specific | | |
| detector-gap mask location | | | visual | | |
| metadata fields present | | | exact/list | | |

## 1D Average Checks

Record values in `docs/templates/matlab_python_comparison_log.csv` or a copied
CSV using the same columns.

Optional helper:

```text
python tools/compare_1d_exports.py --reference matlab_1d.csv --candidate python_1d.csv --json-out comparison_summary.json --csv-out comparison_points.csv
```

| Check | MATLAB value | Python value | Tolerance | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| q-bin count | | | exact | | |
| q range | | | dataset-specific | | |
| peak q positions | | | user-defined | | |
| peak intensity ratio | | | user-defined | | |
| detector-gap kink location | | | visual/numeric | | |
| low-coverage q bins | | | user-defined | | |

## Acceptance Decision

- Accepted for routine use: yes / no
- Blocking differences:
- Follow-up code changes:
- Follow-up MATLAB checks:
