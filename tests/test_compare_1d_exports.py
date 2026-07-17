import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


TOOLS_PATH = Path(__file__).resolve().parents[1] / "tools" / "compare_1d_exports.py"
SPEC = importlib.util.spec_from_file_location("compare_1d_exports", TOOLS_PATH)
compare_1d_exports = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = compare_1d_exports
SPEC.loader.exec_module(compare_1d_exports)


def test_compare_curves_on_matching_q_grid():
    ref = compare_1d_exports.Curve1D(q=np.array([0.1, 0.2]), intensity=np.array([10.0, 20.0]))
    cand = compare_1d_exports.Curve1D(q=np.array([0.1, 0.2]), intensity=np.array([11.0, 18.0]))

    result = compare_1d_exports.compare_curves(ref, cand)

    assert result.compared_points == 2
    np.testing.assert_allclose(result.absolute_diff, np.array([1.0, 2.0]))
    assert result.max_abs_diff == 2.0


def test_compare_curves_requires_interpolation_for_mismatched_q():
    ref = compare_1d_exports.Curve1D(q=np.array([0.1, 0.2]), intensity=np.array([10.0, 20.0]))
    cand = compare_1d_exports.Curve1D(q=np.array([0.1, 0.25]), intensity=np.array([10.0, 25.0]))

    try:
        compare_1d_exports.compare_curves(ref, cand)
    except ValueError as exc:
        assert "q grids differ" in str(exc)
    else:
        raise AssertionError("mismatched q grids should require interpolation")


def test_load_curve_and_write_summary_files(tmp_path):
    ref_path = tmp_path / "matlab.csv"
    cand_path = tmp_path / "python.csv"
    ref_path.write_text("q,intensity,standard_deviation\n0.1,10,1\n0.2,20,2\n", encoding="utf-8")
    cand_path.write_text("q,intensity\n0.1,11\n0.2,18\n", encoding="utf-8")

    ref = compare_1d_exports.load_curve(ref_path)
    cand = compare_1d_exports.load_curve(cand_path)
    result = compare_1d_exports.compare_curves(ref, cand)
    summary = compare_1d_exports.comparison_summary(result, ref_path, cand_path)
    json_path = compare_1d_exports.write_json_summary(tmp_path / "summary.json", summary)
    csv_path = compare_1d_exports.write_pointwise_csv(tmp_path / "points.csv", result)

    assert ref.std is not None
    assert json.loads(json_path.read_text(encoding="utf-8"))["compared_points"] == 2
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("q,reference_intensity")

