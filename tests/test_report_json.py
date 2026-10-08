import json

import numpy as np

from pgixs_python.export import WorkflowReport, save_workflow_report_json


def test_save_workflow_report_json_converts_numpy(tmp_path):
    out = tmp_path / "report.json"

    save_workflow_report_json(
        out,
        WorkflowReport(
            measurement_mode="TrXS",
            input_files=["a.h5"],
            output_files={"two_d_info": "a_2Dinfo.h5"},
            options={"q_bins": np.int32(10)},
            calibration={"beam_center": np.array([1.0, 2.0], dtype=np.float32)},
            diagnostics={"retry_failed_frames": np.array([2], dtype=np.int32)},
        ),
    )

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["measurement_mode"] == "TrXS"
    assert loaded["options"]["q_bins"] == 10
    assert loaded["calibration"]["beam_center"] == [1.0, 2.0]
    assert loaded["diagnostics"]["retry_failed_frames"] == [2]


def _strict_json(text):
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=reject)


def test_workflow_report_writes_nan_as_null(tmp_path):
    from pathlib import Path

    report = WorkflowReport(
        measurement_mode="GIXS",
        input_files=[Path("a.h5")],
        output_files={"h5": None},
        diagnostics={"timestamps": np.array([1.0, np.nan]), "energy": float("nan")},
    )

    payload = _strict_json(save_workflow_report_json(tmp_path / "r.json", report).read_text(encoding="utf-8"))

    assert payload["diagnostics"] == {"timestamps": [1.0, None], "energy": None}
    assert payload["input_files"] == ["a.h5"]
