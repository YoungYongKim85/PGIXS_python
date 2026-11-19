"""The incidence angle a file records fills ai without being typed."""

import h5py
import numpy as np

from pgixs_python.io.h5_metadata import read_incidence_angle
from pgixs_python.ui_backend import console_session as cs
from pgixs_python.ui_backend import gixs2_session as gs

PARAMS = "/entry/instrument/user/params"


def setup_function(_fn):
    gs.reset_session()
    cs.reset_session()


def _frame(path, angle=None, name="IAngle", shape=(48, 48)):
    data = np.random.default_rng(0).uniform(10, 100, size=shape).astype(np.float32)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/entry/data/data", data=data)
        base = h5.create_group(PARAMS)
        base.create_dataset("Counter1", data=np.array([10.0], dtype=np.float32))
        base.create_dataset("TimeStamp", data=np.array([1000.0], dtype=np.float64))
        if angle is not None:
            base.create_dataset(name, data=np.array([angle], dtype=np.float32))
    return path


def _series(tmp_path, angles, start=1):
    return [_frame(tmp_path / f"scan_{start + i:04d}.h5", angle) for i, angle in enumerate(angles)]


def _load_series(tmp_path, angles):
    files = _series(tmp_path, angles)
    result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "detector": "TestDetector"}
    )
    assert result["ok"], result.get("message")
    return result


def test_read_incidence_angle_tells_files_with_and_without_one(tmp_path):
    assert read_incidence_angle(_frame(tmp_path / "i.h5", 0.15)) == 0.15
    assert read_incidence_angle(_frame(tmp_path / "inc.h5", 0.12, name="IncAngle")) == 0.12
    assert read_incidence_angle(_frame(tmp_path / "none.h5")) is None
    assert read_incidence_angle(tmp_path / "missing.h5") is None
    assert read_incidence_angle(tmp_path / "image.tif") is None


def test_console_load_takes_ai_from_the_file_in_gixs_mode_only(tmp_path):
    def load(path, mode):
        return cs.console_load(
            {"directory": str(path.parent), "filename": path.name, "detector": "TestDetector", "mode": mode}
        )

    typed = cs.get_session().calibration["incident_angle_deg"]
    without = load(_frame(tmp_path / "none.h5"), "gixs")
    assert without["ok"], without.get("message")
    assert without["file_incident_angle_deg"] is None
    assert without["calibration"]["incident_angle_deg"] == typed

    assert load(_frame(tmp_path / "with.h5", 0.18), "trxs")["file_incident_angle_deg"] is None

    with_angle = load(tmp_path / "with.h5", "gixs")
    assert with_angle["file_incident_angle_deg"] == 0.18
    assert with_angle["calibration"]["incident_angle_deg"] == 0.18


def test_series_with_one_angle_fills_ai_and_keeps_a_single_qmap(tmp_path):
    result = _load_series(tmp_path, [0.2, 0.2, 0.2])
    assert result["file_incident_angle_deg"] == 0.2
    assert result["file_angles_vary"] is False
    assert gs.get_session().calibration["incident_angle_deg"] == 0.2

    gs.get_session().calibration_loaded = True
    # Ticking Change ai by hand still gives the typed ramp: the files do not disagree with it.
    qcal = gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})
    assert qcal["ok"] and qcal["ai_from_files"] is False
    np.testing.assert_allclose(gs.get_session().aivar, [0.2, 0.25, 0.3])


def test_series_without_angles_or_with_some_missing_keeps_the_typed_ai(tmp_path):
    typed = gs.get_session().calibration["incident_angle_deg"]
    (tmp_path / "a").mkdir()
    none = _load_series(tmp_path / "a", [None, None])
    assert none["file_incident_angle_deg"] is None and none["file_angles_missing"] is None

    (tmp_path / "b").mkdir()
    partial = _load_series(tmp_path / "b", [0.3, None, 0.5])
    assert partial["file_incident_angle_deg"] is None and partial["file_angles_missing"] == 1
    assert partial["file_angles_vary"] is False
    assert gs.get_session().calibration["incident_angle_deg"] == typed


def test_angle_scan_qcal_uses_each_files_own_angle_and_auto_refresh_follows(tmp_path):
    result = _load_series(tmp_path, [0.10, 0.14, 0.30])  # not a linear ramp
    assert result["file_angles_vary"] is True
    assert result["file_angle_range"] == [0.1, 0.3]
    session = gs.get_session()
    session.calibration_loaded = True

    qcal = gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})
    assert qcal["ok"], qcal.get("message")
    assert qcal["ai_from_files"] is True
    np.testing.assert_allclose(session.aivar, [0.10, 0.14, 0.30])

    _series(tmp_path, [0.45], start=4)
    refreshed = gs.gixs2_auto_refresh({"directory": str(tmp_path)})
    assert refreshed["ok"] and refreshed["n_new"] == 1, refreshed
    np.testing.assert_allclose(session.aivar, [0.10, 0.14, 0.30, 0.45])

    # The typed ramp is still there for whoever asks for it.
    ramp = gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05, "incident_angle_deg": 0.1, "ai_from_files": False})
    assert ramp["ai_from_files"] is False
    np.testing.assert_allclose(session.aivar, [0.10, 0.15, 0.20, 0.25])
