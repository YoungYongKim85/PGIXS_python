"""Regressions for the 2026-10-28 review: one check per fix."""

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pgixs_python.calibration.calibration_io import load_standard_calibration
from pgixs_python.processing.masking import detector_mask_for_type
from pgixs_python.ui_backend import console_session as cs
from pgixs_python.ui_backend import gixs2_session as gs
from pgixs_python.ui_backend import sddcal_session as sd
from pgixs_python.ui_backend import trxs2_session as ts

TABS = [pytest.param(gs, "gixs2", id="gixs2"), pytest.param(ts, "trxs2", id="trxs2")]


def _write(folder, names, shape=(40, 30)):
    folder.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(names):
        with h5py.File(folder / name, "w") as h5:
            h5.create_dataset("/entry/data/data", data=np.full(shape, 100.0 + idx, dtype=np.float32))
            h5.create_dataset("/entry/instrument/user/params/Counter1", data=np.array([10.0], dtype=np.float32))
            h5.create_dataset("/entry/instrument/user/params/TimeStamp", data=np.array([float(idx)], dtype=np.float64))


def _import(module, tab, folder, prefix, names):
    module.reset_session()
    assert getattr(module, f"{tab}_list_series_files")({"directory": str(folder), "filename": prefix})["ok"]
    result = getattr(module, f"{tab}_load_series")({"directory": str(folder), "filenames": names, "insitu_mode": "Time"})
    assert result["ok"], result.get("message")


@pytest.mark.parametrize("module, tab", TABS)
def test_auto_refresh_skips_a_file_left_unchecked_at_import(tmp_path, module, tab):
    names = [f"run_{i:04d}.h5" for i in range(1, 6)]
    _write(tmp_path, names)
    _import(module, tab, tmp_path, "run", ["run_0001.h5", "run_0002.h5", "run_0004.h5"])  # 0003 unchecked
    found = getattr(module, f"{tab}_pending_new_files")({"settle_seconds": 0})
    assert found == {"ok": True, "n_new": 1}  # only 0005; not 0003, not 0004 again


@pytest.mark.parametrize("module, tab", TABS)
def test_a_listing_that_is_never_imported_does_not_retarget_auto_refresh(tmp_path, module, tab):
    _write(tmp_path, [f"run_{i:04d}.h5" for i in range(1, 4)] + [f"other_{i:04d}.h5" for i in range(1, 6)])
    _import(module, tab, tmp_path, "run", ["run_0001.h5", "run_0002.h5"])
    assert getattr(module, f"{tab}_list_series_files")({"directory": str(tmp_path), "filename": "other"})["ok"]
    getattr(module, f"{tab}_auto_refresh")({})
    assert [p.name for p in module.get_session().file_paths] == ["run_0001.h5", "run_0002.h5", "run_0003.h5"]


@pytest.mark.parametrize("module, tab", TABS)
def test_auto_refresh_watches_the_series_folder_not_the_directory_box(tmp_path, module, tab):
    _write(tmp_path / "a", [f"run_{i:04d}.h5" for i in range(1, 3)])
    _write(tmp_path / "b", [f"run_{i:04d}.h5" for i in range(1, 6)])  # same names, another sample
    _import(module, tab, tmp_path / "a", "run", ["run_0001.h5", "run_0002.h5"])
    found = getattr(module, f"{tab}_pending_new_files")({"directory": str(tmp_path / "b"), "settle_seconds": 0})
    assert found == {"ok": True, "n_new": 0}


@pytest.mark.parametrize("module, tab", TABS)
def test_saves_are_named_after_the_imported_series(tmp_path, module, tab):
    eiger = detector_mask_for_type("Eiger4M").mask
    _write(tmp_path, ["first_0001.h5", "other_0001.h5", "other_0002.h5"], shape=tuple(reversed(eiger.shape)))
    module.reset_session()
    assert getattr(module, f"{tab}_load")({"directory": str(tmp_path), "filename": "first_0001.h5", "detector": "Eiger4M"})["ok"]
    getattr(module, f"{tab}_list_series_files")({"directory": str(tmp_path), "filename": "other"})
    assert getattr(module, f"{tab}_load_series")(
        {"directory": str(tmp_path), "filenames": ["other_0001.h5", "other_0002.h5"], "insitu_mode": "Time"}
    )["ok"]
    getattr(module, f"{tab}_set_x_axis")({})
    saved = getattr(module, f"{tab}_save_x_axis")({"directory": str(tmp_path / "out")})
    assert saved["ok"] and saved["path"].endswith("other_XAxis.csv")


@pytest.mark.parametrize("module, tab", TABS)
def test_another_detector_at_import_drops_the_old_load_info(tmp_path, module, tab):
    marccd = detector_mask_for_type("MarCCD").mask
    _write(tmp_path, ["mar_0001.h5"], shape=tuple(reversed(marccd.shape)))
    module.reset_session()
    session = module.get_session()
    session.calibration_loaded = True  # an Eiger4M standard, say
    assert getattr(module, f"{tab}_load_series")(
        {"directory": str(tmp_path), "filenames": ["mar_0001.h5"], "insitu_mode": "Time", "detector": "MarCCD"}
    )["ok"]
    assert module.get_session().calibration_loaded is False


def test_a_background_of_another_shape_is_dropped_with_a_new_series(tmp_path):
    _write(tmp_path, ["run_0001.h5"])
    ts.reset_session()
    session = ts.get_session()
    session.background_raw = np.ones((7, 7), dtype=np.float32)
    session.subtract_background = True
    assert ts.trxs2_load_series({"directory": str(tmp_path), "filenames": ["run_0001.h5"], "insitu_mode": "Time"})["ok"]
    session = ts.get_session()
    assert session.background_raw is None and session.background is None and not session.subtract_background


@pytest.mark.parametrize("module, tab", TABS)
def test_bad_numbers_are_messages_not_exceptions(tmp_path, module, tab):
    _write(tmp_path, ["run_0001.h5", "run_0002.h5"])
    _import(module, tab, tmp_path, "run", ["run_0001.h5"])
    assert getattr(module, f"{tab}_set_x_axis")({"time_offset_seconds": "abc"})["ok"] is False
    assert getattr(module, f"{tab}_pending_new_files")({"settle_seconds": "abc"})["ok"] is False


def test_console_normalize_refuses_a_bad_plus_data():
    cs.reset_session() if hasattr(cs, "reset_session") else None
    session = cs.get_session()
    session.data = np.ones((4, 4), dtype=np.float32)
    session.normalized = False
    assert cs.console_normalize({"method": "None", "plus_data": "abc"})["ok"] is False


def test_moving_the_sdd_center_drops_the_fit_it_was_measured_from():
    sd.reset_session()
    session = sd.get_session()
    session.data = np.ones((10, 10), dtype=np.float32)
    session.center = (1000.0, 1000.0)
    session.last_curve = {"kind": "yplot", "x": np.arange(3.0), "y": np.arange(3.0)}
    session.fit_result = {"mode": "single", "center": -599.0, "radius": None}
    assert sd.sddcal_set_center({"y": 1200, "z": 1000})["ok"]
    assert sd.sddcal_apply_fit_to_center({})["ok"] is False


def test_a_null_in_a_calibration_json_is_a_report_not_a_crash(tmp_path):
    for name, raw in (
        ("sdd_null.json", {"pc": [10, 20], "wavelength": 1.3, "pixelsize": 0.075, "sdd": None}),
        ("pc_null.json", {"pc": [None, 20], "wavelength": 1.3, "pixelsize": 0.075, "sdd": 200}),
    ):
        (tmp_path / name).write_text(json.dumps(raw))
        calibration, report = load_standard_calibration(tmp_path / name)
        assert calibration is None and not report.ok and report.message
