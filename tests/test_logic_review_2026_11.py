"""Regressions for the 2026-11 logic review."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from pgixs_python.averaging.cake import accumulation_average
from pgixs_python.calibration.calibration_io import _to_string
from pgixs_python.calibration.center_refine import fine_tune_center
from pgixs_python.calibration.sdd import wavelength_from_energy_kev
from pgixs_python.qmap.gixs import qmap_gixs
from pgixs_python.ui_backend import console_session as cs
from pgixs_python.ui_backend import sddcal_session as sd
from pgixs_python.ui_backend import trxs2_session as ts
from pgixs_python.workflows.common import load_h5_input_stack

QMAP = {"energy_kev": 9.5, "sdd_mm": 100.0, "beam_y": 60.0, "beam_z": 60.0}


def _frame(path, data):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("entry/data/data", data=np.asarray(data, dtype=np.float32)[None, :, :])


def _console(tmp_path, shape=(120, 120), seed=0):
    cs.reset_session()
    path = tmp_path / "frame.h5"
    _frame(path, np.random.default_rng(seed).uniform(10, 100, size=shape))
    assert cs.console_load({"directory": str(tmp_path), "filename": path.name, "detector": "TestDetector"})["ok"]
    cs.get_session().calibration_loaded = True


def _circular():
    result = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "min_coverage": 0.0, "calc_std": True})
    assert result["ok"], result.get("message")
    return np.array(cs.get_session().last_curve["intensity"], dtype=float)


def test_normalize_after_qmap_keeps_the_detector_correction(tmp_path):
    """q-Map Cal. then BKG sub. & Norm. must cut the same curve as the two
    in the other order: the correction used to be dropped while the q-map
    stayed valid."""

    _console(tmp_path)
    assert cs.console_qmap(QMAP)["ok"]
    assert cs.console_normalize({"method": "None", "plus_data": 5.0, "confirm": True})["ok"]
    after = _circular()

    _console(tmp_path)
    assert cs.console_normalize({"method": "None", "plus_data": 5.0, "confirm": True})["ok"]
    assert cs.console_qmap(QMAP)["ok"]
    np.testing.assert_allclose(after, _circular(), rtol=1e-5, equal_nan=True)


def test_a_new_qmap_drops_the_curve_cut_against_the_old_one(tmp_path):
    _console(tmp_path)
    assert cs.console_qmap(QMAP)["ok"]
    _circular()
    assert cs.console_qmap({**QMAP, "sdd_mm": 3000.0})["ok"]
    assert cs.get_session().last_curve is None
    assert not cs.console_save_curve({"out_dir": str(tmp_path)})["ok"]


def test_saved_circular_curve_carries_the_std_column(tmp_path):
    _console(tmp_path)
    assert cs.console_qmap(QMAP)["ok"]
    _circular()
    saved = cs.console_save_curve({"out_dir": str(tmp_path)})
    assert saved["ok"], saved.get("message")
    header = open(saved["curve_path"], encoding="utf-8").readline().lower()
    assert "standard_deviation" in header


def test_zero_energy_is_refused_not_raised(tmp_path):
    assert np.isnan(wavelength_from_energy_kev(0))
    _console(tmp_path)
    result = cs.console_qmap({**QMAP, "energy_kev": 0})
    assert not result["ok"] and "energy" in result["message"].lower()


def test_trxs2_background_keeps_measured_zeros(tmp_path):
    ts.reset_session()
    rng = np.random.default_rng(1)
    for index in range(2):
        _frame(tmp_path / f"S_{index + 1:04d}.h5", rng.poisson(0.3, size=(40, 40)))
    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    background = rng.poisson(0.3, size=(40, 40)).astype(np.float32)
    _frame(bkg_dir / "BKG_0001.h5", background)
    assert ts.trxs2_load_series(
        {"directory": str(tmp_path), "filenames": ["S_0001.h5", "S_0002.h5"], "insitu_mode": "Temperature"}
    )["ok"]
    assert ts.trxs2_background_load({"directory": str(bkg_dir), "filename": "BKG_0001.h5"})["ok"]
    loaded = ts.get_session().background_raw
    assert np.count_nonzero(background == 0) > 0
    assert np.all(np.isfinite(loaded))
    assert abs(float(np.nanmean(loaded)) - float(background.mean())) < 1e-6


def test_trxs2_qcal_drops_results_cut_before_it(tmp_path):
    ts.reset_session()
    for index in range(2):
        _frame(tmp_path / f"S_{index + 1:04d}.h5", np.full((60, 60), 5.0))
    assert ts.trxs2_load_series(
        {"directory": str(tmp_path), "filenames": ["S_0001.h5", "S_0002.h5"], "insitu_mode": "Temperature"}
    )["ok"]
    session = ts.get_session()
    session.last_roi = np.ones((2, 2), dtype=np.float32)  # as an ROI Exec before q-Cal leaves
    assert not ts.trxs2_azimuthal_exec({"pixel": 20.0})["ok"]  # needs q-Cal
    session.calibration.update({"beam_pos": (30.0, 30.0), "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert ts.trxs2_qcal({})["ok"]
    assert session.last_roi is None


def test_sddcal_refuses_a_non_positive_radius_or_energy(tmp_path):
    sd.reset_session()
    path = tmp_path / "ring.h5"
    _frame(path, np.ones((50, 50)))
    assert sd.sddcal_load({"directory": str(tmp_path), "filename": path.name, "detector": "TestDetector"})["ok"]
    for payload in ({"pixel_radius": 0, "energy_kev": 9.5}, {"pixel_radius": -5, "energy_kev": 9.5}, {"pixel_radius": 60, "energy_kev": 0}):
        assert not sd.sddcal_calc_sdd(payload)["ok"]
    assert sd.get_session().sdd_result is None


def test_sddcal_center_move_drops_the_radial_profile(tmp_path):
    sd.reset_session()
    path = tmp_path / "ring.h5"
    _frame(path, np.ones((50, 50)))
    assert sd.sddcal_load({"directory": str(tmp_path), "filename": path.name, "detector": "TestDetector"})["ok"]
    session = sd.get_session()
    session.last_curve = {"kind": "radial", "x": np.arange(5.0), "y": np.ones(5)}
    session.fit_result = {"mode": "single", "center": 3.0, "radius": None}
    y, z = session.center
    sd._move_center(y + 5.0, z)
    assert session.last_curve is None and session.fit_result is None


def test_accumulation_band_off_the_detector_is_invalid():
    image = np.arange(100 * 120, dtype=np.float32).reshape(100, 120)
    q = np.ones_like(image)
    assert accumulation_average(image, q, (60.0, 50.0), 0.0, 0.0, -1.0, 1.0).is_valid
    assert not accumulation_average(image, q, (60.0, 50.0), 0.0, 500.0, -1.0, 1.0).is_valid
    assert not accumulation_average(image, q, (60.0, 50.0), 90.0, -500.0, -1.0, 1.0).is_valid


def test_empty_matlab_char_reads_as_an_empty_string():
    assert _to_string(np.array([], dtype="<U1")) == ""
    assert _to_string(np.array(["Eiger4M"])) == "Eiger4M"


def test_gixs_q_is_continuous_across_the_beam_column():
    k0 = 2 * np.pi / 1.0
    q = qmap_gixs(200, 2000, (100.0, 1900.0), 0.075, 200.0, k0, 0.15)[3]
    np.testing.assert_allclose(q[0, 99], q[0, 98], rtol=1e-4)
    np.testing.assert_allclose(q[0, 99], q[0, 100], rtol=1e-4)


def test_fine_tune_center_reaches_an_off_pixel_centre():
    yy, xx = np.mgrid[1:601, 1:601]
    true = (300.4, 280.3)
    image = 1000 * np.exp(-((np.hypot(xx - true[0], yy - true[1]) - 150) ** 2) / (2 * 3**2)) + 1
    found = fine_tune_center(image, (301.7, 279.5), 150)
    assert abs(found[0] - true[0]) < 0.05 and abs(found[1] - true[1]) < 0.05


def test_workflow_stack_has_the_gui_orientation(tmp_path):
    """The GUI shows an H5 frame as it is stored (rows x cols); the workflow
    stack has to match, since beam centres are (col, row) of that view."""

    raw = np.zeros((40, 30), dtype=np.float32)
    raw[10, 20] = 7.0
    _frame(tmp_path / "f.h5", raw)
    args = argparse.Namespace(
        input=str(tmp_path / "f.h5"), dataset="entry/data/data", frame_index=0, retry_count=0,
        metadata_mode="Time", reference_timestamp=None,
    )
    stack = load_h5_input_stack(args).stack
    assert stack.shape == (1, 40, 30)
    assert stack[0, 10, 20] == 7.0
