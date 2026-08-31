from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pgixs_python.ui_backend import console_session as cs
from pgixs_python.ui_backend import trxs2_session as ts


def setup_function(_fn):
    ts.reset_session()
    cs.reset_session()


def _write_synthetic_series(
    tmp_path, n_frames=3, shape=(200, 200), center=(100.5, 100.5), radii=None, prefix="ABCD", start_counter1=10.0
):
    """A small series of ring-frame .h5 files -- same layout as
    test_ui_backend_gixs2_session.py's own fixture."""

    if radii is None:
        radii = [60.0 + 2.0 * i for i in range(n_frames)]
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(0)

    files = []
    for idx, radius in enumerate(radii):
        data = 10.0 + 500.0 * np.exp(-((r - radius) ** 2) / (2.0 * 3.0**2))
        data = (data + rng.uniform(0.0, 1.0, size=shape)).astype(np.float32)
        h5_file = tmp_path / f"{prefix}_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=data)
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([start_counter1 + idx], dtype=np.float32))
            base.create_dataset("Counter2", data=np.array([5.0 + idx], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([1000.0 + idx * 10.0], dtype=np.float64))
            base.create_dataset("Temp1", data=np.array([25.0 + idx], dtype=np.float32))
            base.create_dataset("Temp2", data=np.array([50.0 + idx], dtype=np.float32))
        files.append(h5_file)
    return files


def _load_series(tmp_path, **kwargs):
    files = _write_synthetic_series(tmp_path, **kwargs)
    result = ts.trxs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert result["ok"], result.get("message")
    return files, result


def _qcal(center, sdd_mm=300.0, energy_kev=9.5):
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": sdd_mm, "energy_kev": energy_kev})
    session.calibration_loaded = True
    result = ts.trxs2_qcal({})
    assert result["ok"], result.get("message")
    return result


# ----------------------------------------------------------------------
# Load / Set-Table
# ----------------------------------------------------------------------
def test_trxs2_load_series_happy_path(tmp_path):
    files, result = _load_series(tmp_path, n_frames=3)

    assert result["n_files"] == 3
    session = ts.get_session()
    assert session.stack.shape[0] == 3
    assert session.counts.shape == (3, 2)
    assert session.timestamps.shape == (3,)
    assert session.timestamps[0] == pytest.approx(0.0)
    assert session.temperatures.shape == (3, 3)
    np.testing.assert_allclose(session.temperatures[:, 0], [25.0, 26.0, 27.0])
    assert session.auto_met == "circular"


def test_trxs2_load_series_rejects_unsupported_insitu_mode_falls_back_to_time(tmp_path):
    files = _write_synthetic_series(tmp_path, n_frames=2)
    result = ts.trxs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Angle"}
    )
    assert result["ok"], result.get("message")
    assert ts.get_session().insitu_mode == "Time"


def test_trxs2_load_requires_directory_and_filename():
    result = ts.trxs2_load({})
    assert not result["ok"]


def test_trxs2_list_series_files_finds_matching_prefix(tmp_path):
    _write_synthetic_series(tmp_path, n_frames=3, prefix="XYZ")
    result = ts.trxs2_list_series_files({"directory": str(tmp_path), "filename": "XYZ_0001.h5"})

    assert result["ok"], result.get("message")
    assert len(result["files"]) == 3


def test_trxs2_load_progress_reports_inactive_by_default():
    assert ts.trxs2_load_progress() == {"ok": True, "active": False}


# ----------------------------------------------------------------------
# Grid Remove
# ----------------------------------------------------------------------
def test_trxs2_grid_remove_corrects_each_frame_individually(tmp_path):
    """Regression check: Grid Remove must run once per frame, not once
    against the averaged display image or once and reused for every
    frame. `grid_remove` derives its correction ratio from each call's
    own reference pixels, so two frames with different content should
    come out with genuinely different corrections -- this cross-checks
    `trxs2_grid_remove`'s whole-stack output against calling `grid_remove`
    directly on each raw frame, one at a time."""

    from pgixs_python.processing.grid_remove import grid_remove

    shape = (2167, 2070)  # real Eiger4M size -- grid_remove's fixed pixel
    # indices require the full detector size to find their reference rows/cols.
    center = (1000.5, 1000.5)
    _load_series(tmp_path, n_frames=2, shape=shape, center=center, radii=[60.0, 400.0])

    session = ts.get_session()
    raw_before = session.stack.copy()

    result = ts.trxs2_grid_remove({})
    assert result["ok"], result.get("message")

    corrected_stack = ts.get_session().stack
    assert corrected_stack.shape == raw_before.shape

    for i in range(raw_before.shape[0]):
        # trxs2_grid_remove applies grid_remove in the untransposed
        # orientation (same convention as console_session.console_grid_remove).
        expected, report = grid_remove(raw_before[i].T, "Eiger4M")
        assert report.applied, report.message
        np.testing.assert_allclose(corrected_stack[i], expected.T)

    # The two frames have very different content (ring radii 60 vs 400),
    # so their own reference-pixel ratios differ -- confirms each frame
    # really got its own correction, not frame 0's reused for both.
    assert not np.allclose(corrected_stack[0], corrected_stack[1], equal_nan=True)


def test_trxs2_grid_remove_requires_a_loaded_series():
    result = ts.trxs2_grid_remove({})

    assert not result["ok"]
    assert "Load a series" in result["message"]


def test_trxs2_grid_remove_refuses_to_run_after_qcal(tmp_path):
    """Regression: running Grid Remove after q-Cal would silently leave
    `corrected_stack` stale (this port never re-derives it from a
    post-hoc-modified raw stack), so downstream Exec calls would keep
    cutting un-grid-removed data despite the 2D preview showing the
    correction applied. Must be refused, not silently accepted."""

    shape = (2167, 2070)
    center = (1000.5, 1000.5)
    _load_series(tmp_path, n_frames=1, shape=shape, center=center, radii=[60.0])
    _qcal(center)

    result = ts.trxs2_grid_remove({})

    assert not result["ok"]
    assert "before q-Cal" in result["message"]


# ----------------------------------------------------------------------
# q-Cal
# ----------------------------------------------------------------------
def test_trxs2_qcal_produces_qmap_and_corrected_stack(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center)
    result = _qcal(center)

    session = ts.get_session()
    assert result["ok"]
    assert session.qmap is not None
    assert "q" in session.qmap
    assert session.corrected_stack is not None
    assert session.corrected_stack.shape == session.stack.shape
    assert session.corrected is True


def test_trxs2_qcal_refuses_a_second_run_without_a_new_series(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center)
    _qcal(center)

    result = ts.trxs2_qcal({})
    assert not result["ok"]
    assert "already corrected" in result["message"].lower()


def test_trxs2_qcal_requires_a_loaded_series():
    result = ts.trxs2_qcal({})
    assert not result["ok"]


def test_trxs2_qcal_requires_load_info_first(tmp_path):
    """Regression: "Load Info에 파일이 들어가 있지 않으면, q-cal이 되지
    않도록 해줘" -- `session.calibration` always holds plausible-looking
    placeholder values even before Load Info ever runs, so q-Cal must
    refuse to run against those rather than silently producing a wrong-
    but-not-obviously-wrong q-map."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center)
    session = ts.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})

    result = ts.trxs2_qcal({})

    assert not result["ok"]
    assert "Load Info" in result["message"]
    assert session.qmap is None


# ----------------------------------------------------------------------
# Background load
# ----------------------------------------------------------------------
def test_trxs2_background_load_averages_a_matched_series(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center)

    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    bkg_files = _write_synthetic_series(bkg_dir, n_frames=3, center=center, prefix="BKG", radii=[90.0, 90.0, 90.0])

    result = ts.trxs2_background_load({"directory": str(bkg_dir), "filename": bkg_files[0].name})

    assert result["ok"], result.get("message")
    assert result["n_files"] == 3
    session = ts.get_session()
    assert session.background is not None
    assert session.background.shape == session.stack.shape[1:]
    assert session.background_counts.shape == (2,)
    # Averaged Counter1 across 3 files starting at 10.0 -> mean(10,11,12) = 11.
    assert session.background_counts[0] == pytest.approx(11.0, abs=0.1)


def test_trxs2_background_load_rejects_shape_mismatch(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, shape=(200, 200))

    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    # A background series with a different detector shape entirely -- every
    # frame gets skipped as a shape mismatch, so nothing loads at all.
    bkg_files = _write_synthetic_series(bkg_dir, n_frames=1, shape=(50, 50), center=(25.5, 25.5), prefix="BKGBAD")

    result = ts.trxs2_background_load({"directory": str(bkg_dir), "filename": bkg_files[0].name})

    assert not result["ok"]
    assert ts.get_session().background is None


def test_trxs2_background_load_requires_a_loaded_series(tmp_path):
    result = ts.trxs2_background_load({"directory": str(tmp_path), "filename": "bkg.h5"})
    assert not result["ok"]


def test_trxs2_background_correction_applied_qcal_first_then_background(tmp_path):
    """q-Cal runs first (background not loaded yet); loading the
    background afterwards must correct it immediately -- the fix over the
    literal MATLAB port, which silently leaves a background loaded *after*
    q-Cal uncorrected."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center)
    _qcal(center)

    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    bkg_files = _write_synthetic_series(bkg_dir, n_frames=1, center=center, prefix="BKG", radii=[90.0])
    result = ts.trxs2_background_load({"directory": str(bkg_dir), "filename": bkg_files[0].name})
    assert result["ok"], result.get("message")

    session = ts.get_session()
    # Compare against the raw (uncorrected) average to confirm the stored
    # background actually differs -- i.e. a correction factor was applied.
    raw = _raw_background_average(bkg_files, session.mask)
    assert not np.allclose(
        np.nan_to_num(session.background), np.nan_to_num(raw), equal_nan=True
    )


def test_trxs2_background_correction_applied_background_first_then_qcal(tmp_path):
    """Background loads first (q-Cal hasn't run yet); q-Cal must correct
    it too, mirroring MATLAB's own `if isfield(param,'bkg'), param.bkg =
    param.bkg .* InvFactor`."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center)

    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    bkg_files = _write_synthetic_series(bkg_dir, n_frames=1, center=center, prefix="BKG", radii=[90.0])
    load_result = ts.trxs2_background_load({"directory": str(bkg_dir), "filename": bkg_files[0].name})
    assert load_result["ok"], load_result.get("message")

    session = ts.get_session()
    raw = _raw_background_average(bkg_files, session.mask)
    # Not yet corrected (q-Cal hasn't run).
    np.testing.assert_allclose(np.nan_to_num(session.background), np.nan_to_num(raw), atol=1e-4)

    _qcal(center)
    session = ts.get_session()
    assert not np.allclose(np.nan_to_num(session.background), np.nan_to_num(raw), equal_nan=True)


def _raw_background_average(bkg_files, mask):
    from pgixs_python.io.image_loader import load_image_file
    from pgixs_python.processing.masking import apply_detector_mask

    frames = []
    for path in bkg_files:
        result = load_image_file(path)
        masked = apply_detector_mask(result.data, mask)
        frames.append(np.ascontiguousarray(masked.T))
    averaged = np.mean(np.stack(frames, axis=0), axis=0)
    averaged[averaged == 0] = np.nan
    return averaged


# ----------------------------------------------------------------------
# Circular exec
# ----------------------------------------------------------------------
def test_trxs2_circular_exec_matches_hand_computed_radius_binned_average(tmp_path):
    from pgixs_python.averaging.radial import radius_average
    from pgixs_python.ui_backend.series_processing import build_circular_radius_map

    center = (100.5, 100.5)
    radii = [60.0, 70.0, 80.0]
    _load_series(tmp_path, n_frames=3, center=center, radii=radii)
    _qcal(center)
    session = ts.get_session()

    result = ts.trxs2_circular_exec({})
    assert result["ok"], result.get("message")

    radius_map = build_circular_radius_map(session.qmap["q"], center)
    for i in range(3):
        expected = radius_average(session.corrected_stack[i], radius_map)
        np.testing.assert_allclose(session.last_1d["intensity"][i], expected.intensity, equal_nan=True)
    assert session.circular_radius_map is not None
    assert session.auto_met == "circular"
    assert session.last_1d["rendered"] is not None


def test_trxs2_circular_exec_requires_qcal_first(tmp_path):
    _load_series(tmp_path, n_frames=2)
    result = ts.trxs2_circular_exec({})
    assert not result["ok"]


def test_trxs2_circular_exec_applies_roi_mask_exclusion(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=1, center=center, radii=[60.0])
    _qcal(center)
    session = ts.get_session()

    unmasked = ts.trxs2_circular_exec({})
    assert unmasked["ok"]
    unmasked_points = int(unmasked["n_points"])

    mask = np.zeros(session.data.shape, dtype=bool)
    mask[:100, :] = True  # keep only the top half
    set_mask_result = ts.trxs2_set_circular_mask({"mask": mask.tolist()})
    assert set_mask_result["ok"], set_mask_result.get("message")

    masked = ts.trxs2_circular_exec({})
    assert masked["ok"], masked.get("message")
    assert int(masked["n_points"]) <= unmasked_points


def test_trxs2_set_circular_mask_clears(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=1, center=center)
    session = ts.get_session()
    mask = np.ones(session.data.shape, dtype=bool)
    assert ts.trxs2_set_circular_mask({"mask": mask.tolist()})["ok"]
    assert session.circular_mask is not None

    clear_result = ts.trxs2_set_circular_mask({"clear": True})
    assert clear_result["ok"]
    assert session.circular_mask is None


def test_trxs2_circular_exec_nulls_bins_below_min_coverage(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=1, center=center, radii=[60.0])
    _qcal(center)

    permissive = ts.trxs2_circular_exec({"min_coverage": 0.0})
    assert permissive["ok"]
    session = ts.get_session()
    assert np.count_nonzero(np.isfinite(session.last_1d["intensity"])) > 0

    ts.reset_session()
    _load_series(tmp_path, n_frames=1, center=center, radii=[60.0])
    _qcal(center)
    strict = ts.trxs2_circular_exec({"min_coverage": 1.01})
    assert strict["ok"]
    session = ts.get_session()
    assert np.count_nonzero(np.isfinite(session.last_1d["intensity"])) == 0


# ----------------------------------------------------------------------
# Line exec (cross-checked against the single-frame console)
# ----------------------------------------------------------------------
def test_trxs2_line_exec_matches_single_frame_console_plain_angle_cut(tmp_path):
    center = (100.5, 100.5)
    radii = [60.0, 70.0]
    _load_series(tmp_path, n_frames=2, center=center, radii=radii)
    _qcal(center)
    session = ts.get_session()

    result = ts.trxs2_line_exec({"angle_deg": 90.0})
    assert result["ok"], result.get("message")

    cs.reset_session()
    console = cs.get_session()
    console.mode = "trxs"
    console.data = session.corrected_stack[0]
    console.qmap = {"q": session.qmap["q"]}
    console.calibration["beam_pos"] = center
    console.calibration_loaded = True
    console_result = cs.console_line_1d({"angle_deg": 90.0})
    assert console_result["ok"], console_result.get("message")

    np.testing.assert_allclose(
        session.last_1d["intensity"][0], console_result["preview"]["intensity"], rtol=1e-4, atol=1e-4
    )
    assert session.auto_met == "line1d"


def test_trxs2_line_exec_requires_qcal_first(tmp_path):
    _load_series(tmp_path, n_frames=2)
    result = ts.trxs2_line_exec({})
    assert not result["ok"]


def test_trxs2_line_exec_computes_background_cut_when_loaded(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    _qcal(center)

    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    bkg_files = _write_synthetic_series(bkg_dir, n_frames=1, center=center, prefix="BKG", radii=[60.0])
    assert ts.trxs2_background_load({"directory": str(bkg_dir), "filename": bkg_files[0].name})["ok"]

    result = ts.trxs2_line_exec({"angle_deg": 90.0})
    assert result["ok"], result.get("message")
    assert ts.get_session().bkg_1d is not None


# ----------------------------------------------------------------------
# Azimuthal exec (cross-checked against the single-frame console)
# ----------------------------------------------------------------------
def test_trxs2_azimuthal_exec_matches_single_frame_console_0_360_result(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = ts.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True

    result = ts.trxs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})
    assert result["ok"], result.get("message")

    cs.reset_session()
    console = cs.get_session()
    console.mode = "trxs"
    console.data = session.stack[0]
    console.qmap = {"q": np.ones_like(session.stack[0])}  # console_azimuthal_1d only needs qmap not-None
    console.calibration.update({"beam_pos": center})
    console.calibration_loaded = True
    console_result = cs.console_azimuthal_1d({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})
    assert console_result["ok"], console_result.get("message")

    np.testing.assert_allclose(
        session.last_azimuthal["intensity"][0], console_result["preview"]["intensity"], rtol=1e-4, atol=1e-4
    )
    assert session.auto_met == "azimuthal"


def test_trxs2_azimuthal_exec_requires_loaded_series():
    result = ts.trxs2_azimuthal_exec({})
    assert not result["ok"]


# ----------------------------------------------------------------------
# ROI exec
# ----------------------------------------------------------------------
def test_trxs2_roi_exec_matches_hand_computed_sum(tmp_path):
    _load_series(tmp_path, n_frames=2)
    session = ts.get_session()
    session.stack = np.ones((2, 10, 10), dtype=np.float32)
    session.corrected_stack = None

    result = ts.trxs2_roi_exec({"rois": [(1, 2, 1, 2)]})

    assert result["ok"], result.get("message")
    assert result["sums"] == [[4.0], [4.0]]
    assert session.auto_met == "roi"


def test_trxs2_roi_exec_requires_loaded_series():
    result = ts.trxs2_roi_exec({"rois": [(1, 2, 1, 2)]})
    assert not result["ok"]


# ----------------------------------------------------------------------
# Subtraction / render / point-normalization
# ----------------------------------------------------------------------
def _setup_circular_with_background(tmp_path, bkg_radius=60.0):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 70.0])
    _qcal(center)

    bkg_dir = tmp_path / "bkg"
    bkg_dir.mkdir()
    bkg_files = _write_synthetic_series(bkg_dir, n_frames=1, center=center, prefix="BKG", radii=[bkg_radius])
    assert ts.trxs2_background_load({"directory": str(bkg_dir), "filename": bkg_files[0].name})["ok"]

    assert ts.trxs2_circular_exec({})["ok"]
    return ts.get_session()


def test_trxs2_render_1d_without_subtraction_equals_raw_normalized(tmp_path):
    session = _setup_circular_with_background(tmp_path)
    session.subtract_background = False
    render = ts.trxs2_render_1d({"normalize": "None"})
    assert render["ok"]
    np.testing.assert_allclose(session.last_1d["rendered"], session.last_1d["intensity"], equal_nan=True)


def test_trxs2_render_1d_with_subtraction_differs_from_unsubtracted(tmp_path):
    session = _setup_circular_with_background(tmp_path)

    ts.trxs2_render_1d({"normalize": "None", "subtract": False})
    unsubtracted = session.last_1d["rendered"].copy()

    ts.trxs2_render_1d({"subtract": True})
    subtracted = session.last_1d["rendered"]

    assert session.subtract_background is True
    assert not np.allclose(np.nan_to_num(unsubtracted), np.nan_to_num(subtracted), equal_nan=True)


def test_trxs2_render_1d_normalize_ic_differs_from_none(tmp_path):
    session = _setup_circular_with_background(tmp_path)
    ts.trxs2_render_1d({"normalize": "None", "subtract": False})
    none_rendered = session.last_1d["rendered"].copy()

    ts.trxs2_render_1d({"normalize": "IC"})
    ic_rendered = session.last_1d["rendered"]

    assert not np.allclose(np.nan_to_num(none_rendered), np.nan_to_num(ic_rendered), equal_nan=True)


def test_trxs2_render_1d_requires_exec_first():
    result = ts.trxs2_render_1d({})
    assert not result["ok"]


def test_trxs2_render_azimuthal_requires_exec_first():
    result = ts.trxs2_render_azimuthal({})
    assert not result["ok"]


# ----------------------------------------------------------------------
# Auto Refresh
# ----------------------------------------------------------------------
def test_trxs2_auto_refresh_reuses_circular_radius_map_and_reapplies_render(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    _qcal(center)
    assert ts.trxs2_circular_exec({})["ok"]
    session = ts.get_session()
    original_radius_map = session.circular_radius_map
    original_rendered = session.last_1d["rendered"].copy()

    _write_synthetic_series(tmp_path, n_frames=5, center=center, radii=[60.0] * 5)

    result = ts.trxs2_auto_refresh({"directory": str(tmp_path)})

    assert result["ok"], result.get("message")
    assert result["n_new"] == 2
    assert result["curve_extended"] is True
    assert session.stack.shape[0] == 5
    # Reused, not rebuilt -- same object identity.
    assert session.circular_radius_map is original_radius_map
    assert session.last_1d["intensity"].shape[0] == 5
    assert session.last_1d["rendered"].shape[0] == 5
    np.testing.assert_allclose(session.last_1d["rendered"][:3], original_rendered, equal_nan=True)


def test_trxs2_auto_refresh_reports_no_new_files(tmp_path):
    _load_series(tmp_path, n_frames=2)
    result = ts.trxs2_auto_refresh({"directory": str(tmp_path)})
    assert result["ok"]
    assert result["n_new"] == 0


def test_trxs2_auto_refresh_requires_initial_series():
    result = ts.trxs2_auto_refresh({})
    assert not result["ok"]


def test_trxs2_auto_refresh_extends_roi_without_touching_render(tmp_path):
    _load_series(tmp_path, n_frames=2, prefix="ABCD")
    session = ts.get_session()
    assert ts.trxs2_roi_exec({"rois": [(1, 2, 1, 2)]})["ok"]

    # Re-writes frames 1-2 (deterministic, same seed) plus 2 brand-new ones.
    _write_synthetic_series(tmp_path, n_frames=4, prefix="ABCD")

    result = ts.trxs2_auto_refresh({"directory": str(tmp_path)})
    assert result["ok"], result.get("message")
    assert result["n_new"] == 2
    assert session.last_roi.shape[0] == 4


# ----------------------------------------------------------------------
# X-axis
# ----------------------------------------------------------------------
def test_trxs2_set_x_axis_temperature_mode(tmp_path):
    _load_series(tmp_path, n_frames=3)
    result = ts.trxs2_set_x_axis({"sub_mode": "", "order": False})
    assert result["ok"], result.get("message")
    np.testing.assert_allclose(result["x_axis"], [25.0, 26.0, 27.0])


def test_trxs2_set_x_axis_order_sorts_ascending(tmp_path):
    files = []
    for idx, temp in enumerate([30.0, 10.0, 20.0]):
        data = np.ones((50, 50), dtype=np.float32)
        h5_file = tmp_path / f"OOO_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=data)
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([1.0], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([float(idx)], dtype=np.float64))
            base.create_dataset("Temp1", data=np.array([temp], dtype=np.float32))
        files.append(h5_file)
    result = ts.trxs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert result["ok"], result.get("message")

    result = ts.trxs2_set_x_axis({"order": True})
    assert result["ok"], result.get("message")
    assert result["x_axis"] == sorted(result["x_axis"])


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
def test_trxs2_save_1d_writes_a_file(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    _qcal(center)
    assert ts.trxs2_circular_exec({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = ts.trxs2_save_1d({"directory": str(out_dir)})

    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()


def test_trxs2_save_1d_std_requires_std_enabled(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    _qcal(center)
    assert ts.trxs2_circular_exec({"calc_std": False})["ok"]

    result = ts.trxs2_save_1d_std({"directory": str(tmp_path)})
    assert not result["ok"]

    ts.reset_session()
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    _qcal(center)
    assert ts.trxs2_circular_exec({"calc_std": True})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = ts.trxs2_save_1d_std({"directory": str(out_dir)})
    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()
    assert "_1D_STD" in Path(result["path"]).name


def test_trxs2_save_creates_a_series_named_subfolder_as_csv(tmp_path):
    center = (100.5, 100.5)
    files, _ = _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 65.0, 70.0], prefix="RUN01")
    _qcal(center)
    assert ts.trxs2_circular_exec({})["ok"]
    assert ts.trxs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})["ok"]
    assert ts.trxs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    for save_fn in (ts.trxs2_save_1d, ts.trxs2_save_azimuthal, ts.trxs2_save_integration, ts.trxs2_save_x_axis):
        result = save_fn({"directory": str(out_dir)})
        assert result["ok"], result.get("message")
        saved_path = Path(result["path"])
        assert saved_path.exists()
        assert saved_path.suffix == ".csv"
        assert saved_path.parent == out_dir / "RUN01"


def test_trxs2_save_integration_uses_active_auto_met_roi(tmp_path):
    _load_series(tmp_path, n_frames=2)
    session = ts.get_session()
    session.stack = np.ones((2, 10, 10), dtype=np.float32)
    session.corrected_stack = None
    assert ts.trxs2_roi_exec({"rois": [(1, 2, 1, 2)]})["ok"]
    assert ts.trxs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = ts.trxs2_save_integration({"directory": str(out_dir)})
    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()


def test_trxs2_save_x_axis_writes_a_file(tmp_path):
    _load_series(tmp_path, n_frames=2)
    assert ts.trxs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = ts.trxs2_save_x_axis({"directory": str(out_dir)})
    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()
