from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pgixs_python.ui_backend import gixs2_session as gs


def setup_function(_fn):
    gs.reset_session()


def _write_synthetic_series(
    tmp_path, n_frames=3, shape=(200, 200), center=(100.5, 100.5), radii=None, prefix="ABCD", start_counter1=10.0
):
    """A small series of ring-frame .h5 files, matching the H5 layout
    `load_h5_series`/`read_h5_metadata` expect (reused from
    test_h5_series.py's own inline fixture pattern)."""

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
    result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert result["ok"], result.get("message")
    return files, result


def test_gixs2_load_and_load_series_survive_a_non_square_detector_mask(tmp_path):
    """Regression: Eiger4M's mask is (2070, 2167) -- genuinely non-square.
    `gixs2_load` stores it in `load_h5_frame`'s own single-transposed
    orientation; if `gixs2_load_series` (or `gixs2_load` itself) ever
    re-transposes it before storing/reusing it, every frame's masking
    multiply silently mismatches shapes, `load_h5_frame_with_retry`
    swallows the error into a same-shaped NaN fallback every time, and the
    averaged image comes out entirely blank with no visible error."""

    from pgixs_python.processing.masking import detector_mask_for_type

    detector_mask = detector_mask_for_type("Eiger4M")
    # `load_h5_frame` transposes the raw on-disk array once before masking,
    # so the raw shape on disk is the mask shape's transpose.
    raw_shape = tuple(reversed(detector_mask.mask.shape))

    rng = np.random.default_rng(0)
    files = []
    for idx in range(3):
        data = (100.0 + rng.uniform(0.0, 5.0, size=raw_shape)).astype(np.float32)
        h5_file = tmp_path / f"pristine_{idx + 1:04d}.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=data)
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([10.0 + idx], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([1000.0 + idx * 10.0], dtype=np.float64))
        files.append(h5_file)

    load_result = gs.gixs2_load({"directory": str(tmp_path), "filename": files[0].name, "detector": "Eiger4M"})
    assert load_result["ok"], load_result.get("message")

    session = gs.get_session()
    # The mask must stay in its native (post-single-transpose) orientation
    # -- the same shape `detector_mask.mask` already has -- not
    # transposed again to match `_SESSION.data`'s own display orientation.
    assert session.mask.shape == detector_mask.mask.shape

    series_result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Time"}
    )
    assert series_result["ok"], series_result.get("message")
    finite_fraction = np.isfinite(session.data).mean()
    assert finite_fraction > 0.99, f"Averaged image is mostly/entirely NaN (finite fraction={finite_fraction})."


def test_gixs2_load_series_parallel_option_matches_serial(tmp_path):
    files = _write_synthetic_series(tmp_path, n_frames=6)
    gs.reset_session()
    serial = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Time", "parallel": False}
    )
    assert serial["ok"], serial.get("message")
    serial_stack = gs.get_session().stack.copy()

    gs.reset_session()
    parallel = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Time", "parallel": True}
    )
    assert parallel["ok"], parallel.get("message")
    np.testing.assert_array_equal(gs.get_session().stack, serial_stack)


def test_gixs2_load_progress_reports_inactive_by_default():
    result = gs.gixs2_load_progress()
    assert result == {"ok": True, "active": False}


def test_gixs2_load_progress_reports_active_state_directly():
    gs.get_session().load_progress = {"current": 2, "total": 5, "label": "ABCD_0002.h5"}
    result = gs.gixs2_load_progress()
    assert result == {"ok": True, "active": True, "current": 2, "total": 5, "label": "ABCD_0002.h5"}


def test_gixs2_load_series_resets_progress_to_inactive_when_done(tmp_path):
    _load_series(tmp_path, n_frames=3)
    result = gs.gixs2_load_progress()
    assert result == {"ok": True, "active": False}


def test_gixs2_load_series_happy_path(tmp_path):
    files, result = _load_series(tmp_path, n_frames=3)

    assert result["n_files"] == 3
    session = gs.get_session()
    assert session.stack.shape[0] == 3
    assert session.counts.shape == (3, 2)
    assert session.timestamps.shape == (3,)
    # Baselined to the first frame's own timestamp.
    assert session.timestamps[0] == pytest.approx(0.0)
    assert session.timestamps[1] == pytest.approx(10.0)
    assert session.temperatures.shape == (3, 3)
    np.testing.assert_allclose(session.temperatures[:, 0], [25.0, 26.0, 27.0])


def test_gixs2_load_requires_directory_and_filename():
    result = gs.gixs2_load({})
    assert not result["ok"]


def test_gixs2_list_series_files_finds_matching_prefix(tmp_path):
    _write_synthetic_series(tmp_path, n_frames=3, prefix="XYZ")
    result = gs.gixs2_list_series_files({"directory": str(tmp_path), "filename": "XYZ_0001.h5"})

    assert result["ok"], result.get("message")
    assert len(result["files"]) == 3


def test_gixs2_qcal_produces_qmap_and_corrected_stack(tmp_path):
    _load_series(tmp_path, n_frames=2, center=(100.5, 100.5))
    session = gs.get_session()
    session.calibration.update({"beam_pos": (100.5, 100.5), "sdd_mm": 300.0, "energy_kev": 9.5})

    result = gs.gixs2_qcal({})

    assert result["ok"], result.get("message")
    assert session.qmap is not None
    assert session.corrected_stack is not None
    assert session.corrected_stack.shape == session.stack.shape


def test_gixs2_qcal_change_ai_produces_per_frame_q_stack(tmp_path):
    _load_series(tmp_path, n_frames=3, center=(100.5, 100.5))
    session = gs.get_session()
    session.calibration.update({"beam_pos": (100.5, 100.5), "sdd_mm": 300.0, "energy_kev": 9.5, "incident_angle_deg": 0.1})

    result = gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})

    assert result["ok"], result.get("message")
    assert result["change_ai"] is True
    assert session.change_ai is True
    assert session.qmap is None
    assert session.q_stack is not None
    assert session.q_stack.shape == session.stack.shape
    assert session.aivar is not None
    np.testing.assert_allclose(session.aivar, [0.1, 0.15, 0.2])
    assert session.corrected_stack is not None
    assert session.corrected_stack.shape == session.stack.shape

    # Each frame's own q-map should independently match a direct
    # `qmap_gixs` call using that frame's own `aivar[i]` -- confirms the
    # per-frame loop is wired correctly. Note |Q| itself is mathematically
    # ai-invariant (rotating two of its three components into each other
    # preserves the magnitude -- see `qmap_gixs`'s own qx/qz rotation), so
    # q_stack's frames are expected to be numerically identical to each
    # other up to float32 rounding; this asserts the recompute is wired to
    # the right per-frame angle, not that the resulting values differ.
    from pgixs_python.calibration.sdd import wavelength_from_energy_kev
    from pgixs_python.qmap.gixs import qmap_gixs

    k0 = 2.0 * np.pi / wavelength_from_energy_kev(9.5)
    rows, cols = session.stack.shape[1:]
    for i, ai in enumerate(session.aivar):
        _, _, _, expected_q, _, _ = qmap_gixs(
            det_y=cols, det_z=rows, beam_pos=(100.5, 100.5), pixel_mm=session.calibration["pixel_mm"],
            sdd_mm=300.0, k0=k0, incident_angle_deg=float(ai),
        )
        np.testing.assert_allclose(session.q_stack[i], expected_q, rtol=1e-5)


def test_gixs2_qcal_switching_back_to_static_clears_change_ai_state(tmp_path):
    _load_series(tmp_path, n_frames=2, center=(100.5, 100.5))
    session = gs.get_session()
    session.calibration.update({"beam_pos": (100.5, 100.5), "sdd_mm": 300.0, "energy_kev": 9.5})
    assert gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})["ok"]

    result = gs.gixs2_qcal({"change_ai": False})

    assert result["ok"], result.get("message")
    assert session.change_ai is False
    assert session.q_stack is None
    assert session.aivar is None
    assert session.qmap is not None


def test_gixs2_linecut_exec_change_ai_uses_per_frame_q_map(tmp_path):
    center = (100.5, 100.5)
    # Same physical radius every frame so a per-frame-varying q-map (from
    # the stepping incident angle) still recovers a consistent Q peak,
    # unlike a real drifting-ring series -- this just confirms the
    # Change-ai code path runs end to end and produces one row per frame.
    _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5, "incident_angle_deg": 0.1})
    assert gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})["ok"]

    result = gs.gixs2_linecut_exec({"angle_deg": 90.0})

    assert result["ok"], result.get("message")
    assert result["n_frames"] == 3
    assert session.last_1d["intensity"].shape[0] == 3


def test_gixs2_auto_refresh_change_ai_extends_aivar_and_q_stack(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5, "incident_angle_deg": 0.1})
    assert gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})["ok"]
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]
    original_intensity = session.last_1d["intensity"].copy()

    _write_synthetic_series(tmp_path, n_frames=5, center=center, radii=[60.0] * 5)

    result = gs.gixs2_auto_refresh({"directory": str(tmp_path)})

    assert result["ok"], result.get("message")
    assert result["n_new"] == 2
    assert session.stack.shape[0] == 5
    assert session.q_stack.shape[0] == 5
    assert session.aivar.shape == (5,)
    # The extension continues the same linear ramp, not a fresh one.
    np.testing.assert_allclose(session.aivar, 0.1 + 0.05 * np.arange(5))
    assert result["curve_extended"] is True
    assert session.last_1d["intensity"].shape[0] == 5
    np.testing.assert_allclose(session.last_1d["intensity"][:3], original_intensity)


def test_gixs2_linecut_exec_requires_qcal_first(tmp_path):
    _load_series(tmp_path, n_frames=2)
    result = gs.gixs2_linecut_exec({})
    assert not result["ok"]


def test_gixs2_linecut_exec_reports_and_clears_load_progress(tmp_path):
    """Exec Cut publishes per-frame progress through the same
    `load_progress` field/`gixs2_load_progress()` poll target series
    loading uses, so the widget's existing progress-dialog machinery works
    for it unchanged -- and clears it again once done, matching
    gixs2_load_series's own before/finally reset."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 70.0, 80.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    assert gs.gixs2_qcal({})["ok"]

    seen = []
    original_process = gs.process_linecut_stack

    def spying_process_linecut_stack(*args, **kwargs):
        real_callback = kwargs.get("progress_callback")

        def spy(completed, total):
            if real_callback is not None:
                real_callback(completed, total)
            seen.append(gs.gixs2_load_progress())

        kwargs["progress_callback"] = spy
        return original_process(*args, **kwargs)

    gs.process_linecut_stack = spying_process_linecut_stack
    try:
        result = gs.gixs2_linecut_exec({"angle_deg": 90.0})
    finally:
        gs.process_linecut_stack = original_process

    assert result["ok"], result.get("message")
    assert seen == [
        {"ok": True, "active": True, "current": 1, "total": 3, "label": "1D cut"},
        {"ok": True, "active": True, "current": 2, "total": 3, "label": "1D cut"},
        {"ok": True, "active": True, "current": 3, "total": 3, "label": "1D cut"},
    ]
    assert gs.gixs2_load_progress() == {"ok": True, "active": False}


def test_gixs2_linecut_exec_end_to_end(tmp_path):
    center = (100.5, 100.5)
    radii = [60.0, 70.0, 80.0]
    _load_series(tmp_path, n_frames=3, center=center, radii=radii)
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    assert gs.gixs2_qcal({})["ok"]

    result = gs.gixs2_linecut_exec({"angle_deg": 90.0})

    assert result["ok"], result.get("message")
    assert result["n_frames"] == 3
    assert session.last_1d is not None
    assert session.last_1d["intensity"].shape[0] == 3
    assert session.auto_met == "linecut"


def test_gixs2_azimuthal_exec_end_to_end(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})

    result = gs.gixs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})

    assert result["ok"], result.get("message")
    assert result["n_frames"] == 2
    assert session.auto_met == "azimuthal"


def test_gixs2_azimuthal_exec_reports_and_clears_load_progress(tmp_path):
    """Same treatment as Exec Cut (see
    test_gixs2_linecut_exec_reports_and_clears_load_progress) -- Exec Azi
    previously ran with no progress feedback at all."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 70.0, 80.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})

    seen = []
    original_process = gs.process_azimuthal_stack

    def spying_process_azimuthal_stack(*args, **kwargs):
        real_callback = kwargs.get("progress_callback")

        def spy(completed, total):
            if real_callback is not None:
                real_callback(completed, total)
            seen.append(gs.gixs2_load_progress())

        kwargs["progress_callback"] = spy
        return original_process(*args, **kwargs)

    gs.process_azimuthal_stack = spying_process_azimuthal_stack
    try:
        result = gs.gixs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})
    finally:
        gs.process_azimuthal_stack = original_process

    assert result["ok"], result.get("message")
    assert seen == [
        {"ok": True, "active": True, "current": 1, "total": 3, "label": "Azimuthal"},
        {"ok": True, "active": True, "current": 2, "total": 3, "label": "Azimuthal"},
        {"ok": True, "active": True, "current": 3, "total": 3, "label": "Azimuthal"},
    ]
    assert gs.gixs2_load_progress() == {"ok": True, "active": False}


def test_gixs2_roi_exec_matches_hand_computed_sum(tmp_path):
    _load_series(tmp_path, n_frames=2)
    session = gs.get_session()
    session.stack = np.ones((2, 10, 10), dtype=np.float32)
    session.corrected_stack = None

    result = gs.gixs2_roi_exec({"rois": [(1, 2, 1, 2)]})

    assert result["ok"], result.get("message")
    assert result["sums"] == [[4.0], [4.0]]
    assert session.auto_met == "roi"


def test_gixs2_save_integration_applies_roi_normalization(tmp_path):
    """Save Integ.'s ROI branch previously always wrote the raw sum,
    ignoring whatever the ROI Normalize combo/payload said -- confirm it
    now applies None/IC/BC via the shared `gixs2_normalized_matrix`."""

    _load_series(tmp_path, n_frames=2)
    session = gs.get_session()
    session.stack = np.ones((2, 10, 10), dtype=np.float32)
    session.corrected_stack = None
    assert gs.gixs2_roi_exec({"rois": [(1, 2, 1, 2)]})["ok"]
    assert gs.gixs2_set_x_axis({})["ok"]

    result_none = gs.gixs2_save_integration({"directory": str(tmp_path), "normalize": "None"})
    assert result_none["ok"], result_none.get("message")
    none_content = Path(result_none["path"]).read_text()

    result_ic = gs.gixs2_save_integration({"directory": str(tmp_path), "normalize": "IC"})
    assert result_ic["ok"], result_ic.get("message")
    ic_content = Path(result_ic["path"]).read_text()

    assert none_content != ic_content


def test_gixs2_set_x_axis_temperature_mode(tmp_path):
    _load_series(tmp_path, n_frames=3)
    result = gs.gixs2_set_x_axis({"sub_mode": "", "order": False})

    assert result["ok"], result.get("message")
    np.testing.assert_allclose(result["x_axis"], [25.0, 26.0, 27.0])


def test_gixs2_set_x_axis_order_sorts_ascending(tmp_path):
    _write_and_load_out_of_order_temperatures(tmp_path)
    result = gs.gixs2_set_x_axis({"order": True})
    assert result["ok"], result.get("message")
    assert result["x_axis"] == sorted(result["x_axis"])


def _write_and_load_out_of_order_temperatures(tmp_path):
    files = []
    yy, xx = np.mgrid[0:50, 0:50]
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
    result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert result["ok"], result.get("message")


def test_gixs2_save_1d_writes_a_file(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    assert gs.gixs2_qcal({})["ok"]
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = gs.gixs2_save_1d({"directory": str(out_dir)})

    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()


def test_gixs2_save_creates_a_series_named_subfolder_as_csv(tmp_path):
    """Every Save button creates a `{prefix}/` subfolder (named after the
    series, matching the Filename field) inside the chosen directory, and
    writes .csv (not .dat) -- covers Save 1D/Azi/Integ/X-Axis together
    since they all now go through the same `_save_path` helper."""

    center = (100.5, 100.5)
    files, _ = _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 65.0, 70.0], prefix="RUN01")
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    assert gs.gixs2_qcal({})["ok"]
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]
    assert gs.gixs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})["ok"]
    assert gs.gixs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    for save_fn in (gs.gixs2_save_1d, gs.gixs2_save_azimuthal, gs.gixs2_save_integration, gs.gixs2_save_x_axis):
        result = save_fn({"directory": str(out_dir)})
        assert result["ok"], result.get("message")
        saved_path = Path(result["path"])
        assert saved_path.exists()
        assert saved_path.suffix == ".csv"
        assert saved_path.parent == out_dir / "RUN01"
        assert saved_path.parent.is_dir()


def test_gixs2_save_integration_uses_active_auto_met(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    assert gs.gixs2_qcal({})["ok"]
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]
    assert gs.gixs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = gs.gixs2_save_integration({"directory": str(out_dir)})

    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()


def test_gixs2_save_x_axis_writes_a_file(tmp_path):
    _load_series(tmp_path, n_frames=2)
    assert gs.gixs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = gs.gixs2_save_x_axis({"directory": str(out_dir)})

    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()


def test_gixs2_auto_refresh_appends_only_new_frames(tmp_path):
    center = (100.5, 100.5)
    files, _ = _load_series(tmp_path, n_frames=3, center=center, radii=[60.0, 60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    assert gs.gixs2_qcal({})["ok"]
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]
    original_intensity = session.last_1d["intensity"].copy()

    # Add 2 more files to the same series prefix.
    _write_synthetic_series(tmp_path, n_frames=5, center=center, radii=[60.0] * 5)[3:5]

    result = gs.gixs2_auto_refresh({"directory": str(tmp_path)})

    assert result["ok"], result.get("message")
    assert result["n_new"] == 2
    assert session.stack.shape[0] == 5
    assert result["curve_extended"] is True
    assert session.last_1d["intensity"].shape[0] == 5
    np.testing.assert_allclose(session.last_1d["intensity"][:3], original_intensity)


def test_gixs2_auto_refresh_reports_no_new_files(tmp_path):
    _load_series(tmp_path, n_frames=2)
    result = gs.gixs2_auto_refresh({"directory": str(tmp_path)})
    assert result["ok"]
    assert result["n_new"] == 0


def test_gixs2_ai_from_pixel_and_back(tmp_path):
    _load_series(tmp_path, n_frames=1)
    session = gs.get_session()
    session.calibration.update({"sdd_mm": 300.0, "pixel_mm": 0.075})

    result = gs.gixs2_ai_from_pixel({"pix": 100.0})
    assert result["ok"], result.get("message")
    ai_deg = result["ai_deg"]

    back = gs.gixs2_ai_from_degree({"ai_deg": ai_deg})
    assert back["ok"], back.get("message")
    assert back["pix"] == pytest.approx(100.0, abs=1.0)


def test_gixs2_z_plot_requires_data_loaded():
    result = gs.gixs2_z_plot({"y_pos": 0.0})
    assert not result["ok"]


def test_gixs2_z_plot_returns_a_curve(tmp_path):
    _load_series(tmp_path, n_frames=1)
    result = gs.gixs2_z_plot({"y_pos": 0.0})
    assert result["ok"], result.get("message")
    assert len(result["x"]) > 0
    assert len(result["y"]) == len(result["x"])
