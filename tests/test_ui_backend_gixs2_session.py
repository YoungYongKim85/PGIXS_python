from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pgixs_python.calibration.sdd import wavelength_from_energy_kev
from pgixs_python.qmap.transform import image_transform_q_gi
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
    # The real EigerMask.mat masks ~6.2% of the detector, so a healthy load
    # sits near 0.94 -- not 1.0. The bug this guards against blanks the image
    # entirely (finite fraction ~0), so compare against the mask's own
    # expected coverage rather than a bare "almost everything survived".
    expected_finite = float(np.isfinite(detector_mask.mask).mean())
    finite_fraction = np.isfinite(session.data).mean()
    assert finite_fraction == pytest.approx(expected_finite, abs=0.01), (
        f"Averaged image is mostly/entirely NaN (finite fraction={finite_fraction}, expected ~{expected_finite})."
    )


@pytest.mark.parametrize("tab", ["gixs2", "trxs2"])
def test_load_series_without_load_still_masks_the_detector(tmp_path, tab):
    """Set/Table -> Import straight away (no Load first) used to import the
    frames unmasked: hot pixels kept, no gap buffer. Given the detector, the
    import now builds that detector's mask itself."""

    from pgixs_python.processing.masking import detector_mask_for_type
    from pgixs_python.ui_backend import trxs2_session as ts

    module = gs if tab == "gixs2" else ts
    detector_mask = detector_mask_for_type("Eiger4M")
    raw = np.full(tuple(reversed(detector_mask.mask.shape)), 100.0, dtype=np.float32)
    names = []
    for idx in range(2):
        name = f"straight_{idx + 1:04d}.h5"
        with h5py.File(tmp_path / name, "w") as h5:
            h5.create_dataset("/entry/data/data", data=raw)
            h5.create_dataset("/entry/instrument/user/params/Counter1", data=np.array([10.0], dtype=np.float32))
        names.append(name)

    module.reset_session()
    load_series = getattr(module, f"{tab}_load_series")
    result = load_series({"directory": str(tmp_path), "filenames": names, "insitu_mode": "Time", "detector": "Eiger4M"})
    assert result["ok"], result.get("message")
    session = module.get_session()
    assert np.isfinite(session.data).mean() == pytest.approx(float(np.isfinite(detector_mask.mask).mean()), abs=0.01)
    assert session.mask is not None and session.gap_buffer is not None and session.gap_buffer.any()


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


def _write_angle_scan_series(tmp_path, angles, prefix="pvk_om_05"):
    """Each file is its own single-frame incident-angle scan step, named
    like the real beamline data that surfaced this bug:
    "pvk_om_05_002_30_0001.h5", "pvk_om_05_004_30_0001.h5", ... -- the
    "_0001" frame counter is 4 digits (normalizePGIXSSeriesPrefix.m/
    _normalize_series_prefix strips only that), but the "_002_30" angle/
    scan segment in between varies file to file, so no single exact
    per-file "normalized prefix" is shared -- only "pvk_om_05" is common,
    as a substring."""

    shape = (64, 64)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r = np.hypot(xx - 31.5, yy - 31.5)
    rng = np.random.default_rng(0)
    files = []
    for idx, angle in enumerate(angles):
        data = (10.0 + 500.0 * np.exp(-((r - 20.0) ** 2) / (2.0 * 3.0**2)) + rng.uniform(0.0, 1.0, size=shape))
        h5_file = tmp_path / f"{prefix}_{angle:03d}_30_0001.h5"
        with h5py.File(h5_file, "w") as h5:
            h5.create_dataset("/entry/data/data", data=data.astype(np.float32))
            base = h5.create_group("/entry/instrument/user/params")
            base.create_dataset("Counter1", data=np.array([10.0 + idx], dtype=np.float32))
            base.create_dataset("Counter2", data=np.array([5.0 + idx], dtype=np.float32))
            base.create_dataset("TimeStamp", data=np.array([1000.0 + idx * 10.0], dtype=np.float64))
            base.create_dataset("Temp1", data=np.array([25.0], dtype=np.float32))
            base.create_dataset("Temp2", data=np.array([50.0], dtype=np.float32))
        files.append(h5_file)
    return files


def test_gixs2_list_series_files_matches_a_hand_typed_substring_prefix(tmp_path):
    """Regression: "Filename에서 내가 임의로 수정한 이름과 똑같은 것을
    Table로 다 불러오게 하고 싶은데, 그게 안되네" -- an earlier version of
    _list_series_files compared each candidate file's own separately-
    normalized prefix for *equality* against the typed prefix, which only
    ever works for a plain "PREFIX_####.h5" series. MATLAB's own
    listPGIXSSeriesFiles.m actually uses `contains(allNames, prefix)` (a
    substring check), which is what lets a shorter, hand-typed prefix like
    "pvk_om_05" catch every file in an angle-scan series whose own middle
    segment varies file to file."""

    _write_angle_scan_series(tmp_path, angles=[2, 4, 6, 8, 10])

    result = gs.gixs2_list_series_files({"directory": str(tmp_path), "filename": "pvk_om_05"})

    assert result["ok"], result.get("message")
    assert len(result["files"]) == 5


def test_gixs2_auto_refresh_finds_new_files_for_a_substring_prefix_series(tmp_path):
    """Auto Refresh re-derives its own match prefix independently of
    Set/Table's -- without remembering the originally hand-typed
    "pvk_om_05", it would fall back to normalizing file_paths[0]'s own
    name ("pvk_om_05_002_30_0001.h5" -> "pvk_om_05_002_30"), which is too
    specific to ever match a *new* angle step's own differently-numbered
    middle segment."""

    files = _write_angle_scan_series(tmp_path, angles=[2, 4, 6])
    list_result = gs.gixs2_list_series_files({"directory": str(tmp_path), "filename": "pvk_om_05"})
    assert list_result["ok"], list_result.get("message")
    load_result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Temperature"}
    )
    assert load_result["ok"], load_result.get("message")

    _write_angle_scan_series(tmp_path, angles=[8, 10])

    result = gs.gixs2_auto_refresh({"directory": str(tmp_path)})

    assert result["ok"], result.get("message")
    assert result["n_new"] == 2
    assert gs.get_session().stack.shape[0] == 5


def test_gixs2_qcal_requires_load_info_first(tmp_path):
    """Regression: "Load Info에 파일이 들어가 있지 않으면, q-cal이 되지
    않도록 해줘" -- `session.calibration` always holds plausible-looking
    placeholder values even before Load Info ever runs, so q-Cal must
    refuse to run against those rather than silently producing a wrong-
    but-not-obviously-wrong q-map."""

    _load_series(tmp_path, n_frames=2, center=(100.5, 100.5))
    session = gs.get_session()
    session.calibration.update({"beam_pos": (100.5, 100.5), "sdd_mm": 300.0, "energy_kev": 9.5})

    result = gs.gixs2_qcal({})

    assert not result["ok"]
    assert "Load Info" in result["message"]
    assert session.qmap is None


def test_gixs2_qcal_produces_qmap_and_a_correction_factor(tmp_path):
    _load_series(tmp_path, n_frames=2, center=(100.5, 100.5))
    session = gs.get_session()
    session.calibration.update({"beam_pos": (100.5, 100.5), "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True

    result = gs.gixs2_qcal({})

    assert result["ok"], result.get("message")
    assert session.qmap is not None
    # The plain path keeps only the factor -- the corrected frames are
    # produced on demand, so no second full stack is held.
    assert session.corrected_stack is None
    assert session.correction_factor is not None
    assert session.correction_factor.shape == session.stack.shape[1:]
    assert gs.corrected_frame(0).shape == session.stack.shape[1:]


def test_gixs2_qcal_change_ai_produces_per_frame_q_stack(tmp_path):
    _load_series(tmp_path, n_frames=3, center=(100.5, 100.5))
    session = gs.get_session()
    session.calibration.update({"beam_pos": (100.5, 100.5), "sdd_mm": 300.0, "energy_kev": 9.5, "incident_angle_deg": 0.1})
    session.calibration_loaded = True

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
    session.calibration_loaded = True
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
    session.calibration_loaded = True
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
    session.calibration_loaded = True
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
    session.calibration_loaded = True
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
    session.calibration_loaded = True
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
    session.calibration_loaded = True

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
    session.calibration_loaded = True

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


def test_gixs2_azimuthal_exec_reflect_missing_wedge_regrids_each_frame(tmp_path):
    """Regression: "Azimuthal에서 우리는 Qxy/Qz로 변형된 맵에서 azimuthal을
    하는것이 아닌 pixel map에서 바로 진행해. 그러다 보니 missing
    wedge파트가 관과돼" -- and a first attempt at a cheap fix (excluding a
    geometry-only "gap" pixel mask from the plain pixel-space cut) turned
    out not to work at all ("그림과 같이 딱히 반영된것 같지 않은데?"), since
    every real pixel really does hold real data -- only genuinely
    regridding each frame onto the Qxy/Qz grid (image_transform_q_gi, the
    same regrid the single-frame console's own Qxy/Qz cut uses) can
    reproduce the real coverage holes. `reflect_missing_wedge=True` must
    hand process_azimuthal_stack a stack of regridded frames, not the
    plain corrected_stack."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({})["ok"]
    # q-Cal keeps only the correction factor now, so build the corrected
    # frames the same way every reduction does.
    original_stack = np.stack([gs.corrected_frame(i) for i in range(session.stack.shape[0])], axis=0)

    captured = {}
    original_process = gs.process_azimuthal_stack

    def spying_process_azimuthal_stack(stack, *args, **kwargs):
        captured["stack"] = stack.copy()
        captured["correction_factor"] = kwargs.get("correction_factor")
        return original_process(stack, *args, **kwargs)

    gs.process_azimuthal_stack = spying_process_azimuthal_stack
    try:
        result = gs.gixs2_azimuthal_exec(
            {"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0, "reflect_missing_wedge": True}
        )
    finally:
        gs.process_azimuthal_stack = original_process

    assert result["ok"], result.get("message")
    regridded_stack = captured["stack"]
    # Genuinely different data (a real regrid), not the plain stack
    # relabeled -- and corrected_stack itself untouched (shared by every
    # other cut type, none of which regrid).
    assert not np.allclose(regridded_stack, original_stack, equal_nan=True)
    # The regrid consumes the correction itself, so it must not be applied
    # a second time inside the reduction...
    assert captured["correction_factor"] is None
    # ...and the session's own raw stack + factor are left untouched, so
    # every other (non-regridding) cut still sees the same frames.
    assert np.allclose(
        np.stack([gs.corrected_frame(i) for i in range(session.stack.shape[0])], axis=0),
        original_stack, equal_nan=True,
    )
    # Matches a hand-rolled regrid of the same frames via the same
    # underlying transform the single-frame console's own Qxy/Qz cut uses.
    expected_0 = image_transform_q_gi(
        original_stack[0], session.qmap["qr"], session.qmap["qz"], center, session.calibration["pixel_mm"],
        wavelength_from_energy_kev(session.calibration["energy_kev"]), session.calibration["sdd_mm"],
    ).image
    np.testing.assert_allclose(regridded_stack[0], expected_0, equal_nan=True)


def test_gixs2_azimuthal_exec_reflect_missing_wedge_requires_plain_qcal(tmp_path):
    """"Change ai" only keeps the scalar |Q| magnitude per frame
    (q_stack), not qr/qz separately, so there is nothing to regrid
    against -- reflect_missing_wedge must fail clearly there instead of
    silently falling back to the plain pixel-space cut."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({"change_ai": True, "ai_step": 0.05})["ok"]

    result = gs.gixs2_azimuthal_exec(
        {"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0, "reflect_missing_wedge": True}
    )

    assert not result["ok"]
    assert "Change ai" in result["message"]


def test_gixs2_azimuthal_exec_default_still_uses_the_plain_pixel_space_cut(tmp_path):
    """The default (reflect_missing_wedge left False/omitted) must be
    byte-identical to the pre-existing plain pixel-space behavior -- this
    is strictly an opt-in second mode, not a change to the default."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({})["ok"]
    raw_stack = session.stack.copy()

    captured = {}
    original_process = gs.process_azimuthal_stack

    def spying_process_azimuthal_stack(stack, *args, **kwargs):
        captured["stack"] = stack
        captured["correction_factor"] = kwargs.get("correction_factor")
        return original_process(stack, *args, **kwargs)

    gs.process_azimuthal_stack = spying_process_azimuthal_stack
    try:
        result = gs.gixs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})
    finally:
        gs.process_azimuthal_stack = original_process

    assert result["ok"], result.get("message")
    # Nothing regridded: the frames handed over are the untouched stack,
    # with the detector correction supplied alongside to be applied
    # per frame during the reduction.
    np.testing.assert_array_equal(captured["stack"], raw_stack)
    np.testing.assert_array_equal(captured["correction_factor"], session.correction_factor)


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


def _load_series_angle_mode(tmp_path, n_frames=3, center=(100.5, 100.5)):
    files = _write_synthetic_series(tmp_path, n_frames=n_frames, center=center, radii=[60.0] * n_frames)
    result = gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files], "insitu_mode": "Angle"}
    )
    assert result["ok"], result.get("message")
    return files, result


def test_gixs2_set_x_axis_angle_mode_prefers_change_ai_ramp_over_logged_metadata(tmp_path):
    """Regression: "angle 모드에서 1D plot에 incidence angle 밸류들이
    제대로 반영이 되지 않아 color가 이상하게 나왔어." Insitu "Angle" mode's
    X-axis used to always read the per-file *logged* "IncAngle" metadata
    column -- unrelated to the ai/ai_step ramp "Change ai" q-Cal actually
    computed and used to build each frame's own q-map. Once Change ai is
    active, the X-axis (and so the 1D/Azimuthal Plot Viewers' colorbar)
    must show that real per-frame incident angle instead, matching
    MATLAB's own "Angle" axis (always param.aivar)."""

    center = (100.5, 100.5)
    _load_series_angle_mode(tmp_path, n_frames=4, center=center)
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    qcal_result = gs.gixs2_qcal({"incident_angle_deg": 0.1, "change_ai": True, "ai_step": 0.05})
    assert qcal_result["ok"], qcal_result.get("message")

    result = gs.gixs2_set_x_axis({})

    assert result["ok"], result.get("message")
    assert result["label"] == "Incidence Angle, deg"
    np.testing.assert_allclose(result["x_axis"], [0.1, 0.15, 0.2, 0.25])


def test_gixs2_set_x_axis_angle_mode_time_submode_still_overrides_change_ai(tmp_path):
    """The Angle insitu mode's own sub-mode selector still offers "Time" as
    an explicit fallback -- Change ai being active shouldn't silently
    steal that choice away from the user."""

    center = (100.5, 100.5)
    _load_series_angle_mode(tmp_path, n_frames=3, center=center)
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({"incident_angle_deg": 0.1, "change_ai": True, "ai_step": 0.05})["ok"]

    result = gs.gixs2_set_x_axis({"sub_mode": "Time"})

    assert result["ok"], result.get("message")
    assert result["label"] != "Incidence Angle, deg"


def test_gixs2_set_x_axis_angle_mode_without_change_ai_falls_back_to_logged_metadata(tmp_path):
    """Without Change ai, there is no per-frame aivar ramp to prefer --
    Angle mode keeps reading the logged metadata column as before."""

    center = (100.5, 100.5)
    _load_series_angle_mode(tmp_path, n_frames=2, center=center)
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({"incident_angle_deg": 0.1, "change_ai": False})["ok"]

    result = gs.gixs2_set_x_axis({})

    assert result["ok"], result.get("message")
    assert result["label"] == "Incidence Angle, deg"
    # These synthetic files never write an "IncAngle" dataset -- the
    # logged-metadata fallback path reads NaN, distinct from the (absent)
    # aivar ramp's own real values, confirming the ramp wasn't used here.
    assert all(np.isnan(v) for v in result["x_axis"])


def test_gixs2_save_1d_writes_a_file(tmp_path):
    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
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
    session.calibration_loaded = True
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
    session.calibration_loaded = True
    assert gs.gixs2_qcal({})["ok"]
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]
    assert gs.gixs2_set_x_axis({})["ok"]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = gs.gixs2_save_integration({"directory": str(out_dir)})

    assert result["ok"], result.get("message")
    assert Path(result["path"]).exists()

    # The 1D branch must honour Normalize too, like the Integration plot.
    raw = np.genfromtxt(result["path"], delimiter=",", skip_header=1)
    session.counts = np.array([[2.0, 1.0], [4.0, 1.0]], dtype=np.float32)
    result_ic = gs.gixs2_save_integration({"directory": str(out_dir), "normalize": "IC"})
    assert result_ic["ok"], result_ic.get("message")
    ic = np.genfromtxt(result_ic["path"], delimiter=",", skip_header=1)
    # Counters relative to the first frame's (2), like MATLAB's RefIC.
    np.testing.assert_allclose(ic[:, -1], raw[:, -1] / np.array([1.0, 2.0]), rtol=1e-5)


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
    session.calibration_loaded = True
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
    session.calibration_loaded = True

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


def test_gixs2_grid_remove_is_refused_after_qcal(tmp_path):
    """Same guard `trxs2_grid_remove` already carries: q-Cal caches
    `corrected_stack` and this port never re-derives it, so a post-q-Cal Grid
    Remove would leave every Exec Cut/Azi/ROI cutting stale, not-grid-removed
    data while the 2D preview showed the correction applied."""

    gs.reset_session()
    _load_series(tmp_path, n_frames=2)
    session = gs.get_session()
    session.calibration.update({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_pos": (100.5, 100.5)})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({"incident_angle_deg": 0.2})["ok"]

    result = gs.gixs2_grid_remove({})

    assert not result["ok"]
    assert "before q-Cal" in result["message"]


def test_sorted_x_axis_reorders_the_saved_frames_with_it(tmp_path):
    """Regression: the Order checkbox sorted `x_axis_values` but nothing ever
    permuted the per-frame data to match, so every frame's intensity was
    plotted and saved against a *different* frame's temperature. MATLAB
    applies `sort_idx` to the data at every plot/save site (`Y = Y(:,
    sort_idx)` in GIXS2_gui.m); `order_frames` is this port's equivalent."""

    gs.reset_session()
    _write_and_load_out_of_order_temperatures(tmp_path)  # temps 30, 10, 20
    session = gs.get_session()

    # One distinct constant curve per frame, so a frame is identifiable by value.
    session.last_1d = {
        "x_axis": np.array([0.1, 0.2], dtype=np.float32),
        "intensity": np.array([[30.0, 30.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        "params": {},
    }
    session.auto_met = "linecut"
    assert gs.gixs2_set_x_axis({"order": True})["ok"]
    np.testing.assert_allclose(session.x_axis_values, [10.0, 20.0, 30.0])

    out_dir = tmp_path / "out"
    assert gs.gixs2_save_integration({"directory": str(out_dir)})["ok"]

    saved = np.loadtxt(next(out_dir.rglob("*_Integ.csv")), delimiter=",", skiprows=1)
    # Each row is (temperature, integrated intensity); frame 30C integrates to
    # 60, 10C to 20, 20C to 40 -- so a correctly-paired file is sorted by
    # temperature *and* still matches each frame's own value.
    np.testing.assert_allclose(saved[:, 0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(saved[:, 1], [20.0, 40.0, 60.0])


def _angle_wedge_gap_buffer(shape, center, lo_deg, hi_deg):
    """A gap-buffer-shaped boolean mask covering one angular wedge, using the
    same angle convention `angular_profile` bins with."""

    rows, cols = shape
    cc, rr = np.meshgrid(np.arange(1, cols + 1, dtype=float), np.arange(1, rows + 1, dtype=float))
    angle = np.degrees(np.arctan2(-(rr - center[1]), -(cc - center[0]))) % 360.0
    return (angle >= lo_deg) & (angle < hi_deg)


def test_gixs2_azimuthal_exec_blanks_the_gap_shoulder_like_every_other_reduction(tmp_path):
    """Regression: Exec Azi was the one reduction that never passed
    `gap_buffer`, so GIXS-series azimuthal curves silently averaged the
    module-gap shoulder that the 1D cut, Auto Refresh's own azimuthal
    replay, and both TrXS-series reductions all exclude. Reported as
    "Reflect missing wedge를 켜면 intensity 값이 달라진다" -- the wedge
    branch regrids through `_q_remap_stack`, which *did* blank the
    shoulder, so the two modes disagreed for that reason on top of the
    regridding itself."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=2, center=center, radii=[60.0, 60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center})
    session.calibration_loaded = True
    session.gap_buffer = _angle_wedge_gap_buffer(session.stack.shape[-2:], center, 40.0, 50.0)

    assert gs.gixs2_azimuthal_exec({"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0})["ok"]

    x_axis = session.last_azimuthal["x_axis"]
    intensity = session.last_azimuthal["intensity"][0]
    inside = (x_axis >= 41.0) & (x_axis <= 49.0)
    outside = (x_axis >= 10.0) & (x_axis <= 30.0)
    # Every bin fully inside the blanked wedge has no pixels left to average.
    assert np.all(~np.isfinite(intensity[inside]))
    # ...and nothing outside it was touched.
    assert np.all(np.isfinite(intensity[outside]))


def test_gixs2_azimuthal_exec_does_not_reapply_the_gap_buffer_after_regridding(tmp_path):
    """The wedge branch blanks the shoulder inside `_q_remap_stack`, before
    regridding. Applying the detector-pixel buffer *again* to the regridded
    frame would blank the wrong cells -- the Q grid is no longer indexed by
    the pixels the mask describes."""

    center = (100.5, 100.5)
    _load_series(tmp_path, n_frames=1, center=center, radii=[60.0])
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({})["ok"]
    session.gap_buffer = _angle_wedge_gap_buffer(session.stack.shape[-2:], center, 40.0, 50.0)

    captured = {}
    original_process = gs.process_azimuthal_stack

    def spy(stack, *args, **kwargs):
        captured["gap_buffer"] = kwargs.get("gap_buffer")
        return original_process(stack, *args, **kwargs)

    gs.process_azimuthal_stack = spy
    try:
        result = gs.gixs2_azimuthal_exec(
            {"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0, "reflect_missing_wedge": True}
        )
    finally:
        gs.process_azimuthal_stack = original_process

    assert result["ok"], result.get("message")
    assert captured["gap_buffer"] is None


def _write_calibration(tmp_path, beam_center):
    from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json

    cal = StandardCalibration(
        beam_center=np.asarray(beam_center, dtype=float), wavelength=1.3051, pixel_size=0.075, sdd=300.0,
        detector="Eiger4M",
    )
    save_standard_calibration_json(tmp_path / "calibration.json", cal)
    return {"directory": str(tmp_path), "filename": "calibration.json"}


def _qcal_series(tmp_path, center=(100.5, 100.5), **kwargs):
    _load_series(tmp_path, center=center, **kwargs)
    session = gs.get_session()
    session.calibration.update({"beam_pos": center, "sdd_mm": 300.0, "energy_kev": 9.5})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({})["ok"]
    return session


def test_gixs2_rerunning_qcal_drops_curves_cut_with_the_old_geometry(tmp_path):
    session = _qcal_series(tmp_path, n_frames=2)
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]
    assert gs.gixs2_roi_exec({"rois": [(1, 2, 1, 2)]})["ok"]

    assert gs.gixs2_qcal({"sdd_mm": 350.0})["ok"]

    assert session.last_1d is None
    assert session.last_roi is None


def test_gixs2_load_info_after_qcal_drops_the_qcal(tmp_path):
    session = _qcal_series(tmp_path, n_frames=2)
    assert gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]

    result = gs.gixs2_load_info(_write_calibration(tmp_path, (90.0, 95.0)))

    assert result["ok"], result.get("message")
    assert "q-Cal" in result["message"]
    assert session.qmap is None and session.correction_factor is None and not session.corrected
    assert session.last_1d is None
    assert not gs.gixs2_linecut_exec({"angle_deg": 90.0})["ok"]


def test_gixs2_auto_refresh_needs_an_imported_series(tmp_path):
    files = _write_synthetic_series(tmp_path, n_frames=3)
    assert gs.gixs2_load({"directory": str(tmp_path), "filename": files[0].name, "detector": "TestDetector"})["ok"]

    result = gs.gixs2_auto_refresh({"directory": str(tmp_path)})

    assert not result["ok"]
    assert gs.get_session().stack.shape[0] == 1


def test_gixs2_auto_refresh_wedge_azimuthal_matches_a_fresh_exec(tmp_path):
    center = (100.5, 100.5)
    files = _write_synthetic_series(tmp_path, n_frames=3, center=center, radii=[60.0, 62.0, 64.0])
    assert gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files[:2]], "insitu_mode": "Temperature"}
    )["ok"]
    session = gs.get_session()
    # A short SDD and steep incidence keep the Q grid far from the pixel
    # grid, so a detector gap mask stamped onto it lands on new pixels.
    session.calibration.update({"beam_pos": center, "sdd_mm": 20.0, "energy_kev": 9.5, "incident_angle_deg": 5.0})
    session.calibration_loaded = True
    assert gs.gixs2_qcal({})["ok"]
    cc, rr = np.meshgrid(np.arange(1, 201, dtype=float), np.arange(1, 201, dtype=float))
    session.gap_buffer = (rr > 60) & (rr < 75)
    payload = {"pixel": 60.0, "range_minus": -5.0, "range_plus": 5.0, "reflect_missing_wedge": True}
    assert gs.gixs2_azimuthal_exec(payload)["ok"]

    assert gs.gixs2_auto_refresh({"directory": str(tmp_path)})["curve_extended"]
    extended = session.last_azimuthal["intensity"].copy()
    assert gs.gixs2_azimuthal_exec(payload)["ok"]

    np.testing.assert_allclose(extended, session.last_azimuthal["intensity"], equal_nan=True)


def test_gixs2_auto_refresh_grid_removes_new_frames_after_grid_remove(tmp_path):
    from pgixs_python.ui_backend.series_processing import grid_remove_stack

    shape = (2167, 2070)  # grid_remove's fixed indices need the full Eiger4M size
    files = _write_synthetic_series(tmp_path, n_frames=2, shape=shape, center=(1000.5, 1000.5), radii=[60.0, 400.0])
    assert gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [files[0].name], "insitu_mode": "Temperature"}
    )["ok"]
    assert gs.gixs2_grid_remove({})["ok"]

    assert gs.gixs2_auto_refresh({"directory": str(tmp_path)})["ok"]

    session = gs.get_session()
    raw = np.ascontiguousarray(np.asarray(h5py.File(files[1], "r")["/entry/data/data"]))
    expected, _ = grid_remove_stack(raw[np.newaxis], "Eiger4M")
    np.testing.assert_allclose(session.stack[1], expected[0], equal_nan=True)


def test_gixs2_incidence_helpers_need_a_loaded_calibration():
    assert not gs.gixs2_ai_from_pixel({"pix": 100.0})["ok"]
    assert not gs.gixs2_ai_from_degree({"ai_deg": 0.2})["ok"]


def test_gixs2_failed_load_keeps_the_current_gap_buffer(tmp_path):
    files = _write_synthetic_series(tmp_path, n_frames=1)
    session = gs.get_session()
    sentinel = np.zeros((2, 2), dtype=bool)
    session.gap_buffer = sentinel

    # The Eiger4M mask cannot fit a 200x200 frame, so this load fails.
    assert not gs.gixs2_load({"directory": str(tmp_path), "filename": files[0].name, "detector": "Eiger4M"})["ok"]

    assert session.gap_buffer is sentinel


def test_gixs2_pending_new_files_counts_only_settled_new_frames(tmp_path):
    import os
    import time

    files = _write_synthetic_series(tmp_path, n_frames=3)
    assert gs.gixs2_load_series(
        {"directory": str(tmp_path), "filenames": [f.name for f in files[:2]], "insitu_mode": "Temperature"}
    )["ok"]

    assert gs.gixs2_pending_new_files({"directory": str(tmp_path)}) == {"ok": True, "n_new": 1}
    # Just written: a watcher must leave it for the next poll.
    assert gs.gixs2_pending_new_files({"directory": str(tmp_path), "settle_seconds": 30.0})["n_new"] == 0
    assert gs.gixs2_auto_refresh({"directory": str(tmp_path), "settle_seconds": 30.0})["n_new"] == 0

    old = time.time() - 60
    os.utime(files[2], (old, old))
    assert gs.gixs2_auto_refresh({"directory": str(tmp_path), "settle_seconds": 30.0})["n_new"] == 1
    assert gs.get_session().stack.shape[0] == 3


@pytest.mark.parametrize("tab", ["gixs2", "trxs2"])
def test_load_series_names_a_detector_that_does_not_fit_the_files(tmp_path, tab):
    """Detector box changed to another detector after Load: the import fails
    and says so in plain words, keeping the loaded series' own mask."""

    from pgixs_python.processing.masking import detector_mask_for_type
    from pgixs_python.ui_backend import trxs2_session as ts

    module = gs if tab == "gixs2" else ts
    eiger = detector_mask_for_type("Eiger4M").mask
    name = "frame_0001.h5"
    with h5py.File(tmp_path / name, "w") as h5:
        h5.create_dataset("/entry/data/data", data=np.full(tuple(reversed(eiger.shape)), 100.0, dtype=np.float32))
        h5.create_dataset("/entry/instrument/user/params/Counter1", data=np.array([10.0], dtype=np.float32))

    module.reset_session()
    assert getattr(module, f"{tab}_load")({"directory": str(tmp_path), "filename": name, "detector": "Eiger4M"})["ok"]
    result = getattr(module, f"{tab}_load_series")(
        {"directory": str(tmp_path), "filenames": [name], "insitu_mode": "Time", "detector": "MarCCD"}
    )
    assert not result["ok"]
    assert result["message"].startswith("Detector MarCCD does not match these files")
    assert "2167 x 2070" in result["message"]
    assert module.get_session().mask.shape == eiger.shape
