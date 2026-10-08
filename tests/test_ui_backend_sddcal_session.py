import h5py
import numpy as np
import pytest

from pgixs_python.calibration.calibration_io import load_standard_calibration, wavelength_to_energy_kev
from pgixs_python.ui_backend import sddcal_session as s


def _write_synthetic_ring_frame(path, shape=(200, 200), center=(100.5, 100.5), ring_radius=60.0, ring_sigma=3.0, seed=0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(seed)
    data = 10.0 + 500.0 * np.exp(-((r - ring_radius) ** 2) / (2.0 * ring_sigma**2))
    data = data + rng.uniform(0.0, 1.0, size=shape)
    data = data.astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])
    return data


def _load(frame_path, **extra):
    payload = {"directory": str(frame_path.parent), "filename": frame_path.name, "detector": "TestDetector"}
    payload.update(extra)
    return s.sddcal_load(payload)


def test_sddcal_load_requires_directory_and_filename():
    s.reset_session()

    payload = s.sddcal_load({})

    assert not payload["ok"]


def test_sddcal_load_returns_shape_and_default_center(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)

    payload = _load(frame_path)

    assert payload["ok"], payload.get("message")
    assert payload["shape"] == [200, 200]
    assert payload["center"] == [100.0, 100.0]  # cols/2, rows/2


def test_sddcal_set_center_requires_loaded_data():
    s.reset_session()

    payload = s.sddcal_set_center({"y": 1.0, "z": 2.0})

    assert not payload["ok"]


def test_sddcal_set_center_updates_session(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)
    _load(frame_path)

    payload = s.sddcal_set_center({"y": 100.5, "z": 100.5})

    assert payload["ok"]
    assert s.get_session().center == (100.5, 100.5)


def test_sddcal_line_cut_requires_loaded_data():
    s.reset_session()

    payload = s.sddcal_line_cut({"direction": "y"})

    assert not payload["ok"]


def test_sddcal_line_cut_y_direction_is_beam_centered(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})

    payload = s.sddcal_line_cut({"direction": "y"})

    assert payload["ok"], payload.get("message")
    curve = s.get_session().last_curve
    assert curve["kind"] == "yplot"
    # Beam-centered: x=0 sits at the beam column.
    zero_idx = int(np.argmin(np.abs(curve["x"])))
    assert abs(curve["x"][zero_idx]) < 1.0


def test_sddcal_radial_profile_requires_loaded_data():
    s.reset_session()

    payload = s.sddcal_radial_profile({})

    assert not payload["ok"]


def test_sddcal_radial_profile_returns_curve(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(100.5, 100.5), ring_radius=60.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})

    payload = s.sddcal_radial_profile({})

    assert payload["ok"], payload.get("message")
    assert s.get_session().last_curve["kind"] == "radial"


def test_sddcal_fit_peak_requires_a_curve():
    s.reset_session()

    payload = s.sddcal_fit_peak({"mode": "single", "c1": 0.0, "c2": 1.0})

    assert not payload["ok"]


def test_sddcal_fit_peak_single_mode_recovers_ring_radius(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(100.5, 100.5), ring_radius=60.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})
    s.sddcal_radial_profile({})

    payload = s.sddcal_fit_peak({"mode": "single", "c1": 40.0, "c2": 80.0})

    assert payload["ok"], payload.get("message")
    assert payload["center"] == pytest.approx(60.0, abs=1.0)
    assert s.get_session().fit_result["mode"] == "single"


def test_sddcal_fit_peak_symmetric_mode_rejects_radial_curve(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})
    s.sddcal_radial_profile({})

    payload = s.sddcal_fit_peak({"mode": "symmetric", "cl1": 0.0, "cl2": 1.0, "cr1": 0.0, "cr2": 1.0})

    assert not payload["ok"]
    assert "line cuts" in payload["message"]


def test_sddcal_fit_peak_symmetric_mode_on_line_cut(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(100.5, 100.5), ring_radius=60.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})
    s.sddcal_line_cut({"direction": "y"})

    # Beam-centered line cut through a ring crosses it at x=+-60.
    payload = s.sddcal_fit_peak({"mode": "symmetric", "cl1": -80.0, "cl2": -40.0, "cr1": 40.0, "cr2": 80.0})

    assert payload["ok"], payload.get("message")
    assert payload["center"] == pytest.approx(0.0, abs=1.0)
    assert payload["radius"] == pytest.approx(60.0, abs=1.0)


def test_sddcal_apply_fit_to_center_requires_a_fit(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)
    _load(frame_path)

    payload = s.sddcal_apply_fit_to_center({})

    assert not payload["ok"]


def test_sddcal_apply_fit_to_center_offsets_the_beam_y(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(105.5, 100.5), ring_radius=60.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})  # deliberately off by 5px in y
    s.sddcal_line_cut({"direction": "y"})
    s.sddcal_fit_peak({"mode": "symmetric", "cl1": -80.0, "cl2": -40.0, "cr1": 40.0, "cr2": 80.0})

    payload = s.sddcal_apply_fit_to_center({})

    assert payload["ok"]
    new_y, new_z = payload["center"]
    assert new_y == pytest.approx(105.5, abs=1.0)
    assert new_z == pytest.approx(100.5, abs=0.01)


def test_sddcal_calc_sdd_requires_loaded_data():
    s.reset_session()

    payload = s.sddcal_calc_sdd({"pixel_radius": 100.0, "energy_kev": 9.5})

    assert not payload["ok"]


def test_sddcal_calc_sdd_matches_reference_calibration(tmp_path):
    """End-to-end against real data: testdata/Cooling/standard's Silver
    behenate frame + its sibling *_standard.json reference values. The
    order-1 ring's fitted radius (~187.84px) and SDD (~644.6mm) come
    straight from that reference file's own Candidates[2] entry."""

    s.reset_session()
    payload = s.sddcal_load({"directory": "testdata/Cooling/standard", "filename": "8p95keV_sb_0001.h5", "detector": "Eiger4M"})
    if not payload["ok"]:
        pytest.skip(f"real test fixture not available: {payload.get('message')}")

    s.sddcal_set_center({"y": 397.76, "z": 1845.46})
    assert s.sddcal_radial_profile({})["ok"]
    fit = s.sddcal_fit_peak({"mode": "single", "c1": 170.0, "c2": 210.0})
    assert fit["ok"], fit.get("message")
    assert fit["center"] == pytest.approx(187.84, abs=1.0)

    energy_kev = wavelength_to_energy_kev(1.3051070195269474)
    calc = s.sddcal_calc_sdd(
        {"pixel_radius": fit["center"], "order": 1, "standard_sample": "Silver behenate", "energy_kev": energy_kev}
    )

    assert calc["ok"], calc.get("message")
    assert calc["sdd_mm"] == pytest.approx(644.61, rel=0.01)


def test_sddcal_auto_run_requires_loaded_data():
    s.reset_session()

    payload = s.sddcal_auto_run({})

    assert not payload["ok"]


def _synthetic_multi_ring_frame(shape=(400, 400), center=(200.5, 200.5), radii=(60.0, 84.8, 103.9, 120.0), seed=0):
    """A few concentric rings at the exact radii Silver behenate's first
    four orders would sit at for a given (arbitrary) SDD -- lets Smart Auto
    Run's full pipeline (center refine -> fine-tune -> candidate search ->
    order assignment) be exercised without needing real detector data."""

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(seed)
    data = 10.0 + 3000.0 * np.exp(-((r - 20.0) ** 2) / (2.0 * 5.0**2))  # near-center feature, must be ignored
    for radius in radii:
        data = data + 600.0 * np.exp(-((r - radius) ** 2) / (2.0 * 2.0**2))
    data = data + rng.uniform(0.0, 1.0, size=shape)
    return data.astype(np.float32)


def test_sddcal_auto_run_end_to_end_on_a_synthetic_multi_ring_frame(tmp_path):
    from pgixs_python.calibration.standards import standard_q_value

    true_sdd = 250.0
    pixel_size_mm = 0.075
    energy_kev = 9.5
    true_center = (200.5, 200.5)

    radii = []
    for order in range(1, 5):
        q = standard_q_value("Silver behenate", order)
        wavelength = 12.398419843320026 / energy_kev
        two_theta = 2.0 * np.arcsin(q * wavelength / (4.0 * np.pi))
        radii.append(true_sdd * np.tan(two_theta) / pixel_size_mm)

    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    data = _synthetic_multi_ring_frame(center=true_center, radii=tuple(radii))
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=data[None, :, :])
    _load(frame_path)
    # A realistic "roughly clicked" starting guess, a few px off.
    s.sddcal_set_center({"y": true_center[0] + 4.0, "z": true_center[1] - 3.0})

    result = s.sddcal_auto_run({"standard_sample": "Silver behenate", "energy_kev": energy_kev})

    assert result["ok"], result.get("message")
    assert result["center"][0] == pytest.approx(true_center[0], abs=2.0)
    assert result["center"][1] == pytest.approx(true_center[1], abs=2.0)
    assert result["sdd_mm"] == pytest.approx(true_sdd, rel=0.05)
    assert result["accepted_count"] >= 2
    assert len(result["candidates"]) >= 2
    assert s.get_session().sdd_result is not None
    assert s.get_session().candidate_report is not None


def test_sddcal_auto_run_on_real_silver_behenate_data_matches_reference_sdd():
    """End-to-end against real data, starting from a realistic close guess
    (not a blind image-center guess -- Smart Auto Run is a local
    refinement, matching MATLAB's own AutoRun_Callback). The reference SDD
    (214.84mm) comes from testdata/Cooling/standard's saved .std
    calibration for this exact frame."""

    s.reset_session()
    payload = s.sddcal_load({"directory": "testdata/Cooling/standard", "filename": "8p95keV_sb_0001.h5", "detector": "Eiger4M"})
    if not payload["ok"]:
        pytest.skip(f"real test fixture not available: {payload.get('message')}")

    s.sddcal_set_center({"y": 400.0, "z": 1842.0})
    energy_kev = wavelength_to_energy_kev(1.3051070195269474)

    result = s.sddcal_auto_run({"standard_sample": "Silver behenate", "energy_kev": energy_kev})

    assert result["ok"], result.get("message")
    assert result["center"][0] == pytest.approx(397.76, abs=3.0)
    assert result["center"][1] == pytest.approx(1845.46, abs=3.0)
    assert result["sdd_mm"] == pytest.approx(214.84, rel=0.01)


def test_sddcal_save_requires_calc_sdd(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)
    _load(frame_path)

    payload = s.sddcal_save({})

    assert not payload["ok"]


def test_sddcal_save_writes_json_and_mat_readable_by_load_info(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(100.5, 100.5), ring_radius=60.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})
    s.sddcal_radial_profile({})
    fit = s.sddcal_fit_peak({"mode": "single", "c1": 40.0, "c2": 80.0})
    assert fit["ok"]
    s.sddcal_calc_sdd({"pixel_radius": fit["center"], "order": 1, "standard_sample": "Silver behenate", "energy_kev": 9.5})

    out_dir = tmp_path / "out"
    payload = s.sddcal_save({"out_dir": str(out_dir)})

    assert payload["ok"], payload.get("message")
    assert payload["json_path"].endswith("_standard.json")
    assert (out_dir / "frame_standard.json").exists()

    # This is the actual cross-tab contract: GIXS/TrXS's own "Load Info"
    # loader must be able to read what SDDcal just saved.
    loaded, report = load_standard_calibration(payload["json_path"], expected_detector="TestDetector")
    assert report.ok, report.message
    assert loaded is not None
    np.testing.assert_allclose(loaded.beam_center, [100.5, 100.5])

    # The session center is already in MATLAB's 1-based pixel coordinates,
    # so .mat/.std carry it unchanged too -- same value in all three files.
    if payload["mat_path"]:
        loaded_mat, report_mat = load_standard_calibration(payload["mat_path"], expected_detector="TestDetector")
        assert report_mat.ok, report_mat.message
        np.testing.assert_allclose(loaded_mat.beam_center, [100.5, 100.5])
    if payload["std_path"]:
        loaded_std, report_std = load_standard_calibration(payload["std_path"], expected_detector="TestDetector")
        assert report_std.ok, report_std.message
        np.testing.assert_allclose(loaded_std.beam_center, [100.5, 100.5])


def test_sddcal_save_writes_the_session_center_verbatim_to_every_format(tmp_path):
    """Regression, superseding an earlier "mat의 경우 1씩 더해야돼" rule: that
    rule assumed the session center is a 0-based numpy index, but it is not.
    Every pixel grid here is `np.arange(1, n+1)` (`pixel_radial`,
    `center_refine`, `qmap`), so a pixel at array index (row, col) sits at
    radius 0 only for center == (col+1, row+1) -- the center is stored in the
    same 1-based coordinates MATLAB uses. MATLAB's `SDDcal_gui.m` writes its
    `.mat`/`.std` from the raw 1-based GUI values too, so adding 1 here shifted
    those files a pixel off on both axes. All three formats now get the
    session center verbatim."""

    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(50.0, 70.0), ring_radius=20.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 12.0, "z": 34.0})
    s.sddcal_calc_sdd({"pixel_radius": 20.0, "order": 1, "standard_sample": "Silver behenate", "energy_kev": 9.5})

    payload = s.sddcal_save({"out_dir": str(tmp_path)})
    assert payload["ok"], payload.get("message")

    loaded_json, _ = load_standard_calibration(payload["json_path"])
    assert loaded_json is not None
    np.testing.assert_allclose(loaded_json.beam_center, [12.0, 34.0])

    loaded_mat, _ = load_standard_calibration(payload["mat_path"])
    assert loaded_mat is not None
    np.testing.assert_allclose(loaded_mat.beam_center, [12.0, 34.0])

    loaded_std, _ = load_standard_calibration(payload["std_path"])
    assert loaded_std is not None
    np.testing.assert_allclose(loaded_std.beam_center, [12.0, 34.0])


def test_sddcal_auto_run_says_so_when_center_refinement_fails(tmp_path, monkeypatch):
    """Both refinement steps fall back to the center the user clicked. That
    center fixes every q axis downstream, so a silent fallback is a wrong
    calibration that looks like a right one."""

    from pgixs_python.calibration.standards import standard_q_value

    energy_kev, true_center = 9.5, (200.5, 200.5)
    wavelength = 12.398419843320026 / energy_kev
    radii = tuple(
        250.0 * np.tan(2.0 * np.arcsin(standard_q_value("Silver behenate", n) * wavelength / (4.0 * np.pi))) / 0.075
        for n in range(1, 5)
    )
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=_synthetic_multi_ring_frame(center=true_center, radii=radii)[None, :, :]
        )
    _load(frame_path)
    s.sddcal_set_center({"y": true_center[0] + 4.0, "z": true_center[1] - 3.0})

    def boom(*args, **kwargs):
        raise RuntimeError("circle fit did not converge")

    monkeypatch.setattr(s, "refine_center_from_ring", boom)
    result = s.sddcal_auto_run({"standard_sample": "Silver behenate", "energy_kev": energy_kev})

    assert result["ok"], result.get("message")
    assert any("Center refinement failed" in w and "did not converge" in w for w in result["warnings"])


def test_sddcal_send_to_main_applies_the_offset_once_and_drops_the_old_sdd(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path, center=(105.5, 100.5), ring_radius=60.0)
    _load(frame_path)
    s.sddcal_set_center({"y": 100.5, "z": 100.5})
    assert s.sddcal_calc_sdd({"pixel_radius": 60.0, "energy_kev": 9.5})["ok"]
    s.sddcal_line_cut({"direction": "y"})
    s.sddcal_fit_peak({"mode": "symmetric", "cl1": -80.0, "cl2": -40.0, "cr1": 40.0, "cr2": 80.0})

    first = s.sddcal_apply_fit_to_center({})
    second = s.sddcal_apply_fit_to_center({})

    assert first["ok"]
    assert not second["ok"]
    assert list(s.get_session().center) == first["center"]
    # The SDD was measured around the old center; Save must not pair them.
    assert not s.sddcal_save({})["ok"]


def test_sddcal_radial_profile_leaves_detector_gaps_out_of_coverage(tmp_path):
    s.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_ring_frame(frame_path)
    _load(frame_path)
    session = s.get_session()
    gap = np.ones(session.data.shape, dtype=np.float32)
    gap[:, 120:130] = np.nan  # a module gap, marked the way the real masks are
    session.mask = gap
    session.data = np.where(np.isfinite(gap), session.data, np.nan).astype(np.float32)

    coverage = np.asarray(s.sddcal_radial_profile({})["preview"]["coverage"], dtype=float)

    assert np.nanmin(coverage) == pytest.approx(1.0)
