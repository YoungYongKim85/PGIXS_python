from pathlib import Path

import h5py
import numpy as np

from pgixs_python.io.image_loader import load_image_file
from pgixs_python.io.two_d_info import load_two_d_info_h5
from pgixs_python.ui_backend import console_session as cs


def _write_synthetic_frame(path, shape=(48, 48), seed=0):
    data = np.random.default_rng(seed).uniform(10, 100, size=shape).astype(np.float32)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("entry/data/data", data=data[None, :, :])
    return data


def _load(frame_path, **extra):
    """console_load helper: splits a full file path into MATLAB's
    Directory + LOAD DATA (filename) pair."""

    payload = {"directory": str(frame_path.parent), "filename": frame_path.name, "detector": "TestDetector"}
    payload.update(extra)
    return cs.console_load(payload)


def test_console_load_requires_directory_and_filename():
    cs.reset_session()

    payload = cs.console_load({})

    assert not payload["ok"]
    assert "directory" in payload["message"].lower()


def test_console_load_reports_missing_file(tmp_path):
    cs.reset_session()

    payload = cs.console_load({"directory": str(tmp_path), "filename": "does_not_exist.h5", "detector": "TestDetector"})

    assert not payload["ok"]
    assert payload["message"]


def test_console_load_returns_preview_and_calibration(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)

    payload = _load(frame_path)

    assert payload["ok"]
    assert payload["shape"] == [48, 48]
    assert payload["calibration"]["energy_kev"] > 0
    assert len(payload["preview"]) == 48
    assert len(payload["preview"][0]) == 48


def test_console_qmap_requires_loaded_data():
    cs.reset_session()

    payload = cs.console_qmap({})

    assert not payload["ok"]
    assert "Load a detector image" in payload["message"]


def test_console_qmap_computes_q_range(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)

    payload = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    assert payload["ok"]
    assert payload["q_range"]["min"] is not None
    assert payload["q_range"]["max"] > payload["q_range"]["min"]
    assert cs.get_session().qmap is not None
    assert cs.get_session().corrected_data is not None


def test_console_qspace_image_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)

    payload = cs.console_qspace_image({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_qspace_image_returns_regridded_preview(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_qspace_image({"mapping": "Qxy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["horizontal_range"]["min"] is not None
    assert payload["vertical_range"]["max"] > payload["vertical_range"]["min"]
    assert len(payload["preview"]) > 0
    # The pixel-space corrected preview and the Q-space regridded preview
    # must actually differ -- this is the concrete regression for "Qxy/Qz
    # selection doesn't change the displayed image."
    pixel_preview = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})["preview"]
    assert payload["preview"] != pixel_preview


def test_console_qspace_image_supports_qy_qz_mapping(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_qspace_image({"mapping": "Qy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["mapping"] == "Qy/Qz"


def test_console_qspace_image_works_for_trxs(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})

    payload = cs.console_qspace_image({})

    assert payload["ok"], payload.get("message")


def test_console_polar_image_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)

    payload = cs.console_polar_image({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_polar_image_returns_polar_preview(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_polar_image({"mapping": "Qxy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["mapping"] == "Qxy/Qz"
    # GIXS restricts the sweep to 0-180deg (see the dedicated test below);
    # the angle axis still exists and is well-formed either way.
    assert payload["horizontal_range"]["min"] is not None
    assert payload["horizontal_range"]["max"] < 360.0
    assert payload["vertical_range"]["max"] > payload["vertical_range"]["min"]
    assert len(payload["preview"]) > 0
    # Must actually differ from the Cartesian Qxy/Qz preview -- otherwise
    # the "Polar" toggle would silently do nothing.
    cartesian_preview = cs.console_qspace_image({"mapping": "Qxy/Qz"})["preview"]
    assert payload["preview"] != cartesian_preview


def test_console_polar_image_supports_qy_qz_mapping(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_polar_image({"mapping": "Qy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["mapping"] == "Qy/Qz"


def test_console_polar_image_gixs_sweeps_only_0_to_180(tmp_path):
    """Regression: GIXS's grazing-incidence geometry only ever scatters
    above the horizon -- the Polar view should sweep 0-180deg, matching
    console_azimuthal_1d's same restriction."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="gixs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_polar_image({"mapping": "Qxy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["horizontal_range"]["min"] >= 0.0
    assert payload["horizontal_range"]["max"] <= 180.0


def test_console_polar_image_trxs_sweeps_full_360(tmp_path):
    """Regression: TrXS's transmission geometry uses the full ring."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})

    payload = cs.console_polar_image({"mapping": "Qy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["horizontal_range"]["min"] >= 0.0
    assert payload["horizontal_range"]["max"] <= 360.0
    assert payload["horizontal_range"]["max"] > 180.0


def test_console_line_1d_requires_qmap():
    cs.reset_session()

    payload = cs.console_line_1d({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_line_1d_returns_curve(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_line_1d({"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0})

    assert payload["ok"]
    assert len(payload["preview"]["q"]) > 0
    assert len(payload["preview"]["q"]) == len(payload["preview"]["intensity"])
    assert cs.get_session().last_curve["kind"] == "line"


def test_console_line_1d_accumulation_returns_curve(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_line_1d({"angle_deg": 90.0, "accumulation": True, "accum_minus": -2.0, "accum_plus": 2.0})

    assert payload["ok"], payload.get("message")
    assert len(payload["preview"]["q"]) > 0


def test_console_line_1d_accumulation_and_cakecut_are_mutually_exclusive(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_line_1d({"accumulation": True, "cakecut": True})

    assert not payload["ok"]
    assert "Accumulation" in payload["message"]


def test_console_line_1d_accumulation_requires_axis_aligned_angle(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_line_1d({"angle_deg": 45.0, "accumulation": True})

    assert not payload["ok"]
    assert "0, 90, or 180" in payload["message"]


def test_console_line_1d_pos_move_shifts_axis_aligned_cut_but_not_diagonal(tmp_path):
    """z/y-pos move only takes effect for axis-aligned angles (0/90/180),
    matching GIXS_gui.m's Anglecut1d_gi -- a diagonal cut always emanates
    from the true beam center regardless of these fields."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    baseline = cs.console_line_1d({"angle_deg": 90.0})
    assert baseline["ok"], baseline.get("message")
    baseline_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    shifted = cs.console_line_1d({"angle_deg": 90.0, "y_pos_move": 5.0})
    assert shifted["ok"], shifted.get("message")
    shifted_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert shifted_intensity.shape != baseline_intensity.shape or not np.allclose(
        shifted_intensity, baseline_intensity, equal_nan=True
    )

    diag_baseline = cs.console_line_1d({"angle_deg": 45.0})
    assert diag_baseline["ok"], diag_baseline.get("message")
    diag_baseline_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    diag_shifted = cs.console_line_1d({"angle_deg": 45.0, "y_pos_move": 5.0, "z_pos_move": 5.0})
    assert diag_shifted["ok"], diag_shifted.get("message")
    diag_shifted_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert diag_shifted_intensity.shape == diag_baseline_intensity.shape
    assert np.allclose(diag_shifted_intensity, diag_baseline_intensity, equal_nan=True)


def test_console_azimuthal_1d_returns_curve(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_azimuthal_1d({"pixel": 10.0, "range_minus": -3.0, "range_plus": 3.0, "n_angle_bins": 16})

    assert payload["ok"]
    assert len(payload["preview"]["angle"]) == 16
    assert cs.get_session().last_curve["kind"] == "azimuthal"


def test_console_azimuthal_1d_gixs_sweeps_only_0_to_180(tmp_path):
    """Regression: GIXS's grazing-incidence geometry only ever scatters
    above the horizon (matches MATLAB GIXS_gui.m's `theta = 0:0.2:180`) --
    the Azimuthal profile should sweep 0-180deg, not the full circle."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="gixs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_azimuthal_1d({"pixel": 10.0, "range_minus": -3.0, "range_plus": 3.0, "n_angle_bins": 16})

    assert payload["ok"]
    angle = np.asarray(payload["preview"]["angle"], dtype=np.float64)
    assert angle.min() >= 0.0
    assert angle.max() <= 180.0


def test_console_azimuthal_1d_trxs_sweeps_full_360(tmp_path):
    """Regression: TrXS's transmission geometry uses the full ring
    (matches MATLAB TrXS_gui.m's `theta = 0:0.2:360`)."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_azimuthal_1d({"pixel": 10.0, "range_minus": -3.0, "range_plus": 3.0, "n_angle_bins": 16})

    assert payload["ok"]
    angle = np.asarray(payload["preview"]["angle"], dtype=np.float64)
    assert angle.min() >= 0.0
    assert angle.max() <= 360.0
    assert angle.max() > 180.0


def test_console_grid_remove_requires_loaded_data():
    cs.reset_session()

    payload = cs.console_grid_remove({})

    assert not payload["ok"]
    assert "Load a detector image" in payload["message"]


def test_console_normalize_requires_loaded_data():
    cs.reset_session()

    payload = cs.console_normalize({})

    assert not payload["ok"]
    assert "Load a detector image" in payload["message"]


def test_console_normalize_none_method_previews_then_commits(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)

    preview = cs.console_normalize({"method": "None", "confirm": False})
    assert preview["ok"]
    assert preview["requires_confirm"]
    assert not cs.get_session().normalized

    committed = cs.console_normalize({"method": "None", "confirm": True})
    assert committed["ok"]
    assert cs.get_session().normalized

    again = cs.console_normalize({"method": "None", "confirm": True})
    assert not again["ok"]
    assert "already" in again["message"]


def test_console_save_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)

    payload = cs.console_save({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_save_writes_files_and_round_trips(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    out_dir = tmp_path / "out"
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})
    cs.console_line_1d({"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0})

    payload = cs.console_save({"out_dir": str(out_dir)})

    assert payload["ok"]
    assert payload["curve_path"] is not None
    info = load_two_d_info_h5(payload["h5_path"])
    assert info.measurement_mode == "GIXS"
    assert info.data2d_pix.shape == (48, 48)


def test_console_save_creates_stem_named_subfolder(tmp_path):
    """console_save and console_save_curve both save into out_dir/<stem>/
    rather than dropping files directly into out_dir -- keeps repeated
    saves (different frames, or the same frame re-saved after changing
    settings) from colliding or piling up loose files together."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    out_dir = tmp_path / "out"
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})
    cs.console_line_1d({"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0})

    payload = cs.console_save({"out_dir": str(out_dir)})

    assert payload["ok"]
    expected_dir = out_dir / "frame"
    assert Path(payload["h5_path"]).parent == expected_dir
    assert Path(payload["curve_path"]).parent == expected_dir
    assert Path(payload["h5_path"]).name == "frame_2Dinfo.h5"


def test_console_save_curve_creates_stem_named_subfolder(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    out_dir = tmp_path / "out"
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})
    cs.console_line_1d({"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0})

    payload = cs.console_save_curve({"out_dir": str(out_dir)})

    assert payload["ok"]
    # Line kind's filename also stamps the reference angle it was cut at
    # (angle_deg=45.0 above) -- see test_curve_filename_labels_kind_and_angle
    # for dedicated coverage of line/azimuthal/circular filenames.
    assert Path(payload["curve_path"]) == out_dir / "frame" / "frame_1D_line_ang45deg.csv"


def test_curve_filename_labels_kind_and_angle(tmp_path):
    """Regression for "x축이 q일때, 어느 angle에서 cut한건지 file이름에 표기, azimuthal은
    azimuthal이라고 표기": a saved 1D profile's filename must say which kind
    of cut it is, and for a line cut, which angle -- otherwise a folder of
    saved profiles is impossible to tell apart without opening each one."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(64, 64))
    _load(frame_path)
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0, "incident_angle_deg": 0.2})

    # Line cut at a negative angle -- filename must stay filesystem-safe.
    cs.console_line_1d({"angle_deg": -30.0})
    payload = cs.console_save_curve({"out_dir": str(tmp_path)})
    assert payload["ok"], payload.get("message")
    assert Path(payload["curve_path"]).name == "frame_1D_line_ang-30deg.csv"

    cs.console_azimuthal_1d({"pixel": 10.0})
    payload = cs.console_save_curve({"out_dir": str(tmp_path)})
    assert payload["ok"], payload.get("message")
    assert Path(payload["curve_path"]).name == "frame_1D_azimuthal.csv"

    _load(frame_path, mode="trxs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0})
    cs.console_circular_average({"angle_min": 0, "angle_max": 360, "min_coverage": 0.0})
    payload = cs.console_save_curve({"out_dir": str(tmp_path)})
    assert payload["ok"], payload.get("message")
    assert Path(payload["curve_path"]).name == "frame_1D_circular.csv"


def test_console_load_defaults_to_gixs_mode(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)

    payload = _load(frame_path)

    assert payload["mode"] == "gixs"
    assert cs.get_session().mode == "gixs"


def test_console_load_accepts_trxs_mode(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)

    payload = _load(frame_path, mode="trxs")

    assert payload["mode"] == "trxs"
    assert cs.get_session().mode == "trxs"


def test_console_qmap_uses_trxs_physics_without_incident_angle(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")

    payload = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})

    assert payload["ok"]
    session = cs.get_session()
    assert "qx" in session.qmap
    assert "qr" not in session.qmap


def test_console_circular_average_requires_qmap():
    cs.reset_session()

    payload = cs.console_circular_average({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_circular_average_returns_curve_and_nulls_low_coverage(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(64, 64))
    _load(frame_path, mode="trxs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0})

    payload = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "min_coverage": 0.99})

    assert payload["ok"]
    assert len(payload["preview"]["q"]) == len(payload["preview"]["intensity"])
    assert cs.get_session().last_curve["kind"] == "circular"


def test_console_save_trxs_mode_has_no_incident_angle(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    out_dir = tmp_path / "out"
    _load(frame_path, mode="trxs")
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})

    payload = cs.console_save({"out_dir": str(out_dir)})

    assert payload["ok"]
    info = load_two_d_info_h5(payload["h5_path"])
    assert info.measurement_mode == "TrXS"
    assert info.incident_angle is None
    assert info.qx_map is not None


def test_console_load_transposes_relative_to_raw_load(tmp_path):
    """A real Eiger4M frame from this beamline displays rotated 90deg unless
    `console_load` applies one more transpose on top of `load_h5_frame`'s
    own transpose. Confirmed against real detector data; locked in here
    with a non-square synthetic frame so a shape mixup shows up as a clear
    axis-length assertion failure."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(30, 50))

    raw = load_image_file(frame_path)
    payload = _load(frame_path)

    assert payload["ok"]
    assert tuple(payload["shape"]) == raw.data.shape[::-1]
    assert cs.get_session().data.shape == raw.data.shape[::-1]


def test_console_grid_remove_works_after_display_transpose(tmp_path):
    """`grid_remove`'s Eiger4M pixel indices are calibrated against the
    untransposed (batch-workflow) orientation. console_grid_remove must
    un-transpose before calling it and re-transpose the result, or this
    either crashes on a real Eiger4M-sized frame or corrects the wrong
    pixels."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    # `load_h5_frame` transposes the raw dataset, so write it pre-transposed
    # (2167, 2070) so the loaded array matches the real Eiger4M fallback
    # mask shape (2070, 2167), exactly like a real detector file.
    _write_synthetic_frame(frame_path, shape=(2167, 2070))
    _load(frame_path, detector="Eiger4M")
    loaded_shape = cs.get_session().data.shape

    payload = cs.console_grid_remove({})

    assert payload["ok"]
    assert cs.get_session().data.shape == loaded_shape


def test_normalize_series_prefix_strips_trailing_index():
    assert cs._normalize_series_prefix("HDH_155_0001.h5") == "HDH_155"
    assert cs._normalize_series_prefix("HDH_155_0002.h5") == "HDH_155"
    assert cs._normalize_series_prefix("standalone.h5") == "standalone"
    assert cs._normalize_series_prefix("run_12.h5") == "run_12"  # only 2 digits, not a series index


def test_list_series_files_matches_prefix_and_extension(tmp_path):
    (tmp_path / "HDH_155_0001.h5").write_bytes(b"")
    (tmp_path / "HDH_155_0002.h5").write_bytes(b"")
    (tmp_path / "HDH_155_0003.h5").write_bytes(b"")
    (tmp_path / "HDH_155_0001.tif").write_bytes(b"")  # different extension, excluded
    (tmp_path / "standard_0001.h5").write_bytes(b"")  # different prefix, excluded
    (tmp_path / "subdir").mkdir()

    files = cs._list_series_files(tmp_path, "HDH_155_0001.h5")

    assert [f.name for f in files] == ["HDH_155_0001.h5", "HDH_155_0002.h5", "HDH_155_0003.h5"]


def test_list_series_files_excludes_prefix_that_is_only_a_substring(tmp_path):
    """Regression: a series like "PTCDIC13_0001.h5" must not sweep up an
    unrelated series such as "PTCDIC13_pPS_0001.h5" just because
    "PTCDIC13" is a substring of "PTCDIC13_pPS" -- the normalized prefixes
    ("PTCDIC13" vs "PTCDIC13_pPS") must match exactly."""

    (tmp_path / "PTCDIC13_0001.h5").write_bytes(b"")
    (tmp_path / "PTCDIC13_0002.h5").write_bytes(b"")
    (tmp_path / "PTCDIC13_0003.h5").write_bytes(b"")
    (tmp_path / "PTCDIC13_pPS_0001.h5").write_bytes(b"")
    (tmp_path / "PTCDIC13_pPS_0002.h5").write_bytes(b"")
    (tmp_path / "PTCDIC13_pPS_0003.h5").write_bytes(b"")

    files = cs._list_series_files(tmp_path, "PTCDIC13_0001.h5")

    assert [f.name for f in files] == ["PTCDIC13_0001.h5", "PTCDIC13_0002.h5", "PTCDIC13_0003.h5"]


def test_console_load_multishot_averages_only_matching_series(tmp_path):
    cs.reset_session()
    frame_a = _write_synthetic_frame(tmp_path / "HDH_155_0001.h5", shape=(20, 20), seed=1)
    frame_b = _write_synthetic_frame(tmp_path / "HDH_155_0002.h5", shape=(20, 20), seed=2)
    _write_synthetic_frame(tmp_path / "standard_0001.h5", shape=(20, 20), seed=3)  # must not be pulled in

    payload = cs.console_load(
        {"directory": str(tmp_path), "filename": "HDH_155_0001.h5", "detector": "TestDetector", "multishot": True}
    )

    assert payload["ok"]
    assert payload["series_file_count"] == 2
    # `load_h5_frame` transposes each raw frame on the way in, and
    # `console_load` transposes the averaged result on the way out -- the
    # two cancel, so the expected value is the plain mean of what was
    # written to disk, no extra transpose.
    expected = np.nanmean(np.stack([frame_a, frame_b], axis=0), axis=0).astype(np.float32)
    np.testing.assert_allclose(cs.get_session().data, expected, rtol=1e-5)


def test_console_list_files_requires_directory():
    payload = cs.console_list_files({})

    assert not payload["ok"]


def test_console_list_files_lists_supported_extensions_sorted(tmp_path):
    (tmp_path / "b.h5").write_bytes(b"")
    (tmp_path / "a.h5").write_bytes(b"")
    (tmp_path / "notes.txt_ignored").write_bytes(b"")
    (tmp_path / "ignored.exe").write_bytes(b"")
    (tmp_path / "subdir").mkdir()

    payload = cs.console_list_files({"directory": str(tmp_path)})

    assert payload["ok"]
    assert payload["files"] == ["a.h5", "b.h5"]


def test_console_list_files_filters_by_requested_suffixes(tmp_path):
    (tmp_path / "frame.h5").write_bytes(b"")
    (tmp_path / "calibration.json").write_bytes(b"{}")
    (tmp_path / "calibration.mat").write_bytes(b"")

    payload = cs.console_list_files({"directory": str(tmp_path), "suffixes": ["json"]})

    assert payload["ok"]
    assert payload["files"] == ["calibration.json"]


def test_console_load_info_requires_directory_and_filename():
    cs.reset_session()

    payload = cs.console_load_info({})

    assert not payload["ok"]
    assert "directory" in payload["message"]


def test_console_load_info_loads_from_directory_and_filename(tmp_path):
    from pgixs_python.calibration.calibration_io import StandardCalibration, save_standard_calibration_json

    cs.reset_session()
    cal = StandardCalibration(
        beam_center=np.array([100.0, 200.0]),
        wavelength=1.3051,
        pixel_size=0.075,
        sdd=300.0,
        detector="Eiger4M",
    )
    json_file = tmp_path / "calibration.json"
    save_standard_calibration_json(json_file, cal)

    payload = cs.console_load_info({"directory": str(tmp_path), "filename": "calibration.json"})

    assert payload["ok"], payload.get("message")
    assert payload["calibration"]["sdd_mm"] == 300.0


def test_console_load_progress_idle_when_nothing_loading():
    cs.reset_session()

    payload = cs.console_load_progress()

    assert payload == {"ok": True, "active": False}


def test_console_load_progress_inactive_after_multishot_completes(tmp_path):
    cs.reset_session()
    _write_synthetic_frame(tmp_path / "HDH_155_0001.h5", shape=(20, 20), seed=1)
    _write_synthetic_frame(tmp_path / "HDH_155_0002.h5", shape=(20, 20), seed=2)

    payload = cs.console_load(
        {"directory": str(tmp_path), "filename": "HDH_155_0001.h5", "detector": "TestDetector", "multishot": True}
    )

    assert payload["ok"]
    # `_load_series_average` updates progress per file during the loop;
    # console_load must clear it again once the request finishes so a
    # poller sees "active: false" and stops.
    assert cs.console_load_progress() == {"ok": True, "active": False}


def test_load_series_average_reports_progress_per_file(tmp_path):
    cs.reset_session()
    _write_synthetic_frame(tmp_path / "HDH_155_0001.h5", shape=(20, 20), seed=1)
    _write_synthetic_frame(tmp_path / "HDH_155_0002.h5", shape=(20, 20), seed=2)
    seen = []
    original_load = cs.load_image_file

    def _spy_load(path):
        seen.append(dict(cs._SESSION.load_progress))
        return original_load(path)

    cs.load_image_file = _spy_load
    try:
        cs._load_series_average(tmp_path / "HDH_155_0001.h5")
    finally:
        cs.load_image_file = original_load

    assert seen == [
        {"current": 1, "total": 2, "label": "HDH_155_0001.h5"},
        {"current": 2, "total": 2, "label": "HDH_155_0002.h5"},
    ]


def test_set_session_isolates_two_independent_sessions(tmp_path):
    """Regression for the desktop app: GIXS and TrXS consoles each own a
    ConsoleSession and swap it into the module global via set_session()
    before dispatching -- they must not clobber each other's loaded data."""

    gixs_frame = tmp_path / "gixs.h5"
    trxs_frame = tmp_path / "trxs.h5"
    _write_synthetic_frame(gixs_frame, shape=(20, 20), seed=1)
    _write_synthetic_frame(trxs_frame, shape=(20, 20), seed=2)

    gixs_session = cs.ConsoleSession(mode="gixs")
    trxs_session = cs.ConsoleSession(mode="trxs")

    cs.set_session(gixs_session)
    gixs_payload = _load(gixs_frame, mode="gixs")
    assert gixs_payload["ok"]
    assert cs.get_session() is gixs_session

    cs.set_session(trxs_session)
    trxs_payload = _load(trxs_frame, mode="trxs")
    assert trxs_payload["ok"]
    assert cs.get_session() is trxs_session

    cs.set_session(gixs_session)
    assert gixs_session.data is not None
    assert gixs_session.source_path == str(gixs_frame)
    assert gixs_session.mode == "gixs"

    cs.set_session(trxs_session)
    assert trxs_session.data is not None
    assert trxs_session.source_path == str(trxs_frame)
    assert trxs_session.mode == "trxs"
