from pathlib import Path

import h5py
import numpy as np
import pytest

from pgixs_python.calibration.sdd import wavelength_from_energy_kev
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
    cs.get_session().calibration_loaded = True

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


def test_console_qmap_requires_load_info_first(tmp_path):
    """Regression: "Load Info에 파일이 들어가 있지 않으면, q-cal이 되지
    않도록 해줘" -- `_SESSION.calibration` always holds plausible-looking
    placeholder values even before Load Info ever runs, so q-Cal must
    refuse to run against those rather than silently producing a wrong-
    but-not-obviously-wrong q-map."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)

    payload = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})

    assert not payload["ok"]
    assert "Load Info" in payload["message"]
    assert cs.get_session().qmap is None


def test_console_qmap_computes_q_range(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True

    payload = cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    assert payload["ok"]
    assert payload["q_range"]["min"] is not None
    assert payload["q_range"]["max"] > payload["q_range"]["min"]
    assert cs.get_session().qmap is not None
    assert cs.get_session().corrected_data is not None


def test_console_qmap_q_uncertainty_requires_both_the_checkbox_and_a_footprint(tmp_path):
    """Regression: GIXS's "q uncertainty" checkbox + FP field were both
    completely unwired -- FP was stored but never used in any calculation,
    and the checkbox was never even read. Confirmed against real MATLAB
    source (calculateGIXSQUncertainty in processGIXS2D.m): footprint is
    treated as a +/-FP/2 spread in SDD, the Q map is recomputed at each
    extreme, and q_std = |Q_max - Q_min| / 4 (the full range treated as
    ~4 sigma). Must only actually compute q_std when *both* the checkbox
    is on and footprint_mm > 0 -- neither alone is enough."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    base_payload = {"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2}

    assert cs.console_qmap(base_payload)["ok"]
    assert "q_std" not in cs.get_session().qmap

    assert cs.console_qmap({**base_payload, "q_uncertainty": True, "footprint_mm": 0.0})["ok"]
    assert "q_std" not in cs.get_session().qmap

    assert cs.console_qmap({**base_payload, "q_uncertainty": False, "footprint_mm": 5.0})["ok"]
    assert "q_std" not in cs.get_session().qmap

    assert cs.console_qmap({**base_payload, "q_uncertainty": True, "footprint_mm": 5.0})["ok"]
    qmap = cs.get_session().qmap
    for key in ("q_std", "qr_std", "qz_std", "qy_std"):
        assert key in qmap
        assert qmap[key].shape == qmap["q"].shape
        assert np.all(qmap[key] >= 0.0)
    # A real, physically meaningful spread -- not all zeros (which a bug
    # collapsing sdd_lo == sdd_hi would silently produce).
    assert np.any(qmap["q_std"] > 0.0)


def test_console_qmap_q_uncertainty_matches_hand_computed_formula(tmp_path):
    """Directly reproduces console_qmap's own q_std math against
    qmap_gixs (the same function it calls internally) to confirm q_std
    isn't just "some positive number" but the actual documented formula.

    Divisor is range/sqrt(12) (a uniform distribution's real std), not
    MATLAB calculateGIXSQUncertainty's own range/4 (~Gaussian 2-sigma) --
    confirmed with the user that typical GIXS footprint illumination
    (sample thickness of order hundreds of nm or less, ~60um vertical
    beam) is effectively uniform across the footprint regardless of
    incidence angle, so the uniform-distribution formula is the physically
    correct one here even though it diverges from the MATLAB original by
    about 13%."""

    from pgixs_python.qmap.gixs import qmap_gixs

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    payload = {
        "energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0,
        "incident_angle_deg": 0.2, "q_uncertainty": True, "footprint_mm": 8.0,
    }
    assert cs.console_qmap(payload)["ok"]
    session = cs.get_session()
    qmap = session.qmap
    pixel_mm = float(session.calibration["pixel_mm"])
    wavelength = wavelength_from_energy_kev(9.5)
    k0 = 2.0 * np.pi / wavelength
    n_rows, n_cols = session.data.shape

    _, qr_lo, qz_lo, q_lo, _, _ = qmap_gixs(
        det_y=n_cols, det_z=n_rows, beam_pos=(24.0, 24.0), pixel_mm=pixel_mm,
        sdd_mm=300.0 - 4.0, k0=k0, incident_angle_deg=0.2,
    )
    _, qr_hi, qz_hi, q_hi, _, _ = qmap_gixs(
        det_y=n_cols, det_z=n_rows, beam_pos=(24.0, 24.0), pixel_mm=pixel_mm,
        sdd_mm=300.0 + 4.0, k0=k0, incident_angle_deg=0.2,
    )
    divisor = np.sqrt(12.0)
    np.testing.assert_allclose(qmap["q_std"], np.abs(q_hi - q_lo) / divisor, rtol=1e-5)
    np.testing.assert_allclose(qmap["qr_std"], np.abs(qr_hi - qr_lo) / divisor, rtol=1e-5)
    np.testing.assert_allclose(qmap["qz_std"], np.abs(qz_hi - qz_lo) / divisor, rtol=1e-5)


def test_console_qspace_image_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True

    payload = cs.console_qspace_image({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_qspace_image_returns_regridded_preview(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_qspace_image({"mapping": "Qy/Qz"})

    assert payload["ok"], payload.get("message")
    assert payload["mapping"] == "Qy/Qz"


def test_console_qspace_image_works_for_trxs(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})

    payload = cs.console_qspace_image({})

    assert payload["ok"], payload.get("message")


def test_console_q_space_point_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True

    payload = cs.console_q_space_point({"col": 24.0, "row": 24.0})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_q_space_point_and_pixel_from_q_space_round_trip(tmp_path):
    """Used to draw a line/accumulation/cake cut overlay directly as a
    straight line in Q-space (console_q_space_point) and to invert a
    Q-space click back to a pixel offset (console_pixel_from_q_space) --
    the two must be exact inverses so a click and the overlay it sets up
    stay consistent. Reported: "Qxy/Qz 맵에서 90도 line cut을... 해보면
    일직선 라인이 아니라 휘어져 나와... 그냥 일직선 라인이 되면 좋을 것
    같아.\""""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    forward = cs.console_q_space_point({"col": 30.0, "row": 15.0})
    assert forward["ok"], forward.get("message")

    inverse = cs.console_pixel_from_q_space({"qx": forward["qx"], "qy": forward["qy"]})
    assert inverse["ok"], inverse.get("message")
    assert inverse["col"] == pytest.approx(30.0)
    assert inverse["row"] == pytest.approx(15.0)


def test_console_polar_image_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True

    payload = cs.console_polar_image({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_polar_image_returns_polar_preview(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_line_1d({"angle_deg": 90.0, "accumulation": True, "accum_minus": -2.0, "accum_plus": 2.0})

    assert payload["ok"], payload.get("message")
    assert len(payload["preview"]["q"]) > 0


def test_console_line_1d_accumulation_and_cakecut_are_mutually_exclusive(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_line_1d({"accumulation": True, "cakecut": True})

    assert not payload["ok"]
    assert "Accumulation" in payload["message"]


def test_console_line_1d_accumulation_requires_axis_aligned_angle(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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


def test_console_line_1d_carries_q_std_when_q_uncertainty_was_computed(tmp_path):
    """Second half of the "q uncertainty" feature: console_line_1d must
    sample q_std along the exact same cut geometry as intensity (reported:
    "1D process에도 column이 생성되게 해줘... 1D result에 x축 errorbar가
    있으면 좋겠어"). Scoped to mapping="Pixel" only -- q_std is a true
    per-pixel map, meaningless against a Q-space-regridded cut."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    qcal = cs.console_qmap(
        {"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2,
         "q_uncertainty": True, "footprint_mm": 5.0}
    )
    assert qcal["ok"], qcal.get("message")

    payload = cs.console_line_1d({"angle_deg": 45.0, "mapping": "Pixel"})
    assert payload["ok"], payload.get("message")
    curve = cs.get_session().last_curve
    assert curve["q_std"] is not None
    assert np.asarray(curve["q_std"]).shape == np.asarray(curve["intensity"]).shape
    assert payload["preview"]["q_std"] is not None
    assert payload.get("warning") is None  # no Cakecut -> no warning

    qspace_payload = cs.console_line_1d({"angle_deg": 45.0, "mapping": "Qy/Qz"})
    assert qspace_payload["ok"], qspace_payload.get("message")
    assert cs.get_session().last_curve["q_std"] is None


def test_console_line_1d_q_std_shape_matches_intensity_with_masked_pixels(tmp_path):
    """Regression, reported live with a real traceback: "GIXS에서
    horizontal cut에서 180으로 angle을 두고 pos move z방향 30을 하여
    라인을 긋고, plot을 하니... ValueError: 'xerr' (shape: (1700,)) must
    be... whose shape matches 'x' (shape: (1690,))".

    Root cause: angle=180 + a nonzero z-pos-move routes through
    accumulation_average, whose own `keep` mask (which output bins
    survive) depends on whether *that specific call's* reduced value
    ends up finite -- not on geometry alone. q_std_map is purely
    geometric and (short of a genuine calibration failure) never NaN, so
    replaying the exact same cut against it unmasked kept extra bins the
    real intensity cut -- run against real detector data with masked/gap
    pixels -- had already dropped, producing a longer q_std curve than
    intensity and crashing errorbar()'s shape check. Reproduces with a
    frame carrying a real NaN region (masked/grid-removed pixels, like
    real GIXS data) intersecting only part of the accumulation band."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    shape = (64, 64)
    data = np.random.default_rng(0).uniform(10, 100, size=shape).astype(np.float32)
    # angle=180 only keeps columns with (1-based col) - beam_y >= 0, i.e.
    # 0-based col index >= 31 for beam_y=32 -- NaN a block *inside* that
    # kept range (cols 40:55) so only part of it gets dropped by the real
    # (masked) data's own cut; a block outside the kept range wouldn't
    # actually reach this bug (confirmed by hand -- an earlier version of
    # this test masked cols 0:20, entirely outside the kept range, and
    # passed without ever exercising the divergence).
    data[0:6, 40:55] = np.nan
    with h5py.File(frame_path, "w") as h5:
        h5.create_dataset("entry/data/data", data=data[None, :, :])
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    qcal = cs.console_qmap(
        {"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0, "incident_angle_deg": 0.2,
         "q_uncertainty": True, "footprint_mm": 5.0}
    )
    assert qcal["ok"], qcal.get("message")

    payload = cs.console_line_1d({"angle_deg": 180.0, "z_pos_move": 30.0, "mapping": "Pixel"})

    assert payload["ok"], payload.get("message")
    curve = cs.get_session().last_curve
    intensity = np.asarray(curve["intensity"])
    q_std = curve.get("q_std")
    assert q_std is not None
    assert np.asarray(q_std).shape == intensity.shape


def test_console_line_1d_excludes_q_uncertainty_entirely_for_cakecut(tmp_path):
    """Regression: "cake_cut에서는 당연히 반영하면 안돼" -- a Cakecut wedge
    spans a real spread of q directions, and averaging per-pixel q_std
    across it the same way intensity is averaged isn't real error
    propagation (variances need combining in quadrature, not averaged
    directly), so it isn't just unreliable, it's not physically
    meaningful. Unlike MATLAB's own Anglecut1d_giplus (which warns
    "Cakecut not working for Q uncertainty!" but still computes and shows
    a value anyway), this port excludes it outright -- no q_std column,
    just a status note explaining why."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    assert cs.console_qmap(
        {"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2,
         "q_uncertainty": True, "footprint_mm": 5.0}
    )["ok"]

    payload = cs.console_line_1d(
        {"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0, "mapping": "Pixel"}
    )

    assert payload["ok"], payload.get("message")
    assert cs.get_session().last_curve["q_std"] is None
    assert payload["preview"]["q_std"] is None
    assert payload.get("warning") is not None
    assert "Cakecut" in payload["warning"]


def test_console_save_curve_writes_a_q_standard_deviation_column(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    assert cs.console_qmap(
        {"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2,
         "q_uncertainty": True, "footprint_mm": 5.0}
    )["ok"]
    assert cs.console_line_1d({"angle_deg": 45.0, "mapping": "Pixel"})["ok"]

    saved = cs.console_save_curve({"out_dir": str(tmp_path / "out")})
    assert saved["ok"], saved.get("message")
    header = Path(saved["curve_path"]).read_text(encoding="utf-8").splitlines()[0]
    assert "q_standard_deviation" in header


def test_console_line_1d_qxyqz_mapping_runs_an_independent_cakecut(tmp_path):
    """Regression: "1D process해서 pixel map에서 cutting하는 것이 Qxy/Qz
    cutting에 반영되서 진행되는데, 그렇게 하지말고 독립적으로 되면
    좋겠어" -- Map format previously only affected the 2D image display;
    every line/accumulation/cake cut ran against the raw pixel data
    regardless of it. "Qxy/Qz" must now run the cut against the actual
    Q-remapped image (port of Anglecut1d_gi.m's `Mapping` branch), giving
    a genuinely different curve from the same cut done in "Pixel" mode."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    pixel_payload = cs.console_line_1d(
        {"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0, "mapping": "Pixel"}
    )
    assert pixel_payload["ok"], pixel_payload.get("message")
    assert pixel_payload["mapping"] == "Pixel"
    pixel_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    qspace_payload = cs.console_line_1d(
        {"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0, "mapping": "Qxy/Qz"}
    )
    assert qspace_payload["ok"], qspace_payload.get("message")
    assert qspace_payload["mapping"] == "Qxy/Qz"
    qspace_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert qspace_intensity.shape != pixel_intensity.shape or not np.allclose(
        qspace_intensity, pixel_intensity, equal_nan=True
    )


def test_console_line_1d_qxyqz_mapping_runs_an_independent_accumulation_cut(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    pixel_payload = cs.console_line_1d(
        {"angle_deg": 90.0, "accumulation": True, "accum_minus": -2.0, "accum_plus": 2.0, "mapping": "Pixel"}
    )
    assert pixel_payload["ok"], pixel_payload.get("message")
    pixel_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    qspace_payload = cs.console_line_1d(
        {"angle_deg": 90.0, "accumulation": True, "accum_minus": -2.0, "accum_plus": 2.0, "mapping": "Qxy/Qz"}
    )
    assert qspace_payload["ok"], qspace_payload.get("message")
    qspace_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert qspace_intensity.shape != pixel_intensity.shape or not np.allclose(
        qspace_intensity, pixel_intensity, equal_nan=True
    )


def test_console_line_1d_qyqz_mapping_also_runs_independently(tmp_path):
    """Same fix, the app's other Q-space Map format option ("Qy/Qz",
    horizontal_q_map = qmap["qy"] instead of qmap["qr"])."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    pixel_payload = cs.console_line_1d(
        {"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0, "mapping": "Pixel"}
    )
    assert pixel_payload["ok"], pixel_payload.get("message")
    pixel_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    qspace_payload = cs.console_line_1d(
        {"angle_deg": 45.0, "cakecut": True, "range_minus": -5.0, "range_plus": 5.0, "mapping": "Qy/Qz"}
    )
    assert qspace_payload["ok"], qspace_payload.get("message")
    qspace_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert qspace_intensity.shape != pixel_intensity.shape or not np.allclose(
        qspace_intensity, pixel_intensity, equal_nan=True
    )


def test_console_line_1d_trxs_reports_the_true_q_not_a_small_angle_approximation(tmp_path):
    """Regression: TrXS's Qy/Qz Line/Cake/Accumulation cut labeled each
    sample with image_transform_q_tr's own small-angle-linear (d_q *
    pixel_index) axis value -- an approximation, off by several percent
    at the detector corners at a moderate SDD and much more at a short
    one (reported: "실제 Qmap이... 광각으로 갈수록 벌어져... 코너 쪽만
    정밀했으면 좋겠어"). TrXS's flat-detector, normal-incidence geometry
    makes the true |Q| an exact closed-form function of radial pixel
    distance alone (2theta = atan(R_px*pixel_mm/sdd_mm), q =
    2*k0*sin(theta)) -- confirmed azimuthally symmetric to <0.5% against
    qmap_trxs's real per-pixel values -- so _q_space_cut_inputs must
    report that instead."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
    beam_y, beam_z = 24.0, 24.0
    sdd_mm = 5.0  # short SDD -- exaggerates the small-angle approximation's error
    energy_kev = 9.5
    assert cs.console_qmap({"energy_kev": energy_kev, "sdd_mm": sdd_mm, "beam_y": beam_y, "beam_z": beam_z})["ok"]

    pixel_mm = float(cs.get_session().calibration["pixel_mm"])
    wavelength_a = wavelength_from_energy_kev(energy_kev)
    k0 = 2.0 * np.pi / wavelength_a

    _, q_magnitude, _, _ = cs._q_space_cut_inputs("Qy/Qz")

    row, col = 0, 0  # the 48x48 frame's far corner from the beam
    radius_px = np.hypot((col + 1) - beam_y, (row + 1) - beam_z)
    two_theta = np.arctan(radius_px * pixel_mm / sdd_mm)
    expected_q = 2.0 * k0 * np.sin(two_theta / 2.0)
    linear_approx_q = (2.0 * np.pi * pixel_mm / (wavelength_a * sdd_mm)) * radius_px

    assert q_magnitude[row, col] == pytest.approx(expected_q, rel=1e-4)
    # Confirms the fix actually changed the reported value (not merely
    # that the new formula matches itself) -- the old linear approximation
    # is meaningfully larger at this short an SDD, this far from the beam.
    assert linear_approx_q > expected_q * 1.05


def test_console_azimuthal_1d_returns_curve(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    payload = cs.console_azimuthal_1d({"pixel": 10.0, "range_minus": -3.0, "range_plus": 3.0, "n_angle_bins": 16})

    assert payload["ok"]
    angle = np.asarray(payload["preview"]["angle"], dtype=np.float64)
    assert angle.min() >= 0.0
    assert angle.max() <= 360.0
    assert angle.max() > 180.0


def test_console_azimuthal_1d_qxyqz_mapping_runs_independently(tmp_path):
    """Regression: the Azimuthal Process ran against the raw pixel data
    regardless of Map format, unlike the line/cake/accumulation cuts fixed
    earlier -- so a "missing wedge" gap plainly visible in the Qxy/Qz 2D
    display (Itrans's masked beamstop column + low-coverage bilinear-splat
    cells) never showed up as a break in the resulting azimuthal curve
    (reported: "90도 부근에 mising wedge가 있음에도 견결되서 그렸어").
    "Qxy/Qz" must now cut the actual Q-remapped (Itrans) image, same as
    console_line_1d's own mapping branch."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0, "incident_angle_deg": 0.2})

    pixel_payload = cs.console_azimuthal_1d(
        {"pixel": 10.0, "range_minus": -3.0, "range_plus": 3.0, "n_angle_bins": 32, "mapping": "Pixel"}
    )
    assert pixel_payload["ok"], pixel_payload.get("message")
    assert pixel_payload["mapping"] == "Pixel"
    pixel_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    qspace_payload = cs.console_azimuthal_1d(
        {"pixel": 10.0, "range_minus": -3.0, "range_plus": 3.0, "n_angle_bins": 32, "mapping": "Qxy/Qz"}
    )
    assert qspace_payload["ok"], qspace_payload.get("message")
    assert qspace_payload["mapping"] == "Qxy/Qz"
    qspace_intensity = np.asarray(cs.get_session().last_curve["intensity"])

    assert qspace_intensity.shape != pixel_intensity.shape or not np.allclose(
        qspace_intensity, pixel_intensity, equal_nan=True
    )


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
    cs.get_session().calibration_loaded = True

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


def test_console_normalize_invalidates_a_stale_corrected_data_from_an_earlier_qcal(tmp_path):
    """Regression: "BKGsub. & Norm을 누르면 바뀐 이미지를 유지하겠냐고
    나오는데, 사실 이미지가 바뀌지 않고 메시지 박스만 나와" -- if q-Map
    Cal. already ran (setting corrected_data from the *pre*-normalization
    image), console_widget.py's own render always prefers corrected_data
    over data when it's set, so committing a background subtraction left
    the displayed image showing the old, stale corrected copy even though
    session.data itself was genuinely updated. confirm=true must drop the
    stale corrected_data so the render falls back to the now-current
    data."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
    assert cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 24.0, "beam_z": 24.0})["ok"]
    assert cs.get_session().corrected_data is not None

    bkg_path = tmp_path / "bkg.h5"
    _write_synthetic_frame(bkg_path, seed=1)
    assert cs.console_background_load({"directory": str(tmp_path), "filename": bkg_path.name})["ok"]

    assert cs.console_normalize({"method": "None", "confirm": False})["ok"]
    committed = cs.console_normalize({"method": "None", "confirm": True})

    assert committed["ok"], committed.get("message")
    assert cs.get_session().corrected_data is None


def test_console_save_requires_qmap(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path)
    cs.get_session().calibration_loaded = True

    payload = cs.console_save({})

    assert not payload["ok"]
    assert "q-Map Cal." in payload["message"]


def test_console_save_writes_files_and_round_trips(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    out_dir = tmp_path / "out"
    _load(frame_path)
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True

    assert payload["mode"] == "gixs"
    assert cs.get_session().mode == "gixs"


def test_console_load_accepts_trxs_mode(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)

    payload = _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True

    assert payload["mode"] == "trxs"
    assert cs.get_session().mode == "trxs"


def test_console_qmap_uses_trxs_physics_without_incident_angle(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True

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
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0})

    payload = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "min_coverage": 0.99})

    assert payload["ok"]
    assert len(payload["preview"]["q"]) == len(payload["preview"]["intensity"])
    assert cs.get_session().last_curve["kind"] == "circular"


def test_console_circular_average_calc_std_populates_last_curve_std(tmp_path):
    """Regression: "Std가 켜고 프로세스 했을때, 1D Result에서 Y축
    errorbar plot이 되면 좋을 것 같아" -- sector_average already computes
    a per-bin std whenever calc_std=True, but console_circular_average
    discarded it before it ever reached last_curve, so the Std checkbox
    had nothing for the widget to plot as error bars. calc_std=True must
    now populate last_curve["std"] (and calc_std=False must leave it
    None, not an all-NaN array, so the widget can tell "not computed"
    apart from "computed but every bin masked out")."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(64, 64))
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0})

    without_std = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "calc_std": False})
    assert without_std["ok"], without_std.get("message")
    assert cs.get_session().last_curve["std"] is None
    assert without_std["preview"]["std"] is None

    with_std = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "calc_std": True})
    assert with_std["ok"], with_std.get("message")
    std = cs.get_session().last_curve["std"]
    assert std is not None
    assert std.shape == np.asarray(cs.get_session().last_curve["intensity"]).shape
    assert np.any(np.isfinite(std))
    assert with_std["preview"]["std"] is not None


def test_console_set_circular_mask_requires_loaded_data():
    cs.reset_session()

    payload = cs.console_set_circular_mask({"mask": [[True]]})

    assert not payload["ok"]
    assert "Load a detector image" in payload["message"]


def test_console_set_circular_mask_rejects_shape_mismatch(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(20, 20))
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True

    payload = cs.console_set_circular_mask({"mask": np.ones((5, 5), dtype=bool)})

    assert not payload["ok"]
    assert "shape" in payload["message"].lower()


def test_console_set_circular_mask_accepts_and_clears(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(20, 20))
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True

    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    payload = cs.console_set_circular_mask({"mask": mask})

    assert payload["ok"], payload.get("message")
    assert payload["pixel_count"] == 25
    assert payload["total_pixels"] == 400
    assert cs.get_session().circular_mask is not None

    cleared = cs.console_set_circular_mask({"clear": True})
    assert cleared["ok"]
    assert cs.get_session().circular_mask is None


def test_console_circular_average_applies_mask(tmp_path):
    """Regression for "circular process에서 mask를 만들 수 있도록 해줘": once a
    mask is set, console_circular_average must actually exclude
    outside-mask pixels (matching TrXS_gui.m's `I2d(BW <= 0) = NaN`), not
    just store the mask inertly."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    data = _write_synthetic_frame(frame_path, shape=(64, 64))
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
    cs.console_qmap({"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0})

    unmasked = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "min_coverage": 0.0})
    assert unmasked["ok"]
    assert not unmasked.get("mask_active")

    # Mask out the whole right half of the detector -- the resulting
    # curve must differ from the unmasked one somewhere.
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :32] = True
    assert cs.console_set_circular_mask({"mask": mask})["ok"]

    masked = cs.console_circular_average({"angle_min": 0, "angle_max": 360, "min_coverage": 0.0})
    assert masked["ok"], masked.get("message")
    assert masked["mask_active"]
    assert masked["preview"]["intensity"] != unmasked["preview"]["intensity"]


def test_console_load_clears_circular_mask(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path, shape=(20, 20))
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
    mask = np.ones((20, 20), dtype=bool)
    assert cs.console_set_circular_mask({"mask": mask})["ok"]
    assert cs.get_session().circular_mask is not None

    frame_path_2 = tmp_path / "frame2.h5"
    _write_synthetic_frame(frame_path_2, shape=(20, 20), seed=7)
    _load(frame_path_2, mode="trxs")
    cs.get_session().calibration_loaded = True

    assert cs.get_session().circular_mask is None


def test_console_save_trxs_mode_has_no_incident_angle(tmp_path):
    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    _write_synthetic_frame(frame_path)
    out_dir = tmp_path / "out"
    _load(frame_path, mode="trxs")
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True

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
    cs.get_session().calibration_loaded = True
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
    cs.get_session().calibration_loaded = True
    assert gixs_payload["ok"]
    assert cs.get_session() is gixs_session

    cs.set_session(trxs_session)
    trxs_payload = _load(trxs_frame, mode="trxs")
    cs.get_session().calibration_loaded = True
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


def test_console_line_1d_offset_cut_is_labelled_by_the_varying_q_component(tmp_path):
    """Regression: an offset axis-aligned cut runs at a fixed nonzero Qz, so
    |Q| along it is inflated by that constant and never reaches 0 even where
    the cut crosses the beam meridian. MATLAB switches the axis to the
    component that actually varies along the cut (`Anglecut1d_gi.m`'s
    `calcQ_Horizontal`: `Qdata = Qr` once `zposmv ~= 0`); this port used to
    label every cut with |Q| regardless, shifting peak positions/d-spacings
    for any offset cut."""

    cs.reset_session()
    frame_path = tmp_path / "frame.h5"
    shape = (64, 64)
    data = np.random.default_rng(1).uniform(10.0, 100.0, size=shape).astype(np.float32)
    with h5py.File(frame_path, "w") as h5:
        h5.create_dataset("entry/data/data", data=data[None, :, :])
    _load(frame_path)
    cs.get_session().calibration_loaded = True
    assert cs.console_qmap(
        {"energy_kev": 9.5, "sdd_mm": 300.0, "beam_y": 32.0, "beam_z": 32.0, "incident_angle_deg": 0.2}
    )["ok"]

    z_pos_move = 30.0
    payload = cs.console_line_1d({"angle_deg": 180.0, "z_pos_move": z_pos_move, "mapping": "Pixel"})
    assert payload["ok"], payload.get("message")
    q_axis = np.asarray(cs.get_session().last_curve["q"], dtype=np.float64)

    qmap = cs.get_session().qmap
    cut_row = int(round(32.0 - z_pos_move)) - 1  # accumulation_average's row_center, 0-based
    row_q = np.abs(qmap["q"][cut_row])
    row_qr = np.abs(qmap["qr"][cut_row])

    # The cut crosses the beam meridian, where the in-plane component is ~0
    # but |Q| is still pinned to the fixed Qz offset.
    assert q_axis.min() < 0.25 * row_q.min()
    # ...and its span follows Qxy, not the (larger) |Q| of the same row.
    assert q_axis.max() == pytest.approx(row_qr.max(), rel=0.02)
    assert q_axis.max() < 0.95 * row_q.max()
