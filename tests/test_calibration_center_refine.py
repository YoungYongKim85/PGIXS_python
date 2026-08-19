import numpy as np
import pytest

from pgixs_python.calibration.center_refine import fine_tune_center, refine_center_from_ring


def _synthetic_ring_frame(shape=(300, 300), center=(150.5, 150.5), ring_radius=100.0, ring_sigma=2.5, seed=0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(xx - (center[0] - 1.0), yy - (center[1] - 1.0))
    rng = np.random.default_rng(seed)
    data = 10.0 + 800.0 * np.exp(-((r - ring_radius) ** 2) / (2.0 * ring_sigma**2))
    data = data + rng.uniform(0.0, 1.0, size=shape)
    return data.astype(np.float32)


def test_refine_center_from_ring_converges_from_a_nearby_guess():
    true_center = (150.5, 150.5)
    data = _synthetic_ring_frame(center=true_center, ring_radius=100.0)

    refined = refine_center_from_ring(data, init_center=(155.0, 146.0))

    assert refined[0] == pytest.approx(true_center[0], abs=1.5)
    assert refined[1] == pytest.approx(true_center[1], abs=1.5)


def test_refine_center_from_ring_returns_init_center_when_frame_too_small():
    # A 20x20 frame's max radius from center never exceeds the >80 (or
    # relaxed >30) threshold -- `valid` stays all-False, hitting the
    # early-return guard before any optimization is attempted.
    data = (10.0 + np.random.default_rng(1).uniform(0.0, 1.0, size=(20, 20))).astype(np.float32)

    refined = refine_center_from_ring(data, init_center=(10.0, 10.0))

    assert refined == (10.0, 10.0)


def test_fine_tune_center_converges_from_a_nearby_guess():
    true_center = (150.5, 150.5)
    data = _synthetic_ring_frame(center=true_center, ring_radius=100.0)

    tuned = fine_tune_center(data, init_center=(154.0, 147.0), target_radius=100.0)

    assert tuned[0] == pytest.approx(true_center[0], abs=1.0)
    assert tuned[1] == pytest.approx(true_center[1], abs=1.0)


def test_fine_tune_center_stops_at_the_image_boundary():
    data = _synthetic_ring_frame()

    # Way outside the frame -- must return immediately, not raise.
    tuned = fine_tune_center(data, init_center=(-100.0, -100.0), target_radius=100.0)

    assert tuned == (-100.0, -100.0)


def test_fine_tune_center_handles_a_degenerate_target_radius():
    data = _synthetic_ring_frame()

    # target_radius so small (or negative) the +/-20 search window has no
    # positive offsets left -- must return the input center unchanged, not raise.
    tuned = fine_tune_center(data, init_center=(150.5, 150.5), target_radius=-30.0)

    assert tuned == (150.5, 150.5)
