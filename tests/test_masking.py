import numpy as np

from pgixs_python.processing.masking import apply_detector_mask, detector_mask_for_type


def test_eiger4m_fallback_mask_matches_transposed_h5_frame_shape():
    """A real Eiger4M H5 frame loads as (2070, 2167) after `load_h5_frame`'s
    transpose (matching MATLAB `loadPGIXSH5Frame`). The fallback mask must
    match that orientation, or `apply_detector_mask` raises a shape
    mismatch for any Eiger4M user without a local `EigerMask.mat` file."""

    detector_mask = detector_mask_for_type("Eiger4M", mask_dir="does/not/exist")

    assert detector_mask.mask.shape == (2070, 2167)
    assert detector_mask.report.fallback_used

    data = np.ones((2070, 2167), dtype=np.float32)
    masked = apply_detector_mask(data, detector_mask.mask)
    assert masked.shape == data.shape
