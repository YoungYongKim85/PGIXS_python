"""TIF images keep the orientation they are read in (MATLAB uses them as read
and its saved beam centres fit them only untransposed); only H5 frames get
the display transpose. Before, an Eiger4M TIF failed to load (mask shape
mismatch) and square-detector TIFs had y and z swapped."""

import h5py
import numpy as np
import pytest

from pgixs_python.processing.masking import detector_mask_for_type, masked_display_frame
from pgixs_python.ui_backend import console_session as cs
from pgixs_python.ui_backend import sddcal_session as sd

tifffile = pytest.importorskip("tifffile")


def _marked(shape, mark):
    data = np.full(shape, 10.0, dtype=np.float32)
    data[mark] = 999.0
    return data


def test_eiger_tif_loads_as_read_with_the_mask_transposed(tmp_path):
    mask = detector_mask_for_type("Eiger4M").mask
    data = _marked(mask.T.shape, (100, 1500))
    tifffile.imwrite(tmp_path / "std_0001.tif", data)

    sd.reset_session()
    result = sd.sddcal_load({"directory": str(tmp_path), "filename": "std_0001.tif", "detector": "Eiger4M"})

    assert result["ok"], result.get("message")
    loaded = sd.get_session().data
    assert loaded.shape == data.shape
    assert loaded[100, 1500] == 999.0
    # Module gaps of the transposed mask are blanked.
    assert np.isnan(loaded[~np.isfinite(mask.T) | (mask.T <= 0)]).all()


def test_square_detector_tif_keeps_y_and_z(tmp_path):
    data = _marked((1920, 1920), (300, 1200))
    tifffile.imwrite(tmp_path / "waxd_0001.tif", data)

    result = cs.console_load({"directory": str(tmp_path), "filename": "waxd_0001.tif", "detector": "MX170HS", "mode": "trxs"})

    assert result["ok"], result.get("message")
    assert cs.get_session().data[300, 1200] == 999.0


def test_h5_frames_are_still_transposed_for_display(tmp_path):
    mask = detector_mask_for_type("Eiger4M").mask
    raw = _marked(mask.T.shape, (100, 1500))  # H5 stores the frame transposed
    with h5py.File(tmp_path / "frame_0001.h5", "w") as h5:
        h5.create_dataset("entry/data/data", data=raw[None])

    from pgixs_python.io.image_loader import load_image_file

    loaded = load_image_file(tmp_path / "frame_0001.h5").data
    shown = masked_display_frame(loaded, mask, tmp_path / "frame_0001.h5")
    assert shown.shape == mask.T.shape
    assert shown[100, 1500] == 999.0
