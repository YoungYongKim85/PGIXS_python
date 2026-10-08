import importlib.util

import numpy as np

from pgixs_python.io.image_loader import load_image_file


def test_load_numeric_csv_image(tmp_path):
    csv_file = tmp_path / "image.csv"
    np.savetxt(csv_file, np.array([[1, 2], [3, 4]], dtype=np.float32), delimiter=",")

    result = load_image_file(csv_file)

    assert result.data.dtype == np.float32
    np.testing.assert_allclose(result.data, [[1, 2], [3, 4]])
    assert result.counts.size == 0
    assert result.metadata.data_size == (2, 2)


def test_load_mat_image_prefers_priority_name(tmp_path):
    if importlib.util.find_spec("scipy") is None:
        import pytest

        pytest.skip("scipy is not installed")

    import scipy.io

    mat_file = tmp_path / "image.mat"
    scipy.io.savemat(
        mat_file,
        {
            "large_background": np.ones((10, 10), dtype=np.float32),
            "data": np.full((2, 3), 7, dtype=np.float32),
        },
    )

    result = load_image_file(mat_file)

    assert result.data.shape == (2, 3)
    assert np.all(result.data == 7)
    assert result.metadata.extra["mat_variable_name"] == "data"


def test_load_tif_image_first_page(tmp_path):
    if importlib.util.find_spec("tifffile") is None:
        import pytest

        pytest.skip("tifffile is not installed")

    import tifffile

    tif_file = tmp_path / "image.tif"
    tifffile.imwrite(tif_file, np.array([[1, 2], [3, 4]], dtype=np.uint16))

    result = load_image_file(tif_file)

    assert result.data.dtype == np.float32
    np.testing.assert_allclose(result.data, [[1, 2], [3, 4]])


def test_v73_mat_loads_in_the_same_orientation_as_v7(tmp_path):
    import pytest

    h5py = pytest.importorskip("h5py")
    scipy_io = pytest.importorskip("scipy.io")

    image = np.arange(12, dtype=np.float64).reshape(3, 4)
    scipy_io.savemat(tmp_path / "v7.mat", {"data": image})

    # A v7.3 file is HDF5 behind a 128-byte MATLAB header, storing each
    # array column-major (so h5py sees it transposed).
    v73 = tmp_path / "v73.mat"
    with h5py.File(v73, "w", userblock_size=512) as h5:
        h5.create_dataset("data", data=image.T)
        h5["data"].attrs["MATLAB_class"] = np.bytes_("double")
    with open(v73, "r+b") as handle:
        handle.write(b"MATLAB 7.3 MAT-file".ljust(116) + b"\x00" * 8 + b"\x00\x02" + b"IM")

    v7_result = load_image_file(tmp_path / "v7.mat")
    v73_result = load_image_file(v73)

    np.testing.assert_array_equal(v73_result.data, v7_result.data)
    assert v73_result.metadata.extra["mat_variable_class"] == "double"
    assert tuple(v73_result.metadata.extra["mat_variable_shape"]) == (3, 4)
