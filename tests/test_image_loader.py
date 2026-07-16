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
