import argparse

import numpy as np

from pgixs_python.io.two_d_info import TwoDInfo, load_two_d_info_h5, save_two_d_info_h5
from pgixs_python.workflows.export_1d_from_h5 import (
    build_q_bins_from_map,
    export_1d_from_two_d_info,
    resolve_1d_output_path,
    save_export_report,
)


def test_build_q_bins_from_map_linear():
    q = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)

    centers, bins, geom = build_q_bins_from_map(q, n_bins=2, q_min=0.0, q_max=1.0, q_scale="linear")

    np.testing.assert_allclose(centers, np.array([0.25, 0.75], dtype=np.float32))
    assert bins[0, 0] == 0
    assert bins[1, 0] == 1
    assert not geom[1, 1]


def test_resolve_1d_output_path_directory():
    out = resolve_1d_output_path("out_dir", "sample_2Dinfo.h5", "GIXS", "csv")

    assert str(out).replace("\\", "/") == "out_dir/sample_2Dinfo_GIXS_1D.csv"


def test_export_1d_from_two_d_info_writes_csv_and_report(tmp_path):
    info = TwoDInfo(
        measurement_mode="TrXS",
        data2d_pix=np.ones((3, 3), dtype=np.float32),
        data2d_q=np.ones((3, 3), dtype=np.float32),
        q_map=np.linspace(0.01, 0.09, 9, dtype=np.float32).reshape(3, 3),
        qz_map=np.zeros((3, 3), dtype=np.float32),
        qz_axis=np.arange(3, dtype=np.float32),
        beam_position=np.array([2.0, 2.0], dtype=np.float32),
        wavelength=1.2,
        sdd=300.0,
        pixel=0.075,
        detector="Eiger4M",
        source_files=["raw.h5"],
    )
    h5_file = tmp_path / "sample_2Dinfo.h5"
    save_two_d_info_h5(h5_file, info)
    loaded = load_two_d_info_h5(h5_file)
    args = argparse.Namespace(
        out=str(tmp_path),
        export_format="csv",
        q_bins=4,
        q_min=None,
        q_max=None,
        q_scale="linear",
        export_std=True,
        export_diagnostics=True,
        export_figure=False,
        report_json=True,
    )

    data_path, figure_path = export_1d_from_two_d_info(loaded, h5_file, args)
    report_path = save_export_report(loaded, h5_file, args, data_path, figure_path)

    assert data_path.exists()
    assert data_path.read_text(encoding="utf-8").splitlines()[0] == (
        "q,intensity,standard_deviation,valid_count,coverage"
    )
    assert report_path is not None
    assert report_path.exists()
