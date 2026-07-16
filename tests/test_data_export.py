import numpy as np

from pgixs_python.export import OneDExport, build_1d_table, save_1d_data


def test_build_1d_table_defaults_to_q_intensity_std():
    table, columns = build_1d_table(
        OneDExport(
            x=np.array([0.1, 0.2], dtype=np.float32),
            intensity=np.array([10, 20], dtype=np.float32),
            std=np.array([1, 2], dtype=np.float32),
            counts=np.array([5, 6], dtype=np.float32),
            coverage=np.array([0.5, 0.6], dtype=np.float32),
        )
    )

    assert columns == ["q", "intensity", "standard_deviation"]
    assert table.shape == (2, 3)


def test_build_1d_table_can_include_diagnostics():
    table, columns = build_1d_table(
        OneDExport(
            x=np.array([0.1], dtype=np.float32),
            intensity=np.array([10], dtype=np.float32),
            counts=np.array([5], dtype=np.float32),
            coverage=np.array([0.5], dtype=np.float32),
        ),
        include_diagnostics=True,
    )

    assert columns == ["q", "intensity", "valid_count", "coverage"]
    assert table.shape == (1, 4)


def test_save_1d_data_csv(tmp_path):
    out = tmp_path / "curve.csv"

    save_1d_data(
        out,
        OneDExport(
            x=np.array([0.1, 0.2], dtype=np.float32),
            intensity=np.array([10, 20], dtype=np.float32),
        ),
    )

    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "q,intensity"
    assert "," in text.splitlines()[1]
