import numpy as np

from pgixs_python.export import OneDExport, build_1d_table, save_1d_data
from pgixs_python.export.data_export import save_series_data


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


def test_save_series_data_x_label_containing_delimiter_does_not_shift_columns(tmp_path):
    """Regression: GIXS2's X-axis label is built as e.g. "Time, s" (see
    series_processing.py's `_time_axis`) -- writing that verbatim into a
    comma-delimited CSV header split it into two fields ("Time" and " s"),
    silently shifting every ROI/frame column one place to the right of its
    actual data (reported: ROI1..ROI5 headers no longer lined up with their
    columns)."""

    out = tmp_path / "series.csv"

    save_series_data(
        out,
        x_axis=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        x_label="Time, s",
        matrix=np.arange(15, dtype=np.float32).reshape(5, 3),
        column_labels=["ROI1", "ROI2", "ROI3", "ROI4", "ROI5"],
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header == ["Time  s", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
    for line in lines[1:]:
        assert len(line.split(",")) == len(header)


def test_save_1d_data_keeps_10_digits_and_a_comma_label_in_one_csv_field(tmp_path):
    from pgixs_python.export.data_export import OneDExport, save_1d_data

    path = save_1d_data(
        tmp_path / "c.csv",
        OneDExport(x=np.array([0.0123456789, 0.5]), intensity=np.array([1e8 + 1, 2.0]), x_label="q, 1/A"),
    )

    header, first = path.read_text(encoding="utf-8").splitlines()[:2]
    assert header.split(",") == ["q  1/A", "intensity"]
    assert first == "0.0123456789,100000001"
