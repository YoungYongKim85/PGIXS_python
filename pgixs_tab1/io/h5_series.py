"""H5 detector series loading for PGIXS Tab1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pgixs_tab1.io.h5_loader import DEFAULT_DETECTOR_DATASET, H5RetryReport, load_h5_frame_with_retry
from pgixs_tab1.io.h5_metadata import H5Metadata, read_h5_metadata


@dataclass
class H5FrameRecord:
    """Per-file diagnostics for an H5 detector frame."""

    file_path: Path
    frame_number: int
    ok: bool
    retry_report: H5RetryReport
    metadata: H5Metadata | None = None
    metadata_ok: bool = True
    metadata_message: str = ""


@dataclass
class H5SeriesLoadResult:
    """Loaded H5 stack plus MATLAB-compatible sidecar diagnostics."""

    stack: np.ndarray
    records: list[H5FrameRecord]
    counts: np.ndarray
    timestamps: np.ndarray
    temperatures: np.ndarray
    retry_failed_frames: np.ndarray
    metadata_failed_frames: np.ndarray


def load_h5_series(
    file_paths: Iterable[str | Path],
    dataset_path: str = DEFAULT_DETECTOR_DATASET,
    mask: np.ndarray | None = None,
    saturated_value: float = np.nan,
    fallback_size: tuple[int, int] | None = None,
    retry_count: int = 1,
    metadata_mode: str = "",
    reference_timestamp: float | None = None,
) -> H5SeriesLoadResult:
    """Load a list of H5 detector files into a `(frame, row, col)` stack.

    Failed frame loads retry once by default, then become a NaN image when a
    fallback size is known. Frame and metadata failure sidecars use 1-based
    frame numbers to match existing MATLAB diagnostics.
    """

    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("load_h5_series requires at least one H5 file.")

    if fallback_size is None and mask is not None:
        fallback_size = tuple(int(v) for v in np.asarray(mask).shape)  # type: ignore[assignment]

    frames: list[np.ndarray] = []
    records: list[H5FrameRecord] = []
    counts = np.full((len(paths), 2), np.nan, dtype=np.float64)
    timestamps = np.full(len(paths), np.nan, dtype=np.float64)
    temperatures = np.full((len(paths), 3), np.nan, dtype=np.float64)
    retry_failed: list[int] = []
    metadata_failed: list[int] = []

    known_size = fallback_size
    for index, path in enumerate(paths, start=1):
        data, retry_report = load_h5_frame_with_retry(
            path,
            dataset_path=dataset_path,
            mask=mask,
            saturated_value=saturated_value,
            fallback_size=known_size,
            retry_count=retry_count,
        )

        if known_size is None:
            known_size = tuple(int(v) for v in data.shape)
        elif tuple(data.shape) != tuple(known_size):
            raise ValueError(
                f"H5 frame shape mismatch at frame {index}: expected {known_size}, got {tuple(data.shape)}"
            )

        if not retry_report.ok:
            retry_failed.append(index)

        metadata = None
        metadata_ok = True
        metadata_message = ""
        try:
            metadata = read_h5_metadata(path, mode=metadata_mode, reference_timestamp=reference_timestamp)
            counts[index - 1, 0] = metadata.counter1
            counts[index - 1, 1] = metadata.counter_bc
            timestamps[index - 1] = metadata.timestamp
            temperatures[index - 1, :] = metadata.temperature
        except Exception as exc:  # noqa: BLE001 - diagnostics must keep the original reason.
            metadata_ok = False
            metadata_message = str(exc)
            metadata_failed.append(index)

        frames.append(data)
        records.append(
            H5FrameRecord(
                file_path=path,
                frame_number=index,
                ok=retry_report.ok,
                retry_report=retry_report,
                metadata=metadata,
                metadata_ok=metadata_ok,
                metadata_message=metadata_message,
            )
        )

    stack = np.stack(frames, axis=0).astype(np.float32, copy=False)
    return H5SeriesLoadResult(
        stack=stack,
        records=records,
        counts=counts,
        timestamps=timestamps,
        temperatures=temperatures,
        retry_failed_frames=np.asarray(retry_failed, dtype=np.int32),
        metadata_failed_frames=np.asarray(metadata_failed, dtype=np.int32),
    )
