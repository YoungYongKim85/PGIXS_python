"""H5 detector series loading for PGIXS Tab1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from pgixs_python.io.h5_loader import DEFAULT_DETECTOR_DATASET, H5RetryReport, load_h5_frame_with_retry
from pgixs_python.io.h5_metadata import H5Metadata, read_h5_metadata


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


def _load_one_frame(
    path: Path,
    dataset_path: str,
    frame_index: int,
    mask: np.ndarray | None,
    saturated_value: float,
    fallback_size: tuple[int, int] | None,
    retry_count: int,
    metadata_mode: str,
    reference_timestamp: float | None,
) -> tuple[np.ndarray, H5RetryReport, H5Metadata | None, bool, str]:
    """Load one frame's data plus its metadata -- the unit of work shared
    by both the serial loop and the parallel thread pool below."""

    data, retry_report = load_h5_frame_with_retry(
        path,
        dataset_path=dataset_path,
        frame_index=frame_index,
        mask=mask,
        saturated_value=saturated_value,
        fallback_size=fallback_size,
        retry_count=retry_count,
    )
    metadata = None
    metadata_ok = True
    metadata_message = ""
    try:
        metadata = read_h5_metadata(path, mode=metadata_mode, reference_timestamp=reference_timestamp)
    except Exception as exc:  # noqa: BLE001 - diagnostics must keep the original reason.
        metadata_ok = False
        metadata_message = str(exc)
    return data, retry_report, metadata, metadata_ok, metadata_message


def load_h5_series(
    file_paths: Iterable[str | Path],
    dataset_path: str = DEFAULT_DETECTOR_DATASET,
    frame_index: int = 0,
    mask: np.ndarray | None = None,
    saturated_value: float = np.nan,
    fallback_size: tuple[int, int] | None = None,
    retry_count: int = 1,
    metadata_mode: str = "",
    reference_timestamp: float | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> H5SeriesLoadResult:
    """Load a list of H5 detector files into a `(frame, row, col)` stack.

    Failed frame loads retry once by default, then become a NaN image when a
    fallback size is known. Frame and metadata failure sidecars use 1-based
    frame numbers to match existing MATLAB diagnostics.

    `progress_callback(completed, total, filename)`, if given, is called
    once per file right after it finishes loading (`completed` is 1-based)
    -- lets a caller drive a progress dialog (matches MATLAB
    `uiprogressdlg`'s per-file update in `Loadingdata()`) without this
    module knowing anything about UI. Under `parallel=True`, files finish
    out of load order, so `completed` counts *files done so far*, not a
    specific frame index -- matches MATLAB's own `parfor` path, which
    drops per-file progress for an indeterminate spinner instead.

    `parallel=True` loads files concurrently on a `ThreadPoolExecutor`
    (matches MATLAB's `ParallelCom`/`parfor` toggle in `Loadingdata()`) --
    file I/O and each frame's own decompression are the dominant cost, and
    HDF5's C library is thread-safe for concurrent reads of *different*
    files through separate handles, which is what each worker opens.
    Output order is always by input order regardless of completion order.
    """

    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("load_h5_series requires at least one H5 file.")

    if fallback_size is None and mask is not None:
        fallback_size = tuple(int(v) for v in np.asarray(mask).shape)  # type: ignore[assignment]

    frames: list[np.ndarray | None] = [None] * len(paths)
    records: list[H5FrameRecord | None] = [None] * len(paths)
    counts = np.full((len(paths), 2), np.nan, dtype=np.float64)
    timestamps = np.full(len(paths), np.nan, dtype=np.float64)
    temperatures = np.full((len(paths), 3), np.nan, dtype=np.float64)
    retry_failed: list[int] = []
    metadata_failed: list[int] = []
    known_size = fallback_size

    def _store(index0: int, path: Path, data: np.ndarray, retry_report, metadata, metadata_ok, metadata_message) -> None:
        frames[index0] = data
        records[index0] = H5FrameRecord(
            file_path=path,
            frame_number=index0 + 1,
            ok=retry_report.ok,
            retry_report=retry_report,
            metadata=metadata,
            metadata_ok=metadata_ok,
            metadata_message=metadata_message,
        )
        if not retry_report.ok:
            retry_failed.append(index0 + 1)
        if metadata is not None:
            counts[index0, 0] = metadata.counter1
            counts[index0, 1] = metadata.counter_bc
            timestamps[index0] = metadata.timestamp
            temperatures[index0, :] = metadata.temperature
        if not metadata_ok:
            metadata_failed.append(index0 + 1)

    if not parallel or len(paths) < 2:
        for index0, path in enumerate(paths):
            data, retry_report, metadata, metadata_ok, metadata_message = _load_one_frame(
                path, dataset_path, frame_index, mask, saturated_value, known_size, retry_count, metadata_mode, reference_timestamp
            )
            if known_size is None:
                known_size = tuple(int(v) for v in data.shape)
            elif tuple(data.shape) != tuple(known_size):
                raise ValueError(
                    f"H5 frame shape mismatch at frame {index0 + 1}: expected {known_size}, got {tuple(data.shape)}"
                )
            _store(index0, path, data, retry_report, metadata, metadata_ok, metadata_message)
            if progress_callback is not None:
                progress_callback(index0 + 1, len(paths), path.name)
    else:
        # A NaN-fallback shape must be known before dispatching workers --
        # unlike the serial loop, they can't learn it from each other as
        # they go, since they may run in any order. Load the first file up
        # front (synchronously) when no shape is known yet; the rest go to
        # the pool.
        start_index0 = 0
        if known_size is None:
            data, retry_report, metadata, metadata_ok, metadata_message = _load_one_frame(
                paths[0], dataset_path, frame_index, mask, saturated_value, None, retry_count, metadata_mode, reference_timestamp
            )
            known_size = tuple(int(v) for v in data.shape)
            _store(0, paths[0], data, retry_report, metadata, metadata_ok, metadata_message)
            if progress_callback is not None:
                progress_callback(1, len(paths), paths[0].name)
            start_index0 = 1

        completed = start_index0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _load_one_frame, path, dataset_path, frame_index, mask, saturated_value, known_size,
                    retry_count, metadata_mode, reference_timestamp,
                ): index0
                for index0, path in enumerate(paths)
                if index0 >= start_index0
            }
            for future in as_completed(future_to_index):
                index0 = future_to_index[future]
                data, retry_report, metadata, metadata_ok, metadata_message = future.result()
                if tuple(data.shape) != tuple(known_size):
                    raise ValueError(
                        f"H5 frame shape mismatch at frame {index0 + 1}: expected {known_size}, got {tuple(data.shape)}"
                    )
                _store(index0, paths[index0], data, retry_report, metadata, metadata_ok, metadata_message)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(paths), paths[index0].name)

    stack = np.stack(frames, axis=0).astype(np.float32, copy=False)  # type: ignore[arg-type]
    return H5SeriesLoadResult(
        stack=stack,
        records=records,  # type: ignore[arg-type]
        counts=counts,
        timestamps=timestamps,
        temperatures=temperatures,
        retry_failed_frames=np.asarray(sorted(retry_failed), dtype=np.int32),
        metadata_failed_frames=np.asarray(sorted(metadata_failed), dtype=np.int32),
    )
