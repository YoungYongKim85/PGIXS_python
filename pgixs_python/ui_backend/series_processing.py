"""Series-orchestration layer for GIXS2 (and later TrXS2): loops the
existing single-frame averaging primitives (`cake_average`,
`accumulation_average`, `angular_profile`) across every frame of an
in-memory detector stack, and builds/sorts the experiment X-axis
(Time/Temperature/Angle) those per-frame curves get plotted against.

Ports the *orchestration* MATLAB `processGIXSLinecutStack` /
`processGIXSAzimuthalStack` / `processPGIXSROIStack` /
`computePGIXSExperimentXAxis` / `sortPGIXSXAxis` perform -- the actual
per-frame geometry/averaging math is unchanged from the single-frame
console (`console_session.py`), which is already validated against
MATLAB. `process_linecut_stack` supports both static geometry (one
q-map/angle-map for the whole series) and "Change ai" angle-scanning
geometry (a per-frame q-map, via `q_map_stack`); `process_azimuthal_stack`/
`process_roi_stack` are pixel-space-only and never need a per-frame q-map.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pgixs_python.averaging.angular import angular_profile, make_sector_mask
from pgixs_python.averaging.cake import CakeAverageResult, accumulation_average, cake_average
from pgixs_python.averaging.radial import RadiusBinMap, build_radius_bin_map, radius_average
from pgixs_python.processing.corrections import apply_detector_correction
from pgixs_python.processing.grid_remove import grid_remove


def apply_correction_to(
    values: np.ndarray,
    correction_factor: np.ndarray | None,
    gap_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Divide by the detector correction factor, or pass through when there
    is none.

    Series q-Cal keeps only the factor (one detector-sized array) rather than
    a fully corrected copy of the stack, so the correction is applied here, on
    the one frame (or ROI sub-block) actually being reduced. `correction_factor`
    broadcasts over any leading frame axis; the NaN-preserving semantics match
    `apply_detector_correction`, which the single-frame consoles still use.
    "Change ai" is the exception -- its factor differs per frame, so that path
    materializes the corrected stack up front and passes `None` here.

    `gap_buffer` (detector-shaped boolean, from `dilate_gap_mask`) blanks the
    pixels bordering a module gap, which can carry biased counts. They stay in
    the geometry so a bin sitting mostly on a gap edge reads as poorly sampled
    rather than as fully sampled from fewer pixels. ROI sums deliberately do
    not pass it -- MATLAB applies the buffer only in its averaging paths."""

    if correction_factor is None and gap_buffer is None:
        return values
    array = np.asarray(values, dtype=np.float32)
    if gap_buffer is not None and gap_buffer.shape == array.shape[-2:]:
        array = array.copy()
        array[..., gap_buffer] = np.nan
    if correction_factor is None:
        return array
    factor = np.asarray(correction_factor, dtype=np.float32)
    if array.shape == factor.shape:
        return apply_detector_correction(array, factor)
    broadcast = np.broadcast_to(factor, array.shape)
    out = np.full(array.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(array) & np.isfinite(broadcast) & (broadcast != 0)
    out[valid] = array[valid] / broadcast[valid]
    return out


@dataclass
class SeriesCurveResult:
    x_axis: np.ndarray
    intensity: np.ndarray  # (n_frames, n_bins)
    coverage: np.ndarray  # (n_frames, n_bins)


@dataclass
class SeriesCircularResult:
    x_axis: np.ndarray
    intensity: np.ndarray  # (n_frames, n_bins)
    coverage: np.ndarray  # (n_frames, n_bins)
    std: np.ndarray | None  # (n_frames, n_bins), or None when calc_std=False


def grid_remove_stack(stack: np.ndarray, detector: str) -> tuple[np.ndarray | None, str]:
    """Grid-remove every display-orientation frame on its own (each frame's
    own reference-pixel ratio). `grid_remove` takes the `load_h5_frame`
    orientation, hence the transposes. Returns (None, message) if any frame
    can't be corrected, so a series never ends up half-corrected."""

    frames = []
    message = ""
    for frame in stack:
        corrected, report = grid_remove(frame.T, detector)
        if not report.applied:
            return None, report.message or "Grid remove is not applicable."
        message = report.message
        frames.append(np.ascontiguousarray(corrected.T))
    return np.stack(frames, axis=0), message


def counter_reference(counts: np.ndarray | None, method: str) -> float:
    """The first frame's IC/BC counter, which MATLAB divides every counter
    by (`normalizePGIXSCounters` with `RefIC`/`RefBC` read from files{1}),
    so a normalized curve stays on the scale of the first frame's raw one.
    1 when the first frame has no usable counter (MATLAB keeps the raw
    counter then)."""

    if counts is None or np.shape(counts)[0] == 0:
        return 1.0
    value = float(counts[0, 0 if method == "IC" else 1])
    return value if np.isfinite(value) and value != 0 else 1.0


def normalize_rows_by_counter(matrix: np.ndarray, counts: np.ndarray | None, method: str) -> np.ndarray:
    """Divide each frame's row by its IC ("IC") or BC (otherwise) counter,
    taken relative to the first frame's (`counter_reference`).
    A frame without a usable counter (missing, NaN or 0) comes out all-NaN
    rather than raw: a raw row next to normalized ones is off by the
    counter value."""

    out = np.full(np.shape(matrix), np.nan, dtype=np.float32)
    if counts is None:
        return out
    column = 0 if method == "IC" else 1
    reference = counter_reference(counts, method)
    for i, row in enumerate(np.asarray(matrix, dtype=np.float32)):
        count = float(counts[i, column]) if i < counts.shape[0] else float("nan")
        if np.isfinite(count) and count != 0:
            out[i] = row / np.float32(count / reference)
    return out


def _match_to_reference(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
    """Place a static-geometry frame's curve onto the shared axis by exact Q
    value, NaN where this frame has no point. Accumulation cuts drop a
    column whose band reads all-NaN in that one frame, so frames can share
    a length yet not a Q axis; every Q value still comes from the same
    q-map, so exact matching is safe (no interpolation across the gap)."""

    lookup = dict(zip(np.asarray(x).tolist(), np.asarray(y).tolist()))
    return np.array([lookup.get(v, np.nan) for v in np.asarray(ref_x).tolist()], dtype=np.float32)


def _align_to_reference(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
    """Port of `alignLineToRef.m`: resample a (possibly differently-Q-valued)
    frame's own curve onto the shared reference axis by linear
    interpolation, `NaN` outside its own range. Used only in "Change ai"
    mode, where each frame's own Q-axis differs slightly (the same pixel
    radius maps to a different Q as the incident angle steps)."""

    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.full(ref_x.shape, np.nan, dtype=np.float32)
    ux, idx = np.unique(x[finite], return_index=True)
    uy = y[finite][idx]
    if ux.size < 2:
        return np.full(ref_x.shape, np.nan, dtype=np.float32)
    return np.interp(ref_x, ux, uy, left=np.nan, right=np.nan).astype(np.float32)


def process_linecut_stack(
    stack: np.ndarray,
    q_map: np.ndarray | None,
    angle_map: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
    cakecut: bool = False,
    range_minus: float = -1.0,
    range_plus: float = 1.0,
    use_median: bool = False,
    y_pos_move: float = 0.0,
    z_pos_move: float = 0.0,
    accumulation: bool = False,
    accum_minus: float = -1.0,
    accum_plus: float = 1.0,
    q_map_stack: np.ndarray | None = None,
    ref_axis: np.ndarray | None = None,
    correction_factor: np.ndarray | None = None,
    gap_buffer: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> SeriesCurveResult:
    """1D Plot Process across a whole series -- same dispatch as
    `console_session.console_line_1d` (plain angle cut / Cakecut wedge /
    Accumulation band), run once per frame.

    `progress_callback(completed, total)`, when given, is called after
    every frame -- lets a caller drive a progress dialog for a real series
    (hundreds to thousands of frames), same convention as
    `load_h5_series`'s own `progress_callback`.

    `parallel=True` runs each frame's own `cake_average`/
    `accumulation_average` call on a `ThreadPoolExecutor` (same toggle
    convention as `load_h5_series`'s own `parallel` option) -- only that
    per-frame compute step is parallelized; the reference-axis selection/
    alignment pass below still runs single-threaded, in frame order, over
    the completed results, so the output is bit-for-bit identical to the
    serial path regardless of completion order.

    `q_map_stack`, an optional `(n_frames, rows, cols)` array, is "Change
    ai" mode (port of `processGIXSLinecutStack.m`'s `ChangeAngle` path):
    each frame gets its *own* Q-map (a different incident angle maps the
    same pixel radius to a different Q), so frame 0's own Q-axis becomes
    the shared reference and every other frame's curve is linearly
    interpolated onto it (`alignLineToRef.m`) rather than requiring an
    exact shape/value match. Static mode (`q_map_stack=None`) keeps the
    original fast path: one shared Q-map, one shared axis, no interpolation
    needed since every frame's Q-axis is then identical by construction.
    `angle_map` (purely pixel-geometry, independent of Q/incident angle)
    never varies per frame either way.

    `ref_axis`, when given, is used as the shared axis instead of deriving
    it from this call's own first valid frame (port of `opts.RefAxis`):
    Auto Refresh, in Change-ai mode, must align newly-appended frames onto
    the *original* series' reference axis, not a fresh one built from just
    the new frames (whose own first frame's Q-axis has already drifted
    away from the series' first frame, since `ai` keeps stepping).
    """

    axis_aligned = angle_deg in (0.0, 90.0, 180.0)
    if accumulation and not axis_aligned:
        raise ValueError("Accumulation cut only supports angle 0, 90, or 180 degrees.")

    n_frames = stack.shape[0]
    change_ai = q_map_stack is not None
    if not change_ai and q_map is None:
        raise ValueError("q_map is required unless q_map_stack is given.")

    def _run_frame(i: int) -> CakeAverageResult:
        frame = apply_correction_to(stack[i], correction_factor, gap_buffer)
        frame_q_map = q_map_stack[i] if change_ai else q_map
        if accumulation:
            perp_offset = z_pos_move if angle_deg in (0.0, 180.0) else y_pos_move
            return accumulation_average(
                frame, frame_q_map, center=center, angle_deg=angle_deg,
                perp_offset=perp_offset, band_minus=accum_minus, band_plus=accum_plus, use_median=use_median,
            )
        if cakecut:
            return cake_average(
                frame, frame_q_map, angle_map, center=center, reference_angle=angle_deg,
                angle_range=(range_minus, range_plus), use_median=use_median,
            )
        if axis_aligned and (y_pos_move != 0.0 or z_pos_move != 0.0):
            perp_offset = z_pos_move if angle_deg in (0.0, 180.0) else y_pos_move
            return accumulation_average(
                frame, frame_q_map, center=center, angle_deg=angle_deg,
                perp_offset=perp_offset, band_minus=0.0, band_plus=0.0, use_median=use_median,
            )
        return cake_average(
            frame, frame_q_map, angle_map, center=center, reference_angle=angle_deg,
            angle_range=(-1.0, 1.0), use_median=use_median,
        )

    results: list[CakeAverageResult] = [None] * n_frames  # type: ignore[list-item]
    if parallel and n_frames > 1:
        progress_lock = threading.Lock()
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_frame, i): i for i in range(n_frames)}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                if progress_callback is not None:
                    with progress_lock:
                        completed += 1
                        progress_callback(completed, n_frames)
    else:
        for i in range(n_frames):
            results[i] = _run_frame(i)
            if progress_callback is not None:
                progress_callback(i + 1, n_frames)

    x_axis: np.ndarray | None = np.asarray(ref_axis, dtype=np.float32) if ref_axis is not None else None
    if x_axis is None and not change_ai:
        valid = [r for r in results if r.is_valid]
        if valid and any(not np.array_equal(r.q_axis, valid[0].q_axis) for r in valid[1:]):
            # Frames disagree on which points they kept: use the union of
            # their Q values, ordered by distance from the beam like each cut.
            r_of_q: dict[float, float] = {}
            for r in valid:
                r_of_q.update(zip(r.q_axis.tolist(), r.r_axis.tolist()))
            x_axis = np.asarray(sorted(r_of_q, key=r_of_q.__getitem__), dtype=np.float32)
    intensity_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []
    # Change-ai frames invalid before the first valid one: their NaN rows
    # can only be sized once that frame fixes the axis.
    n_leading_invalid = 0

    for i, result in enumerate(results):
        if not result.is_valid:
            if change_ai:
                # Matches alignLineToRef's own tolerance for a dropped
                # frame: an all-NaN row rather than aborting the series.
                if x_axis is None:
                    n_leading_invalid += 1
                    continue
                intensity_rows.append(np.full(x_axis.shape, np.nan, dtype=np.float32))
                coverage_rows.append(np.full(x_axis.shape, np.nan, dtype=np.float32))
                continue
            raise ValueError(f"No pixels selected for this line/cake-cut geometry at frame {i + 1}.")

        if x_axis is None:
            x_axis = result.q_axis
            intensity_rows.extend([np.full(x_axis.shape, np.nan, dtype=np.float32)] * n_leading_invalid)
            coverage_rows.extend([np.full(x_axis.shape, np.nan, dtype=np.float32)] * n_leading_invalid)
            intensity_rows.append(result.intensity)
            coverage_rows.append(result.coverage)
        elif change_ai:
            intensity_rows.append(_align_to_reference(result.q_axis, result.intensity, x_axis))
            coverage_rows.append(_align_to_reference(result.q_axis, result.coverage, x_axis))
        elif not np.array_equal(result.q_axis, x_axis):
            intensity_rows.append(_match_to_reference(result.q_axis, result.intensity, x_axis))
            coverage_rows.append(_match_to_reference(result.q_axis, result.coverage, x_axis))
        else:
            intensity_rows.append(result.intensity)
            coverage_rows.append(result.coverage)

    if x_axis is None:
        raise ValueError("No pixels selected for this line/cake-cut geometry in any frame.")
    return SeriesCurveResult(x_axis=x_axis, intensity=np.vstack(intensity_rows), coverage=np.vstack(coverage_rows))


def build_circular_radius_map(
    q_map: np.ndarray,
    center: tuple[float, float],
    angle_range: tuple[float, float] = (0.0, 360.0),
    mask: np.ndarray | None = None,
) -> RadiusBinMap:
    """Precompute the once-per-series radius bin map Circular Average loops
    across every frame of a stack -- port of `buildPGIXSRadiusBinMap`'s own
    call site in `CircularAve2_Callback` (MATLAB's `radiusPrep`, explicitly
    built once and reused for speed rather than recomputed per frame).

    `angle_range` restricts the sector the same way `make_sector_mask`
    does elsewhere in this codebase (equal start/end means full circle);
    `mask`, when given, is a user-drawn ROI mask (True = keep pixel,
    matching `console_set_circular_mask`'s convention) ANDed onto the
    sector -- both are combined with `isfinite(q) & q > 0` before the one
    `build_radius_bin_map` call, matching `CircularAve2_Callback`'s own
    `geom_valid = sector_mask & (BW > 0) & isfinite(param.Q) & (param.Q > 0)`.
    """

    q = np.asarray(q_map, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("q_map must be a 2D array.")

    geom_valid = make_sector_mask(q.shape, center, angle_range)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != q.shape:
            raise ValueError("mask must match q_map shape.")
        geom_valid = geom_valid & mask_arr

    return build_radius_bin_map(q, center, geom_valid)


def process_circular_stack(
    stack: np.ndarray,
    radius_map: RadiusBinMap,
    use_median: bool = False,
    calc_std: bool = False,
    min_coverage: float = 0.80,
    progress_callback: Callable[[int, int], None] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    correction_factor: np.ndarray | None = None,
    gap_buffer: np.ndarray | None = None,
) -> SeriesCircularResult:
    """Circular Average across a whole series -- same q-binned average as
    `console_session.console_circular_average`, run once per frame against
    a single precomputed `radius_map` (built once by
    `build_circular_radius_map`, not rebuilt per frame -- the speed-critical
    "radiusPrep" reuse `CircularAve2_Callback` itself relies on).

    Per-frame bins below `min_coverage` are nulled to NaN, matching
    `console_circular_average`'s own coverage-threshold behavior (MATLAB's
    `CircularAve2_Callback` has no such threshold -- this mirrors the
    single-frame TrXS console's own already-established coverage handling
    instead, a deliberate simplification/enhancement over the literal
    MATLAB port, same as `console_circular_average`'s docstring already
    documents).

    `progress_callback`/`parallel`/`max_workers` follow the same
    conventions as `process_linecut_stack`'s own.
    """

    if not radius_map.is_valid:
        raise ValueError("No valid radius bins for this circular-average geometry.")

    n_frames = stack.shape[0]

    def _run_frame(i: int):
        return radius_average(
            apply_correction_to(stack[i], correction_factor, gap_buffer), radius_map, use_median=use_median, calc_std=calc_std
        )

    results = [None] * n_frames
    if parallel and n_frames > 1:
        progress_lock = threading.Lock()
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_frame, i): i for i in range(n_frames)}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                if progress_callback is not None:
                    with progress_lock:
                        completed += 1
                        progress_callback(completed, n_frames)
    else:
        for i in range(n_frames):
            results[i] = _run_frame(i)
            if progress_callback is not None:
                progress_callback(i + 1, n_frames)

    intensity_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []
    std_rows: list[np.ndarray] | None = [] if calc_std else None

    for result in results:
        intensity = np.asarray(result.intensity, dtype=np.float32).copy()
        coverage = np.asarray(result.coverage, dtype=np.float32)
        below = np.isfinite(coverage) & (coverage < min_coverage)
        intensity[below] = np.nan
        intensity_rows.append(intensity)
        coverage_rows.append(coverage)
        if calc_std:
            std_rows.append(result.std)

    return SeriesCircularResult(
        x_axis=radius_map.q_axis,
        intensity=np.vstack(intensity_rows),
        coverage=np.vstack(coverage_rows),
        std=np.vstack(std_rows) if calc_std else None,
    )


def process_azimuthal_stack(
    stack: np.ndarray,
    center: tuple[float, float],
    radius_range: tuple[float, float],
    n_angle_bins: int = 180,
    use_median: bool = False,
    angle_range: tuple[float, float] = (0.0, 180.0),
    progress_callback: Callable[[int, int], None] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    correction_factor: np.ndarray | None = None,
    gap_buffer: np.ndarray | None = None,
) -> SeriesCurveResult:
    """Azimuthal Process across a whole series -- same as
    `console_session.console_azimuthal_1d`, run once per frame.

    `progress_callback(completed, total)`, when given, is called after
    every frame -- same convention as `process_linecut_stack`'s own.

    `parallel`/`max_workers` follow the exact same `ThreadPoolExecutor`
    convention as `process_linecut_stack`/`process_circular_stack`: only
    the per-frame `angular_profile` call is parallelized, and results are
    reassembled below in frame order, so the output is bit-for-bit
    identical to the serial path regardless of completion order."""

    n_frames = stack.shape[0]

    def _run_frame(i: int):
        return angular_profile(
            apply_correction_to(stack[i], correction_factor, gap_buffer), center=center, radius_range=radius_range,
            n_angle_bins=n_angle_bins, use_median=use_median, angle_range=angle_range,
        )

    results = [None] * n_frames
    if parallel and n_frames > 1:
        progress_lock = threading.Lock()
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_frame, i): i for i in range(n_frames)}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                if progress_callback is not None:
                    with progress_lock:
                        completed += 1
                        progress_callback(completed, n_frames)
    else:
        for i in range(n_frames):
            results[i] = _run_frame(i)
            if progress_callback is not None:
                progress_callback(i + 1, n_frames)

    x_axis: np.ndarray | None = None
    intensity_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []
    for i, result in enumerate(results):
        if x_axis is None:
            x_axis = result.angle_axis
        elif result.angle_axis.shape != x_axis.shape:
            raise ValueError(
                f"Frame {i + 1} produced a {result.angle_axis.shape[0]}-point curve, "
                f"expected {x_axis.shape[0]} (geometry must stay fixed across the series)."
            )
        intensity_rows.append(result.intensity)
        coverage_rows.append(result.coverage)

    if x_axis is None:
        raise ValueError("Azimuthal Process needs at least one frame.")
    return SeriesCurveResult(x_axis=x_axis, intensity=np.vstack(intensity_rows), coverage=np.vstack(coverage_rows))


def process_roi_stack(
    stack: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    correction_factor: np.ndarray | None = None,
) -> np.ndarray:
    """Region-of-Interest sums across a whole series.

    `rois` are `(r1, r2, c1, c2)` 1-based inclusive pixel-index rectangles,
    matching MATLAB `param.ROI(:,:) = [r1 r2 c1 c2]`. Returns an
    `(n_frames, n_roi)` matrix of `nansum` over each rectangle per frame.
    """

    n_frames, n_rows, n_cols = stack.shape
    sums = np.zeros((n_frames, len(rois)), dtype=np.float64)
    for j, (r1, r2, c1, c2) in enumerate(rois):
        row_start, row_end = max(1, r1), min(n_rows, r2)
        col_start, col_end = max(1, c1), min(n_cols, c2)
        if row_start > row_end or col_start > col_end:
            sums[:, j] = np.nan
            continue
        region = stack[:, row_start - 1 : row_end, col_start - 1 : col_end]
        if correction_factor is not None:
            region = apply_correction_to(
                region, correction_factor[row_start - 1 : row_end, col_start - 1 : col_end]
            )
        with np.errstate(all="ignore"):
            sums[:, j] = np.nansum(region, axis=(1, 2))
    return sums


_TIME_UNIT_DIVISORS = {"second": 1.0, "min": 60.0, "hour": 3600.0}
_TIME_UNIT_ABBREV = {"second": "s", "min": "min", "hour": "h"}


def average_exposure_seconds(timestamps: np.ndarray) -> float:
    """Average interval between consecutive frame timestamps -- stands in
    for each frame's own exposure duration, which isn't a field the H5
    metadata actually records. A back-to-back series' frame-to-frame
    interval is essentially its exposure time; averaging across the whole
    series (rather than trusting just the first interval) is more robust
    against the first frame's own timing being atypical (shutter warm-up,
    a longer/shorter first exposure, clock jitter) -- exactly the case
    where a single data point "may not know the real exposure time"."""

    ts = np.asarray(timestamps, dtype=np.float64)
    finite = np.sort(ts[np.isfinite(ts)])
    if finite.size < 2:
        return 0.0
    diffs = np.diff(finite)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs))


# MATLAB's Exp. time / time-offset field (`GX2.hExptime`, `TX2.hExptime`)
# starts at 1.
DEFAULT_TIME_OFFSET = 1.0


def _time_axis(
    timestamps: np.ndarray,
    time_offset: float | None,
    time_divisor: float,
    time_unit: str,
    offset_after_unit: bool = False,
) -> tuple[np.ndarray, str]:
    """`computePGIXSExperimentXAxis.m`'s time axis. GIXS adds the offset in
    seconds before the unit change, `(dt + ExpTime) / unit`; TrXS
    (`offset_after_unit`) adds it in the display unit, `dt / unit + offset`."""

    ts = np.asarray(timestamps, dtype=np.float64)
    offset = DEFAULT_TIME_OFFSET if time_offset is None else float(time_offset)
    # Missing timestamps are filled with the series' own frame interval.
    interval = average_exposure_seconds(ts)
    finite = np.isfinite(ts)
    if not finite.any():
        # No readable timestamp at all: the frame number is the only honest
        # axis (a ramp built from a 0 s average interval was all zeros).
        return np.arange(1, ts.size + 1, dtype=np.float64), "Frame"
    if not finite.all():
        # Fill only the frames whose timestamp is missing -- from their
        # neighbours, or the average interval past either end -- instead of
        # throwing away every real timestamp (and its gaps) for a ramp.
        index = np.arange(ts.size, dtype=np.float64)
        known_index, known = index[finite], ts[finite]
        filled = np.interp(index, known_index, known)
        before, after = index < known_index[0], index > known_index[-1]
        filled[before] = known[0] - (known_index[0] - index[before]) * interval
        filled[after] = known[-1] + (index[after] - known_index[-1]) * interval
        ts = filled
    # Normally already baseline-relative (the session subtracts the first
    # frame's timestamp at load time, matching MATLAB's
    # `timestamps - timestamps(1)`), so this is a no-op; it only matters when
    # the first frame had none and the rest are still absolute epoch times.
    ts = ts - ts[finite][0]
    axis = ts / time_divisor + offset if offset_after_unit else (ts + offset) / time_divisor
    label = f"Time, {_TIME_UNIT_ABBREV.get(time_unit, time_unit)}"
    return axis, label


def compute_experiment_x_axis(
    exp_mode: str,
    timestamps: np.ndarray,
    temperatures: np.ndarray,
    *,
    sub_mode: str = "",
    time_offset_seconds: float | None = None,
    time_unit: str = "second",
    offset_after_unit: bool = False,
) -> tuple[np.ndarray, str]:
    """Port of `computePGIXSExperimentXAxis.m`'s GIXS branch: build the
    series X-axis for whichever insitu mode (`exp_mode`: "Time" /
    "Temperature" / "LowTemp." / "Angle") and sub-mode (`sub_mode`, the
    `Tmode`/`Amode` selector: "Temperature1"/"2"/"3", "LowTemp1"/"2", or
    "Time" to fall back to the time axis from within a Temperature/Angle
    series) is active. `temperatures` is the `(n_frames, 3)` array
    `load_h5_series`/`read_h5_metadata` already fill per insitu mode
    (Temp1-3 for "Temperature", LowTemp1-2 for "LowTemp.", or the per-file
    logged incidence angle in column 0 for "Angle" -- this last one is an
    intentional adaptation, not a literal MATLAB port: MATLAB's own
    "Angle" axis reads `param.aivar`, a *computed* angle ramp that only
    exists after the "Change ai" q-Cal correction (deferred to Phase 2);
    using the per-file logged angle here is the natural Phase-1-available
    equivalent for an angle-series experiment).

    `time_offset_seconds` is MATLAB's Exp. time field (default 1). It is
    seconds for GIXS; with `offset_after_unit` (TrXS) it is in `time_unit`
    and added after the unit change -- see `_time_axis`.
    """

    time_divisor = _TIME_UNIT_DIVISORS.get(time_unit, 1.0)

    if exp_mode == "Time" or sub_mode == "Time":
        return _time_axis(timestamps, time_offset_seconds, time_divisor, time_unit, offset_after_unit)

    temps = np.asarray(temperatures, dtype=np.float64)
    n_frames = temps.shape[0] if temps.ndim == 2 else 0

    if exp_mode == "Temperature":
        column, label = {"Temperature2": (1, "Temperature 2"), "Temperature3": (2, "Temperature 3")}.get(
            sub_mode, (0, "Temperature")
        )
    elif exp_mode == "LowTemp.":
        column, label = {"LowTemp2": (1, "Low Temperature 2")}.get(sub_mode, (0, "Low Temperature"))
    elif exp_mode == "Angle":
        column, label = 0, "Incidence Angle, deg"
    else:
        raise ValueError(f"Unknown x-axis mode: {exp_mode}")

    if temps.ndim != 2 or temps.shape[1] <= column or n_frames == 0:
        raise ValueError(f"Temperature data has no column for x-axis mode '{exp_mode}'/'{sub_mode}'.")
    values = temps[:, column]
    if not np.isfinite(values).any():
        # The files never logged this value (e.g. no Angle field): number the
        # frames, as MATLAB's computePGIXSExperimentXAxis falls back to
        # 1:nFrames, instead of an all-NaN axis that plots nothing.
        return np.arange(1, n_frames + 1, dtype=np.float64), "Frame"
    return values, label


def integrate_frames(matrix: np.ndarray, axis_values: np.ndarray, axis_range=None) -> np.ndarray:
    """Per-frame sum over the bins whose Q/angle lies in `axis_range`
    (lo, hi), as MATLAB's Integration plot and Save Integ. sum only between
    qMin-qMax / aziMin-aziMax. No range, or hi <= lo, sums every bin."""

    values = np.asarray(matrix, dtype=np.float64)
    if axis_range is not None:
        lo, hi = (float(v) for v in axis_range)
        if hi > lo:
            axis = np.asarray(axis_values, dtype=np.float64)
            values = values[:, (axis >= lo) & (axis <= hi)]
    return np.nansum(values, axis=1)


def normalize_at_point(matrix: np.ndarray, axis_values: np.ndarray, point: float) -> np.ndarray:
    """MATLAB's Norm(Pt): divide each frame (row) by its own value at the
    bin nearest `point`, so every curve passes through 1 there. A frame
    whose value there is NaN or 0 is left unscaled
    (`sanitizePGIXSScaleFactors`)."""

    values = np.asarray(matrix, dtype=np.float64)
    axis = np.asarray(axis_values, dtype=np.float64)
    index = int(np.nanargmin(np.abs(axis - float(point))))
    factors = values[:, index].copy()
    factors[~np.isfinite(factors) | (factors == 0)] = 1.0
    return values / factors[:, np.newaxis]


def frames_without_value_at(matrix: np.ndarray, axis_values: np.ndarray, point: float) -> int:
    """How many frames `normalize_at_point` leaves unscaled (NaN or 0 at
    the point, e.g. a detector gap), so the UI can say so."""

    values = np.asarray(matrix, dtype=np.float64)
    axis = np.asarray(axis_values, dtype=np.float64)
    column = values[:, int(np.nanargmin(np.abs(axis - float(point))))]
    return int(np.count_nonzero(~np.isfinite(column) | (column == 0)))


def norm_point_message(point: float, unscaled: int, n_frames: int) -> str:
    if unscaled == 0:
        return f"Norm(Pt): every frame scaled to 1 at Q = {point:.4g} 1/A."
    return (
        f"Norm(Pt) at Q = {point:.4g} 1/A: {unscaled} of {n_frames} frame(s) have no value there "
        "(gap or mask) and were left unscaled -- pick another Q."
    )


def nearest_axis_value(axis_values: np.ndarray, point: float) -> float:
    axis = np.asarray(axis_values, dtype=np.float64)
    return float(axis[int(np.nanargmin(np.abs(axis - float(point))))])


def sort_x_axis(values: np.ndarray, should_sort: bool) -> np.ndarray:
    """Order indices for `values`, matching `sortPGIXSXAxis.m`: the
    T.Order/A.Order checkbox is a plain on/off for an ascending numeric
    sort -- off leaves the series in its original file-load order
    (identity indices), there is no separate "descending" option."""

    if not should_sort:
        return np.arange(len(values))
    return np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
