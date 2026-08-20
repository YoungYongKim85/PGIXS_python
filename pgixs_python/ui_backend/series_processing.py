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

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pgixs_python.averaging.angular import angular_profile
from pgixs_python.averaging.cake import accumulation_average, cake_average


@dataclass
class SeriesCurveResult:
    x_axis: np.ndarray
    intensity: np.ndarray  # (n_frames, n_bins)
    coverage: np.ndarray  # (n_frames, n_bins)


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
    progress_callback: Callable[[int, int], None] | None = None,
) -> SeriesCurveResult:
    """1D Plot Process across a whole series -- same dispatch as
    `console_session.console_line_1d` (plain angle cut / Cakecut wedge /
    Accumulation band), run once per frame.

    `progress_callback(completed, total)`, when given, is called after
    every frame -- lets a caller drive a progress dialog for a real series
    (hundreds to thousands of frames), same convention as
    `load_h5_series`'s own `progress_callback`.

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
    x_axis: np.ndarray | None = np.asarray(ref_axis, dtype=np.float32) if ref_axis is not None else None
    intensity_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []

    for i in range(n_frames):
        frame = stack[i]
        frame_q_map = q_map_stack[i] if change_ai else q_map
        if accumulation:
            perp_offset = z_pos_move if angle_deg in (0.0, 180.0) else y_pos_move
            result = accumulation_average(
                frame, frame_q_map, center=center, angle_deg=angle_deg,
                perp_offset=perp_offset, band_minus=accum_minus, band_plus=accum_plus, use_median=use_median,
            )
        elif cakecut:
            result = cake_average(
                frame, frame_q_map, angle_map, center=center, reference_angle=angle_deg,
                angle_range=(range_minus, range_plus), use_median=use_median,
            )
        elif axis_aligned and (y_pos_move != 0.0 or z_pos_move != 0.0):
            perp_offset = z_pos_move if angle_deg in (0.0, 180.0) else y_pos_move
            result = accumulation_average(
                frame, frame_q_map, center=center, angle_deg=angle_deg,
                perp_offset=perp_offset, band_minus=0.0, band_plus=0.0, use_median=use_median,
            )
        else:
            result = cake_average(
                frame, frame_q_map, angle_map, center=center, reference_angle=angle_deg,
                angle_range=(-1.0, 1.0), use_median=use_median,
            )

        if not result.is_valid:
            if change_ai:
                # Matches alignLineToRef's own tolerance for a dropped
                # frame: an all-NaN row rather than aborting the series.
                if x_axis is None:
                    continue
                intensity_rows.append(np.full(x_axis.shape, np.nan, dtype=np.float32))
                coverage_rows.append(np.full(x_axis.shape, np.nan, dtype=np.float32))
                continue
            raise ValueError(f"No pixels selected for this line/cake-cut geometry at frame {i + 1}.")

        if x_axis is None:
            x_axis = result.q_axis
            intensity_rows.append(result.intensity)
            coverage_rows.append(result.coverage)
        elif change_ai:
            intensity_rows.append(_align_to_reference(result.q_axis, result.intensity, x_axis))
            coverage_rows.append(_align_to_reference(result.q_axis, result.coverage, x_axis))
        elif result.q_axis.shape != x_axis.shape:
            raise ValueError(
                f"Frame {i + 1} produced a {result.q_axis.shape[0]}-point curve, "
                f"expected {x_axis.shape[0]} (geometry must stay fixed across the series)."
            )
        else:
            intensity_rows.append(result.intensity)
            coverage_rows.append(result.coverage)

        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    if x_axis is None:
        raise ValueError("No pixels selected for this line/cake-cut geometry in any frame.")
    return SeriesCurveResult(x_axis=x_axis, intensity=np.vstack(intensity_rows), coverage=np.vstack(coverage_rows))


def process_azimuthal_stack(
    stack: np.ndarray,
    center: tuple[float, float],
    radius_range: tuple[float, float],
    n_angle_bins: int = 180,
    use_median: bool = False,
    angle_range: tuple[float, float] = (0.0, 180.0),
) -> SeriesCurveResult:
    """Azimuthal Process across a whole series -- same as
    `console_session.console_azimuthal_1d`, run once per frame."""

    n_frames = stack.shape[0]
    x_axis: np.ndarray | None = None
    intensity_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []

    for i in range(n_frames):
        result = angular_profile(
            stack[i], center=center, radius_range=radius_range,
            n_angle_bins=n_angle_bins, use_median=use_median, angle_range=angle_range,
        )
        if x_axis is None:
            x_axis = result.angle_axis
        elif result.angle_axis.shape != x_axis.shape:
            raise ValueError(
                f"Frame {i + 1} produced a {result.angle_axis.shape[0]}-point curve, "
                f"expected {x_axis.shape[0]} (geometry must stay fixed across the series)."
            )
        intensity_rows.append(result.intensity)
        coverage_rows.append(result.coverage)

    assert x_axis is not None
    return SeriesCurveResult(x_axis=x_axis, intensity=np.vstack(intensity_rows), coverage=np.vstack(coverage_rows))


def process_roi_stack(stack: np.ndarray, rois: list[tuple[int, int, int, int]]) -> np.ndarray:
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
        with np.errstate(all="ignore"):
            sums[:, j] = np.nansum(region, axis=(1, 2))
    return sums


_TIME_UNIT_DIVISORS = {"second": 1.0, "min": 60.0, "hour": 3600.0}


def _time_axis(timestamps: np.ndarray, time_offset_seconds: float, time_divisor: float) -> tuple[np.ndarray, str]:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size and np.all(np.isfinite(ts)):
        # `timestamps` is expected already baseline-relative (load_h5_series's
        # own `reference_timestamp` param, seeded from the first frame) --
        # matches MATLAB's `timestamps - timestamps(1)` done once at compute
        # time; doing it at load time instead keeps this function simple.
        axis = (ts + time_offset_seconds) / time_divisor
    else:
        n = ts.size
        axis = (np.arange(n, dtype=np.float64) * time_offset_seconds + time_offset_seconds) / time_divisor
    return axis, "Time"


def compute_experiment_x_axis(
    exp_mode: str,
    timestamps: np.ndarray,
    temperatures: np.ndarray,
    *,
    sub_mode: str = "",
    time_offset_seconds: float = 0.0,
    time_unit: str = "second",
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
    """

    time_divisor = _TIME_UNIT_DIVISORS.get(time_unit, 1.0)

    if exp_mode == "Time" or sub_mode == "Time":
        return _time_axis(timestamps, time_offset_seconds, time_divisor)

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
    return temps[:, column], label


def sort_x_axis(values: np.ndarray, should_sort: bool) -> np.ndarray:
    """Order indices for `values`, matching `sortPGIXSXAxis.m`: the
    T.Order/A.Order checkbox is a plain on/off for an ascending numeric
    sort -- off leaves the series in its original file-load order
    (identity indices), there is no separate "descending" option."""

    if not should_sort:
        return np.arange(len(values))
    return np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
