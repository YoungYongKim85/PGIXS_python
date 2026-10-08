"""Normalization and background subtraction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizationReport:
    success: bool
    message: str = ""
    method: str = "None"


def normalize_by_count(data: np.ndarray, count: float | None) -> np.ndarray:
    """Normalize by a scalar count while preserving NaNs."""

    arr = np.asarray(data, dtype=np.float32)
    if count is None or not np.isfinite(count) or count == 0:
        return arr.copy()
    return arr / np.float32(count)


def normalize_counters(
    counts: np.ndarray | list[float] | tuple[float, ...],
    reference_counts: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Normalize IC/BC counters with safe fallbacks."""

    out = np.asarray(counts, dtype=np.float32).reshape(-1).copy()
    reference = np.asarray(reference_counts, dtype=np.float32).reshape(-1)
    for idx in range(out.size):
        if idx < reference.size and np.isfinite(reference[idx]) and reference[idx] != 0:
            out[idx] = out[idx] / reference[idx]
    return out


def normalization_value(
    source_values: np.ndarray | list[float] | tuple[float, ...],
    method: str,
    frame_index: int = 0,
) -> float:
    """Return the scalar normalization value for a method and optional frame index."""

    values = np.asarray(source_values, dtype=np.float32)
    if values.ndim == 0:
        return float(values)
    idx = 0 if _normalize_method(method) == "IC" else 1
    if values.ndim == 1:
        return float(values[idx]) if idx < values.size else float("nan")
    if idx >= values.shape[1] or frame_index >= values.shape[0]:
        return float("nan")
    return float(values[int(frame_index), idx])


def process_normalization(
    data: np.ndarray,
    background: np.ndarray | None = None,
    counts: np.ndarray | list[float] | tuple[float, ...] | None = None,
    background_counts: np.ndarray | list[float] | tuple[float, ...] | None = None,
    method: str = "None",
    plus_data: float = 1.0,
    clamp_negative: bool = True,
) -> tuple[np.ndarray, NormalizationReport]:
    """Apply background subtraction and IC/BC normalization."""

    image = np.asarray(data, dtype=np.float32)
    bkg = None if background is None else np.asarray(background, dtype=np.float32)
    if bkg is not None and bkg.shape != image.shape:
        raise ValueError("background must match data shape.")

    norm_method = _normalize_method(method)
    report = NormalizationReport(success=True, message="", method=norm_method)
    scale = np.float32(plus_data)

    if norm_method == "NONE":
        out = image - scale * bkg if bkg is not None else image.copy()
    elif norm_method in {"IC", "BC"}:
        idx = 0 if norm_method == "IC" else 1
        ok, message = _has_valid_counter(counts, idx, f"{norm_method} value is 0 or missing!")
        if not ok:
            return image.copy(), NormalizationReport(success=False, message=message, method=norm_method)
        count_value = np.float32(np.asarray(counts, dtype=np.float32).reshape(-1)[idx])
        if bkg is not None:
            ok, message = _has_valid_counter(background_counts, idx, f"Background {norm_method} value is 0!")
            if not ok:
                return image.copy(), NormalizationReport(success=False, message=message, method=norm_method)
            bkg_count = np.float32(np.asarray(background_counts, dtype=np.float32).reshape(-1)[idx])
            out = (image / count_value) - (scale * bkg / bkg_count)
        else:
            out = image / count_value
    else:
        # Reporting success here let a typo ("ic ", "I0") pass as a
        # normalized result.
        return image.copy(), NormalizationReport(
            success=False, message=f"Unknown normalization method: {method!r}.", method=norm_method
        )

    out = out.astype(np.float32, copy=False)
    if clamp_negative:
        out[out < 0] = 0
    return out, report


def _has_valid_counter(
    counts: np.ndarray | list[float] | tuple[float, ...] | None,
    index: int,
    message: str,
) -> tuple[bool, str]:
    if counts is None:
        return False, message
    arr = np.asarray(counts, dtype=np.float32).reshape(-1)
    ok = index < arr.size and np.isfinite(arr[index]) and arr[index] != 0
    return bool(ok), "" if ok else message


def _normalize_method(method: str) -> str:
    text = str(method or "None").strip().upper()
    if text in {"", "NONE", "NO", "OFF"}:
        return "NONE"
    if text in {"IC", "INCIDENT", "COUNTER1"}:
        return "IC"
    if text in {"BC", "BEAM", "COUNTER2", "COUNTER3"}:
        return "BC"
    return text
