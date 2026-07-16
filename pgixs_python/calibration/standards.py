"""Standard sample q values for PGIXS SDD calibration."""

from __future__ import annotations


STANDARD_BASE_Q = {
    "sebs": 0.01916,
    "silver behenate": 0.1052,
    "mesoporous silica": 0.059264,
    "sucrose": 0.5933,
}


def standard_q_value(sample_name: str, order: int | float = 1) -> float:
    """Return expected q value for a standard sample ring order."""

    ring_order = float(order)
    if not ring_order > 0:
        raise ValueError("Standard ring order must be positive.")

    key = _normalize_sample_name(sample_name)
    if key not in STANDARD_BASE_Q:
        known = ", ".join(sorted(STANDARD_BASE_Q))
        raise ValueError(f"Unknown SDD standard sample: {sample_name}. Known samples: {known}")
    return STANDARD_BASE_Q[key] * ring_order


def _normalize_sample_name(sample_name: str) -> str:
    return " ".join(str(sample_name).strip().lower().split())
