"""Processing helpers for normalization, corrections, and masking."""

from pgixs_python.processing.corrections import (
    DetectorCorrectionInputs,
    apply_detector_correction,
    detector_correction_factor,
    kapton_efficiency,
    load_attenuation_table,
    polarization,
    silicon_efficiency,
    solid_angle,
)
from pgixs_python.processing.grid_remove import GridRemoveReport, grid_remove
from pgixs_python.processing.masking import (
    DetectorMask,
    DetectorMaskReport,
    apply_detector_mask,
    detector_mask_for_type,
    mask_zero_to_nan,
)
from pgixs_python.processing.normalization import (
    NormalizationReport,
    normalization_value,
    normalize_by_count,
    normalize_counters,
    process_normalization,
)

__all__ = [
    "DetectorCorrectionInputs",
    "DetectorMask",
    "DetectorMaskReport",
    "GridRemoveReport",
    "apply_detector_correction",
    "apply_detector_mask",
    "detector_correction_factor",
    "detector_mask_for_type",
    "grid_remove",
    "kapton_efficiency",
    "load_attenuation_table",
    "mask_zero_to_nan",
    "NormalizationReport",
    "normalization_value",
    "normalize_by_count",
    "normalize_counters",
    "polarization",
    "process_normalization",
    "silicon_efficiency",
    "solid_angle",
]
