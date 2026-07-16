"""1D averaging helpers."""

from pgixs_python.averaging.angular import SectorAverageResult, find_coverage_drop_bins, make_sector_mask, sector_average
from pgixs_python.averaging.cake import (
    CakeAverageResult,
    angle_in_relative_sector,
    cake_average,
    normalize_angle180,
    pixel_angle_map,
)
from pgixs_python.averaging.radial import RadiusAverageResult, RadiusBinMap, build_radius_bin_map, radius_average
from pgixs_python.averaging.qbin import QBinAverageResult, mean_by_bin, qbin_average, summarize_coverage

__all__ = [
    "CakeAverageResult",
    "QBinAverageResult",
    "RadiusAverageResult",
    "RadiusBinMap",
    "SectorAverageResult",
    "angle_in_relative_sector",
    "build_radius_bin_map",
    "cake_average",
    "find_coverage_drop_bins",
    "make_sector_mask",
    "mean_by_bin",
    "normalize_angle180",
    "pixel_angle_map",
    "qbin_average",
    "radius_average",
    "sector_average",
    "summarize_coverage",
]
