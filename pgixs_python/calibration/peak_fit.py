"""Gaussian peak fitting for SDD calibration ring profiles.

Ports the Gaussian-plus-constant-background model MATLAB's `fitPGIXSGaussianPeak.m`
and `OneDplotpixel_gui.m`/`OneDplotpixelR_gui.m`'s inline fits share
(`a*exp(-((x-b)^2)/(2c^2))+d`, with the same Lower/Upper bounds), using
`scipy.optimize.curve_fit` instead of MATLAB's Curve Fitting Toolbox.

MATLAB seeds the fit from a quick `fit(x, y, 'gauss1')` toolbox pre-fit
(`OneDplotpixel_gui.m`) or a fixed `sigma=2.0` guess (`fitPGIXSGaussianPeak.m`);
neither is available here, so both call sites are served by a single
moment-estimation initial guess -- a standard, dependency-free substitute for
seeding a single-peak Gaussian fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class GaussianFitResult:
    success: bool
    message: str
    amplitude: float
    center: float
    sigma: float
    offset: float
    r_square: float
    fit_x: np.ndarray
    fit_y: np.ndarray


def _gaussian(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a * np.exp(-((x - b) ** 2) / (2.0 * c**2)) + d


def _failed(message: str) -> GaussianFitResult:
    empty = np.array([], dtype=np.float64)
    return GaussianFitResult(
        success=False,
        message=message,
        amplitude=float("nan"),
        center=float("nan"),
        sigma=float("nan"),
        offset=float("nan"),
        r_square=float("nan"),
        fit_x=empty,
        fit_y=empty,
    )


def fit_gaussian_peak(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_center: float | None = None,
    half_width: float | None = None,
    n_curve_points: int = 100,
) -> GaussianFitResult:
    """Fit one Gaussian peak with a constant background.

    If `window_center`/`half_width` are given, `x`/`y` are first restricted
    to `[window_center - half_width, window_center + half_width]` (matches
    `fitPGIXSGaussianPeak.m`'s ROI clipping, used for center-refinement and
    ring-candidate fits); otherwise fits the caller's `x`/`y` directly
    (matches `OneDplotpixel_gui.m`/`OneDplotpixelR_gui.m`'s manual
    range-selection fits, where the caller already sliced the data).
    """

    x_all = np.asarray(x, dtype=np.float64).reshape(-1)
    y_all = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_all.shape != y_all.shape:
        raise ValueError("x and y must have the same shape.")

    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if window_center is not None and half_width is not None:
        finite = finite & (x_all >= window_center - half_width) & (x_all <= window_center + half_width)
    x_roi = x_all[finite]
    y_roi = y_all[finite]

    if x_roi.size < 4:
        return _failed("Not enough finite points for Gaussian peak fit.")

    offset0 = float(np.min(y_roi))
    amp0 = float(np.max(y_roi) - offset0)
    if not np.isfinite(amp0) or amp0 <= 0:
        return _failed("Peak amplitude guess is not positive.")

    y_pos = np.clip(y_roi - offset0, 0.0, None)
    total = float(np.sum(y_pos))
    if total > 0:
        center0 = float(np.sum(x_roi * y_pos) / total)
        sigma0 = float(np.sqrt(np.sum(y_pos * (x_roi - center0) ** 2) / total))
        sigma0 = max(sigma0, 0.5)
    else:
        center0 = float(x_roi[int(np.argmax(y_roi))])
        sigma0 = max(float(x_roi.max() - x_roi.min()) / 6.0, 0.5)

    x_min, x_max = float(x_roi.min()), float(x_roi.max())
    span = max(abs(x_max - x_min), 0.2)
    lower = [0.0, x_min, 0.1, 0.0]
    upper = [np.inf, x_max, span, float(np.max(y_roi))]
    p0 = [amp0, np.clip(center0, x_min, x_max), np.clip(sigma0, 0.1, span), offset0]

    try:
        popt, _ = curve_fit(_gaussian, x_roi, y_roi, p0=p0, bounds=(lower, upper), maxfev=5000)
    except (RuntimeError, ValueError) as exc:
        return _failed(str(exc))

    amplitude, center, sigma, offset = (float(v) for v in popt)
    if not np.isfinite(center):
        return _failed("Gaussian fit returned a non-finite center.")

    fitted = _gaussian(x_roi, *popt)
    ss_res = float(np.sum((y_roi - fitted) ** 2))
    ss_tot = float(np.sum((y_roi - np.mean(y_roi)) ** 2))
    r_square = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fit_x = np.linspace(x_min, x_max, int(n_curve_points))
    fit_y = _gaussian(fit_x, *popt)

    return GaussianFitResult(
        success=True,
        message="",
        amplitude=amplitude,
        center=center,
        sigma=sigma,
        offset=offset,
        r_square=r_square,
        fit_x=fit_x,
        fit_y=fit_y,
    )
