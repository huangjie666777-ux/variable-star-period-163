"""Per-frame zero-point calibration and target magnitudes (requirement 3).

Reference stars with known magnitudes and exposure-normalised fluxes give
a per-frame zero point; transparency variations are absorbed by the zero
point while target variability is preserved.  The target never takes part
in the calibration.
"""
from __future__ import annotations

import numpy as np

MIN_CALIBRATORS = 3
REJECT_NSIGMA = 4.0
MIN_REJECT_WIDTH = 0.05  # mag; never reject tighter than this


class CalibrationError(ValueError):
    """Calibration failure; carries indices of rejected reference stars."""

    def __init__(self, message, rejected=None):
        super().__init__(message)
        self.rejected = list(rejected or [])


def mag_error(rate: float, rate_err: float) -> float:
    """Flux-rate error propagated to magnitudes."""
    return 2.5 / np.log(10.0) * rate_err / rate


def estimate_zero_point(mags, rates, rate_errs):
    """Robust per-frame zero point from reference stars.

    zp_i = mag_i + 2.5*log10(rate_i); outliers are rejected by a
    median/MAD rule (never tighter than MIN_REJECT_WIDTH mag).

    Returns (zp, zp_err, kept_idx, rejected_idx).
    Raises CalibrationError when fewer than MIN_CALIBRATORS remain.
    """
    mags = np.asarray(mags, dtype=float)
    rates = np.asarray(rates, dtype=float)
    rate_errs = np.asarray(rate_errs, dtype=float)
    zp_i = mags + 2.5 * np.log10(rates)

    keep = np.ones(len(zp_i), dtype=bool)
    for _ in range(10):
        vals = zp_i[keep]
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        width = max(REJECT_NSIGMA * mad, MIN_REJECT_WIDTH)
        new_keep = np.abs(zp_i - med) <= width
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if keep.sum() < MIN_CALIBRATORS:
        raise CalibrationError(
            f"fewer than {MIN_CALIBRATORS} reference stars remain after "
            f"outlier rejection ({int(keep.sum())} left)",
            rejected=np.flatnonzero(~keep).tolist())

    kept = np.flatnonzero(keep)
    vals = zp_i[kept]
    zp = float(np.median(vals))
    scatter = 1.4826 * float(np.median(np.abs(vals - zp))) / np.sqrt(len(vals))
    phot = float(np.median(mag_error(rates[kept], rate_errs[kept])))
    zp_err = float(np.hypot(scatter, phot))
    return zp, zp_err, kept.tolist(), np.flatnonzero(~keep).tolist()


def target_magnitude(rate: float, rate_err: float,
                     zp: float, zp_err: float) -> tuple[float, float]:
    """Target magnitude with fully propagated error."""
    mag = zp - 2.5 * np.log10(rate)
    err = float(np.hypot(mag_error(rate, rate_err), zp_err))
    return float(mag), err

