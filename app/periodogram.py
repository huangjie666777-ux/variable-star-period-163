"""Weighted multi-night period search with per-night constant offsets.

At each trial frequency the model is

    mag(t) = c_night(t) + a*sin(2*pi*f*t) + b*cos(2*pi*f*t)

fitted jointly by weighted least squares (weights 1/err^2).  The nightly
constants are fitted simultaneously with the shared sinusoid -- the data
are never pre-subtracted by nightly means.  Power is the relative weighted
residual reduction with respect to the nightly-constants-only model.
Times are used relative to the earliest valid MJD to keep the sinusoid
argument accurate in float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .samples import SampleSet

FREQ_STEP_OVERSAMPLE = 5   # df <= 1/(5 * span)
MAX_FREQ_POINTS = 20000
MAX_PEAKS = 3


class PeriodogramError(ValueError):
    """Caller-facing period search failure (returned as 422)."""


@dataclass
class JointFit:
    chi2: float | None
    params: np.ndarray | None     # [c_0..c_{k-1}, a, b]
    degenerate_reason: str | None = None


@dataclass
class Peak:
    frequency: float
    period: float
    power: float
    boundary: bool = False


@dataclass
class PeriodogramResult:
    frequencies: np.ndarray
    power: np.ndarray             # NaN where the fit was degenerate
    window_power: np.ndarray
    degenerate: list[dict] = field(default_factory=list)
    peaks: list[Peak] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    df: float = 0.0


def _design(t: np.ndarray, night_col: np.ndarray, n_nights: int,
            freq: float | None) -> np.ndarray:
    cols = [(night_col == k).astype(float) for k in range(n_nights)]
    if freq is not None:
        phase = 2.0 * np.pi * freq * t
        cols += [np.sin(phase), np.cos(phase)]
    return np.column_stack(cols)


def _weighted_fit(design: np.ndarray, y: np.ndarray,
                  sqrt_w: np.ndarray) -> JointFit:
    a = design * sqrt_w[:, None]
    b = y * sqrt_w
    params, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < design.shape[1]:
        return JointFit(chi2=None, params=None,
                        degenerate_reason=(
                            f"design matrix rank deficient "
                            f"({rank}/{design.shape[1]})"))
    resid = b - a @ params
    return JointFit(chi2=float(resid @ resid), params=params)


def compute_periodogram(samples: SampleSet, period_min: float,
                        period_max: float) -> PeriodogramResult:
    """Scan the frequency axis implied by the period bounds."""
    mjd = np.array([p.mjd for p in samples.points])
    mag = np.array([p.mag for p in samples.points])
    err = np.array([p.mag_err for p in samples.points])
    nights = samples.night_ids
    night_col = np.array([nights.index(p.night_id) for p in samples.points])

    t = mjd - mjd.min()          # relative time keeps float64 precision
    span = float(t.max())
    sqrt_w = 1.0 / err

    f_lo, f_hi = 1.0 / period_max, 1.0 / period_min
    df = 1.0 / (FREQ_STEP_OVERSAMPLE * span)
    n_freq = int(np.floor((f_hi - f_lo) / df)) + 1
    if n_freq > MAX_FREQ_POINTS:
        raise PeriodogramError(
            f"frequency grid would need {n_freq} points "
            f"(limit {MAX_FREQ_POINTS}); widen the frequency step by "
            f"narrowing the period bounds")
    freqs = f_lo + df * np.arange(n_freq)

    null = _weighted_fit(_design(t, night_col, len(nights), None),
                         mag, sqrt_w)
    if null.chi2 is None or null.chi2 <= 0:
        raise PeriodogramError(
            "nightly-constants-only model is degenerate or fits exactly; "
            "cannot normalise the periodogram power")

    power = np.full(n_freq, np.nan)
    window = np.full(n_freq, np.nan)
    degenerate: list[dict] = []
    ones = np.ones_like(mag)
    for i, f in enumerate(freqs):
        design = _design(t, night_col, len(nights), float(f))
        fit = _weighted_fit(design, mag, sqrt_w)
        if fit.chi2 is None:
            degenerate.append({"frequency": float(f),
                               "reason": fit.degenerate_reason})
        else:
            power[i] = max(0.0, (null.chi2 - fit.chi2) / null.chi2)
        win = _weighted_fit(design, ones, sqrt_w)
        win_null = _weighted_fit(_design(t, night_col, len(nights), None),
                                 ones, sqrt_w)
        if win.chi2 is not None and win_null.chi2 and win_null.chi2 > 0:
            window[i] = max(0.0, (win_null.chi2 - win.chi2) / win_null.chi2)

    result = PeriodogramResult(frequencies=freqs, power=power,
                               window_power=window, degenerate=degenerate,
                               df=df)
    result.peaks = find_peaks(freqs, power, span)
    result.warnings = _warnings(result.peaks, span, f_lo, f_hi, df)
    return result


def find_peaks(freqs: np.ndarray, power: np.ndarray,
               span: float) -> list[Peak]:
    """Up to MAX_PEAKS local maxima separated by >= 1/span in frequency.

    Sorted by power; equal power prefers the longer period.  Grid-edge
    maxima are kept but flagged as boundary peaks.
    """
    min_sep = 1.0 / span
    candidates: list[tuple[int, bool]] = []
    for i in range(len(freqs)):
        p = power[i]
        if not np.isfinite(p):
            continue
        left = power[i - 1] if i > 0 else np.nan
        right = power[i + 1] if i < len(freqs) - 1 else np.nan
        boundary = i == 0 or i == len(freqs) - 1
        ok_left = not np.isfinite(left) or p >= left
        ok_right = not np.isfinite(right) or p >= right
        if boundary:
            inner_ok = ok_right if i == 0 else ok_left
            if inner_ok:
                candidates.append((i, True))
        elif p >= left and p >= right and (p > left or p > right):
            candidates.append((i, False))
    # Power descending; ties resolved towards the longer period (lower f).
    candidates.sort(key=lambda c: (-power[c[0]], freqs[c[0]]))
    chosen: list[Peak] = []
    for idx, boundary in candidates:
        if len(chosen) >= MAX_PEAKS:
            break
        if any(abs(freqs[idx] - pk.frequency) < min_sep for pk in chosen):
            continue
        chosen.append(Peak(frequency=float(freqs[idx]),
                           period=float(1.0 / freqs[idx]),
                           power=float(power[idx]), boundary=boundary))
    return chosen


def _warnings(peaks: list[Peak], span: float, f_lo: float, f_hi: float,
              df: float) -> list[str]:
    warns: list[str] = []
    for pk in peaks:
        near_edge = pk.boundary or min(abs(pk.frequency - f_lo),
                                       abs(pk.frequency - f_hi)) <= df
        if near_edge:
            warns.append(
                f"peak at period {pk.period:.6f} d lies at the edge of the "
                f"searched range; the true maximum may fall outside")
        if span < 2.0 * pk.period:
            warns.append(
                f"baseline {span:.2f} d covers fewer than two cycles of "
                f"period {pk.period:.6f} d")
    if peaks:
        warns.append("the highest peak is a candidate, not a confirmed "
                     "period")
    return warns


def phase_fold(samples: SampleSet, period: float):
    """Phase, night-offset-corrected mag, model value and residual per point.

    Phase zero is the earliest valid MJD.  Returns (rows, model) where rows
    is a list of dicts aligned with samples.points and model holds the
    joint-fit coefficients.
    """
    mjd = np.array([p.mjd for p in samples.points])
    mag = np.array([p.mag for p in samples.points])
    err = np.array([p.mag_err for p in samples.points])
    nights = samples.night_ids
    night_col = np.array([nights.index(p.night_id) for p in samples.points])
    t0 = float(mjd.min())
    t = mjd - t0
    freq = 1.0 / period

    design = _design(t, night_col, len(nights), freq)
    fit = _weighted_fit(design, mag, 1.0 / err)
    if fit.params is None:
        raise PeriodogramError(
            f"joint fit at the candidate period is degenerate: "
            f"{fit.degenerate_reason}")
    offsets = fit.params[:len(nights)]
    a_sin, b_cos = fit.params[-2], fit.params[-1]
    model = design @ fit.params
    phase = (t * freq) % 1.0
    rows = []
    for i, p in enumerate(samples.points):
        rows.append({
            "night_id": p.night_id,
            "batch_index": p.batch_index,
            "frame_index": p.frame_index,
            "filename": p.filename,
            "mjd": p.mjd,
            "mag": p.mag,
            "mag_err": p.mag_err,
            "phase": float(phase[i]),
            "detrended_mag": float(mag[i] - offsets[night_col[i]]),
            "model_mag": float(model[i]),
            "residual": float(mag[i] - model[i]),
        })
    model_info = {
        "period": period,
        "phase_zero_mjd": t0,
        "amplitude": float(np.hypot(a_sin, b_cos)),
        "night_offsets": {nights[k]: float(offsets[k])
                          for k in range(len(nights))},
    }
    return rows, model_info
