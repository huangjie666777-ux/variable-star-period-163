"""Multi-night period search for differential photometry results.

Sample aggregation, joint weighted fits (per-night constants plus a shared
sin/cos at each trial frequency), candidate peak selection and phase
folding.  Nightly zero-point offsets are fitted jointly with the sinusoid
-- nightly means are never subtracted beforehand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import (MAX_PERIOD_FREQUENCIES, MAX_PERIOD_POINTS,
                     MIN_PERIOD_POINTS, PeriodSearchParams)

# Grid step: at most one fifth of the natural frequency resolution 1/span.
STEP_SAFETY = 5.0
MAX_CANDIDATES = 3


class PeriodError(ValueError):
    """Invalid period-search input; mapped to HTTP 422 by the app."""


@dataclass
class Point:
    """One usable magnitude with full provenance."""
    mjd: float
    mag: float
    err: float
    night: int           # index into the night_ids list
    night_id: str
    batch_index: int
    frame_index: int
    filename: str


@dataclass
class Excluded:
    batch_index: int
    night_id: str
    frame_index: Optional[int]
    filename: Optional[str]
    reason: str


@dataclass
class Fit:
    rss: Optional[float]
    degenerate_reason: Optional[str]
    coeffs: Optional[np.ndarray] = None  # [sin, cos, c_night0, ...]


def collect_points(params: PeriodSearchParams):
    """Gather usable points across batches; record exclusions with reasons."""
    points: list[Point] = []
    excluded: list[Excluded] = []
    night_ids: list[str] = []
    night_of: dict[str, int] = {}
    for bi, batch in enumerate(params.batches):
        for fr in batch.photometry.frames:
            prov = dict(batch_index=bi, night_id=batch.night_id,
                        frame_index=fr.index, filename=fr.filename)
            if fr.status != "ok":
                reason = f"frame status is {fr.status!r}"
                if fr.reason:
                    reason += f" ({fr.reason})"
                excluded.append(Excluded(reason=reason, **prov))
                continue
            if fr.mjd is None or not math.isfinite(fr.mjd):
                excluded.append(Excluded(reason="mjd missing or not finite",
                                         **prov))
                continue
            if fr.mag is None or not math.isfinite(fr.mag):
                excluded.append(Excluded(reason="mag missing or not finite",
                                         **prov))
                continue
            if fr.mag_err is None or not math.isfinite(fr.mag_err) \
                    or fr.mag_err <= 0:
                excluded.append(Excluded(reason="mag_err not positive",
                                         **prov))
                continue
            if batch.night_id not in night_of:
                night_of[batch.night_id] = len(night_ids)
                night_ids.append(batch.night_id)
            points.append(Point(mjd=fr.mjd, mag=fr.mag, err=fr.mag_err,
                                night=night_of[batch.night_id],
                                night_id=batch.night_id,
                                batch_index=bi, frame_index=fr.index,
                                filename=fr.filename))
    if not (MIN_PERIOD_POINTS <= len(points) <= MAX_PERIOD_POINTS):
        raise PeriodError(
            f"need {MIN_PERIOD_POINTS}-{MAX_PERIOD_POINTS} usable points, "
            f"got {len(points)}")
    if len(night_ids) < 2:
        raise PeriodError("usable points must span at least 2 distinct "
                          f"nights, got {len(night_ids)}")
    mjds = np.array([p.mjd for p in points])
    span = float(mjds.max() - mjds.min())
    if span <= 0:
        raise PeriodError("usable points must cover a positive time span")
    return points, excluded, night_ids, span


def frequency_grid(period_min: float, period_max: float,
                   span: float) -> np.ndarray:
    """Uniform frequency grid for the period bounds (period in days)."""
    f_lo = 1.0 / period_max
    f_hi = 1.0 / period_min
    df = 1.0 / (STEP_SAFETY * span)
    n = int(math.ceil((f_hi - f_lo) / df)) + 1
    n = max(n, 2)
    if n > MAX_PERIOD_FREQUENCIES:
        raise PeriodError(
            f"period range needs {n} frequency points at the required step "
            f"(limit {MAX_PERIOD_FREQUENCIES}); narrow the period bounds")
    return np.linspace(f_lo, f_hi, n)


def _design(t: np.ndarray, nights: np.ndarray, n_nights: int,
            freq: float) -> np.ndarray:
    """sin/cos shared by all nights plus one constant column per night."""
    x = np.zeros((len(t), 2 + n_nights))
    x[:, 0] = np.sin(2.0 * np.pi * freq * t)
    x[:, 1] = np.cos(2.0 * np.pi * freq * t)
    x[np.arange(len(t)), 2 + nights] = 1.0
    return x


def _fit_frequency(t, mag, w, nights, n_nights, freq) -> Fit:
    """Weighted joint fit at one frequency; power = relative RSS reduction
    versus the night-constants-only model."""
    x = _design(t, nights, n_nights, freq)
    xw = x * w[:, None]
    yw = mag * w
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return Fit(rss=None, degenerate_reason="least squares failed")
    if rank < x.shape[1]:
        return Fit(rss=None,
                   degenerate_reason="design matrix rank deficient")
    rss1 = float(np.sum((yw - xw @ coeffs) ** 2))
    if not math.isfinite(rss1):
        return Fit(rss=None, degenerate_reason="non-finite residual sum")
    return Fit(rss=rss1, degenerate_reason=None, coeffs=coeffs)


def search(points: list[Point], n_nights: int, freqs: np.ndarray):
    """Scan the grid.

    Times are relative to the earliest point to keep 2*pi*f*t small and
    avoid floating-point loss at large absolute MJDs.  Returns
    (powers, degenerate_count, degenerate_reason, mjd0, rss0).
    """
    mjd0 = min(p.mjd for p in points)
    t = np.array([p.mjd - mjd0 for p in points])
    mag = np.array([p.mag for p in points])
    err = np.array([p.err for p in points])
    nights = np.array([p.night for p in points])
    w = 1.0 / err  # weights proportional to 1/err^2 enter via whitening

    x0 = np.zeros((len(t), n_nights))
    x0[np.arange(len(t)), nights] = 1.0
    x0w = x0 * w[:, None]
    c0, _, rank0, _ = np.linalg.lstsq(x0w, mag * w, rcond=None)
    if rank0 < n_nights:
        raise PeriodError("night-constant model is rank deficient")
    rss0 = float(np.sum((mag * w - x0w @ c0) ** 2))
    if rss0 <= 0 or not math.isfinite(rss0):
        raise PeriodError("night-constant model fits exactly (zero weighted "
                          "scatter); no period search possible")

    powers: list[Optional[float]] = []
    n_degenerate = 0
    degenerate_reason: Optional[str] = None
    for f in freqs:
        fit = _fit_frequency(t, mag, w, nights, n_nights, float(f))
        if fit.degenerate_reason is not None:
            powers.append(None)
            n_degenerate += 1
            if degenerate_reason is None:
                degenerate_reason = fit.degenerate_reason
        else:
            powers.append((rss0 - fit.rss) / rss0)
    return powers, n_degenerate, degenerate_reason, mjd0


def window_powers(points: list[Point], freqs: np.ndarray) -> list[float]:
    """Sampling window: normalised weighted power of the time sampling."""
    mjd0 = min(p.mjd for p in points)
    t = np.array([p.mjd - mjd0 for p in points])
    w = np.array([1.0 / p.err ** 2 for p in points])
    norm = float(w.sum() ** 2)
    out = np.empty(len(freqs))
    chunk = 2000
    for lo in range(0, len(freqs), chunk):
        f = freqs[lo:lo + chunk]
        e = np.exp(2j * np.pi * np.outer(f, t))
        out[lo:lo + chunk] = np.abs(e @ w) ** 2 / norm
    return out.tolist()


def find_peaks(freqs: np.ndarray, powers: list[Optional[float]],
               span: float) -> list[int]:
    """Up to MAX_CANDIDATES local maxima separated by >= 1/span in
    frequency, best power first; ties go to the longer period."""
    min_sep = 1.0 / span
    idx = []
    for i in range(len(freqs)):
        p = powers[i]
        if p is None:
            continue
        left = powers[i - 1] if i > 0 else None
        right = powers[i + 1] if i < len(freqs) - 1 else None
        if (left is None or p > left) and (right is None or p >= right):
            idx.append(i)
    # Higher power first; on ties the lower frequency (longer period) wins.
    idx.sort(key=lambda i: (-powers[i], freqs[i]))
    chosen: list[int] = []
    for i in idx:
        if all(abs(freqs[i] - freqs[j]) >= min_sep for j in chosen):
            chosen.append(i)
        if len(chosen) == MAX_CANDIDATES:
            break
    return chosen


def fold(points: list[Point], n_nights: int, freq: float, mjd0: float):
    """Refit at the candidate frequency and fold around the phase zero
    (earliest usable MJD)."""
    t = np.array([p.mjd - mjd0 for p in points])
    mag = np.array([p.mag for p in points])
    err = np.array([p.err for p in points])
    nights = np.array([p.night for p in points])
    w = 1.0 / err
    x = _design(t, nights, n_nights, freq)
    coeffs, _, rank, _ = np.linalg.lstsq(x * w[:, None], mag * w, rcond=None)
    if rank < x.shape[1]:
        raise PeriodError("joint fit at the candidate frequency is "
                          "rank deficient")
    model = x @ coeffs
    night_const = coeffs[2 + nights]
    rows = []
    for i, p in enumerate(points):
        rows.append({
            "batch_index": p.batch_index,
            "night_id": p.night_id,
            "frame_index": p.frame_index,
            "filename": p.filename,
            "mjd": p.mjd,
            "phase": float((p.mjd - mjd0) * freq % 1.0),
            "mag": p.mag,
            "mag_err": p.err,
            "mag_night_zero_removed": float(mag[i] - night_const[i]),
            "model_mag": float(model[i]),
            "residual": float(mag[i] - model[i]),
        })
    return rows
