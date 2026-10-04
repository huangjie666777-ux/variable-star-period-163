"""Collect valid multi-night light-curve points from photometry batches.

Each batch is an existing /api/photometry response tagged with a night id.
Only frames that succeeded and carry a finite MJD, a finite magnitude and a
positive magnitude error take part in the period search; every excluded
frame keeps its reason and its batch/frame provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import PeriodBatch

MIN_POINTS = 20
MAX_POINTS = 2000
MIN_NIGHTS = 2


class SampleError(ValueError):
    """Caller-facing sample collection failure (returned as 422)."""


@dataclass
class SamplePoint:
    night_id: str
    batch_index: int
    frame_index: int
    filename: str
    mjd: float
    mag: float
    mag_err: float


@dataclass
class ExcludedPoint:
    night_id: str
    batch_index: int
    frame_index: int
    filename: str
    reason: str
    mjd: float | None = None


@dataclass
class SampleSet:
    target_id: str
    points: list[SamplePoint] = field(default_factory=list)
    excluded: list[ExcludedPoint] = field(default_factory=list)

    @property
    def night_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for p in self.points:
            seen.setdefault(p.night_id)
        return list(seen)

    @property
    def span_days(self) -> float:
        mjds = [p.mjd for p in self.points]
        return max(mjds) - min(mjds)


def _frame_exclusion_reason(status, reason, mjd, mag, mag_err):
    """None when the frame is usable, else the first disqualifying reason."""
    if status != "ok":
        return f"frame failed: {reason or 'unknown reason'}"
    if mjd is None or not np.isfinite(mjd):
        return "missing or non-finite MJD"
    if mag is None or not np.isfinite(mag):
        return "missing or non-finite magnitude"
    if mag_err is None or not np.isfinite(mag_err) or mag_err <= 0:
        return "magnitude error must be positive and finite"
    return None


def collect_samples(batches: list[PeriodBatch]) -> SampleSet:
    """Merge batches into one validated multi-night sample.

    Raises SampleError when the merged sample cannot support a period
    search (too few points, too many points, fewer than two nights, or a
    non-positive time span).
    """
    target_ids = {b.photometry.target_id for b in batches}
    if len(target_ids) != 1:
        raise SampleError(
            f"all batches must share one target_id, got {sorted(target_ids)}")
    target_id = target_ids.pop()

    samples = SampleSet(target_id=target_id)
    for b_idx, batch in enumerate(batches):
        for f_idx, frame in enumerate(batch.photometry.frames):
            why = _frame_exclusion_reason(frame.status, frame.reason,
                                          frame.mjd, frame.mag, frame.mag_err)
            if why is not None:
                samples.excluded.append(ExcludedPoint(
                    night_id=batch.night_id, batch_index=b_idx,
                    frame_index=frame.index, filename=frame.filename,
                    reason=why,
                    mjd=frame.mjd if frame.mjd is not None
                    and np.isfinite(frame.mjd) else None))
                continue
            samples.points.append(SamplePoint(
                night_id=batch.night_id, batch_index=b_idx,
                frame_index=frame.index, filename=frame.filename,
                mjd=float(frame.mjd), mag=float(frame.mag),
                mag_err=float(frame.mag_err)))

    n = len(samples.points)
    if n < MIN_POINTS:
        raise SampleError(
            f"need at least {MIN_POINTS} valid points, got {n} "
            f"({len(samples.excluded)} frames excluded)")
    if n > MAX_POINTS:
        raise SampleError(
            f"at most {MAX_POINTS} valid points supported, got {n}")
    n_nights = len(samples.night_ids)
    if n_nights < MIN_NIGHTS:
        raise SampleError(
            f"valid points must span at least {MIN_NIGHTS} nights, "
            f"got {n_nights}")
    if samples.span_days <= 0:
        raise SampleError("time span of valid points must be positive")
    return samples
