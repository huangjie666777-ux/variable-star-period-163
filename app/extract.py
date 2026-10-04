"""Source extraction: background/noise estimation and centroiding (requirement 2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .models import MAX_DETECTED_SOURCES


class ExtractionError(ValueError):
    pass


@dataclass
class Source:
    x: float  # 0-based centroid column
    y: float  # 0-based centroid row
    flux: float  # background-subtracted summed weight


def estimate_background(data: np.ndarray) -> tuple[float, float]:
    """Median background and robust (MAD-based) noise sigma from finite pixels."""
    finite = data[np.isfinite(data)]
    if finite.size < 100:
        raise ExtractionError("image has too few finite pixels")
    # Subsample for speed on large frames.
    if finite.size > 2_000_000:
        idx = np.random.default_rng(0).choice(finite.size, 2_000_000, replace=False)
        finite = finite[idx]
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        raise ExtractionError("cannot estimate image noise (degenerate pixel values)")
    return med, sigma


def extract_sources(
    data: np.ndarray,
    threshold_sigma: float = 5.0,
    saturation: float | None = None,
    edge_margin: int = 3,
) -> tuple[list[Source], float, float]:
    """Detect local sources and compute background-subtracted weighted centroids.

    Excludes non-finite pixels, saturated sources, edge-truncated sources and
    isolated single-pixel (hot) detections.
    """
    if data.ndim != 2:
        raise ExtractionError("image must be a 2D array")
    background, sigma = estimate_background(data)
    threshold = background + threshold_sigma * sigma

    finite_mask = np.isfinite(data)
    work = np.where(finite_mask, data, -np.inf)
    det = work > threshold
    # Fill single-pixel holes so non-finite pixels inside a source do not split it.
    det = ndimage.binary_closing(det, structure=np.ones((3, 3)))

    labels, nlab = ndimage.label(det)
    if nlab == 0:
        return [], background, sigma

    ny, nx = data.shape
    sources: list[Source] = []
    slices = ndimage.find_objects(labels)
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        ys, xs = sl
        npix = (labels[sl] == lab).sum()
        if npix < 3:  # isolated hot pixel / cosmic ray
            continue
        # Edge-truncated sources are incomplete: reject.
        if (ys.start < edge_margin or xs.start < edge_margin
                or ys.stop > ny - edge_margin or xs.stop > nx - edge_margin):
            continue
        region = data[sl]
        mask = labels[sl] == lab
        pix = region[mask]
        if not np.all(np.isfinite(pix)):
            continue  # non-finite pixels inside the source footprint
        if saturation is not None and np.any(pix >= saturation):
            continue  # saturated source
        weights = pix - background
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            continue
        yy, xx = np.nonzero(mask)
        cx = float((xx + xs.start) @ weights / total)
        cy = float((yy + ys.start) @ weights / total)
        sources.append(Source(x=cx, y=cy, flux=total))

    sources.sort(key=lambda s: s.flux, reverse=True)
    del sources[MAX_DETECTED_SOURCES:]
    return sources, background, sigma

