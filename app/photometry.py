"""Aperture photometry on solved frames (requirement 2).

Pixel-center-in-aperture summation, robust local background from an
annulus with other detected sources excluded, and a noise model combining
Poisson, read-noise and background-estimation terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_ANNULUS_PIXELS = 10


@dataclass
class ApertureResult:
    flux: float | None          # background-subtracted flux in ADU
    flux_err: float | None      # 1-sigma error in ADU
    background: float | None    # estimated local background per pixel (ADU)
    n_aperture: int
    n_annulus: int
    flag: str | None            # None when the measurement is usable


def measure_aperture(
    data: np.ndarray,
    x: float,
    y: float,
    r_aperture: float,
    r_inner: float,
    r_outer: float,
    gain: float,
    read_noise: float,
    other_sources: np.ndarray,
    saturation: float | None = None,
) -> ApertureResult:
    """Measure one star at 0-based pixel position (x, y).

    other_sources: (K, 2) array of other detected-source centroids used to
    mask the background annulus and to flag crowding.  Returns an
    ApertureResult with flag set (and no flux) when unusable.
    """
    ny, nx = data.shape
    bad = ApertureResult(None, None, None, 0, 0, None)

    # Aperture and annulus must lie fully inside the frame.
    if (x - r_outer < 0 or x + r_outer > nx - 1
            or y - r_outer < 0 or y + r_outer > ny - 1):
        bad.flag = "out_of_bounds"
        return bad

    x0 = int(np.floor(x - r_outer))
    x1 = int(np.ceil(x + r_outer)) + 1
    y0 = int(np.floor(y - r_outer))
    y1 = int(np.ceil(y + r_outer)) + 1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - x, yy - y)
    ap_mask = dist <= r_aperture
    ann_mask = (dist >= r_inner) & (dist <= r_outer)
    cut = data[y0:y1, x0:x1]

    ap_pix = cut[ap_mask]
    if not np.all(np.isfinite(ap_pix)):
        bad.flag = "non_finite"
        return bad
    if saturation is not None and np.any(ap_pix >= saturation):
        bad.flag = "saturated"
        return bad

    # Crowding: another detected source inside the aperture.
    if len(other_sources):
        d = np.hypot(other_sources[:, 0] - x, other_sources[:, 1] - y)
        if np.any(d <= r_aperture):
            bad.flag = "crowded"
            return bad
        # Exclude annulus pixels near any other detected source.
        ay = yy[ann_mask].ravel()[:, None]
        ax = xx[ann_mask].ravel()[:, None]
        near = np.hypot(ax - other_sources[None, :, 0],
                        ay - other_sources[None, :, 1]) <= r_aperture
        keep = ~near.any(axis=1)
    else:
        keep = np.ones(int(ann_mask.sum()), dtype=bool)

    ann_pix = cut[ann_mask]
    keep &= np.isfinite(ann_pix)
    ann_pix = ann_pix[keep]
    if ann_pix.size < MIN_ANNULUS_PIXELS:
        bad.flag = "background_annulus_depleted"
        return bad

    # Robust local background: median with MAD scatter.
    bkg = float(np.median(ann_pix))
    mad = float(np.median(np.abs(ann_pix - bkg)))
    bkg_std = 1.4826 * mad

    n_ap = int(ap_mask.sum())
    flux = float(ap_pix.sum() - bkg * n_ap)
    if not np.isfinite(flux) or flux <= 0:
        bad.flag = "non_positive_flux"
        return bad

    # Noise model in electrons, then back to ADU.
    total_e = gain * float(ap_pix.sum())          # source + background Poisson
    var_e = (total_e
             + n_ap * read_noise ** 2
             + n_ap ** 2 * (np.pi / 2.0) * (gain * bkg_std) ** 2 / ann_pix.size)
    flux_err = float(np.sqrt(var_e) / gain)

    return ApertureResult(flux=flux, flux_err=flux_err, background=bkg,
                          n_aperture=n_ap, n_annulus=int(ann_pix.size),
                          flag=None)

