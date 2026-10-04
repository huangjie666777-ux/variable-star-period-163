"""Robust affine refinement between tangent-plane and pixel coordinates (req. 4)."""
from __future__ import annotations

import numpy as np


class FitError(ValueError):
    pass


def _lstsq_affine(src: np.ndarray, dst: np.ndarray):
    n = len(src)
    M = np.hstack([src, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(M, dst, rcond=None)
    return coef[:2].T, coef[2]


def _check_geometry(cat_xy: np.ndarray):
    """Reject collinear / degenerate pair geometry."""
    c = cat_xy - cat_xy.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    if len(sv) < 2 or sv[0] <= 0 or sv[1] / sv[0] < 1e-3:
        raise FitError("matched pairs are (nearly) collinear; cannot fit 2D transform")


def robust_fit(src_xy: np.ndarray, cat_xy: np.ndarray, pairs,
               rms_max: float, max_pairs: int):
    """Iteratively fit pixel = A @ tangent + b, rejecting outliers.

    Returns (A, b, kept_pairs, residuals_arcsec, rms_arcsec).
    Raises FitError on insufficiency, degeneracy or RMS above the caller limit.
    """
    if len(pairs) < 6:
        raise FitError("fewer than 6 consistent pairs; refusing to solve")
    pairs = list(pairs)[:max_pairs]
    si = np.array([p[0] for p in pairs])
    ci = np.array([p[1] for p in pairs])
    keep = np.ones(len(pairs), dtype=bool)

    A = b = None
    for _ in range(10):
        s, c = si[keep], ci[keep]
        if len(s) < 6:
            raise FitError("fewer than 6 pairs remain after outlier rejection")
        _check_geometry(cat_xy[c])
        A, b = _lstsq_affine(cat_xy[c], src_xy[s])
        pred = cat_xy[ci] @ A.T + b
        resid = np.hypot(*(pred - src_xy[si]).T)
        r = resid[keep]
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med)) + 1e-9
        new_keep = resid < med + 4.0 * mad
        if new_keep.sum() < 6:
            new_keep = keep.copy()
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    s, c = si[keep], ci[keep]
    _check_geometry(cat_xy[c])
    A, b = _lstsq_affine(cat_xy[c], src_xy[s])
    pred = cat_xy[c] @ A.T + b
    resid = np.hypot(*(pred - src_xy[s]).T)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if not np.isfinite(rms):
        raise FitError("non-finite residual RMS")
    if rms > rms_max:
        raise FitError(
            f"residual RMS {rms:.3f} arcsec exceeds caller limit {rms_max:.3f} arcsec"
        )
    kept = [(pairs[i][0], pairs[i][1]) for i in np.flatnonzero(keep)]
    return A, b, kept, resid, rms

