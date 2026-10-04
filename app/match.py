"""Tangent-plane projection and order-free geometric asterism matching (req. 1 & 3)."""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

# Search budget (requirement 4): bounded work, deterministic failure.
MAX_TRI_STARS_SRC = 18       # brightest detected sources used for triangles
MAX_TRI_STARS_CAT = 30       # catalog stars used for triangles
MATCH_TIME_BUDGET_S = 20.0
INVARIANT_RTOL = 0.02        # relative tolerance on triangle side ratios


class MatchError(ValueError):
    pass


def project_tangent(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic (TAN) projection to tangent plane, in arcsec.

    Raises MatchError for stars at/behind 90 deg from the projection center
    (they cannot be represented in a TAN projection).
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    cos_c = (np.sin(dec0) * np.sin(dec)
             + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
    if np.any(cos_c <= 1e-6):
        raise MatchError(
            "catalog contains stars at or behind 90 deg from the projection center"
        )
    xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
    eta = ((np.cos(dec0) * np.sin(dec)
            - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c)
    return np.rad2deg(xi) * 3600.0, np.rad2deg(eta) * 3600.0  # arcsec


def _triangles(pts: np.ndarray, max_stars: int):
    """Yield (inv1, inv2, (v0, v1, v2)) for triangles of the first max_stars.

    Invariants are scale- and mirror-invariant side-length ratios.  Vertices
    are returned canonically ordered by their opposite side length (shortest
    to longest), so two matched triangles imply an exact vertex mapping.
    """
    p = pts[:max_stars]
    n = len(p)
    out = []
    for i, j, k in itertools.combinations(range(n), 3):
        # side opposite each vertex
        opp = [(np.hypot(*(p[j] - p[k])), i),
               (np.hypot(*(p[k] - p[i])), j),
               (np.hypot(*(p[i] - p[j])), k)]
        opp.sort(key=lambda t: t[0])
        d = [o[0] for o in opp]
        if d[2] <= 0 or d[0] / d[2] < 0.05:
            continue  # degenerate / nearly collinear
        out.append((d[1] / d[2], d[0] / d[2], (opp[0][1], opp[1][1], opp[2][1])))
    return out


def _fit_affine(src: np.ndarray, dst: np.ndarray):
    """Least-squares dst = A @ src + b. src/dst: (N, 2)."""
    n = len(src)
    M = np.hstack([src, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(M, dst, rcond=None)
    A = coef[:2].T
    b = coef[2]
    return A, b


def _vote_transforms(src_xy, cat_xy, src_tris, cat_tris,
                     scale_min, scale_max, deadline):
    """Vote on (scale, translation) transform hypotheses from triangle matches.

    Returns up to 20 (votes, A, b) hypotheses, strongest first.
    """
    keys = {}
    for a, b, idx in cat_tris:
        keys.setdefault((round(a / INVARIANT_RTOL), round(b / INVARIANT_RTOL)),
                        []).append(idx)
    ballots: dict[tuple, list] = {}
    for a, b, sidx in src_tris:
        if time.monotonic() > deadline:
            raise MatchError("matching search budget exhausted (triangle voting)")
        ka, kb = round(a / INVARIANT_RTOL), round(b / INVARIANT_RTOL)
        sp = src_xy[list(sidx)]
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for cidx in keys.get((ka + da, kb + db), ()):
                    cp = cat_xy[list(cidx)]
                    try:
                        A, tb = _fit_affine(cp, sp)
                    except np.linalg.LinAlgError:
                        continue
                    det = np.linalg.det(A)
                    scale = float(np.sqrt(abs(det)))
                    if not (scale_min <= scale <= scale_max):
                        continue
                    key = (round(np.log(scale) / 0.01),
                           round(tb[0] / 3.0), round(tb[1] / 3.0))
                    ballots.setdefault(key, []).append((A, tb))
    ranked = sorted(ballots.items(), key=lambda kv: -len(kv[1]))[:20]
    out = []
    for key, hyps in ranked:
        A = np.mean([h[0] for h in hyps], axis=0)
        tb = np.mean([h[1] for h in hyps], axis=0)
        out.append((len(hyps), A, tb))
    return out


def match(src_xy: np.ndarray, cat_xy: np.ndarray,
          scale_min: float, scale_max: float):
    """Find source<->catalog correspondences on the tangent plane.

    src_xy: (N, 2) pixel centroids; cat_xy: (M, 2) tangent-plane arcsec.
    Returns (pairs, A, b): list of (src_idx, cat_idx) and the affine
    pixel = A @ tangent_arcsec + b consistent with them (may be mirrored).
    """
    deadline = time.monotonic() + MATCH_TIME_BUDGET_S
    if len(src_xy) < 3 or len(cat_xy) < 3:
        raise MatchError("too few sources or catalog stars for geometric matching")

    src_tris = _triangles(src_xy, MAX_TRI_STARS_SRC)
    cat_tris = _triangles(cat_xy, MAX_TRI_STARS_CAT)
    if not src_tris or not cat_tris:
        raise MatchError("could not build triangle asterisms (degenerate geometry)")

    hyps = _vote_transforms(src_xy, cat_xy, src_tris, cat_tris,
                            scale_min, scale_max, deadline)
    if not hyps:
        raise MatchError("no matching triangle asterisms found")

    # Geometric verification: refine each hypothesis and count inliers.
    best = None
    for votes, A, b in hyps:
        if time.monotonic() > deadline:
            raise MatchError("matching search budget exhausted (verification)")
        for _ in range(3):
            pred = cat_xy @ A.T + b
            dist = np.sqrt(((pred[:, None, :] - src_xy[None, :, :]) ** 2).sum(-1))
            scale = float(np.sqrt(abs(np.linalg.det(A))))
            tol = 3.0 * scale  # 3 pixels
            ci, si = np.nonzero(dist < tol)
            if len(ci) < 3:
                break
            try:
                A, b = _fit_affine(cat_xy[ci], src_xy[si])
            except np.linalg.LinAlgError:
                break
        pred = cat_xy @ A.T + b
        dist = np.sqrt(((pred[:, None, :] - src_xy[None, :, :]) ** 2).sum(-1))
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        tol = 3.0 * scale
        n_inl = int((dist.min(axis=1) < tol).sum())
        if best is None or n_inl > best[0]:
            best = (n_inl, A, b, tol)
    if best is None or best[0] < 6:
        raise MatchError("geometric verification failed (no consistent asterism)")

    _, A, b, tol = best
    # One-to-one assignment on all inliers, greedily by nearest distance.
    pred = cat_xy @ A.T + b
    dist = np.sqrt(((pred[:, None, :] - src_xy[None, :, :]) ** 2).sum(-1))
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    used_s, used_c, pairs = set(), set(), []
    for ci, si in order:
        if dist[ci, si] >= tol:
            break
        if ci in used_c or si in used_s:
            continue
        used_c.add(ci)
        used_s.add(si)
        pairs.append((int(si), int(ci)))
    if len(pairs) < 6:
        raise MatchError("fewer than 6 one-to-one pairs after matching")
    return pairs, A, b


def assign_pairs(src_xy: np.ndarray, cat_xy: np.ndarray,
                 A: np.ndarray, b: np.ndarray, tol_arcsec: float):
    """One-to-one greedy nearest-neighbour assignment under a given transform."""
    pred = cat_xy @ A.T + b
    dist = np.sqrt(((pred[:, None, :] - src_xy[None, :, :]) ** 2).sum(-1))
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    used_s, used_c, pairs = set(), set(), []
    for ci, si in order:
        if dist[ci, si] >= tol_arcsec:
            break
        if ci in used_c or si in used_s:
            continue
        used_c.add(ci)
        used_s.add(si)
        pairs.append((int(si), int(ci)))
    return pairs
