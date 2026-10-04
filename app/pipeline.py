"""Shared solve pipeline: extraction -> projection -> matching -> robust fit.

Used by both the /api/solve endpoint and the differential photometry
pipeline so every frame is solved with the exact same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import ExtractionError, extract_sources
from .fit import FitError, robust_fit
from .match import MatchError, assign_pairs, match, project_tangent
from .models import SolveParams
from .wcsbuild import affine_to_wcs_header


class PipelineError(ValueError):
    """Any deterministic, caller-facing solve failure."""


@dataclass
class SolveResult:
    A: np.ndarray          # pixel = A @ tangent_arcsec + b
    b: np.ndarray
    kept_pairs: list       # [(src_idx, cat_idx), ...]
    residuals: np.ndarray  # arcsec, aligned with kept_pairs
    rms_arcsec: float
    mirrored: bool
    wcs: dict
    sources: list          # detected Source objects (centroids, flux)
    src_xy: np.ndarray     # (N, 2) detected centroids
    cat_xy: np.ndarray     # (M, 2) catalog tangent-plane arcsec


def solve_field(data: np.ndarray, p: SolveParams) -> SolveResult:
    """Run the full plate-solve pipeline on one 2D image.

    Raises PipelineError on any failure (extraction, matching, fit, RMS).
    """
    try:
        sources, _, _ = extract_sources(
            data, threshold_sigma=p.threshold_sigma, saturation=p.saturation)
    except ExtractionError as exc:
        raise PipelineError(f"extraction failed: {exc}") from exc
    if len(sources) < 6:
        raise PipelineError(
            f"only {len(sources)} usable sources detected; need at least 6")

    ra = [s.ra for s in p.catalog]
    dec = [s.dec for s in p.catalog]
    try:
        xi, eta = project_tangent(ra, dec, p.center_ra, p.center_dec)
    except MatchError as exc:
        raise PipelineError(str(exc)) from exc
    cat_xy = np.column_stack([xi, eta])

    src_xy = np.array([[s.x, s.y] for s in sources])
    try:
        pairs, _, _ = match(src_xy, cat_xy,
                            1.0 / p.pixel_scale_max, 1.0 / p.pixel_scale_min)
    except MatchError as exc:
        raise PipelineError(f"matching failed: {exc}") from exc

    try:
        A, b, kept, resid, rms = robust_fit(src_xy, cat_xy, pairs,
                                            p.rms_max, p.max_pairs)
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        pairs2 = assign_pairs(src_xy, cat_xy, A, b, 3.0 * scale)
        if len(pairs2) > len(kept):
            A, b, kept, resid, rms = robust_fit(src_xy, cat_xy, pairs2,
                                                p.rms_max, p.max_pairs)
    except FitError as exc:
        raise PipelineError(f"fit failed: {exc}") from exc

    wcs = affine_to_wcs_header(A, b, p.center_ra, p.center_dec)
    return SolveResult(A=A, b=b, kept_pairs=kept, residuals=resid,
                       rms_arcsec=rms, mirrored=bool(np.linalg.det(A) < 0),
                       wcs=wcs, sources=sources, src_xy=src_xy, cat_xy=cat_xy)

