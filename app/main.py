"""FastAPI entry point: /api/solve and solved-FITS download (requirements 1, 5, 6)."""
from __future__ import annotations

import io
import uuid

import numpy as np
from astropy.io import fits
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .extract import ExtractionError, extract_sources
from .fit import FitError, robust_fit
from .match import MatchError, assign_pairs, match, project_tangent
from .models import (MAX_FITS_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE,
                     PairOut, SolveParams, SolveResponse)
from .wcsbuild import affine_to_wcs_header, solved_fits_bytes

app = FastAPI(title="Star Field Plate Solver")

# In-memory store of solved FITS products, keyed by solve_id.
_RESULTS: dict[str, bytes] = {}


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


@app.post("/api/solve", response_model=SolveResponse)
async def solve(image: UploadFile = File(...), params: str = Form(...)):
    # ---- Parse and validate parameters (requirement 1) ----
    try:
        p = SolveParams.model_validate_json(params)
    except Exception as exc:
        _err(422, f"invalid parameters: {exc}")

    blob = await image.read()
    if len(blob) > MAX_FITS_BYTES:
        _err(413, "FITS file too large")
    try:
        hdul = fits.open(io.BytesIO(blob), memmap=False)
        hdu = hdul[0]
        if not isinstance(hdu, fits.PrimaryHDU) or hdu.data is None:
            _err(422, "FITS must contain a primary HDU with image data")
        data = np.asarray(hdu.data, dtype=np.float64)
    except HTTPException:
        raise
    except Exception:
        _err(422, "could not read FITS file")
    if data.ndim != 2:
        _err(422, "primary HDU data must be a 2D image")
    if data.size > MAX_IMAGE_PIXELS or max(data.shape) > MAX_IMAGE_SIDE:
        _err(413, f"image exceeds limit of {MAX_IMAGE_SIDE}px side / "
                  f"{MAX_IMAGE_PIXELS} pixels")

    # ---- Source extraction (requirement 2) ----
    try:
        sources, background, sigma = extract_sources(
            data, threshold_sigma=p.threshold_sigma, saturation=p.saturation)
    except ExtractionError as exc:
        _err(422, f"extraction failed: {exc}")
    if len(sources) < 6:
        _err(422, f"only {len(sources)} usable sources detected; need at least 6")

    # ---- Catalog projection (requirement 1: reject stars behind center) ----
    ra = [s.ra for s in p.catalog]
    dec = [s.dec for s in p.catalog]
    try:
        xi, eta = project_tangent(ra, dec, p.center_ra, p.center_dec)
    except MatchError as exc:
        _err(422, str(exc))
    cat_xy = np.column_stack([xi, eta])

    # ---- Geometric matching (requirement 3) ----
    src_xy = np.array([[s.x, s.y] for s in sources])
    try:
        # Caller gives arcsec/pixel; the matcher works in px/arcsec.
        pairs, _, _ = match(src_xy, cat_xy,
                            1.0 / p.pixel_scale_max, 1.0 / p.pixel_scale_min)
    except MatchError as exc:
        _err(422, f"matching failed: {exc}")

    # ---- Robust affine fit and RMS gate (requirement 4) ----
    try:
        A, b, kept, resid, rms = robust_fit(src_xy, cat_xy, pairs,
                                            p.rms_max, p.max_pairs)
        # Second assignment round under the refined transform, then refit.
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        pairs2 = assign_pairs(src_xy, cat_xy, A, b, 3.0 * scale)
        if len(pairs2) > len(kept):
            A, b, kept, resid, rms = robust_fit(src_xy, cat_xy, pairs2,
                                                p.rms_max, p.max_pairs)
    except FitError as exc:
        _err(422, f"fit failed: {exc}")

    wcs = affine_to_wcs_header(A, b, p.center_ra, p.center_dec)
    mirrored = bool(np.linalg.det(A) < 0)

    out_pairs = []
    for (si, ci), r in zip(kept, resid):
        out_pairs.append(PairOut(id=p.catalog[ci].id, x=float(src_xy[si, 0]),
                                 y=float(src_xy[si, 1]),
                                 residual_arcsec=float(r)))

    solve_id = uuid.uuid4().hex
    _RESULTS[solve_id] = solved_fits_bytes(hdu, wcs)
    hdul.close()
    return SolveResponse(solve_id=solve_id, n_pairs=len(out_pairs),
                         rms_arcsec=rms, mirrored=mirrored,
                         pairs=out_pairs, wcs=wcs)


@app.get("/api/solve/{solve_id}/fits")
def download_fits(solve_id: str):
    blob = _RESULTS.get(solve_id)
    if blob is None:
        _err(404, "unknown solve_id")
    return Response(content=blob, media_type="application/fits",
                    headers={"Content-Disposition":
                             "attachment; filename=solved.fits"})

