"""FastAPI entry point: plate solving, differential photometry, downloads."""
from __future__ import annotations

import csv
import io
import uuid
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.time import Time
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .calibrate import CalibrationError, estimate_zero_point, target_magnitude
from .models import (MAX_FITS_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE,
                     MAX_PHOTOMETRY_FRAMES, MIN_PHOTOMETRY_FRAMES,
                     ExcludedPoint, ExcludedReference, FoldedPoint,
                     FrameResult, PairOut, PeriodCandidate,
                     PeriodSearchParams, PeriodSearchResponse,
                     PhotometryParams, PhotometryResponse, SolveParams,
                     SolveResponse)
from .period import PeriodError, collect_points, find_peaks, fold,     frequency_grid, search, window_powers
from .photometry import measure_aperture
from .pipeline import PipelineError, solve_field
from .wcsbuild import solved_fits_bytes

app = FastAPI(title="Star Field Plate Solver + Differential Photometry")

# In-memory stores keyed by result id.
_RESULTS: dict[str, bytes] = {}
_CSV_RESULTS: dict[str, bytes] = {}
_PERIOD_CSV: dict[str, bytes] = {}


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def _read_fits_image(blob: bytes) -> tuple[np.ndarray, fits.PrimaryHDU, fits.HDUList]:
    """Validate size and open a 2D primary-HDU FITS image."""
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
    return data, hdu, hdul


@app.post("/api/solve", response_model=SolveResponse)
async def solve(image: UploadFile = File(...), params: str = Form(...)):
    # ---- Parse and validate parameters (requirement 1) ----
    try:
        p = SolveParams.model_validate_json(params)
    except Exception as exc:
        _err(422, f"invalid parameters: {exc}")

    blob = await image.read()
    data, hdu, hdul = _read_fits_image(blob)

    # ---- Shared solve pipeline (requirements 1-4) ----
    try:
        res = solve_field(data, p)
    except PipelineError as exc:
        _err(422, str(exc))

    out_pairs = []
    for (si, ci), r in zip(res.kept_pairs, res.residuals):
        out_pairs.append(PairOut(id=p.catalog[ci].id,
                                 x=float(res.src_xy[si, 0]),
                                 y=float(res.src_xy[si, 1]),
                                 residual_arcsec=float(r)))

    solve_id = uuid.uuid4().hex
    _RESULTS[solve_id] = solved_fits_bytes(hdu, res.wcs)
    hdul.close()
    return SolveResponse(solve_id=solve_id, n_pairs=len(out_pairs),
                         rms_arcsec=res.rms_arcsec, mirrored=res.mirrored,
                         pairs=out_pairs, wcs=res.wcs)


@app.get("/api/solve/{solve_id}/fits")
def download_fits(solve_id: str):
    blob = _RESULTS.get(solve_id)
    if blob is None:
        _err(404, "unknown solve_id")
    return Response(content=blob, media_type="application/fits",
                    headers={"Content-Disposition":
                             "attachment; filename=solved.fits"})


# ---------------------------------------------------------------------------
# Differential aperture photometry (requirements 1-4)
# ---------------------------------------------------------------------------

def _exposure_midpoint_mjd(header: fits.Header) -> tuple[float, float]:
    """UTC exposure midpoint (MJD) and EXPTIME; raises ValueError if invalid."""
    date_obs = header.get("DATE-OBS")
    exptime = header.get("EXPTIME")
    if date_obs is None:
        raise ValueError("missing DATE-OBS header")
    try:
        t = Time(str(date_obs), scale="utc")
    except Exception as exc:
        raise ValueError(f"invalid DATE-OBS: {date_obs!r}") from exc
    try:
        exptime = float(exptime)
    except (TypeError, ValueError) as exc:
        raise ValueError("missing or non-numeric EXPTIME header") from exc
    if not np.isfinite(exptime) or exptime <= 0:
        raise ValueError(f"EXPTIME must be positive, got {exptime}")
    return float(t.mjd + exptime / 2.0 / 86400.0), exptime


class FrameFailure(PipelineError):
    """Frame failure that still carries known reference-star details."""

    def __init__(self, reason: str,
                 references_used: Optional[list] = None,
                 references_excluded: Optional[list] = None):
        super().__init__(reason)
        self.references_used = references_used or []
        self.references_excluded = references_excluded or []


def _measure_frame(data: np.ndarray, p: PhotometryParams,
                   mjd: float, exptime: float) -> FrameResult:
    """Solve one frame, photometer target + references, calibrate."""
    res = solve_field(data, p.solve)  # raises PipelineError

    # Project every catalog star to pixels with the solved transform.
    pix = res.cat_xy @ res.A.T + res.b
    cat_ids = [s.id for s in p.solve.catalog]
    pos = {cid: pix[i] for i, cid in enumerate(cat_ids)}
    det_xy = res.src_xy  # detected centroids, for crowding/annulus masking

    def measure(star_id: str):
        x, y = pos[star_id]
        d = np.hypot(det_xy[:, 0] - x, det_xy[:, 1] - y)
        # Everything except the star's own detection (centroid noise and
        # residual astrometric error can shift it by more than a pixel).
        others = det_xy[d > 2.0]
        return measure_aperture(
            data, x, y, p.aperture_radius, p.annulus_inner, p.annulus_outer,
            p.gain, p.read_noise, others, saturation=p.solve.saturation)

    excluded: list[ExcludedReference] = []
    ok_mags: list[float] = []
    ok_rates: list[float] = []
    ok_rate_errs: list[float] = []
    ok_ids: list[str] = []
    for ref in p.references:
        r = measure(ref.id)
        if r.flag is not None:
            excluded.append(ExcludedReference(id=ref.id, reason=r.flag))
            continue
        ok_mags.append(ref.mag)
        ok_rates.append(r.flux / exptime)
        ok_rate_errs.append(r.flux_err / exptime)
        ok_ids.append(ref.id)

    if len(ok_ids) < 3:
        raise FrameFailure(
            f"fewer than 3 usable reference stars ({len(ok_ids)} available)",
            references_excluded=excluded)

    # Zero point from valid references; robust outlier rejection.
    try:
        zp, zp_err, kept, dropped = estimate_zero_point(
            ok_mags, ok_rates, ok_rate_errs)
    except CalibrationError as exc:
        for i in exc.rejected:
            excluded.append(ExcludedReference(id=ok_ids[i],
                                              reason="zero_point_outlier"))
        raise FrameFailure(str(exc), references_excluded=excluded) from exc
    for i in dropped:
        excluded.append(ExcludedReference(id=ok_ids[i],
                                          reason="zero_point_outlier"))
    used = [ok_ids[i] for i in kept]

    # Target last: it never participates in the calibration.
    t = measure(p.target_id)
    if t.flag is not None:
        raise FrameFailure(f"target measurement failed: {t.flag}",
                           references_used=used,
                           references_excluded=excluded)
    rate = t.flux / exptime
    rate_err = t.flux_err / exptime
    mag, mag_err = target_magnitude(rate, rate_err, zp, zp_err)

    return FrameResult(index=-1, filename="", status="ok", mjd=mjd,
                       exptime=exptime, flux_rate=rate, flux_rate_err=rate_err,
                       mag=mag, mag_err=mag_err, zero_point=zp,
                       zero_point_err=zp_err, references_used=used,
                       references_excluded=excluded)


@app.post("/api/photometry", response_model=PhotometryResponse)
async def photometry(images: List[UploadFile] = File(...),
                     params: str = Form(...)):
    try:
        p = PhotometryParams.model_validate_json(params)
    except Exception as exc:
        _err(422, f"invalid parameters: {exc}")
    if not (MIN_PHOTOMETRY_FRAMES <= len(images) <= MAX_PHOTOMETRY_FRAMES):
        _err(422, f"need {MIN_PHOTOMETRY_FRAMES}-{MAX_PHOTOMETRY_FRAMES} "
                  f"frames, got {len(images)}")

    frames: list[FrameResult] = []
    for idx, up in enumerate(images):
        name = up.filename or f"frame{idx}"
        blob = await up.read()
        mjd: Optional[float] = None
        exptime: Optional[float] = None
        try:
            data, _, hdul = _read_fits_image(blob)
            try:
                mjd, exptime = _exposure_midpoint_mjd(hdul[0].header)
            finally:
                hdul.close()
            fr = _measure_frame(data, p, mjd, exptime)
            fr.index = idx
            fr.filename = name
        except HTTPException:
            raise
        except (PipelineError, ValueError) as exc:
            # Single-frame failure: keep the record (including any known
            # MJD/EXPTIME and reference-star details), continue with the rest.
            failed = FrameResult(index=idx, filename=name, status="failed",
                                 reason=str(exc), mjd=mjd, exptime=exptime)
            if isinstance(exc, FrameFailure):
                failed.references_used = exc.references_used
                failed.references_excluded = exc.references_excluded
            frames.append(failed)
            continue
        frames.append(fr)

    # Sort by exposure midpoint; failed frames without MJD keep input order.
    frames.sort(key=lambda f: (f.mjd is None, f.mjd if f.mjd is not None
                               else f.index))

    photometry_id = uuid.uuid4().hex
    _CSV_RESULTS[photometry_id] = _frames_to_csv(frames).encode()
    return PhotometryResponse(photometry_id=photometry_id,
                              target_id=p.target_id,
                              n_frames=len(frames),
                              n_ok=sum(f.status == "ok" for f in frames),
                              frames=frames)


_CSV_COLUMNS = ("index,filename,status,reason,mjd,exptime,flux_rate,"
                "flux_rate_err,mag,mag_err,zero_point,zero_point_err,"
                "references_used,references_excluded")


def _frames_to_csv(frames: list[FrameResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS.split(","))
    for f in frames:
        used = ";".join(f.references_used)
        excl = ";".join(f"{e.id}:{e.reason}" for e in f.references_excluded)
        vals = [f.index, f.filename, f.status, f.reason, f.mjd, f.exptime,
                f.flux_rate, f.flux_rate_err, f.mag, f.mag_err,
                f.zero_point, f.zero_point_err, used, excl]
        writer.writerow(["" if v is None else v for v in vals])
    return buf.getvalue()


@app.get("/api/photometry/{photometry_id}/csv")
def download_csv(photometry_id: str):
    blob = _CSV_RESULTS.get(photometry_id)
    if blob is None:
        _err(404, "unknown photometry_id")
    return Response(content=blob, media_type="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=photometry.csv"})


# ---------------------------------------------------------------------------
# Multi-night period search (requirement 5)
# ---------------------------------------------------------------------------

_FOLD_CSV_COLUMNS = ("batch_index,night_id,frame_index,filename,mjd,phase,"
                     "mag,mag_err,mag_night_zero_removed,model_mag,residual")


@app.post("/api/period", response_model=PeriodSearchResponse)
def period_search(p: PeriodSearchParams):
    # ---- Aggregate usable points across nights (with exclusion reasons) ----
    try:
        points, excluded, night_ids, span = collect_points(p)
        freqs = frequency_grid(p.period_min_days, p.period_max_days, span)
        # ---- Joint per-night-constant + shared sinusoid scan ----
        powers, n_degen, degen_reason, mjd0 = search(points, len(night_ids),
                                                     freqs)
        window = window_powers(points, freqs)
        peak_idx = find_peaks(freqs, powers, span)

        candidates = [PeriodCandidate(frequency=float(freqs[i]),
                                      period_days=float(1.0 / freqs[i]),
                                      power=float(powers[i]))
                      for i in peak_idx]
        warnings: list[str] = []
        no_candidate_reason: Optional[str] = None
        best: Optional[PeriodCandidate] = None
        phase_zero_mjd: Optional[float] = None
        fold_rows: list[dict] = []
        if not candidates:
            no_candidate_reason = (
                "no local power maximum found"
                + (f" ({n_degen} degenerate fits: {degen_reason})"
                   if n_degen else "")
                + "; the highest power is not a confirmed period")
        else:
            best = candidates[0]
            warnings.append("the highest-power candidate is not a confirmed "
                            "period; check the window function and aliases")
            if peak_idx[0] == 0 or peak_idx[0] == len(freqs) - 1:
                warnings.append("strongest candidate sits at the edge of the "
                                "searched frequency range; widen the period "
                                "bounds")
            if span < 2.0 * best.period_days:
                warnings.append(
                    f"baseline {span:.2f} d covers fewer than two cycles of "
                    f"the {best.period_days:.4f} d candidate")
            phase_zero_mjd = mjd0
            fold_rows = fold(points, len(night_ids), best.frequency, mjd0)
    except PeriodError as exc:
        _err(422, str(exc))

    period_id = uuid.uuid4().hex
    if fold_rows:
        _PERIOD_CSV[period_id] = _fold_to_csv(fold_rows).encode()
    return PeriodSearchResponse(
        period_id=period_id,
        target_id=p.batches[0].photometry.target_id,
        n_points=len(points),
        n_nights=len(night_ids),
        mjd_span_days=span,
        period_min_days=p.period_min_days,
        period_max_days=p.period_max_days,
        excluded_points=[ExcludedPoint(**vars(e)) for e in excluded],
        frequencies=[float(f) for f in freqs],
        powers=powers,
        window_powers=window,
        n_degenerate_fits=n_degen,
        degenerate_reason=degen_reason,
        candidates=candidates,
        no_candidate_reason=no_candidate_reason,
        warnings=warnings,
        phase_zero_mjd=phase_zero_mjd,
        best_candidate=best,
        fold_points=[FoldedPoint(**r) for r in fold_rows],
    )


def _fold_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_FOLD_CSV_COLUMNS.split(","))
    for r in rows:
        writer.writerow([r["batch_index"], r["night_id"],
                         r["frame_index"], r["filename"], r["mjd"],
                         r["phase"], r["mag"], r["mag_err"],
                         r["mag_night_zero_removed"], r["model_mag"],
                         r["residual"]])
    return buf.getvalue()


@app.get("/api/period/{period_id}/csv")
def download_period_csv(period_id: str):
    blob = _PERIOD_CSV.get(period_id)
    if blob is None:
        _err(404, "unknown period_id or no folded light curve available")
    return Response(content=blob, media_type="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=period_fold.csv"})
