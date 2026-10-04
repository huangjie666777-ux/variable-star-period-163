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
                     CandidateModel, DegenerateFitOut, ExcludedPointOut,
                     ExcludedReference, FrameResult, PairOut, PeakOut,
                     PeriodogramParams, PeriodogramResponse, PhasedPointOut,
                     PhotometryParams, PhotometryResponse, SolveParams,
                     SolveResponse)
from .periodogram import (PeriodogramError, compute_periodogram, phase_fold)
from .photometry import measure_aperture
from .pipeline import PipelineError, solve_field
from .samples import SampleError, collect_samples
from .wcsbuild import solved_fits_bytes

app = FastAPI(title="Star Field Plate Solver + Differential Photometry")

# In-memory stores keyed by result id.
_RESULTS: dict[str, bytes] = {}
_CSV_RESULTS: dict[str, bytes] = {}
_PERIODOGRAM_CSV: dict[str, bytes] = {}


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


def _measure_frame(data: np.ndarray, p: PhotometryParams,
                   mjd: float, exptime: float) -> FrameResult:
    """Solve one frame, photometer target + references, calibrate.

    Failures keep whatever was already known (MJD, EXPTIME, excluded
    reference stars) instead of dropping the whole record.
    """
    fr = FrameResult(index=-1, filename="", status="failed",
                     mjd=mjd, exptime=exptime)
    try:
        res = solve_field(data, p.solve)  # raises PipelineError
    except PipelineError as exc:
        fr.reason = str(exc)
        return fr

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

    fr.references_excluded = excluded
    if len(ok_ids) < 3:
        fr.reason = (
            f"fewer than 3 usable reference stars ({len(ok_ids)} available)")
        return fr

    # Zero point from valid references; robust outlier rejection.
    try:
        zp, zp_err, kept, dropped = estimate_zero_point(
            ok_mags, ok_rates, ok_rate_errs)
    except CalibrationError as exc:
        fr.reason = str(exc)
        return fr
    for i in dropped:
        excluded.append(ExcludedReference(id=ok_ids[i],
                                          reason="zero_point_outlier"))
    used = [ok_ids[i] for i in kept]
    fr.references_excluded = excluded

    # Target last: it never participates in the calibration.
    t = measure(p.target_id)
    if t.flag is not None:
        fr.reason = f"target measurement failed: {t.flag}"
        return fr
    rate = t.flux / exptime
    rate_err = t.flux_err / exptime
    mag, mag_err = target_magnitude(rate, rate_err, zp, zp_err)

    fr.status = "ok"
    fr.flux_rate = rate
    fr.flux_rate_err = rate_err
    fr.mag = mag
    fr.mag_err = mag_err
    fr.zero_point = zp
    fr.zero_point_err = zp_err
    fr.references_used = used
    return fr


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
            # Single-frame failure: keep the record, continue with the rest.
            frames.append(FrameResult(index=idx, filename=name,
                                      status="failed", reason=str(exc)))
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

_PHASED_CSV_COLUMNS = ("night_id,batch_index,frame_index,filename,mjd,mag,"
                       "mag_err,phase,detrended_mag,model_mag,residual")


def _phased_to_csv(rows: list[PhasedPointOut]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_PHASED_CSV_COLUMNS.split(","))
    for r in rows:
        writer.writerow([r.night_id, r.batch_index, r.frame_index,
                         r.filename, r.mjd, r.mag, r.mag_err, r.phase,
                         r.detrended_mag, r.model_mag, r.residual])
    return buf.getvalue()


@app.post("/api/periodogram", response_model=PeriodogramResponse)
async def periodogram(params: str = Form(...)):
    try:
        p = PeriodogramParams.model_validate_json(params)
    except Exception as exc:
        _err(422, f"invalid parameters: {exc}")

    # Sample collection (app/samples.py).
    try:
        samples = collect_samples(p.batches)
    except SampleError as exc:
        _err(422, str(exc))

    # Joint weighted fit over the frequency grid (app/periodogram.py).
    try:
        result = compute_periodogram(samples, p.period_min, p.period_max)
    except PeriodogramError as exc:
        _err(422, str(exc))

    peaks = [PeakOut(frequency=pk.frequency, period=pk.period,
                     power=pk.power, boundary=pk.boundary)
             for pk in result.peaks]

    # Phase-fold on the strongest candidate, if any.
    candidate_period = None
    no_candidate_reason = None
    candidate_model = None
    phased: list[PhasedPointOut] | None = None
    if not result.peaks:
        no_candidate_reason = ("no local power maximum found in the "
                               "searched frequency range")
    else:
        candidate_period = result.peaks[0].period
        try:
            rows, model_info = phase_fold(samples, candidate_period)
            candidate_model = CandidateModel(**model_info)
            phased = [PhasedPointOut(**row) for row in rows]
        except PeriodogramError as exc:
            no_candidate_reason = str(exc)
            candidate_period = None

    periodogram_id = uuid.uuid4().hex
    if phased is not None:
        _PERIODOGRAM_CSV[periodogram_id] = _phased_to_csv(phased).encode()

    return PeriodogramResponse(
        periodogram_id=periodogram_id,
        target_id=samples.target_id,
        n_points=len(samples.points),
        n_nights=len(samples.night_ids),
        n_excluded=len(samples.excluded),
        span_days=samples.span_days,
        df=result.df,
        frequencies=[float(f) for f in result.frequencies],
        power=[None if not np.isfinite(v) else float(v)
               for v in result.power],
        window_power=[None if not np.isfinite(v) else float(v)
                      for v in result.window_power],
        peaks=peaks,
        warnings=result.warnings,
        degenerate_fits=[DegenerateFitOut(**d) for d in result.degenerate],
        excluded=[ExcludedPointOut(**vars(e)) for e in samples.excluded],
        candidate_period=candidate_period,
        no_candidate_reason=no_candidate_reason,
        candidate_model=candidate_model,
        phased_points=phased)


@app.get("/api/periodogram/{periodogram_id}/csv")
def download_periodogram_csv(periodogram_id: str):
    blob = _PERIODOGRAM_CSV.get(periodogram_id)
    if blob is None:
        _err(404, "unknown periodogram_id or no phased points available")
    return Response(content=blob, media_type="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=periodogram_phased.csv"})
