"""Request/response schemas and server-side limits for the plate solver."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Server-side resource limits (requirement 1).
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGE_SIDE = 8192
MAX_CATALOG_STARS = 2000
MAX_DETECTED_SOURCES = 500
MAX_FITS_BYTES = 64 * 1024 * 1024
MIN_PHOTOMETRY_FRAMES = 2
MAX_PHOTOMETRY_FRAMES = 20


class CatalogStar(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    ra: float = Field(ge=0.0, lt=360.0, description="ICRS RA in degrees")
    dec: float = Field(ge=-90.0, le=90.0, description="ICRS Dec in degrees")


class SolveParams(BaseModel):
    catalog: List[CatalogStar] = Field(min_length=6, max_length=MAX_CATALOG_STARS)
    center_ra: float = Field(ge=0.0, lt=360.0)
    center_dec: float = Field(ge=-90.0, le=90.0)
    pixel_scale_min: float = Field(gt=0.0, le=3600.0, description="arcsec/pixel")
    pixel_scale_max: float = Field(gt=0.0, le=3600.0, description="arcsec/pixel")
    rms_max: float = Field(gt=0.0, le=3600.0, description="max acceptable RMS, arcsec")
    threshold_sigma: float = Field(default=5.0, gt=0.0, le=100.0)
    saturation: Optional[float] = Field(
        default=None, description="pixels >= this are treated as saturated"
    )
    max_pairs: int = Field(default=200, ge=6, le=1000)

    @field_validator("pixel_scale_max")
    @classmethod
    def _scale_order(cls, v, info):
        lo = info.data.get("pixel_scale_min")
        if lo is not None and v < lo:
            raise ValueError("pixel_scale_max must be >= pixel_scale_min")
        return v

    @field_validator("catalog")
    @classmethod
    def _unique_ids(cls, v):
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog ids must be unique")
        return v


class PairOut(BaseModel):
    id: str
    x: float  # 0-based pixel centroid
    y: float
    residual_arcsec: float


class SolveResponse(BaseModel):
    solve_id: str
    n_pairs: int
    rms_arcsec: float
    mirrored: bool
    pairs: List[PairOut]
    wcs: dict


# ---------------------------------------------------------------------------
# Differential aperture photometry (requirements 1-4)
# ---------------------------------------------------------------------------

class ReferenceStar(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    mag: float = Field(gt=-30.0, lt=30.0, description="known catalog magnitude")


class PhotometryParams(BaseModel):
    solve: SolveParams
    target_id: str = Field(min_length=1, max_length=64)
    references: List[ReferenceStar] = Field(min_length=3, max_length=50)
    aperture_radius: float = Field(gt=0.0, le=200.0, description="pixels")
    annulus_inner: float = Field(gt=0.0, le=500.0, description="pixels")
    annulus_outer: float = Field(gt=0.0, le=1000.0, description="pixels")
    gain: float = Field(gt=0.0, le=1e6, description="electrons per ADU")
    read_noise: float = Field(ge=0.0, le=1e4, description="electrons")

    @model_validator(mode="after")
    def _check_relations(self):
        if not (self.aperture_radius < self.annulus_inner < self.annulus_outer):
            raise ValueError(
                "radii must satisfy aperture_radius < annulus_inner < annulus_outer")
        cat_ids = {s.id for s in self.solve.catalog}
        if self.target_id not in cat_ids:
            raise ValueError("target_id is not present in the solve catalog")
        ref_ids = [r.id for r in self.references]
        if len(set(ref_ids)) != len(ref_ids):
            raise ValueError("reference star ids must be unique")
        if self.target_id in ref_ids:
            raise ValueError("target star must not appear among the references")
        missing = [i for i in ref_ids if i not in cat_ids]
        if missing:
            raise ValueError(f"reference ids not in catalog: {missing}")
        return self


class ExcludedReference(BaseModel):
    id: str
    reason: str


class FrameResult(BaseModel):
    index: int
    filename: str
    status: str  # "ok" or "failed"
    reason: Optional[str] = None
    mjd: Optional[float] = None
    exptime: Optional[float] = None
    flux_rate: Optional[float] = None
    flux_rate_err: Optional[float] = None
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    zero_point: Optional[float] = None
    zero_point_err: Optional[float] = None
    references_used: List[str] = []
    references_excluded: List[ExcludedReference] = []


class PhotometryResponse(BaseModel):
    photometry_id: str
    target_id: str = ""
    n_frames: int
    n_ok: int
    frames: List[FrameResult]


# ---------------------------------------------------------------------------
# Multi-night period search (requirement 5)
# ---------------------------------------------------------------------------

class PeriodBatch(BaseModel):
    night_id: str = Field(min_length=1, max_length=64)
    photometry: PhotometryResponse


class PeriodogramParams(BaseModel):
    batches: List[PeriodBatch] = Field(min_length=1, max_length=100)
    period_min: float = Field(gt=0.0, le=1e6, description="days")
    period_max: float = Field(gt=0.0, le=1e7, description="days")

    @model_validator(mode="after")
    def _check_period_order(self):
        if self.period_max <= self.period_min:
            raise ValueError("period_max must be greater than period_min")
        return self


class ExcludedPointOut(BaseModel):
    night_id: str
    batch_index: int
    frame_index: int
    filename: str
    reason: str
    mjd: Optional[float] = None


class PeakOut(BaseModel):
    frequency: float          # cycles/day
    period: float             # days
    power: float
    boundary: bool = False


class DegenerateFitOut(BaseModel):
    frequency: float
    reason: str


class PhasedPointOut(BaseModel):
    night_id: str
    batch_index: int
    frame_index: int
    filename: str
    mjd: float
    mag: float
    mag_err: float
    phase: float
    detrended_mag: float      # mag minus this night's fitted constant
    model_mag: float          # full joint model (night constant + sinusoid)
    residual: float


class CandidateModel(BaseModel):
    period: float
    phase_zero_mjd: float
    amplitude: float
    night_offsets: dict


class PeriodogramResponse(BaseModel):
    periodogram_id: str
    target_id: str
    n_points: int
    n_nights: int
    n_excluded: int
    span_days: float
    df: float
    frequencies: List[float]
    power: List[Optional[float]]     # null where the fit was degenerate
    window_power: List[Optional[float]]
    peaks: List[PeakOut]
    warnings: List[str]
    degenerate_fits: List[DegenerateFitOut]
    excluded: List[ExcludedPointOut]
    candidate_period: Optional[float] = None
    no_candidate_reason: Optional[str] = None
    candidate_model: Optional[CandidateModel] = None
    phased_points: Optional[List[PhasedPointOut]] = None
