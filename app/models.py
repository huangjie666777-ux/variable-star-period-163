"""Request/response schemas and server-side limits for the plate solver."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Server-side resource limits (requirement 1).
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGE_SIDE = 8192
MAX_CATALOG_STARS = 2000
MAX_DETECTED_SOURCES = 500
MAX_FITS_BYTES = 64 * 1024 * 1024


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

