"""Build a TAN WCS from the solved affine and merge it into the FITS HDU (req. 5)."""
from __future__ import annotations

import re

import numpy as np
from astropy.io import fits

# Old-WCS keywords that must be removed to avoid conflicting solutions.
# Numbered keywords are matched exactly (e.g. PC1_1, CDELT2, PV2_3) so that
# unrelated observing headers that merely share a prefix (PCOUNT, PSFREF,
# CTYPE-like custom keys, ...) are preserved.
_OLD_WCS_PATTERNS = tuple(re.compile(p) for p in (
    r"CD\d+_\d+", r"PC\d+_\d+", r"PV\d+_\d+", r"PS\d+_\d+",
    r"CDELT\d+", r"CROTA\d+", r"CRPIX\d+", r"CRVAL\d+",
    r"CTYPE\d+", r"CUNIT\d+",
    r"A_\d+_\d+", r"B_\d+_\d+", r"AP_\d+_\d+", r"BP_\d+_\d+",
))
_OLD_WCS_EXACT = {"WCSAXES", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "EPOCH"}


def affine_to_wcs_header(A: np.ndarray, b: np.ndarray,
                         center_ra: float, center_dec: float) -> dict:
    """Convert pixel = A @ tangent_arcsec + b into a TAN WCS header.

    Pixels are 0-based internally; FITS CRPIX is 1-based, hence the +1 shift.
    The inverse affine maps pixel offsets to tangent-plane degrees.
    """
    Ainv = np.linalg.inv(A) / 3600.0  # deg per pixel
    crpix = b + 1.0  # FITS 1-based reference pixel of the tangent point
    return {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": float(center_ra),
        "CRVAL2": float(center_dec),
        "CRPIX1": float(crpix[0]),
        "CRPIX2": float(crpix[1]),
        "CD1_1": float(Ainv[0, 0]),
        "CD1_2": float(Ainv[0, 1]),
        "CD2_1": float(Ainv[1, 0]),
        "CD2_2": float(Ainv[1, 1]),
        "RADESYS": "ICRS",
    }


def strip_old_wcs(header: fits.Header) -> None:
    for key in list(header.keys()):
        if key in _OLD_WCS_EXACT:
            del header[key]
            continue
        if key in ("COMMENT", "HISTORY"):
            continue
        if any(p.fullmatch(key) for p in _OLD_WCS_PATTERNS):
            try:
                del header[key]
            except KeyError:
                pass


def solved_fits_bytes(hdu: fits.PrimaryHDU, wcs_header: dict) -> bytes:
    """Return FITS bytes with original pixels/headers plus the new WCS only."""
    import io

    out = fits.PrimaryHDU(data=hdu.data, header=hdu.header.copy())
    strip_old_wcs(out.header)
    for k, v in wcs_header.items():
        out.header[k] = v
    buf = io.BytesIO()
    out.writeto(buf, overwrite=True)
    return buf.getvalue()
