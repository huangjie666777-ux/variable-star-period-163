"""End-to-end tests on synthetic star fields: rotation, dropouts, false sources."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from fastapi.testclient import TestClient

from app.main import app
from app.match import project_tangent

CENTER_RA, CENTER_DEC = 150.0, 20.0
SCALE = 1.3  # arcsec/pixel
NY, NX = 400, 400


def make_field(seed=0, rot_deg=35.0, mirror=False, n_stars=60,
               n_dropout=8, n_false=6):
    rng = np.random.default_rng(seed)
    # Random catalog stars within ~0.4 deg of the center.
    ras = CENTER_RA + rng.uniform(-0.06, 0.06, n_stars) / np.cos(np.deg2rad(CENTER_DEC))
    decs = CENTER_DEC + rng.uniform(-0.06, 0.06, n_stars)
    ids = [f"star{i:03d}" for i in range(n_stars)]
    xi, eta = project_tangent(ras, decs, CENTER_RA, CENTER_DEC)
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if mirror:
        R = R @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    A = R / SCALE  # 1 arcsec on sky = 1/SCALE px -> SCALE arcsec/pixel
    t = np.array([NX / 2.0, NY / 2.0])
    pix = np.column_stack([xi, eta]) @ A.T + t
    inside = ((pix[:, 0] > 10) & (pix[:, 0] < NX - 10)
              & (pix[:, 1] > 10) & (pix[:, 1] < NY - 10))
    # Drop some in-FOV stars to simulate missed detections.
    idx = np.flatnonzero(inside)
    rng.shuffle(idx)
    kept = idx[n_dropout:]
    img = rng.normal(1000.0, 5.0, (NY, NX))
    fluxes = rng.uniform(2000, 40000, n_stars)
    yy, xx = np.mgrid[0:NY, 0:NX]
    for i in kept:
        img += fluxes[i] * np.exp(-((xx - pix[i, 0]) ** 2
                                    + (yy - pix[i, 1]) ** 2) / (2 * 1.5 ** 2))
    # False sources: noise spikes not in the catalog.
    for _ in range(n_false):
        fx, fy = rng.uniform(20, NX - 20), rng.uniform(20, NY - 20)
        img += 30000 * np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * 1.2 ** 2))
    # A hot pixel that must be rejected.
    img[50, 50] += 50000
    hdu = fits.PrimaryHDU(data=img.astype(np.float32))
    hdu.header["TELESCOP"] = "SYNTH"
    hdu.header["CRPIX1"] = 1.0  # stale, conflicting WCS to be stripped
    hdu.header["CRVAL1"] = 0.0
    buf = io.BytesIO()
    hdu.writeto(buf)
    catalog = [{"id": i, "ra": float(r), "dec": float(d)}
               for i, r, d in zip(ids, ras, decs)]
    params = {
        "catalog": catalog,
        "center_ra": CENTER_RA,
        "center_dec": CENTER_DEC,
        "pixel_scale_min": 1.0,
        "pixel_scale_max": 1.6,
        "rms_max": 0.5,
    }
    return buf.getvalue(), params, pix, inside


def solve(client, fits_bytes, params):
    r = client.post("/api/solve",
                    files={"image": ("img.fits", fits_bytes, "application/fits")},
                    data={"params": json.dumps(params)})
    return r


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_solve_rotated_field_with_dropouts_and_false_sources(client):
    fits_bytes, params, pix, inside = make_field()
    r = solve(client, fits_bytes, params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_pairs"] >= 12
    assert body["rms_arcsec"] <= 0.5
    assert not body["mirrored"]
    # Returned WCS must map pixels back to the right sky positions.
    w = WCS({k: v for k, v in body["wcs"].items()})
    cat = {c["id"]: c for c in params["catalog"]}
    for pr in body["pairs"][:10]:
        ra, dec = w.all_pix2world([[pr["x"] + 1, pr["y"] + 1]], 1)[0]
        c = cat[pr["id"]]
        sep = np.hypot((ra - c["ra"]) * np.cos(np.deg2rad(c["dec"])), dec - c["dec"])
        assert sep * 3600 < 1.0
    # Download the solved FITS and verify WCS + preserved data/headers.
    r2 = client.get(f"/api/solve/{body['solve_id']}/fits")
    assert r2.status_code == 200
    hdul = fits.open(io.BytesIO(r2.content))
    h = hdul[0].header
    assert h["TELESCOP"] == "SYNTH"
    assert h["CTYPE1"] == "RA---TAN"
    assert hdul[0].data.shape == (NY, NX)
    det_cd = abs(h["CD1_1"] * h["CD2_2"] - h["CD1_2"] * h["CD2_1"])
    assert 3600 * np.sqrt(det_cd) == pytest.approx(SCALE, rel=0.05)


def test_solve_mirrored_field(client):
    fits_bytes, params, _, _ = make_field(seed=3, rot_deg=80.0, mirror=True)
    r = solve(client, fits_bytes, params)
    assert r.status_code == 200, r.text
    assert r.json()["mirrored"]


def test_reject_star_behind_projection_center(client):
    fits_bytes, params, _, _ = make_field()
    params["catalog"].append({"id": "far", "ra": (CENTER_RA + 180) % 360,
                              "dec": -CENTER_DEC})
    r = solve(client, fits_bytes, params)
    assert r.status_code == 422
    assert "90 deg" in r.json()["detail"]


def test_reject_impossible_rms(client):
    fits_bytes, params, _, _ = make_field()
    params["rms_max"] = 1e-4
    r = solve(client, fits_bytes, params)
    assert r.status_code == 422
    assert "RMS" in r.json()["detail"]


def test_reject_bad_params(client):
    fits_bytes, params, _, _ = make_field()
    params["pixel_scale_max"] = 0.5  # < min
    r = solve(client, fits_bytes, params)
    assert r.status_code == 422

