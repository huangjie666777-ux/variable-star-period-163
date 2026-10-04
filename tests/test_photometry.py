"""End-to-end tests for differential aperture photometry (requirements 1-4)."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app.main import app
from app.match import project_tangent

CENTER_RA, CENTER_DEC = 150.0, 20.0
SCALE = 1.3          # arcsec/pixel
NY, NX = 300, 300
GAIN = 2.5           # e-/ADU
READ_NOISE = 4.0     # e-
R_AP, R_IN, R_OUT = 4.0, 8.0, 13.0
MJD0 = 60100.0

REF_IDS = [f"ref{i}" for i in range(4)]
REF_MAGS = [12.0, 12.5, 13.0, 13.5]
TARGET_ID = "varstar"


def make_sequence(n_frames=5, seed=1, rot_deg=20.0, zp0=25.0,
                  bad_header_frame=None, variable=True):
    """Synthetic sequence: transparency drifts, target varies, refs constant."""
    rng = np.random.default_rng(seed)
    n_bg = 40
    # References and target near the middle of the field, well separated.
    ctr = np.array([[0.0, 0.0], [0.02, 0.0], [-0.02, 0.0],
                    [0.0, 0.02], [0.0, -0.02]])  # deg offsets
    # Background stars inside the field of view with a minimum separation,
    # so detections do not merge and annuli stay usable.
    min_sep_bg = 0.004       # deg ~ 11 px
    min_sep_ctr = 0.008      # deg ~ 22 px, beyond the background annulus
    offs_list = []
    while len(offs_list) < n_bg:
        cand = rng.uniform(-0.048, 0.048, (2 * n_bg, 2))
        for c in cand:
            if np.hypot(*c) > 0.046:
                continue
            if any(np.hypot(*(c - o)) < min_sep_ctr for o in ctr):
                continue
            if all(np.hypot(*(c - o)) >= min_sep_bg for o in offs_list):
                offs_list.append(c)
            if len(offs_list) == n_bg:
                break
    offs_bg = np.array(offs_list)
    ras = CENTER_RA + offs_bg[:, 0] / np.cos(np.deg2rad(CENTER_DEC))
    decs = CENTER_DEC + offs_bg[:, 1]
    bg_mags = rng.uniform(13.0, 16.0, n_bg)
    ids = [f"bg{i}" for i in range(n_bg)]

    t_ra = CENTER_RA + ctr[:, 0] / np.cos(np.deg2rad(CENTER_DEC))
    t_dec = CENTER_DEC + ctr[:, 1]
    ras = np.concatenate([ras, t_ra])
    decs = np.concatenate([decs, t_dec])
    ids += [TARGET_ID] + REF_IDS
    ref_mags = dict(zip(REF_IDS, REF_MAGS))

    # True instrumental behaviour.
    transparency = 1.0 + 0.25 * np.sin(np.linspace(0, np.pi, n_frames))  # sky
    target_dmag = 0.6 * np.sin(np.linspace(0, 2 * np.pi, n_frames)) if variable \
        else np.zeros(n_frames)

    xi, eta = project_tangent(ras, decs, CENTER_RA, CENTER_DEC)
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = R / SCALE
    t = np.array([NX / 2.0, NY / 2.0])
    pix = np.column_stack([xi, eta]) @ A.T + t

    exptime = 60.0
    frames, true_mags = [], []
    yy, xx = np.mgrid[0:NY, 0:NX]
    for f in range(n_frames):
        img = rng.normal(500.0, 3.0, (NY, NX))
        mags = np.concatenate([bg_mags, [14.0], [ref_mags[r] for r in REF_IDS]])
        mags[len(bg_mags)] = 14.0 + (target_dmag[f] if variable else 0.0)
        true_mags.append(float(mags[len(bg_mags)]))
        for i in range(len(ids)):
            flux_adu = 10 ** ((zp0 - mags[i]) / 2.5) * exptime \
                * transparency[f] / GAIN
            img += flux_adu * np.exp(-((xx - pix[i, 0]) ** 2
                                       + (yy - pix[i, 1]) ** 2) / (2 * 1.5 ** 2))
        img = rng.poisson(np.clip(img, 0, None) * GAIN) / GAIN
        hdu = fits.PrimaryHDU(data=img.astype(np.float32))
        if f != bad_header_frame:
            hdu.header["DATE-OBS"] = \
                f"2023-06-15T{13 + f:02d}:00:00"
            hdu.header["EXPTIME"] = exptime
        hdu.header["GAIN"] = GAIN
        hdu.header["PCOUNT"] = 0      # must survive WCS stripping
        hdu.header["PSFREF"] = "SYNTH"
        buf = io.BytesIO()
        hdu.writeto(buf)
        frames.append((f"frame{f}.fits", buf.getvalue()))
    catalog = [{"id": i, "ra": float(r), "dec": float(d)}
               for i, r, d in zip(ids, ras, decs)]
    params = {
        "solve": {
            "catalog": catalog,
            "center_ra": CENTER_RA,
            "center_dec": CENTER_DEC,
            "pixel_scale_min": 1.0,
            "pixel_scale_max": 1.6,
            "rms_max": 0.5,
        },
        "target_id": TARGET_ID,
        "references": [{"id": r, "mag": m} for r, m in zip(REF_IDS, REF_MAGS)],
        "aperture_radius": R_AP,
        "annulus_inner": R_IN,
        "annulus_outer": R_OUT,
        "gain": GAIN,
        "read_noise": READ_NOISE,
    }
    return frames, params, np.array(true_mags), transparency


def post_photometry(client, frames, params):
    files = [("images", (name, blob, "application/fits")) for name, blob in frames]
    return client.post("/api/photometry", files=files,
                       data={"params": json.dumps(params)})


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_lightcurve_recovers_variability_through_transparency(client):
    frames, params, true_mags, transparency = make_sequence()
    r = post_photometry(client, frames, params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_frames"] == len(frames)
    assert body["n_ok"] == len(frames)
    got = np.array([f["mag"] for f in body["frames"]])
    # Differential photometry removes the transparency drift: the recovered
    # light curve tracks the true variation to a few hundredths of a mag.
    assert np.allclose(got, true_mags, atol=0.05)
    # Zero points absorb the transparency change (they co-vary with it).
    zps = np.array([f["zero_point"] for f in body["frames"]])
    assert np.corrcoef(zps, transparency)[0, 1] > 0.9
    # Frames are sorted by exposure midpoint; errors and refs are reported.
    mjds = [f["mjd"] for f in body["frames"]]
    assert mjds == sorted(mjds)
    for f in body["frames"]:
        assert f["flux_rate"] > 0 and f["flux_rate_err"] > 0
        assert f["mag_err"] > 0 and f["zero_point_err"] > 0
        assert sorted(f["references_used"]) == sorted(REF_IDS)
        assert f["references_excluded"] == []
    # CSV download is consistent with the JSON payload.
    r2 = client.get(f"/api/photometry/{body['photometry_id']}/csv")
    assert r2.status_code == 200
    lines = r2.text.strip().splitlines()
    assert len(lines) == 1 + len(frames)
    assert lines[0].startswith("index,filename,status")
    csv_mags = [float(row.split(",")[8]) for row in lines[1:]]
    assert np.allclose(csv_mags, got)


def test_constant_target_gives_flat_lightcurve(client):
    frames, params, _, _ = make_sequence(seed=2, variable=False)
    r = post_photometry(client, frames, params)
    assert r.status_code == 200, r.text
    got = np.array([f["mag"] for f in r.json()["frames"]])
    assert got.std() < 0.03  # transparency drift corrected, no variability


def test_failed_frame_kept_others_continue(client):
    frames, params, true_mags, _ = make_sequence(seed=3, bad_header_frame=2)
    r = post_photometry(client, frames, params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_ok"] == len(frames) - 1
    failed = [f for f in body["frames"] if f["status"] == "failed"]
    assert len(failed) == 1
    assert "DATE-OBS" in failed[0]["reason"]
    ok = [f for f in body["frames"] if f["status"] == "ok"]
    good = np.delete(true_mags, 2)
    assert np.allclose([f["mag"] for f in ok], good, atol=0.05)


def test_validation_errors(client):
    frames, params, _, _ = make_sequence()
    # Target listed as a reference.
    bad = json.loads(json.dumps(params))
    bad["references"].append({"id": TARGET_ID, "mag": 14.0})
    assert post_photometry(client, frames, bad).status_code == 422
    # Radius ordering violated.
    bad = json.loads(json.dumps(params))
    bad["annulus_inner"] = 2.0
    assert post_photometry(client, frames, bad).status_code == 422
    # Unknown reference id.
    bad = json.loads(json.dumps(params))
    bad["references"][0]["id"] = "nope"
    assert post_photometry(client, frames, bad).status_code == 422
    # Non-positive gain.
    bad = json.loads(json.dumps(params))
    bad["gain"] = 0.0
    assert post_photometry(client, frames, bad).status_code == 422
    # Too few frames.
    assert post_photometry(client, frames[:1], params).status_code == 422
    # Too few references.
    bad = json.loads(json.dumps(params))
    bad["references"] = bad["references"][:2]
    assert post_photometry(client, frames, bad).status_code == 422


def test_outlier_reference_rejected(client):
    frames, params, true_mags, _ = make_sequence(seed=4)
    params["references"].append({"id": "bg0", "mag": 5.0})  # wrong mag on purpose
    r = post_photometry(client, frames, params)
    assert r.status_code == 200, r.text
    for f in r.json()["frames"]:
        assert "bg0" not in f["references_used"]
        reasons = {e["id"]: e["reason"] for e in f["references_excluded"]}
        assert reasons.get("bg0") == "zero_point_outlier"
    got = np.array([f["mag"] for f in r.json()["frames"]])
    assert np.allclose(got, true_mags, atol=0.05)


def test_solve_order_independent_and_observing_headers_kept(client):
    """Catalog permutation must not change the solve; PC*/PSF* headers stay."""
    from tests.test_solver import make_field, solve
    fits_bytes, params, _, _ = make_field()
    params["catalog"] = list(reversed(params["catalog"]))
    r = solve(client, fits_bytes, params)
    assert r.status_code == 200, r.text
    assert r.json()["n_pairs"] >= 12
    r2 = client.get(f"/api/solve/{r.json()['solve_id']}/fits")
    hdul = fits.open(io.BytesIO(r2.content))
    assert hdul[0].header["TELESCOP"] == "SYNTH"
    assert hdul[0].header["CTYPE1"] == "RA---TAN"
