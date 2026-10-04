"""Light-curve demo: synthetic variable star + drifting transparency.

Generates 6 calibrated FITS frames (same filter) in which the target varies
by +-0.3 mag while the sky transparency drifts by +-0.25 mag, then posts
them to a running server and prints the recovered light curve.

Usage:
    .venv/bin/python examples/lightcurve_demo.py [--url http://127.0.0.1:8152]
"""
from __future__ import annotations

import argparse
import io
import json

import httpx
import numpy as np
from astropy.io import fits

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.match import project_tangent

CENTER_RA, CENTER_DEC = 150.0, 20.0
SCALE = 1.3
NY, NX = 300, 300
GAIN, READ_NOISE = 2.5, 4.0
N_FRAMES = 6


def build_frames(seed=7):
    rng = np.random.default_rng(seed)
    n_bg = 40
    ang = rng.uniform(0, 2 * np.pi, n_bg)
    rad = rng.uniform(0.032, 0.048, n_bg)
    ras = CENTER_RA + rad * np.cos(ang) / np.cos(np.deg2rad(CENTER_DEC))
    decs = CENTER_DEC + rad * np.sin(ang)
    bg_mags = rng.uniform(13.0, 16.0, n_bg)
    ids = [f"bg{i}" for i in range(n_bg)]

    offs = np.array([[0.0, 0.0], [0.02, 0.0], [-0.02, 0.0], [0.0, 0.02], [0.0, -0.02]])
    extra_ids = ["varstar", "ref0", "ref1", "ref2", "ref3"]
    extra_mags = [14.0, 12.0, 12.5, 13.0, 13.5]
    ras = np.concatenate([ras, CENTER_RA + offs[:, 0] / np.cos(np.deg2rad(CENTER_DEC))])
    decs = np.concatenate([decs, CENTER_DEC + offs[:, 1]])
    ids += extra_ids

    transparency = 1.0 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES))
    target_true = 14.0 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, N_FRAMES))

    xi, eta = project_tangent(ras, decs, CENTER_RA, CENTER_DEC)
    th = np.deg2rad(20.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pix = np.column_stack([xi, eta]) @ (R / SCALE).T + np.array([NX / 2, NY / 2])

    frames = []
    yy, xx = np.mgrid[0:NY, 0:NX]
    for f in range(N_FRAMES):
        img = rng.normal(500.0, 3.0, (NY, NX))
        mags = np.concatenate([bg_mags, extra_mags])
        mags[len(bg_mags)] = target_true[f]
        for i in range(len(ids)):
            flux = 10 ** ((25.0 - mags[i]) / 2.5) * 60.0 * transparency[f] / GAIN
            img += flux * np.exp(-((xx - pix[i, 0]) ** 2
                                   + (yy - pix[i, 1]) ** 2) / (2 * 1.5 ** 2))
        img = rng.poisson(np.clip(img, 0, None) * GAIN) / GAIN
        hdu = fits.PrimaryHDU(data=img.astype(np.float32))
        hdu.header["DATE-OBS"] = f"2023-06-15T{13 + f:02d}:30:00"
        hdu.header["EXPTIME"] = 60.0
        hdu.header["FILTER"] = "V"
        buf = io.BytesIO()
        hdu.writeto(buf)
        frames.append((f"night1_{f:02d}.fits", buf.getvalue()))

    catalog = [{"id": i, "ra": float(r), "dec": float(d)}
               for i, r, d in zip(ids, ras, decs)]
    params = {
        "solve": {"catalog": catalog, "center_ra": CENTER_RA,
                  "center_dec": CENTER_DEC, "pixel_scale_min": 1.0,
                  "pixel_scale_max": 1.6, "rms_max": 0.5},
        "target_id": "varstar",
        "references": [{"id": f"ref{i}", "mag": m}
                       for i, m in enumerate([12.0, 12.5, 13.0, 13.5])],
        "aperture_radius": 4.0, "annulus_inner": 8.0, "annulus_outer": 13.0,
        "gain": GAIN, "read_noise": READ_NOISE,
    }
    return frames, params, target_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8152")
    args = ap.parse_args()

    frames, params, target_true = build_frames()
    files = [("images", (n, b, "application/fits")) for n, b in frames]
    r = httpx.post(f"{args.url}/api/photometry",
                   files=files, data={"params": json.dumps(params)},
                   timeout=120.0)
    r.raise_for_status()
    body = r.json()
    print(f"photometry_id: {body['photometry_id']}  "
          f"ok {body['n_ok']}/{body['n_frames']}")
    print(f"{'MJD':>14} {'mag':>7} {'err':>6} {'true':>7} {'zp':>7}")
    for fr, tm in zip(body["frames"], target_true):
        print(f"{fr['mjd']:14.5f} {fr['mag']:7.3f} {fr['mag_err']:6.3f} "
              f"{tm:7.3f} {fr['zero_point']:7.3f}")
    csv = httpx.get(f"{args.url}/api/photometry/{body['photometry_id']}/csv")
    csv.raise_for_status()
    out = Path(__file__).parent / "lightcurve.csv"
    out.write_text(csv.text)
    print(f"CSV saved to {out}")


if __name__ == "__main__":
    main()

