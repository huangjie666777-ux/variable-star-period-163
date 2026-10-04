"""Multi-night period-search demo: synthetic variable star over 3 nights.

Generates 3 nights of calibrated FITS frames (target varies sinusoidally
with a 0.6 d period, transparency drifts, each night gets an extra
zero-point offset), runs differential photometry per night, then posts
the three photometry responses to /api/period and prints the candidates
and the folded light curve.

Usage:
    .venv/bin/python examples/period_demo.py [--url http://127.0.0.1:8152]
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
N_NIGHTS = 3
FRAMES_PER_NIGHT = 8
TRUE_PERIOD = 0.6          # days
MJD0 = 60100.0


def build_night(night, seed=11):
    """One night of synthetic frames; returns (frames, params, mjds)."""
    rng = np.random.default_rng(seed + night)
    n_bg = 40
    ang = rng.uniform(0, 2 * np.pi, n_bg)
    rad = rng.uniform(0.032, 0.048, n_bg)
    ras = CENTER_RA + rad * np.cos(ang) / np.cos(np.deg2rad(CENTER_DEC))
    decs = CENTER_DEC + rad * np.sin(ang)
    bg_mags = rng.uniform(13.0, 16.0, n_bg)
    ids = [f"bg{i}" for i in range(n_bg)]

    offs = np.array([[0.0, 0.0], [0.02, 0.0], [-0.02, 0.0],
                     [0.0, 0.02], [0.0, -0.02]])
    extra_ids = ["varstar", "ref0", "ref1", "ref2", "ref3"]
    extra_mags = [14.0, 12.0, 12.5, 13.0, 13.5]
    ras = np.concatenate(
        [ras, CENTER_RA + offs[:, 0] / np.cos(np.deg2rad(CENTER_DEC))])
    decs = np.concatenate([decs, CENTER_DEC + offs[:, 1]])
    ids += extra_ids

    xi, eta = project_tangent(ras, decs, CENTER_RA, CENTER_DEC)
    th = np.deg2rad(20.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pix = np.column_stack([xi, eta]) @ (R / SCALE).T + np.array([NX / 2, NY / 2])

    night_zp_offset = [0.0, 0.2, -0.15][night]   # nightly zero-point shift
    mjds = MJD0 + night * 1.05 + 0.3 * np.arange(FRAMES_PER_NIGHT) \
        / FRAMES_PER_NIGHT
    frames = []
    yy, xx = np.mgrid[0:NY, 0:NX]
    for k, mjd in enumerate(mjds):
        img = rng.normal(500.0, 3.0, (NY, NX))
        transparency = 1.0 + 0.2 * np.sin(2 * np.pi * (mjd - MJD0) / 1.3)
        target_mag = (14.0 + night_zp_offset
                      + 0.35 * np.sin(2 * np.pi * (mjd - MJD0) / TRUE_PERIOD))
        mags = np.concatenate([bg_mags, extra_mags])
        mags[len(bg_mags)] = target_mag
        for i in range(len(ids)):
            flux = 10 ** ((25.0 - mags[i]) / 2.5) * 60.0 * transparency / GAIN
            img += flux * np.exp(-((xx - pix[i, 0]) ** 2
                                   + (yy - pix[i, 1]) ** 2) / (2 * 1.5 ** 2))
        img = rng.poisson(np.clip(img, 0, None) * GAIN) / GAIN
        hdu = fits.PrimaryHDU(data=img.astype(np.float32))
        from astropy.time import Time
        hdu.header["DATE-OBS"] = Time(mjd, format="mjd").isot
        hdu.header["EXPTIME"] = 60.0
        hdu.header["FILTER"] = "V"
        buf = io.BytesIO()
        hdu.writeto(buf)
        frames.append((f"night{night + 1}_{k:02d}.fits", buf.getvalue()))

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
    return frames, params, mjds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8152")
    args = ap.parse_args()

    batches = []
    for night in range(N_NIGHTS):
        frames, params, _ = build_night(night)
        files = [("images", (n, b, "application/fits")) for n, b in frames]
        r = httpx.post(f"{args.url}/api/photometry", files=files,
                       data={"params": json.dumps(params)}, timeout=300.0)
        r.raise_for_status()
        body = r.json()
        print(f"night{night + 1}: photometry_id={body['photometry_id']}  "
              f"ok {body['n_ok']}/{body['n_frames']}")
        batches.append({"night_id": f"night{night + 1}", "photometry": body})

    r = httpx.post(f"{args.url}/api/period",
                   json={"batches": batches, "period_min_days": 0.2,
                         "period_max_days": 2.0}, timeout=300.0)
    r.raise_for_status()
    body = r.json()
    print(f"\nperiod_id: {body['period_id']}  points={body['n_points']}  "
          f"nights={body['n_nights']}  span={body['mjd_span_days']:.2f} d")
    print(f"true period: {TRUE_PERIOD} d")
    for c in body["candidates"]:
        print(f"  candidate: P={c['period_days']:.4f} d  "
              f"f={c['frequency']:.4f}/d  power={c['power']:.3f}")
    for w in body["warnings"]:
        print(f"  warning: {w}")
    print(f"{'phase':>6} {'mag':>7} {'night-zero-removed':>18} {'model':>7} "
          f"{'resid':>7}")
    for fp in body["fold_points"][:10]:
        print(f"{fp['phase']:6.3f} {fp['mag']:7.3f} "
              f"{fp['mag_night_zero_removed']:18.3f} {fp['model_mag']:7.3f} "
              f"{fp['residual']:7.3f}")
    csv = httpx.get(f"{args.url}/api/period/{body['period_id']}/csv")
    csv.raise_for_status()
    out = Path(__file__).parent / "period_fold.csv"
    out.write_text(csv.text)
    print(f"fold CSV saved to {out}")


if __name__ == "__main__":
    main()
