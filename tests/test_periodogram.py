"""Tests for the multi-night period search (requirement 5)."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

MJD0 = 60100.0
TRUE_PERIOD = 0.62


def make_batches(n_nights=3, per_night=8, period=TRUE_PERIOD, seed=3,
                 bad_frames=()):
    """Synthetic photometry responses: sinusoid + per-night offsets."""
    rng = np.random.default_rng(seed)
    batches = []
    for night in range(n_nights):
        frames = []
        offset = 0.15 * (night - 1)
        for f in range(per_night):
            mjd = MJD0 + night + (f + 0.5) / 24.0
            mag = 14.0 + 0.4 * np.sin(2 * np.pi * (mjd - MJD0) / period) \
                + offset
            mag += rng.normal(0.0, 0.02)
            if (night, f) in bad_frames:
                frames.append({"index": f, "filename": f"n{night}_{f}.fits",
                               "status": "failed", "reason": "solve failed",
                               "mjd": mjd})
                continue
            frames.append({"index": f, "filename": f"n{night}_{f}.fits",
                           "status": "ok", "mjd": mjd, "exptime": 60.0,
                           "mag": float(mag), "mag_err": 0.02})
        batches.append({
            "night_id": f"night{night}",
            "photometry": {"photometry_id": f"ph{night}",
                           "target_id": "varstar",
                           "n_frames": len(frames),
                           "n_ok": sum(fr["status"] == "ok"
                                       for fr in frames),
                           "frames": frames},
        })
    return batches


def post(client, batches, pmin=0.3, pmax=2.0):
    params = {"batches": batches, "period_min": pmin, "period_max": pmax}
    return client.post("/api/periodogram",
                       data={"params": json.dumps(params)})


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_recovers_true_period(client):
    r = post(client, make_batches())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_points"] == 24
    assert body["n_nights"] == 3
    assert body["candidate_period"] is not None
    assert abs(body["candidate_period"] - TRUE_PERIOD) < 2 * body["df"] \
        / body["candidate_period"] ** 2 + 0.01
    top = body["peaks"][0]
    assert abs(top["period"] - body["candidate_period"]) < 1e-12
    assert top["power"] > 0.9
    # Frequency grid: step <= 1/(5*span), within 20000 points.
    freqs = body["frequencies"]
    assert len(freqs) == len(body["power"]) == len(body["window_power"])
    assert len(freqs) <= 20000
    steps = np.diff(freqs)
    assert np.all(steps <= 1.0 / (5.0 * body["span_days"]) * 1.0000001)
    assert min(freqs) >= 1.0 / 2.0 - 1e-12
    assert max(freqs) <= 1.0 / 0.3 + 1e-12
    # Window power peaks near zero frequency offset structure, sane range.
    assert all(0.0 <= w <= 1.0 for w in body["window_power"]
               if w is not None)
    # "Candidate, not confirmed" disclaimer is present.
    assert any("not a confirmed period" in w for w in body["warnings"])
    # Phased points carry provenance and model values.
    pts = body["phased_points"]
    assert len(pts) == body["n_points"]
    p0 = pts[0]
    for key in ("night_id", "batch_index", "frame_index", "filename",
                "mjd", "mag", "mag_err", "phase", "detrended_mag",
                "model_mag", "residual"):
        assert key in p0
    assert all(0.0 <= p["phase"] < 1.0 for p in pts)
    assert abs(body["candidate_model"]["phase_zero_mjd"]
               - min(p["mjd"] for p in pts)) < 1e-9
    # CSV matches the JSON phased points.
    csv_r = client.get(f"/api/periodogram/{body['periodogram_id']}/csv")
    assert csv_r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_r.text)))
    assert len(rows) == len(pts)
    assert rows[0]["night_id"] == pts[0]["night_id"]
    assert abs(float(rows[0]["phase"]) - pts[0]["phase"]) < 1e-9


def test_failed_frames_excluded_with_reason_and_provenance(client):
    batches = make_batches(per_night=9, bad_frames={(0, 0), (1, 3)})
    r = post(client, batches)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_points"] == 25
    assert body["n_excluded"] == 2
    reasons = {e["filename"]: e for e in body["excluded"]}
    assert reasons["n0_0.fits"]["reason"].startswith("frame failed")
    assert reasons["n0_0.fits"]["night_id"] == "night0"
    assert reasons["n1_3.fits"]["mjd"] is not None


def test_peak_spacing_and_order(client):
    body = post(client, make_batches(), pmin=0.2, pmax=5.0).json()
    peaks = body["peaks"]
    assert len(peaks) <= 3
    powers = [p["power"] for p in peaks]
    assert powers == sorted(powers, reverse=True)
    min_sep = 1.0 / body["span_days"]
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            assert abs(peaks[i]["frequency"] - peaks[j]["frequency"]) \
                >= min_sep - 1e-12


def test_validation_errors(client):
    # Too few valid points.
    r = post(client, make_batches(n_nights=2, per_night=5))
    assert r.status_code == 422
    assert "at least 20" in r.json()["detail"]
    # Single night.
    r = post(client, make_batches(n_nights=1, per_night=30))
    assert r.status_code == 422
    assert "at least 2 nights" in r.json()["detail"]
    # Reversed period bounds.
    r = post(client, make_batches(), pmin=2.0, pmax=0.3)
    assert r.status_code == 422
    # Mixed target ids.
    batches = make_batches()
    batches[1]["photometry"]["target_id"] = "other"
    r = post(client, batches)
    assert r.status_code == 422
    assert "target_id" in r.json()["detail"]
    # Frequency grid exceeding 20000 points: span ~2.04 d -> df tiny,
    # so a very wide period range overflows the limit.
    r = post(client, make_batches(), pmin=0.0001, pmax=100.0)
    assert r.status_code == 422
    assert "20000" in r.json()["detail"]


def test_constant_star_no_strong_peak_and_baseline_warning(client):
    batches = make_batches(period=1e9)   # effectively constant
    body = post(client, batches, pmin=1.5, pmax=2.0).json()
    assert body["n_points"] == 24
    if body["peaks"]:
        assert body["peaks"][0]["power"] < 0.5
    # Baseline ~2.04 d < 2 * period for any peak above 1.02 d.
    assert any("fewer than two cycles" in w for w in body["warnings"]) \
        or not body["peaks"]
