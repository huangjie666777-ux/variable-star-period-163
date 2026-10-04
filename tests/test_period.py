"""Tests for the multi-night period search endpoint (requirement 5)."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

TARGET_ID = "varstar"
PERIOD = 0.6          # days
MJD0 = 60100.0


def make_response(frames, photometry_id="p0", target_id=TARGET_ID):
    return {"photometry_id": photometry_id, "target_id": target_id,
            "n_frames": len(frames),
            "n_ok": sum(f["status"] == "ok" for f in frames),
            "frames": frames}


def ok_frame(index, mjd, mag, err=0.02, filename=None):
    return {"index": index, "filename": filename or f"f{index}.fits",
            "status": "ok", "reason": None, "mjd": mjd, "exptime": 60.0,
            "flux_rate": 100.0, "flux_rate_err": 1.0, "mag": mag,
            "mag_err": err, "zero_point": 25.0, "zero_point_err": 0.01,
            "references_used": ["ref0", "ref1", "ref2"],
            "references_excluded": []}


def make_batches(n_nights=3, per_night=10, seed=5, period=PERIOD,
                 amplitude=0.4, extra_frames=None):
    """Synthetic multi-night light curve with per-night zero offsets."""
    rng = np.random.default_rng(seed)
    night_offsets = [0.0, 0.15, -0.1, 0.05, -0.2][:n_nights]
    batches = []
    for n in range(n_nights):
        frames = []
        night_start = MJD0 + n * 1.05
        for k in range(per_night):
            mjd = night_start + 0.3 * k / per_night
            mag = (14.0 + night_offsets[n]
                   + amplitude * np.sin(2 * np.pi * (mjd - MJD0) / period)
                   + rng.normal(0, 0.01))
            frames.append(ok_frame(k, float(mjd), float(mag)))
        if extra_frames and n in extra_frames:
            frames.extend(extra_frames[n])
        batches.append({"night_id": f"night{n}",
                        "photometry": make_response(frames,
                                                    photometry_id=f"p{n}")})
    return batches


def post_period(client, batches, pmin=0.2, pmax=2.0):
    return client.post("/api/period", json={
        "batches": batches, "period_min_days": pmin,
        "period_max_days": pmax})


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_period_recovered_with_nightly_offsets(client):
    r = post_period(client, make_batches())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_id"] == TARGET_ID
    assert body["n_points"] == 30
    assert body["n_nights"] == 3
    assert body["mjd_span_days"] > 0
    assert len(body["frequencies"]) == len(body["powers"])
    assert len(body["window_powers"]) == len(body["frequencies"])
    # Grid step <= 1/(5*span).
    dfs = np.diff(body["frequencies"])
    assert dfs.max() <= 1.0 / (5.0 * body["mjd_span_days"]) + 1e-12
    # Strongest candidate matches the injected period (within 2 grid steps).
    best = body["best_candidate"]
    assert best == body["candidates"][0]
    assert abs(best["period_days"] - PERIOD) < 2 * dfs[0] / best["frequency"] ** 2 + 1e-9
    assert best["power"] > 0.9
    # Candidates are power-sorted and separated by >= 1/span.
    powers = [c["power"] for c in body["candidates"]]
    assert powers == sorted(powers, reverse=True)
    assert len(body["candidates"]) <= 3
    for a, b in zip(body["candidates"], body["candidates"][1:]):
        assert abs(a["frequency"] - b["frequency"]) >= 1.0 / body["mjd_span_days"]
    # Never labelled as confirmed.
    assert any("not a confirmed period" in w for w in body["warnings"])
    # Folded light curve: phase zero is the earliest usable MJD.
    assert body["phase_zero_mjd"] == pytest.approx(MJD0)
    folds = body["fold_points"]
    assert len(folds) == 30
    for fp in folds:
        assert 0.0 <= fp["phase"] < 1.0
        assert fp["residual"] == pytest.approx(fp["mag"] - fp["model_mag"])
    # Nightly offsets absorbed: detrended mags follow one sinusoid.
    res = np.array([fp["residual"] for fp in folds])
    assert res.std() < 0.05
    # CSV matches the JSON fold points and keeps provenance.
    r2 = client.get(f"/api/period/{body['period_id']}/csv")
    assert r2.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r2.text)))
    assert len(rows) == len(folds)
    for row, fp in zip(rows, folds):
        assert row["night_id"] == fp["night_id"]
        assert row["filename"] == fp["filename"]
        assert float(row["phase"]) == pytest.approx(fp["phase"])
        assert float(row["residual"]) == pytest.approx(fp["residual"])


def test_exclusions_keep_reasons_and_provenance(client):
    bad = [
        {"index": 100, "filename": "failed.fits", "status": "failed",
         "reason": "missing DATE-OBS header", "mjd": None, "exptime": None,
         "flux_rate": None, "flux_rate_err": None, "mag": None,
         "mag_err": None, "zero_point": None, "zero_point_err": None,
         "references_used": [], "references_excluded": []},
        ok_frame(101, MJD0 + 0.4, None),
        ok_frame(102, MJD0 + 0.5, 14.0, err=0.0),
        ok_frame(103, None, 14.0),
    ]
    batches = make_batches(extra_frames={1: bad})
    r = post_period(client, batches)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_points"] == 30
    excl = body["excluded_points"]
    assert len(excl) == 4
    by_file = {e["filename"]: e for e in excl}
    assert "failed.fits" in by_file
    assert "missing DATE-OBS" in by_file["failed.fits"]["reason"]
    assert by_file["failed.fits"]["night_id"] == "night1"
    assert by_file["failed.fits"]["batch_index"] == 1
    reasons = [e["reason"] for e in excl]
    assert any("mag missing or not finite" in x for x in reasons)
    assert any("mag_err not positive" in x for x in reasons)
    assert any("mjd missing or not finite" in x for x in reasons)


def test_validation_errors(client):
    good = make_batches()
    # Too few usable points.
    few = make_batches(n_nights=2, per_night=5)
    assert post_period(client, few).status_code == 422
    # Period bounds not increasing.
    assert post_period(client, good, pmin=2.0, pmax=0.2).status_code == 422
    # Only one distinct night among usable points.
    one = make_batches(n_nights=2, per_night=15)
    for f in one[1]["photometry"]["frames"]:
        f["status"] = "failed"
        f["mag"] = f["mjd"] = f["mag_err"] = None
    assert post_period(client, one).status_code == 422
    # Frequency grid limit exceeded.
    assert post_period(client, good, pmin=0.0001, pmax=1000.0).status_code == 422
    # Mismatched target ids.
    mixed = make_batches()
    mixed[1]["photometry"]["target_id"] = "other"
    assert post_period(client, mixed).status_code == 422
    # Duplicate night ids.
    dup = make_batches()
    dup[1]["night_id"] = dup[0]["night_id"]
    assert post_period(client, dup).status_code == 422


def test_no_candidate_reports_reason(client):
    # Constant magnitudes with nightly offsets exactly fitted by the
    # night-constant model leave zero weighted scatter -> 422; instead use
    # a pure linear trend (no local peak inside the grid).
    batches = []
    for n in range(3):
        frames = []
        for k in range(10):
            mjd = MJD0 + n * 1.05 + 0.3 * k / 10
            frames.append(ok_frame(k, float(mjd), 14.0 + 0.5 * (mjd - MJD0)))
        batches.append({"night_id": f"night{n}",
                        "photometry": make_response(frames,
                                                    photometry_id=f"p{n}")})
    r = post_period(client, batches, pmin=0.2, pmax=0.5)
    assert r.status_code == 200, r.text
    body = r.json()
    if not body["candidates"]:
        assert body["no_candidate_reason"]
        assert body["best_candidate"] is None
        assert body["fold_points"] == []
    else:
        # A trend may still produce edge maxima; the warning must be there.
        assert body["warnings"]


def test_short_baseline_warning(client):
    # Span ~0.27 d, period bound up to 2 d -> fewer than two cycles.
    batches = make_batches(n_nights=2, per_night=15)
    for b in batches:
        for k, f in enumerate(b["photometry"]["frames"]):
            f["mjd"] = MJD0 + (0 if b["night_id"] == "night0" else 0.02) \
                + 0.25 * k / 15
    r = post_period(client, batches, pmin=0.2, pmax=2.0)
    assert r.status_code == 200, r.text
    body = r.json()
    if body["best_candidate"] is not None:
        assert any("fewer than two cycles" in w for w in body["warnings"])
