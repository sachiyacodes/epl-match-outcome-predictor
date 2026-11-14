from datetime import date
from functools import lru_cache
import hashlib
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import ROOT
from .features import feature_rows
from .train import temperature_scale

app = FastAPI(title="Premier League Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@lru_cache(maxsize=1)
def artifact():
    path = ROOT / "artifacts/model.joblib"
    pointer = ROOT / "artifacts/active.json"
    if pointer.exists():
        try:
            active = json.loads(pointer.read_text(encoding="utf-8"))
            path = (ROOT / active["path"]).resolve()
            if active["version"] != 2 or not path.is_relative_to((ROOT / "artifacts").resolve()):
                raise ValueError("Invalid artifact pointer")
            if hashlib.sha256(path.read_bytes()).hexdigest() != active["sha256"]:
                raise ValueError("Artifact checksum mismatch")
        except (ValueError, KeyError, OSError) as exc:
            raise HTTPException(503, "Active model failed validation; restore a verified artifact pointer") from exc
    if not path.exists():
        raise HTTPException(503, "Model unavailable. Run ingestion and training first.")
    # Load only locally generated artifacts; pickle/joblib is not safe for untrusted files.
    return joblib.load(path)


class Fixture(BaseModel):
    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    match_date: date


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ROOT / "artifacts/active.json").exists() or (ROOT / "artifacts/model.joblib").exists()}


@app.get("/teams")
def teams():
    bundle = artifact()
    matches = bundle["matches"]
    latest = matches[matches.season == matches.iloc[-1].season]
    return dict(teams=sorted(set(latest.home_team) | set(latest.away_team)),
                data_through=bundle["metrics"]["data_through"])


@app.get("/metrics")
def metrics():
    bundle = artifact()
    result = dict(bundle["metrics"])
    if bundle.get("version") != 2:
        result.update(model_version="legacy-v1", audit_status="superseded-unverified-chronology",
                      warning="Legacy benchmark includes known upstream date defects; run audited experiments before relying on its scores.")
    return result


@app.post("/predict")
def predict(fixture: Fixture):
    bundle = artifact()
    matches = bundle["matches"]
    known = set(matches.home_team) | set(matches.away_team)
    if fixture.home_team == fixture.away_team:
        raise HTTPException(422, "Choose two different teams")
    if fixture.home_team not in known or fixture.away_team not in known:
        raise HTTPException(422, "Unknown team; use names returned by /teams")
    if pd.Timestamp(fixture.match_date) <= matches.date.max():
        raise HTTPException(422, "Prediction date must follow data_through; use held-out test metrics for historical evaluation")
    if bundle.get("version") == 2:
        from .experiments import matrix
        if bundle["config"]["kind"] == "score":
            x = pd.DataFrame([dict(home_team=fixture.home_team, away_team=fixture.away_team)])
        else:
            x = matrix(matches, bundle["config"], (fixture.home_team, fixture.away_team, fixture.match_date)).to_numpy(dtype=float)
    else:
        x = feature_rows(matches, (fixture.home_team, fixture.away_team, fixture.match_date))
    p = temperature_scale(bundle["model"].predict_proba(x), bundle["temperature"])[0]
    return dict(probabilities=dict(zip(bundle["model"].classes_.tolist(), p.tolist())),
                data_through=bundle["metrics"]["data_through"], match_date=str(fixture.match_date),
                model_version=bundle["metrics"].get("model_version", "legacy-v1"),
                trained_through=bundle["metrics"].get("trained_through"),
                calibration_period=bundle.get("calibration_period"),
                audit_status=bundle["metrics"].get("audit_status", "superseded-unverified-chronology"),
                warning=("Historical results only; no live fixtures, injuries, lineups or current-season updates."
                         if bundle.get("version") == 2 else
                         "Legacy model includes known source date defects. Run the audited pipeline; this prediction is provisional."))
