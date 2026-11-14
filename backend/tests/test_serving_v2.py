"""Synthetic serving fixtures verify routing and rollback, not model performance."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from epl import api
from epl.experiments import matrix


@pytest.fixture
def stored_model(tmp_path, monkeypatch):
    rows = pd.DataFrame(dict(date=pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
                             home_team=["Alpha"] * 3, away_team=["Beta"] * 3,
                             home_goals=[2, 1, 0], away_goals=[0, 1, 1],
                             result=["H", "D", "A"], season=["2019-20"] * 3))
    config = {"kind": "logistic", "bundle": "baseline", "C": 1.0}
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(np.arange(90).reshape(9, 10), ["A", "D", "H"] * 3)
    folder = tmp_path / "artifacts/research"
    folder.mkdir(parents=True)
    path = folder / "baseline.joblib"
    bundle = dict(version=2, model=model, config=config, matches=rows, temperature=1., calibration_period="2018-19",
                  metrics=dict(data_through="2020-01-15", trained_through="2020-01-15",
                               model_version="audited-v2", audit_status="passed"))
    joblib.dump(bundle, path)
    pointer = {"version": 2, "path": "artifacts/research/baseline.joblib",
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "artifacts/active.json").write_text(json.dumps(pointer))
    api.artifact.cache_clear()
    monkeypatch.setattr(api, "ROOT", tmp_path)
    yield TestClient(api.app), bundle, pointer, tmp_path
    api.artifact.cache_clear()


def test_verified_pointer_serves_matching_features_and_additive_metadata(stored_model):
    client, bundle, _, _ = stored_model
    fixture = {"home_team": "Alpha", "away_team": "Beta", "match_date": "2020-02-01"}
    response = client.post("/predict", json=fixture)
    assert response.status_code == 200
    body = response.json()
    x = matrix(bundle["matches"], bundle["config"], ("Alpha", "Beta", fixture["match_date"])).to_numpy()
    expected = bundle["model"].predict_proba(x)[0]
    assert [body["probabilities"][key] for key in ["A", "D", "H"]] == pytest.approx(expected)
    assert body["audit_status"] == "passed"
    assert body["model_version"] == "audited-v2"
    assert body["calibration_period"] == "2018-19"
    assert client.get("/metrics").json()["audit_status"] == "passed"


def test_corrupt_active_model_fails_closed_instead_of_using_legacy(stored_model):
    client, _, pointer, directory = stored_model
    (directory / pointer["path"]).write_bytes(b"tampered")
    response = client.get("/teams")
    assert response.status_code == 503
    assert "failed validation" in response.json()["detail"]


def test_artifact_pointer_cannot_escape_artifacts_directory(stored_model):
    client, _, pointer, directory = stored_model
    pointer["path"] = "outside.joblib"
    (directory / "artifacts/active.json").write_text(json.dumps(pointer))
    assert client.get("/metrics").status_code == 503
