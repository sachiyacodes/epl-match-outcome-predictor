"""Small synthetic fixtures test invariants; these are not historical EPL data."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from epl import api
from epl import train as training
from epl.data import load_matches, parse_season
from epl.features import FEATURES, feature_rows
from epl.train import LABELS, evaluate, temperature_scale


def match(day, home="Alpha", away="Beta", score=(2, 0)):
    return {"date": day, "team1": home, "team2": away, "score": {"ft": list(score)}}


def season(matches):
    return {"name": "English Premier League test fixture", "matches": matches}


@pytest.fixture
def matches():
    return parse_season(season([
        match("2020-01-01"),
        match("2020-01-01", "Gamma", "Delta", (1, 1)),
        match("2020-01-08", "Alpha", "Gamma", (0, 1)),
        match("2020-01-09", "Beta", "Delta", (1, 1)),
        match("2020-01-15", "Gamma", "Beta", (3, 0)),
    ]), "2019-20")


def test_own_and_future_results_do_not_change_pre_match_features(matches):
    before = feature_rows(matches)
    changed = matches.copy()
    changed.loc[2:, ["home_goals", "away_goals"]] = [12, 0]
    assert_frame_equal(before.iloc[:3], feature_rows(changed).iloc[:3])
    assert not before.iloc[4].equals(feature_rows(changed).iloc[4])


def test_same_day_results_never_enter_prediction_features(matches):
    fixture = ("Alpha", "Beta", "2020-01-08")
    before = feature_rows(matches, fixture)
    changed = matches.copy()
    changed.loc[changed.date >= "2020-01-08", ["home_goals", "away_goals"]] = [9, 0]
    assert_frame_equal(before, feature_rows(changed, fixture))
    assert before.loc[0, "home_points"] == 3
    assert before.loc[0, "home_history"] == 1
    assert before.loc[0, "away_points"] == 0


def test_unseen_teams_receive_explicit_cold_start_features(matches):
    row = feature_rows(matches, ("New home", "New away", "2020-02-01")).iloc[0]
    assert row.tolist() == [1.0, 1.3, 1.3, 0, 7] * 2


def test_history_is_limited_to_last_ten_results():
    frame = parse_season(season([
        match(str(day.date()), away=f"Opponent {i}", score=(i, 0))
        for i, day in enumerate(pd.date_range("2020-01-01", periods=12))
    ]), "2019-20")
    row = feature_rows(frame, ("Alpha", "New", "2020-04-01")).iloc[0]
    assert row.home_history == 10
    assert row.home_scored == pytest.approx(np.mean(range(2, 12)))
    assert row.home_rest_days == 60


def test_parser_excludes_unfinished_matches_and_derives_labels():
    scheduled = {"date": "2020-01-04", "team1": "Alpha", "team2": "Beta"}
    frame = parse_season(season([
        match("2020-01-01", score=(2, 0)),
        match("2020-01-02", score=(1, 1)),
        match("2020-01-03", score=(0, 1)), scheduled,
    ]), "2019-20")
    assert frame.result.tolist() == ["H", "D", "A"]
    assert len(frame) == 3


def test_team_aliases_preserve_history_across_seasons():
    old = parse_season(season([
        match("2020-01-01", "Manchester City", "Manchester United", (2, 0)),
    ]), "2019-20")
    modern = parse_season(season([
        match("2020-08-01", "Manchester City FC", "Manchester United FC", (0, 1)),
    ]), "2020-21")
    frame = pd.concat([old, modern], ignore_index=True)
    assert frame.home_team.unique().tolist() == ["Manchester City FC"]
    row = feature_rows(frame).iloc[1]
    assert row.home_history == 1
    assert row.home_points == 3
    assert row.away_history == 1
    assert row.away_points == 0


@pytest.mark.parametrize("score", [[-1, 0], [True, 0], [1], [1, 2, 3], [1.0, 0]])
def test_parser_rejects_invalid_scores(score):
    with pytest.raises(ValueError, match="Invalid full-time score"):
        parse_season(season([match("2020-01-01", score=score)]), "2019-20")


def test_parser_rejects_invalid_dates_and_same_teams():
    with pytest.raises(ValueError):
        parse_season(season([match("not-a-date")]), "2019-20")
    with pytest.raises(ValueError, match="Invalid teams"):
        parse_season(season([match("2020-01-01", home="Alpha", away="Alpha")]), "2019-20")
    with pytest.raises(ValueError, match="Expected English Premier League"):
        parse_season({"name": "Another competition", "matches": []}, "2019-20")


def write_manifest(directory, payload):
    content = json.dumps(payload).encode()
    (directory / "test.json").write_bytes(content)
    manifest = {"files": [{"file": "test.json", "season": "2019-20",
                            "sha256": hashlib.sha256(content).hexdigest()}]}
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_raw_file_checksum_is_verified(tmp_path):
    write_manifest(tmp_path, season([match("2020-01-01")]))
    assert len(load_matches(tmp_path)) == 1
    (tmp_path / "test.json").write_text("tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_matches(tmp_path)


def test_duplicate_fixture_and_same_day_appearance_are_rejected(tmp_path):
    write_manifest(tmp_path, season([match("2020-01-01"), match("2020-01-02")]))
    with pytest.raises(ValueError, match="Duplicate fixtures"):
        load_matches(tmp_path)
    write_manifest(tmp_path, season([
        match("2020-01-01"), match("2020-01-01", "Alpha", "Gamma"),
    ]))
    with pytest.raises(ValueError, match="Team appears twice"):
        load_matches(tmp_path)


def test_temperature_scaling_preserves_valid_probability_distribution():
    values = np.array([[.2, .3, .5], [.7, .2, .1]])
    np.testing.assert_allclose(temperature_scale(values, 1), values)
    softened = temperature_scale(values, 2)
    np.testing.assert_allclose(softened.sum(axis=1), 1)
    assert np.all(softened > 0)
    assert softened[0].max() < values[0].max()


def test_metrics_map_alphabetical_classes_and_report_calibration():
    probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    result = evaluate(["A", "D", "H"], probabilities)
    assert LABELS == ["A", "D", "H"]
    assert result["accuracy"] == 1
    assert result["log_loss"] == pytest.approx(-np.log(.8))
    assert result["brier_multiclass"] == pytest.approx(.06)
    assert result["expected_calibration_error"] == pytest.approx(.2)
    assert set(result["calibration"]) == {"A", "D", "H"}


def test_calibration_error_includes_probability_one():
    probabilities = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    result = evaluate(["A", "D", "H"], probabilities)
    assert result["expected_calibration_error"] == pytest.approx(1.0)


def test_training_scaler_fits_only_earliest_split(monkeypatch, tmp_path):
    rows = []
    for year in (2019, 2020, 2021):
        for i, day in enumerate(pd.date_range(f"{year}-01-01", periods=33)):
            score = [(2, 0), (1, 1), (0, 2)][i % 3]
            rows.append(match(str(day.date()), score=score))
    frame = parse_season(season(rows), "synthetic-only")
    monkeypatch.setattr(training, "load_matches", lambda: frame)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/manifest.json").write_text('{"synthetic_test_fixture": true}')
    metrics = training.train(validation_start="2020-01-01", test_start="2021-01-01")
    import joblib
    bundle = joblib.load(tmp_path / "artifacts/model.joblib")
    scaler = bundle["model"].named_steps["standardscaler"]
    expected = feature_rows(frame).iloc[:33].mean().to_numpy()
    assert scaler.n_samples_seen_ == 33
    np.testing.assert_allclose(scaler.mean_, expected)
    assert [metrics["splits"][name]["n"] for name in ("train", "validation", "test")] == [33, 33, 33]
    assert metrics["splits"]["train"]["end"] < metrics["splits"]["validation"]["start"]
    assert metrics["splits"]["validation"]["end"] < metrics["splits"]["test"]["start"]


@pytest.fixture
def client_bundle(monkeypatch, matches):
    # Fit a real small classifier on explicitly synthetic unit-test features.
    x = pd.DataFrame(np.arange(90).reshape(9, 10), columns=FEATURES)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x, ["H", "D", "A"] * 3)
    bundle = {"model": model, "temperature": 1.0, "matches": matches,
              "metrics": {"data_through": "2020-01-15", "test_only": True}}
    monkeypatch.setattr(api, "artifact", lambda: bundle)
    return TestClient(api.app), bundle


def test_prediction_success_uses_model_class_order(client_bundle):
    client, bundle = client_bundle
    fixture = {"home_team": "Alpha", "away_team": "Beta", "match_date": "2020-01-16"}
    response = client.post("/predict", json=fixture)
    assert response.status_code == 200
    body = response.json()
    x = feature_rows(bundle["matches"], ("Alpha", "Beta", "2020-01-16"))
    expected = dict(zip(bundle["model"].classes_, bundle["model"].predict_proba(x)[0]))
    assert body["probabilities"] == pytest.approx(expected)
    assert sum(body["probabilities"].values()) == pytest.approx(1)
    assert body["data_through"] == "2020-01-15"
    assert client.get("/teams").json()["teams"] == ["Alpha", "Beta", "Delta", "Gamma"]
    assert client.get("/metrics").json()["test_only"] is True


@pytest.mark.parametrize("changes", [
    {"home_team": "Alpha", "away_team": "Alpha"},
    {"home_team": "Unknown"},
    {"match_date": "2020-01-15"},
    {"match_date": "2019-01-01"},
    {"match_date": "invalid"},
    {"home_team": ""},
])
def test_prediction_rejects_invalid_fixture(client_bundle, changes):
    client, _ = client_bundle
    fixture = {"home_team": "Alpha", "away_team": "Beta", "match_date": "2020-01-16"}
    assert client.post("/predict", json=fixture | changes).status_code == 422


def test_missing_artifact_returns_actionable_503(monkeypatch, tmp_path):
    api.artifact.cache_clear()
    monkeypatch.setattr(api, "ROOT", tmp_path)
    try:
        client = TestClient(api.app)
        assert client.get("/health").json() == {"status": "ok", "model_ready": False}
        response = client.get("/teams")
        assert response.status_code == 503
        assert "ingestion and training" in response.json()["detail"]
    finally:
        api.artifact.cache_clear()
