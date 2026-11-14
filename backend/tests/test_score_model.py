"""Synthetic unit fixtures exercise score-model invariants, not real EPL performance."""
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from epl import score_model
from epl.score_model import ScoreModel


@pytest.fixture
def matches():
    rows = []
    clubs = ["Alpha", "Beta", "Gamma", "Delta"]
    rng = np.random.default_rng(42)
    for i in range(120):
        home = clubs[i % 4]
        away = clubs[(i + 1 + (i // 4) % 3) % 4]
        rows.append(dict(date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                         home_team=home, away_team=away, home_goals=int(rng.poisson(1.7)),
                         away_goals=int(rng.poisson(1.1)), season="2019-20"))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("corrected", [False, True])
def test_score_model_fits_valid_probabilities_and_identifiable_strengths(matches, corrected, tmp_path):
    model = ScoreModel(corrected=corrected).fit(matches)
    assert abs(model.attack_.mean()) < 1e-12
    assert abs(model.defence_.mean()) < 1e-12
    assert model.classes_.tolist() == ["A", "D", "H"]
    fixtures = pd.DataFrame(dict(home_team=["Alpha", "New home"], away_team=["Beta", "New away"]))
    probabilities = model.predict_proba(fixtures)
    assert probabilities.shape == (2, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities > 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    joblib.dump(model, tmp_path / "model.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "model.joblib").predict_proba(fixtures), probabilities)


def test_negative_low_score_dependence_increases_draw_probability(matches):
    model = ScoreModel(corrected=False).fit(matches)
    fixture = pd.DataFrame([dict(home_team="Alpha", away_team="Beta")])
    independent = model.predict_proba(fixture)
    model.rho_ = -.01
    corrected = model.predict_proba(fixture)
    assert corrected[0, 1] > independent[0, 1]
    np.testing.assert_allclose(corrected.sum(axis=1), 1)


def test_cutoff_and_five_season_window_are_enforced(matches):
    old = matches.copy()
    old["date"] -= pd.DateOffset(years=8)
    old["season"] = "2011-12"
    history = [old]
    for year in range(2015, 2021):
        frame = matches.copy()
        frame["date"] += pd.DateOffset(years=year - 2020)
        frame["season"] = f"{year}-{str(year + 1)[2:]}"
        history.append(frame)
    model = ScoreModel(corrected=False).fit(pd.concat(history), cutoff="2020-03-01")
    assert model.training_seasons_ == ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
    assert model.training_cutoff_ < "2020-03-01"


def test_rho_transform_gradient_matches_finite_difference():
    parameters = np.array([.2, -.4, .3, .1, -.3, .4, -.2, .1, .3, .2, -.4])
    def rho(value):
        return ScoreModel._rho(value, value[:4] - value[:4].mean(), value[4:8] - value[4:8].mean())[0]
    _, derivative = ScoreModel._rho(parameters, parameters[:4] - parameters[:4].mean(), parameters[4:8] - parameters[4:8].mean())
    numerical = approx_derivative(rho, parameters).ravel()
    np.testing.assert_allclose(derivative, numerical, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("corrected", [False, True])
def test_likelihood_gradient_matches_finite_difference(matches, monkeypatch, corrected):
    original = score_model.minimize

    def checked_minimize(objective, initial, **kwargs):
        probe = initial + np.random.default_rng(8).normal(0, .05, len(initial))
        _, gradient = objective(probe)
        numerical = approx_derivative(lambda value: objective(value)[0], probe).ravel()
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)
        return original(objective, initial, **kwargs)

    monkeypatch.setattr(score_model, "minimize", checked_minimize)
    ScoreModel(corrected=corrected).fit(matches)


def test_optimization_failure_is_explicit(matches, monkeypatch):
    monkeypatch.setattr(score_model, "minimize", lambda *a, **k: SimpleNamespace(success=False, message="test failure"))
    with pytest.raises(ValueError, match="optimization failed: test failure"):
        ScoreModel().fit(matches)


def test_invalid_data_and_prediction_corrections_are_rejected(matches):
    with pytest.raises(ValueError, match="half_life"):
        ScoreModel(half_life=0)
    with pytest.raises(ValueError, match="not fitted"):
        ScoreModel().predict_proba(matches)
    invalid = matches.copy()
    invalid.loc[0, "home_goals"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        ScoreModel().fit(invalid)
    model = ScoreModel(corrected=False).fit(matches)
    model.rho_ = 10
    with pytest.raises(ValueError, match="prediction correction"):
        model.predict_proba(matches.iloc[:1])
