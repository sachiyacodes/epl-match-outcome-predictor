"""Synthetic unit fixtures prove experimental boundaries, not football accuracy."""
import json

import numpy as np
import pandas as pd
import pytest

from epl import experiments as exp


def test_gate_requires_gain_probability_guardrail_and_majority():
    baseline = {"pooled": {"accuracy": .50, "log_loss": 1.},
                "seasons": {str(i): {"accuracy": .5} for i in range(7)}}
    report = {"pooled": {"accuracy": .52, "log_loss": 1.},
              "seasons": {str(i): {"accuracy": .51 if i < 4 else .49} for i in range(7)}}
    assert exp.qualifies(report, baseline)
    report["pooled"]["accuracy"] = .505
    assert not exp.qualifies(report, baseline)
    report["pooled"].update(accuracy=.52, log_loss=1.02)
    assert not exp.qualifies(report, baseline)
    report["pooled"]["log_loss"] = 1.
    report["seasons"]["3"]["accuracy"] = .49
    assert not exp.qualifies(report, baseline)


def test_monthly_training_cutoffs_never_include_the_prediction_month(monkeypatch):
    matches = pd.DataFrame({"date": pd.to_datetime(["2017-12-31", "2018-08-03", "2018-08-15", "2018-09-01"]),
                            "season": ["2017-18", "2018-19", "2018-19", "2018-19"],
                            "result": ["A", "D", "H", "A"]})
    seen = []

    class Model:
        def predict_proba(self, x):
            return np.tile([.3, .3, .4], (len(x), 1))

    def fit(frame, config, x, cutoff):
        seen.append((cutoff, frame[frame.date < cutoff].date.max()))
        return Model()

    monkeypatch.setattr(exp, "fit_model", fit)
    indices, p, cutoffs = exp.rolling_predict(matches, exp.baseline_config(), [2018], np.zeros((4, 10)))
    assert indices.tolist() == [1, 2, 3]
    assert cutoffs.tolist() == ["2018-08-01", "2018-08-01", "2018-09-01"]
    assert all(last < cutoff for cutoff, last in seen)
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_temperature_never_changes_argmax_and_insufficient_data_uses_identity():
    p = np.tile([.5, .2, .3], (330, 1))
    y = np.array(["A", "D", "H"] * 110)
    t = exp.calibration_temperature(y, p)
    np.testing.assert_array_equal(p.argmax(axis=1), exp.temperature_scale(p, t).argmax(axis=1))
    assert exp.calibration_temperature(y[:30], p[:30]) == 1
    assert exp.calibration_temperature(np.array(["A"] * 330), p) == 1


def test_paired_bootstrap_is_reproducible_and_keeps_pairs():
    dates = pd.date_range("2020-01-01", periods=60)
    y = np.array(["A", "D", "H"] * 20)
    wrong = np.eye(3)[[(i + 1) % 3 for i in range(60)]] * .8 + .2 / 3
    right = np.eye(3)[[i % 3 for i in range(60)]] * .8 + .2 / 3
    a = exp.paired_bootstrap(dates, y, wrong, right)
    assert a == exp.paired_bootstrap(dates, y, wrong, right)
    assert a["accuracy_ci95"] == [1., 1.]
    same = exp.paired_bootstrap(dates, y, right, right)
    assert same["accuracy_ci95"] == [0., 0.]
    assert same["log_loss_ci95"] == [0., 0.]


def test_atomic_json_rejects_non_finite_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    exp.write_json(path, {"valid": True})
    with pytest.raises(ValueError):
        exp.write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}


def test_output_cannot_escape_repository_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="must stay inside"):
        exp.run(tmp_path / "outside")
    assert not (tmp_path / "outside").exists()


def test_publish_rejects_changed_sealed_probabilities_before_fitting(tmp_path, monkeypatch):
    decision = {"locked": True}
    result = {"selection_sha256": exp.digest(decision), "ledger_sha256": {"baseline": "not-the-file-hash"}}
    (tmp_path / "confirmation-baseline.csv").write_text("raw_A,raw_D,raw_H\n.1,.2,.7\n")
    monkeypatch.setattr(exp, "fit_model", lambda *args: pytest.fail("Must not fit using a tampered ledger"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        exp.publish(pd.DataFrame(), tmp_path, decision, result, {})


def test_publish_rejects_changed_selection_before_any_artifact_writes(tmp_path):
    result = {"selection_sha256": exp.digest({"original": True})}
    with pytest.raises(ValueError, match="Sealed selection differs"):
        exp.publish(pd.DataFrame(), tmp_path, {"changed": True}, result, {})
    assert not list(tmp_path.iterdir())
