"""Chronological train, validation calibration, untouched final test."""
import argparse
import json

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, load_matches
from .features import FEATURES, feature_rows

LABELS = ["A", "D", "H"]


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def evaluate(y, probabilities):
    truth = np.array([[int(value == label) for label in LABELS] for value in y])
    curves = {}
    for i, label in enumerate(LABELS):
        actual, predicted = calibration_curve(truth[:, i], probabilities[:, i], n_bins=10)
        curves[label] = dict(mean_predicted=predicted.tolist(), fraction_positive=actual.tolist())
    confidence = probabilities.max(axis=1)
    correct = np.array(LABELS)[probabilities.argmax(axis=1)] == np.array(y)
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (confidence >= low) & ((confidence <= high) if index == 9 else (confidence < high))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return dict(log_loss=float(log_loss(y, probabilities, labels=LABELS)),
                accuracy=float(accuracy_score(y, np.array(LABELS)[probabilities.argmax(axis=1)])),
                brier_multiclass=float(np.mean(np.sum((probabilities - truth) ** 2, axis=1))),
                expected_calibration_error=float(ece), calibration=curves)


def train(validation_start="2022-07-01", test_start="2023-07-01"):
    matches = load_matches()
    x = feature_rows(matches)
    y = matches.result
    train_mask = matches.date < validation_start
    val_mask = (matches.date >= validation_start) & (matches.date < test_start)
    test_mask = matches.date >= test_start
    for name, mask in (("train", train_mask), ("validation", val_mask), ("test", test_mask)):
        if mask.sum() < 30 or set(y[mask]) != set(LABELS):
            raise ValueError(f"{name} needs >=30 matches and all three outcomes; check boundaries")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))
    model.fit(x[train_mask], y[train_mask])
    assert model.classes_.tolist() == LABELS
    val_p = model.predict_proba(x[val_mask])
    fit = minimize_scalar(lambda t: log_loss(y[val_mask], temperature_scale(val_p, t), labels=LABELS),
                          bounds=(0.25, 4.0), method="bounded")
    temperature = float(fit.x)
    if log_loss(y[val_mask], temperature_scale(val_p, temperature), labels=LABELS) > log_loss(y[val_mask], val_p, labels=LABELS):
        temperature = 1.0
    prior = np.array([(y[train_mask] == label).mean() for label in LABELS])
    metrics = dict(model="multinomial logistic regression", features=FEATURES, temperature=temperature,
                   data_through=str(matches.date.max().date()), labels=LABELS,
                   source_manifest=json.loads((ROOT / "data/raw/manifest.json").read_text()), splits={})
    for name, mask in (("train", train_mask), ("validation", val_mask), ("test", test_mask)):
        raw = model.predict_proba(x[mask])
        metrics["splits"][name] = dict(n=int(mask.sum()), start=str(matches.date[mask].min().date()),
            end=str(matches.date[mask].max().date()), raw=evaluate(y[mask], raw),
            calibrated=evaluate(y[mask], temperature_scale(raw, temperature)),
            class_prior=evaluate(y[mask], np.tile(prior, (int(mask.sum()), 1))))
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    joblib.dump(dict(model=model, temperature=temperature, matches=matches, metrics=metrics), output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({name: report["calibrated"]["log_loss"] for name, report in metrics["splits"].items()}, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-start", default="2022-07-01")
    parser.add_argument("--test-start", default="2023-07-01")
    args = parser.parse_args()
    train(args.validation_start, args.test_start)
