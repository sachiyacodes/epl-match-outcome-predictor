"""Audited, resumable calendar-month experiments and a locked confirmation season.

The historical artifact is never overwritten. Selection uses development outcomes
only; confirmation is written once and cannot be used to choose another model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import ROOT
from .features import feature_rows as baseline_features
from .train import LABELS, evaluate, temperature_scale

DEVELOPMENT_YEARS = tuple(range(2018, 2025))
CONFIRMATION_YEAR = 2025
SEED = 20261114


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def season_name(year):
    return f"{year}-{str(year + 1)[2:]}"


def diagnostics(y, p):
    result = evaluate(y, p)
    predicted = np.array(LABELS)[p.argmax(axis=1)]
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    result.update(confusion_matrix=confusion_matrix(y, predicted, labels=LABELS).tolist(),
                  per_class={label: report[label] for label in LABELS},
                  macro_f1=report["macro avg"]["f1-score"],
                  predicted_counts={label: int((predicted == label).sum()) for label in LABELS})
    return result


def promoted_fixtures(matches, indices):
    previous_teams = {}
    for season, frame in matches.groupby("season", sort=True):
        previous_teams[int(season[:4])] = set(frame.home_team) | set(frame.away_team)
    return np.array([row.home_team not in previous_teams.get(int(row.season[:4]) - 1, set())
                     or row.away_team not in previous_teams.get(int(row.season[:4]) - 1, set())
                     for row in matches.iloc[indices].itertuples()])


def promotion_breakdown(matches, indices, p):
    mask = promoted_fixtures(matches, indices)
    return {name: dict(n=int(selected.sum()), **diagnostics(matches.iloc[indices].result.to_numpy()[selected], p[selected]))
            for name, selected in (("involves_promoted_or_returning", mask), ("established_teams", ~mask))
            if selected.any()}


def corrected_legacy_comparison(matches):
    """Reproduce the original frozen procedure with only source dates repaired."""
    frame = matches[matches.season.isin([season_name(year) for year in range(2014, 2025)])].reset_index(drop=True)
    x = baseline_features(frame).to_numpy(dtype=float)
    model = fit_model(frame, baseline_config(), x, pd.Timestamp("2022-07-01"))
    calibration = (frame.date >= "2022-07-01") & (frame.date < "2023-07-01")
    t = calibration_temperature(frame.result[calibration].to_numpy(), model.predict_proba(x[calibration]))
    test = frame.date >= "2023-07-01"
    raw = model.predict_proba(x[test])
    return dict(note="Original frozen procedure on repaired dates; old inspected test is development data now",
                n=int(test.sum()), raw=diagnostics(frame.result[test], raw),
                calibrated=diagnostics(frame.result[test], temperature_scale(raw, t)), temperature=t)


def baseline_config():
    return dict(kind="logistic", bundle="baseline", C=1.0)


def logistic_configs():
    configs = []
    for bundle in ("elo", "rich"):
        for k in (10, 20, 40):
            for offset in (0, 50, 100):
                for c in (.1, 1., 10.):
                    configs.append(dict(kind="logistic", bundle=bundle, k=k, home_offset=offset, C=c))
    return configs


def boost_configs(feature_config):
    features = {key: feature_config[key] for key in ("bundle", "k", "home_offset")}
    return [dict(kind="boosting", **features, max_leaf_nodes=leaves,
                 min_samples_leaf=minimum, l2_regularization=l2)
            for leaves in (7, 15) for minimum in (20, 50) for l2 in (1., 10.)]


def score_configs():
    return [dict(kind="score", half_life=half_life, corrected=corrected)
            for corrected in (False, True) for half_life in (180, 365, 730)]


def matrix(matches, config, fixture=None):
    if config["kind"] == "score":
        return None
    if config["bundle"] == "baseline":
        return baseline_features(matches, fixture)
    from .advanced import feature_rows
    return feature_rows(matches, fixture, k=config["k"], home_offset=config["home_offset"], bundle=config["bundle"])


def fit_model(matches, config, x, cutoff):
    train_mask = (matches.date < pd.Timestamp(cutoff)).to_numpy()
    history = matches.loc[train_mask]
    if len(history) < 100 or set(history.result) != set(LABELS):
        raise ValueError("Insufficient chronological training history")
    if config["kind"] == "score":
        from .score_model import ScoreModel
        return ScoreModel(half_life=config["half_life"], corrected=config["corrected"]).fit(history, cutoff=cutoff)
    if config["kind"] == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=config["C"], solver="lbfgs", max_iter=2000, tol=1e-5))
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=200, early_stopping=False, random_state=SEED,
            max_leaf_nodes=config["max_leaf_nodes"], min_samples_leaf=config["min_samples_leaf"],
            l2_regularization=config["l2_regularization"])
    model.fit(x[train_mask], history.result.to_numpy())
    if model.classes_.tolist() != LABELS:
        raise ValueError("Unexpected probability class order")
    return model


def rolling_predict(matches, config, years, x=None):
    """Refit at calendar-month boundaries; no fitting includes its own month."""
    if x is None and config["kind"] != "score":
        x = matrix(matches, config).to_numpy(dtype=float)
    target = matches.season.isin([season_name(year) for year in years]).to_numpy()
    indices = np.flatnonzero(target)
    probabilities = np.empty((len(indices), 3))
    cutoffs = np.empty(len(indices), dtype="U10")
    months = matches.loc[target, "date"].dt.to_period("M").to_numpy()
    for month in sorted(set(months)):
        origin = month.start_time
        local = months == month
        positions = indices[local]
        model = fit_model(matches, config, x, origin)
        inputs = matches.iloc[positions] if config["kind"] == "score" else x[positions]
        p = np.asarray(model.predict_proba(inputs))
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("Invalid probabilities")
        probabilities[local] = p
        cutoffs[local] = str(origin.date())
    return indices, probabilities, cutoffs


def summary(matches, indices, p, config):
    subset = matches.iloc[indices]
    return dict(config=config, pooled=diagnostics(subset.result.to_numpy(), p),
                promoted_team_breakdown=promotion_breakdown(matches, indices, p), seasons={
        season: diagnostics(subset.result.to_numpy()[mask], p[mask])
        for season in sorted(set(subset.season))
        for mask in [(subset.season == season).to_numpy()]})


def candidate_key(report):
    # Fewer features / a linear model wins an exact metric tie.
    complexity = {"logistic": 0, "score": 1, "boosting": 2}[report["config"]["kind"]]
    return (-report["pooled"]["accuracy"], report["pooled"]["log_loss"], complexity,
            digest(report["config"]))


def qualifies(report, baseline):
    seasons = sorted(set(report["seasons"]) & set(baseline["seasons"]))
    non_worse = sum(report["seasons"][s]["accuracy"] >= baseline["seasons"][s]["accuracy"] for s in seasons)
    return (report["pooled"]["accuracy"] >= baseline["pooled"]["accuracy"] + .01
            and report["pooled"]["log_loss"] <= baseline["pooled"]["log_loss"] + .01
            and non_worse > len(seasons) / 2)


def calibration_temperature(y, p):
    if len(y) < 300 or set(y) != set(LABELS):
        return 1.0
    fit = minimize_scalar(lambda t: log_loss(y, temperature_scale(p, t), labels=LABELS),
                          bounds=(.25, 4.), method="bounded")
    if not fit.success or not np.isfinite(fit.x):
        return 1.0
    return float(fit.x) if fit.fun <= log_loss(y, p, labels=LABELS) else 1.0


def paired_bootstrap(dates, y, baseline, challenger, n_resamples=2000):
    """Pair observations and resample calendar weeks, retaining within-week groups."""
    weeks = pd.to_datetime(dates).to_period("W").astype(str)
    blocks = [np.flatnonzero(weeks == week) for week in sorted(set(weeks))]
    truth = np.array([LABELS.index(label) for label in y])
    base_correct = baseline.argmax(axis=1) == truth
    new_correct = challenger.argmax(axis=1) == truth
    delta_correct = new_correct.astype(float) - base_correct
    delta_loss = -np.log(np.clip(challenger[np.arange(len(y)), truth], 1e-15, 1)) + np.log(
        np.clip(baseline[np.arange(len(y)), truth], 1e-15, 1))
    sums = np.array([[delta_correct[b].sum(), delta_loss[b].sum(), len(b)] for b in blocks])
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    totals = sums[samples].sum(axis=1)
    differences = totals[:, :2] / totals[:, 2, None]
    return dict(method="paired calendar-week bootstrap", seed=SEED, resamples=n_resamples,
                accuracy_difference=float(delta_correct.mean()),
                accuracy_ci95=np.quantile(differences[:, 0], [.025, .975]).tolist(),
                log_loss_difference=float(delta_loss.mean()),
                log_loss_ci95=np.quantile(differences[:, 1], [.025, .975]).tolist())


def make_protocol(matches, audit):
    code = {}
    for name in ("experiments.py", "advanced.py", "score_model.py", "features.py", "train.py", "data.py", "audit.py"):
        code[name] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    import sklearn
    return dict(version=2, code_sha256=code, sklearn=sklearn.__version__, seed=SEED,
                data_hash=hashlib.sha256(pd.util.hash_pandas_object(matches, index=True).values.tobytes()).hexdigest(),
                audit=audit, development_years=list(DEVELOPMENT_YEARS), confirmation_year=CONFIRMATION_YEAR,
                logistic=logistic_configs(), boosting="8 configs on development-selected logistic feature settings",
                score=score_configs(), calibration="prior-season monthly out-of-time predictions; scalar T in [.25,4]",
                selection=dict(min_accuracy_gain=.01, max_log_loss_regression=.01, majority_seasons=True),
                promotion=dict(positive_accuracy_gain=True, accuracy_ci95_lower_gt_zero=True,
                               max_calibrated_log_loss_regression=.01), bootstrap_resamples=2000)


def save_ledger(path, matches, indices, p, cutoffs, temperatures=None):
    ledger = matches.iloc[indices][["date", "season", "home_team", "away_team", "result"]].copy()
    ledger["fit_cutoff"] = cutoffs
    for i, label in enumerate(LABELS):
        ledger[f"raw_{label}"] = p[:, i]
    if temperatures is not None:
        ledger["temperature"] = temperatures
    ledger.to_csv(path, index=False)


def run(output=None, stage="all"):
    from .audit import load_audited
    output = Path(output or ROOT / "artifacts/research-v2").resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Experiment output must stay inside this repository's artifacts directory")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    matches = load_audited()
    audit = json.loads((ROOT / "data/audited/audit.json").read_text(encoding="utf-8"))
    protocol = make_protocol(matches, audit)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Protocol/code/data changed. Use another --output directory; never reuse inspected confirmation.")
    else:
        write_json(protocol_path, protocol)
    legacy_path = output / "corrected-legacy.json"
    if not legacy_path.exists():
        write_json(legacy_path, corrected_legacy_comparison(matches))
    features = {}

    def development(config):
        key = digest(config)
        path = cache / f"{key}.npz"
        report_path = cache / f"{key}.json"
        if report_path.exists():
            return json.loads(report_path.read_text()), path
        try:
            feature_key = digest({k: config[k] for k in ("bundle", "k", "home_offset") if k in config})
            if config["kind"] != "score" and feature_key not in features:
                features[feature_key] = matrix(matches, config).to_numpy(dtype=float)
            indices, p, cutoffs = rolling_predict(matches, config, DEVELOPMENT_YEARS, features.get(feature_key))
            np.savez_compressed(path, indices=indices, probabilities=p, cutoffs=cutoffs)
            report = summary(matches, indices, p, config)
        except (ValueError, FloatingPointError) as exc:
            report = dict(config=config, disqualified=str(exc))
        write_json(report_path, report)
        if "pooled" in report:
            print(f"Development {config}: accuracy={report['pooled']['accuracy']:.4f}, log_loss={report['pooled']['log_loss']:.4f}", flush=True)
        else:
            print(f"Disqualified {config}: {report['disqualified']}", flush=True)
        return report, path

    baseline, baseline_path = development(baseline_config())
    if "pooled" not in baseline:
        raise ValueError("Baseline experiment failed")
    reports = []
    for config in logistic_configs():
        reports.append(development(config)[0])
    best_logistic = min((r for r in reports if "pooled" in r), key=candidate_key)
    for config in boost_configs(best_logistic["config"]) + score_configs():
        reports.append(development(config)[0])
    eligible = [r for r in reports if "pooled" in r and qualifies(r, baseline)]
    selected = min(eligible, key=candidate_key) if eligible else baseline
    decision = dict(protocol_sha256=digest(protocol), baseline=baseline, candidates=reports,
                    selected=selected, development_qualified=bool(eligible),
                    reason="Development gates passed" if eligible else "No challenger passed development gates; retain corrected baseline")
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != decision:
        raise ValueError("Selection differs from previously frozen decision")
    write_json(selection_path, decision)
    for name, report in (("baseline", baseline), ("selected", selected)):
        data = np.load(cache / f"{digest(report['config'])}.npz")
        save_ledger(output / f"development-{name}.csv", matches, data["indices"], data["probabilities"], data["cutoffs"])
    if stage == "develop":
        return decision
    final_path = output / "confirmation.json"
    if final_path.exists():
        print("Confirmation already sealed; reusing stored predictions without another evaluation.", flush=True)
        result = json.loads(final_path.read_text())
        publish(matches, output, decision, result, audit)
        return result
    # Selection is committed to disk before any confirmation metrics are computed.
    return confirm(matches, output, decision, cache, audit)


def confirm(matches, output, decision, cache, audit):
    reports, raw_predictions, calibrated_predictions, fitted_temperatures = {}, {}, {}, {}
    subset = matches[matches.season == season_name(CONFIRMATION_YEAR)]
    if len(subset) != 380:
        raise ValueError("Confirmation requires the complete 380-match season")
    for name in ("baseline", "selected"):
        config = decision[name]["config"]
        cached = np.load(cache / f"{digest(config)}.npz")
        dev = matches.iloc[cached["indices"]]
        previous = (dev.season == season_name(CONFIRMATION_YEAR - 1)).to_numpy()
        t = calibration_temperature(dev.result.to_numpy()[previous], cached["probabilities"][previous])
        indices, raw, cutoffs = rolling_predict(matches, config, [CONFIRMATION_YEAR])
        calibrated = temperature_scale(raw, t)
        fitted_temperatures[name] = t
        raw_predictions[name], calibrated_predictions[name] = raw, calibrated
        reports[name] = dict(config=config, temperature=t, raw=diagnostics(subset.result, raw),
                             calibrated=diagnostics(subset.result, calibrated),
                             promoted_team_breakdown=promotion_breakdown(matches, indices, calibrated))
        save_ledger(output / f"confirmation-{name}.csv", matches, indices, raw, cutoffs, t)
    interval = paired_bootstrap(subset.date.to_numpy(), subset.result.to_numpy(),
                                calibrated_predictions["baseline"], calibrated_predictions["selected"])
    gain = reports["selected"]["calibrated"]["accuracy"] - reports["baseline"]["calibrated"]["accuracy"]
    loss_ok = reports["selected"]["calibrated"]["log_loss"] <= reports["baseline"]["calibrated"]["log_loss"] + .01
    promoted = bool(decision["development_qualified"] and gain > 0 and interval["accuracy_ci95"][0] > 0 and loss_ok)
    active = "selected" if promoted else "baseline"
    result = dict(version=2, season=season_name(CONFIRMATION_YEAR), n=len(subset),
                  start=str(subset.date.min().date()), end=str(subset.date.max().date()),
                  protocol_sha256=decision["protocol_sha256"], reports=reports, uncertainty=interval,
                  promoted=promoted, active=active,
                  reason="All promotion gates passed" if promoted else "Improvement not confirmed by every gate; corrected baseline retained")
    result["selection_sha256"] = digest(decision)
    result["ledger_sha256"] = {name: hashlib.sha256((output / f"confirmation-{name}.csv").read_bytes()).hexdigest()
                               for name in ("baseline", "selected")}
    write_json(output / "confirmation.json", result)
    publish(matches, output, decision, result, audit)
    print(json.dumps(result, indent=2), flush=True)
    return result


def publish(matches, output, decision, result, audit):
    """Resume artifact creation from sealed predictions, never re-score confirmation."""
    if result["selection_sha256"] != digest(decision):
        raise ValueError("Sealed selection differs from the development decision")
    for name in ("baseline", "selected"):
        content = (output / f"confirmation-{name}.csv").read_bytes()
        if hashlib.sha256(content).hexdigest() != result["ledger_sha256"][name]:
            raise ValueError("Sealed confirmation ledger checksum mismatch")
    subset = matches[matches.season == season_name(CONFIRMATION_YEAR)]
    active, promoted = result["active"], result["promoted"]
    reports, interval = result["reports"], result["uncertainty"]
    # Final fit includes all observed data after the locked evaluation is complete.
    # Evaluate monthly predictions above, never the refitted artifact's in-sample outputs.
    calibration_season = season_name(CONFIRMATION_YEAR)
    cutoff = matches.date.max() + pd.Timedelta(days=1)
    for name in ("baseline", "selected"):
        config = decision[name]["config"]
        x = matrix(matches, config)
        model = fit_model(matches, config, None if x is None else x.to_numpy(dtype=float), cutoff)
        ledger = pd.read_csv(output / f"confirmation-{name}.csv")
        expected = subset[["date", "season", "home_team", "away_team", "result"]].reset_index(drop=True)
        actual = ledger[["date", "season", "home_team", "away_team", "result"]].copy()
        actual["date"] = pd.to_datetime(actual.date)
        # A sealed result must not be paired with a different prediction ledger.
        if not np.array_equal(expected.astype(str).to_numpy(), actual.astype(str).to_numpy()):
            raise ValueError("Sealed confirmation ledger differs from audited fixture identities")
        raw = ledger[[f"raw_{label}" for label in LABELS]].to_numpy()
        t = calibration_temperature(subset.result.to_numpy(), raw)
        prior_history = matches[matches.date < subset.date.min()]
        prior = np.array([(prior_history.result == label).mean() for label in LABELS])
        metrics = dict(model=config["kind"], model_version="audited-v2", config=config,
                       data_through=str(matches.date.max().date()), trained_through=str(matches.date.max().date()),
                       calibration_period=calibration_season, evaluation_calibration_period=season_name(CONFIRMATION_YEAR - 1),
                       audit_status=audit["status"], audit=audit,
                       temperature=t, labels=LABELS, evaluation="monthly out-of-time confirmation; final artifact refitted after evaluation",
                       splits={"test": dict(n=len(subset), start=result["start"], end=result["end"],
                               raw=reports[name]["raw"], calibrated=reports[name]["calibrated"],
                               class_prior=diagnostics(subset.result, np.tile(prior, (len(subset), 1))))},
                       experiment=dict(development_qualified=decision["development_qualified"], promoted=promoted,
                                       active=active, reason=result["reason"], uncertainty=interval,
                                       baseline=reports["baseline"], challenger=reports["selected"],
                                       development_baseline=decision["baseline"], development_selected=decision["selected"]))
        bundle = dict(version=2, model=model, config=config, temperature=t, matches=matches,
                      metrics=metrics, feature_version=config.get("bundle", "score"),
                      training_cutoff=str(cutoff.date()), calibration_period=calibration_season)
        joblib.dump(bundle, output / f"{name}.joblib")
        write_json(output / f"{name}-metrics.json", metrics)
    path = output / f"{active}.joblib"
    pointer = dict(version=2, path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   reason=result["reason"])
    write_json(ROOT / "artifacts/active.json", pointer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("develop", "all"), default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.stage)
