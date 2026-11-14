"""Isolated, fixed-protocol Wyscout shot-quality research; never promotes a model.

Data: Pappalardo et al. (2019), doi:10.1038/s41597-019-0247-7,
CC BY 4.0. Derived probabilities are this project's estimates, not Wyscout xG.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import ROOT
from .features import feature_rows
from .train import LABELS, evaluate

SOURCES = {
    "matches.zip": ("https://ndownloader.figshare.com/files/14464622", "c8f92bb7533e5c127e043cee764c991b5c25b4f5e70a65be931baae0b1765ce9"),
    "teams.json": ("https://ndownloader.figshare.com/files/15073697", "9f7a4a3b3d92c0be33f40613ad6e6eb4316c3b9771ec74c61a22c9b8ece23a4d"),
    "events.zip": ("https://ndownloader.figshare.com/files/14464685", "877e015b716ffdeea18f04418e3f24fed307ed03c37ff305cabe1f47c4822a45"),
}
SHOT_COLUMNS = ["distance_metres", "angle_radians", "header", "unknown_body_part"]
XG_COLUMNS = ["home_prior_xg_for", "home_prior_xg_against", "home_xg_history",
              "away_prior_xg_for", "away_prior_xg_against", "away_xg_history"]


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def acquire(cache=None):
    """Verify immutable original files; cached data permits offline replay."""
    cache = Path(cache or ROOT / "data/external/wyscout")
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "Pappalardo et al. / Wyscout", "license": "CC BY 4.0",
                "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871/", "files": {}}
    for name, (url, expected) in SOURCES.items():
        path = cache / name
        if not path.exists():
            with urlopen(url, timeout=120) as response:
                content = response.read()
            if sha256(content) != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
            temporary = cache / f"{name}.shot-quality.tmp"
            temporary.write_bytes(content)
            temporary.replace(path)
        content = path.read_bytes()
        if sha256(content) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
        manifest["files"][name] = dict(url=url, sha256=expected, bytes=len(content))
    (cache / "shot-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache, manifest


def source_matches(payload, teams):
    names = {int(team["wyId"]): team["name"] for team in teams}
    records = []
    for match in payload:
        if match["status"] != "Played":
            raise ValueError("Research requires completed source fixtures")
        sides = {item["side"]: (int(team), item) for team, item in match["teamsData"].items()}
        if set(sides) != {"home", "away"}:
            raise ValueError("Expected one home and one away club")
        home_id, home = sides["home"]
        away_id, away = sides["away"]
        hg, ag = home["score"], away["score"]
        if home_id == away_id or any(type(n) is not int or n < 0 for n in (hg, ag)):
            raise ValueError("Invalid fixture identity or score")
        day = pd.to_datetime(match["dateutc"], utc=True, errors="raise").tz_convert(None).normalize()
        records.append(dict(match_id=int(match["wyId"]), date=day,
                            home_team=names[home_id], away_team=names[away_id],
                            home_id=home_id, away_id=away_id, home_goals=hg, away_goals=ag,
                            result="H" if hg > ag else "A" if hg < ag else "D", season="2017-18"))
    frame = pd.DataFrame(records).sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    if frame.match_id.duplicated().any() or frame.duplicated(["home_id", "away_id"]).any():
        raise ValueError("Duplicate source fixture")
    return frame


def shot_geometry(x, y):
    if not np.isfinite([x, y]).all() or not (0 <= x <= 100 and 0 <= y <= 100):
        raise ValueError("Shot position must be finite percentages in [0,100]")
    dx, dy = (100 - float(x)) * 1.05, (float(y) - 50) * .68
    return float(np.hypot(dx, dy)), float(np.arctan2(7.32 * dx, dx * dx + dy * dy - 3.66 ** 2))


def shot_features(event):
    """Only location/body context enter X; outcome tags never become predictors."""
    positions = event.get("positions")
    if not isinstance(positions, list) or not positions or not {"x", "y"}.issubset(positions[0]):
        raise ValueError("Shot lacks start position")
    distance, angle = shot_geometry(positions[0]["x"], positions[0]["y"])
    tags = {tag["id"] for tag in event.get("tags", [])}
    return [distance, angle, int(403 in tags), int(not tags.intersection({401, 402, 403}))]


def extract_shots(events, matches):
    fixtures = {row.match_id: row for row in matches.itertuples()}
    records, penalties = [], 0
    seen = set()
    for event in events:
        if event.get("subEventName") == "Penalty":
            penalties += 1
            continue
        if event.get("eventName") != "Shot" and event.get("subEventName") != "Free kick shot":
            continue
        if event["id"] in seen:
            raise ValueError("Duplicate shot event")
        seen.add(event["id"])
        fixture = fixtures.get(event["matchId"])
        if fixture is None or event["teamId"] not in {fixture.home_id, fixture.away_id}:
            raise ValueError("Shot does not belong to a known fixture/team")
        tags = {tag["id"] for tag in event.get("tags", [])}
        values = shot_features(event)
        records.append(dict(event_id=event["id"], match_id=fixture.match_id, date=fixture.date,
                            team_id=event["teamId"], goal=int(101 in tags),
                            **dict(zip(SHOT_COLUMNS, values))))
    if not records:
        raise ValueError("No non-penalty shots found")
    return pd.DataFrame(records).sort_values(["date", "match_id", "event_id"]).reset_index(drop=True), penalties


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1., solver="lbfgs", max_iter=2000))


def out_of_time_shots(shots, start="2017-11-01", end="2018-06-01"):
    target = shots[(shots.date >= start) & (shots.date < end)].copy()
    target["estimated_xg"] = np.nan
    target["fit_cutoff"] = ""
    for month in sorted(target.date.dt.to_period("M").unique()):
        cutoff = month.start_time
        train = shots[shots.date < cutoff]
        if len(train) < 100 or set(train.goal) != {0, 1}:
            raise ValueError("Shot warmup requires >=100 shots and both outcomes")
        model = classifier().fit(train[SHOT_COLUMNS].to_numpy(), train.goal.to_numpy())
        mask = target.date.dt.to_period("M") == month
        target.loc[mask, "estimated_xg"] = model.predict_proba(target.loc[mask, SHOT_COLUMNS].to_numpy())[:, 1]
        target.loc[mask, "fit_cutoff"] = str(cutoff.date())
    if target.empty or not np.isfinite(target.estimated_xg).all():
        raise ValueError("No finite out-of-time shot estimates")
    return target


def aggregate_xg(matches, predicted_shots, start="2017-11-01", end="2018-06-01"):
    totals = predicted_shots.groupby(["match_id", "team_id"]).estimated_xg.sum().to_dict()
    rows = []
    for row in matches[(matches.date >= start) & (matches.date < end)].itertuples():
        # A team with zero non-penalty attempts has zero non-penalty estimated xG.
        rows.append(dict(match_id=row.match_id, date=row.date, home_xg=totals.get((row.match_id, row.home_id), 0.),
                         away_xg=totals.get((row.match_id, row.away_id), 0.)))
    return pd.DataFrame(rows)


def lagged_xg_features(matches, xg):
    values = {row.match_id: (row.home_xg, row.away_xg) for row in xg.itertuples()}
    history = defaultdict(lambda: deque(maxlen=5))
    rows = []
    for day, batch in matches.groupby("date", sort=True):
        for row in batch.itertuples():
            features = []
            for team in (row.home_id, row.away_id):
                previous = history[team]
                means = np.mean(previous, axis=0).tolist() if previous else [1.3, 1.3]
                features += [*means, len(previous)]
            rows.append(features)
        for row in batch.itertuples():
            if row.match_id in values:
                home, away = values[row.match_id]
                history[row.home_id].append((home, away))
                history[row.away_id].append((away, home))
    return pd.DataFrame(rows, columns=XG_COLUMNS)


def outcome_comparison(matches, xg):
    baseline = feature_rows(matches).to_numpy()
    enhanced = np.column_stack([baseline, lagged_xg_features(matches, xg).to_numpy()])
    target = (matches.date >= "2018-02-01") & (matches.date < "2018-06-01")
    indices = np.flatnonzero(target)
    outputs = {"goals_only": np.empty((len(indices), 3)), "goals_plus_lagged_xg": np.empty((len(indices), 3))}
    cutoffs = np.empty(len(indices), dtype="U10")
    for month in sorted(matches.loc[target, "date"].dt.to_period("M").unique()):
        origin = month.start_time
        train = (matches.date >= "2017-11-01") & (matches.date < origin)
        if train.sum() < 30 or set(matches.loc[train, "result"]) != set(LABELS):
            raise ValueError("Outcome warmup requires >=30 matches and all classes")
        local = (matches.iloc[indices].date.dt.to_period("M") == month).to_numpy()
        for name, matrix in (("goals_only", baseline), ("goals_plus_lagged_xg", enhanced)):
            model = classifier().fit(matrix[train], matches.loc[train, "result"])
            if model.classes_.tolist() != LABELS:
                raise ValueError("Unexpected outcome probability order")
            outputs[name][local] = model.predict_proba(matrix[indices[local]])
        cutoffs[local] = str(origin.date())
    if not len(indices):
        raise ValueError("Empty fixed evaluation period")
    metrics = {name: evaluate(matches.iloc[indices].result.to_numpy(), p) for name, p in outputs.items()}
    ledger = matches.iloc[indices][["match_id", "date", "home_team", "away_team", "result"]].copy()
    ledger["fit_cutoff"] = cutoffs
    for name, probabilities in outputs.items():
        for column, label in enumerate(LABELS):
            ledger[f"{name}_{label}"] = probabilities[:, column]
    return metrics, ledger


def run(cache=None, output=None):
    output = Path(output or ROOT / "artifacts/shot-quality").resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Research output must stay under artifacts")
    directory, sources = acquire(cache)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=1, research_only=True, no_promotion=True, source=sources,
                    code_sha256=sha256(Path(__file__).read_bytes()), shot_features=SHOT_COLUMNS,
                    shot_warmup="2017-08-01 through 2017-10-31", shot_predictions="2017-11-01 through 2018-05-31",
                    outcome_warmup="2017-11-01 through 2018-01-31", outcome_evaluation="2018-02-01 through 2018-05-31",
                    schedule="monthly refits using strictly earlier dates; all same-day features before updates",
                    excluded="penalties", regularization_C=1., calibration="none", tuning="none")
    protocol_path, report_path = output / "protocol.json", output / "report.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Research code/source/protocol changed; use a new --output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for name, expected in report["derived_file_sha256"].items():
            if sha256((output / name).read_bytes()) != expected:
                raise ValueError("Saved research derivative checksum mismatch")
        return report
    with ZipFile(directory / "matches.zip") as archive:
        matches = source_matches(json.loads(archive.read("matches_England.json")),
                                 json.loads((directory / "teams.json").read_bytes()))
    if len(matches) != 380:
        raise ValueError("Expected the complete 380-match EPL source season")
    with ZipFile(directory / "events.zip") as archive:
        events = json.loads(archive.read("events_England.json"))
    if len(events) != 643150:
        raise ValueError("Unexpected England event count")
    shots, penalties = extract_shots(events, matches)
    del events
    predicted = out_of_time_shots(shots)
    xg = aggregate_xg(matches, predicted)
    metrics, ledger = outcome_comparison(matches, xg)
    predicted.to_csv(output / "out-of-time-shots.csv", index=False)
    xg.to_csv(output / "match-estimated-xg.csv", index=False)
    ledger.to_csv(output / "outcome-predictions.csv", index=False)
    report = dict(version=1, research_only=True, serving_model_changed=False,
                  scope="One historical EPL season; exploratory within-season result, not modern confirmation",
                  attribution="Pappalardo et al. (2019), Wyscout, doi:10.1038/s41597-019-0247-7; CC BY 4.0. Estimated xG is a derived project model, not provider-supplied xG.",
                  source_matches=len(matches), source_events=643150, non_penalty_shots=len(shots), penalties_excluded=penalties,
                  out_of_time_shots=len(predicted), xg_matches=len(xg), evaluation_matches=len(ledger),
                  evaluation_start=str(ledger.date.min().date()), evaluation_end=str(ledger.date.max().date()),
                  shot_model=dict(log_loss=float(log_loss(predicted.goal, predicted.estimated_xg, labels=[0, 1])),
                                  brier=float(brier_score_loss(predicted.goal, predicted.estimated_xg))),
                  outcome_models=metrics,
                  accuracy_difference=float(metrics["goals_plus_lagged_xg"]["accuracy"] - metrics["goals_only"]["accuracy"]),
                  log_loss_difference=float(metrics["goals_plus_lagged_xg"]["log_loss"] - metrics["goals_only"]["log_loss"]),
                  limitations=["One 2017/18 season cannot establish present-day forecast quality", "Small evaluation set and no external-season confirmation", "No hyperparameter tuning, modern model promotion, or betting-profitability claim", "Penalties excluded; estimates are non-penalty shot quality", "Coordinates use a nominal 105x68 metre pitch", "Final lineups and same-match goal/xG outcomes are not pre-match inputs"],
                  derived_file_sha256={name: sha256((output / name).read_bytes()) for name in
                                       ("out-of-time-shots.csv", "match-estimated-xg.csv", "outcome-predictions.csv")})
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(run(args.cache, args.output), indent=2))
