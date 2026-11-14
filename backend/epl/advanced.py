"""Opponent-adjusted, chronological features shared by experiments and serving."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .features import FEATURES, team_features

_PRIOR = np.array([1.0, 1.3, 1.3, 0.0])


def _season_year(day):
    return day.year if day.month >= 8 else day.year - 1


def feature_rows(matches, fixture=None, *, k=20, home_offset=50, bundle="rich"):
    """Return pre-match rows in input order, or one strictly historical fixture row.

    Elo uses a 400-point logistic scale. New-season ratings retain 75% of
    their deviation from 1500; teams absent in the preceding season reset.
    """
    if bundle not in {"elo", "rich"}:
        raise ValueError("bundle must be 'elo' or 'rich'")
    if not np.isfinite(k) or k <= 0 or not np.isfinite(home_offset):
        raise ValueError("Elo parameters must be finite and k positive")
    required = {"date", "season", "home_team", "away_team", "home_goals", "away_goals"}
    if not required.issubset(matches.columns):
        raise ValueError("Missing match columns")
    if fixture is not None:
        if len(fixture) != 3 or fixture[0] == fixture[1]:
            raise ValueError("Fixture must contain distinct home and away teams and a date")
        fixture_day = pd.Timestamp(fixture[2])
        if pd.isna(fixture_day):
            raise ValueError("Invalid fixture date")

    ratings = defaultdict(lambda: 1500.0)
    legacy = defaultdict(lambda: deque(maxlen=10))
    recent = defaultdict(lambda: deque(maxlen=20))
    venue = defaultdict(lambda: deque(maxlen=5))
    last_seen = {}
    season = None
    season_teams = set()
    previous_teams = set()

    def advance(year):
        nonlocal season, season_teams, previous_teams
        if season == year:
            return
        previous_teams = season_teams if season is not None and year == season + 1 else set()
        for team in ratings:
            ratings[team] = 1500 + .75 * (ratings[team] - 1500) if team in previous_teams else 1500.0
        season_teams = set()
        season = year

    def describe(team, side, day):
        days = (day - last_seen[team]).days if team in last_seen else None
        if days is not None and days > 365:
            recent[team].clear()
            venue[(team, "home")].clear()
            venue[(team, "away")].clear()
        history = list(recent[team])
        values = np.array([row[:4] for row in history]) if history else np.empty((0, 4))
        output = {}
        for count in (5, 10, 20):
            means = values[-count:].mean(axis=0) if len(values) else _PRIOR
            for name, value in zip(("points", "scored", "conceded", "goal_difference"), means):
                output[f"form_{count}_{name}"] = float(value)
        overall = values.mean(axis=0) if len(values) else _PRIOR
        at_venue = list(venue[(team, side)])
        venue_sum = np.sum(at_venue, axis=0) if at_venue else np.zeros(4)
        shrunk = (venue_sum + 5 * overall) / (len(at_venue) + 5)
        for name, value in zip(("points", "scored", "conceded", "goal_difference"), shrunk):
            output[f"venue_{name}"] = float(value)
        output.update(history=len(history), venue_history=len(at_venue),
                      rest_days=min(days, 60) if days is not None else 7,
                      days_since_appearance=min(days, 365) if days is not None else 365,
                      returning_or_promoted=int(team not in previous_teams),
                      opponent_strength=float(np.mean([row[4] for row in history[-5:]])) if history else 1500.0)
        return output

    def row_features(home, away, day):
        home_rating, away_rating = ratings[home], ratings[away]
        if bundle == "rich":
            for team in (home, away):
                if team in last_seen and (day - last_seen[team]).days > 365:
                    legacy[team].clear()
            home_values, away_values = describe(home, "home", day), describe(away, "away", day)
        output = dict(zip(FEATURES, team_features(legacy[home], day) + team_features(legacy[away], day)))
        output.update(home_elo=home_rating, away_elo=away_rating,
                      elo_difference=home_rating + home_offset - away_rating)
        if bundle == "rich":
            output.update({f"home_{name}": value for name, value in home_values.items()})
            output.update({f"away_{name}": value for name, value in away_values.items()})
            output.update({f"difference_{name}": home_values[name] - away_values[name] for name in home_values})
        return output

    ordered = matches.copy()
    ordered["_position"] = np.arange(len(ordered))
    ordered["date"] = pd.to_datetime(ordered.date, errors="raise")
    ordered = ordered.sort_values(["date", "home_team", "away_team"], kind="stable")
    output = [None] * len(matches)
    for day, batch in ordered.groupby("date", sort=False):
        if fixture is not None and day >= fixture_day:
            break
        years = {int(str(value).split("-")[0]) for value in batch.season}
        if len(years) != 1:
            raise ValueError("One date cannot contain multiple seasons")
        year = next(iter(years))
        advance(year)
        updates = []
        for row in batch.to_dict("records"):
            home, away = row["home_team"], row["away_team"]
            output[row["_position"]] = row_features(home, away, day)
            hg, ag = row["home_goals"], row["away_goals"]
            home_rating, away_rating = ratings[home], ratings[away]
            expected = 1 / (1 + 10 ** ((away_rating - home_rating - home_offset) / 400))
            actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            updates.append((home, away, hg, ag, home_rating, away_rating, k * (actual - expected)))
        # Capture every opponent rating before revealing any same-day result.
        for home, away, hg, ag, home_rating, away_rating, change in updates:
            ratings[home] += change
            ratings[away] -= change
            hp, ap = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
            legacy[home].append((hp, hg, ag, day))
            legacy[away].append((ap, ag, hg, day))
            home_result = (hp, hg, ag, hg - ag)
            away_result = (ap, ag, hg, ag - hg)
            recent[home].append((*home_result, away_rating))
            recent[away].append((*away_result, home_rating))
            venue[(home, "home")].append(home_result)
            venue[(away, "away")].append(away_result)
            last_seen[home] = last_seen[away] = day
            season_teams.update((home, away))
    if fixture is not None:
        advance(_season_year(fixture_day))
        return pd.DataFrame([row_features(fixture[0], fixture[1], fixture_day)])
    if not output:
        return pd.DataFrame()
    return pd.DataFrame(output, index=matches.index)
