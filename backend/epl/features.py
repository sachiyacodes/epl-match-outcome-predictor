"""All features depend only on results strictly before the fixture date."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

FEATURES = [f"{side}_{name}" for side in ("home", "away")
            for name in ("points", "scored", "conceded", "history", "rest_days")]


def team_features(history, day):
    if not history:
        return [1.0, 1.3, 1.3, 0, 7]
    values = np.array([row[:3] for row in history])
    rest = min((day - history[-1][3]).days, 60)
    return [*values.mean(axis=0), len(history), rest]


def feature_rows(matches, fixture=None):
    history = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for day, batch in matches.groupby("date", sort=True):
        if fixture and day >= pd.Timestamp(fixture[2]):
            break
        # Produce every same-day feature before revealing any same-day outcome.
        for row in batch.itertuples():
            rows.append(team_features(history[row.home_team], day) +
                        team_features(history[row.away_team], day))
        for row in batch.itertuples():
            hg, ag = row.home_goals, row.away_goals
            history[row.home_team].append((3 if hg > ag else 1 if hg == ag else 0, hg, ag, day))
            history[row.away_team].append((3 if ag > hg else 1 if hg == ag else 0, ag, hg, day))
    if fixture:
        home, away, date = fixture
        day = pd.Timestamp(date)
        return pd.DataFrame([team_features(history[home], day) + team_features(history[away], day)], columns=FEATURES)
    return pd.DataFrame(rows, columns=FEATURES)
