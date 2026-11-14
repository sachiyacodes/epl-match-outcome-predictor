"""Synthetic cases test chronology and geometry, never football performance claims."""
import numpy as np
import pandas as pd
import pytest

from epl.shot_quality import SHOT_COLUMNS, aggregate_xg, extract_shots, lagged_xg_features, out_of_time_shots, shot_features, shot_geometry
from epl import shot_quality


def event(event_id=1, match_id=1, team=1, goal=False):
    return dict(id=event_id, matchId=match_id, teamId=team, eventName="Shot", subEventName="Shot",
                positions=[{"x": 88, "y": 50}], tags=[{"id": 402}] + ([{"id": 101}] if goal else []))


def fixtures():
    return pd.DataFrame(dict(match_id=[1, 2, 3], date=pd.to_datetime(["2017-11-01", "2017-11-08", "2017-11-15"]),
                             home_id=[1, 1, 1], away_id=[2, 3, 2]))


def test_geometry_uses_goal_angle_and_distance_in_metres():
    distance, angle = shot_geometry(90, 50)
    assert distance == pytest.approx(10.5)
    assert angle == pytest.approx(2 * np.arctan(3.66 / 10.5))
    assert shot_geometry(90, 20)[1] < angle
    assert shot_geometry(80, 50)[0] > distance


@pytest.mark.parametrize("x,y", [(101, 50), (-1, 50), (50, 101), (np.nan, 50), (50, np.inf)])
def test_invalid_shot_coordinates_fail_closed(x, y):
    with pytest.raises(ValueError, match="finite percentages"):
        shot_geometry(x, y)


def test_goal_and_outcome_tags_never_enter_shot_predictors():
    plain, scored = event(), event(goal=True)
    scored["tags"] += [{"id": 1801}, {"id": 1205}]
    assert shot_features(plain) == shot_features(scored)
    shots, penalties = extract_shots([scored], fixtures())
    assert shots.goal.tolist() == [1]
    assert penalties == 0
    assert not any("goal" in column for column in SHOT_COLUMNS)


def test_penalties_are_excluded_and_unknown_fixture_is_rejected():
    penalty = event(2)
    penalty["subEventName"] = "Penalty"
    shots, excluded = extract_shots([event(), penalty], fixtures())
    assert len(shots) == 1
    assert excluded == 1
    with pytest.raises(ValueError, match="known fixture/team"):
        extract_shots([event(match_id=99)], fixtures())


def test_same_and_future_match_xg_cannot_change_earlier_features():
    matches = fixtures()
    xg = pd.DataFrame(dict(match_id=[1, 2, 3], home_xg=[2., 3., 4.], away_xg=[.5, 1., 2.]))
    before = lagged_xg_features(matches, xg)
    changed = xg.copy()
    changed.loc[1:, "home_xg"] = 50
    after = lagged_xg_features(matches, changed)
    pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
    assert before.iloc[0].home_xg_history == 0
    assert before.iloc[1].home_prior_xg_for == 2
    assert before.iloc[2].home_prior_xg_for != after.iloc[2].home_prior_xg_for


def test_every_same_day_feature_precedes_updates():
    matches = fixtures()
    matches.loc[1, "date"] = matches.loc[0, "date"]
    xg = pd.DataFrame(dict(match_id=[1, 2], home_xg=[10., 20.], away_xg=[1., 2.]))
    values = lagged_xg_features(matches, xg)
    assert values.home_xg_history.tolist() == [0, 0, 2]
    assert values.iloc[1].home_prior_xg_for == 1.3


def test_shot_monthly_fit_excludes_own_and_future_outcomes():
    rows = []
    for month in range(8, 13):
        for i in range(120):
            rows.append(dict(date=pd.Timestamp(2017, month, 10), goal=int(i % 8 == 0),
                             distance_metres=5 + i % 30, angle_radians=.1 + (i % 10) / 10,
                             header=int(i % 4 == 0), unknown_body_part=0))
    shots = pd.DataFrame(rows)
    before = out_of_time_shots(shots, end="2018-01-01")
    changed = shots.copy()
    changed.loc[changed.date >= "2017-12-01", "goal"] = 1
    after = out_of_time_shots(changed, end="2018-01-01")
    np.testing.assert_allclose(before.estimated_xg, after.estimated_xg)
    assert all(pd.Timestamp(cutoff) < day for cutoff, day in zip(before.fit_cutoff, before.date))


def test_zero_shots_means_zero_non_penalty_xg_not_missing_match():
    matches = fixtures()
    predicted = pd.DataFrame(dict(match_id=[1], team_id=[1], estimated_xg=[.4]))
    xg = aggregate_xg(matches, predicted)
    assert len(xg) == 3
    assert xg.iloc[0].home_xg == .4
    assert xg.iloc[0].away_xg == 0


def test_cached_source_tampering_fails_before_network_access(tmp_path, monkeypatch):
    monkeypatch.setattr(shot_quality, "SOURCES", {"fixture.bin": ("https://example.invalid", "expected")})
    (tmp_path / "fixture.bin").write_bytes(b"synthetic test fixture")
    monkeypatch.setattr(shot_quality, "urlopen", lambda *args, **kwargs: pytest.fail("Cached tampering must not trigger download"))
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        shot_quality.acquire(tmp_path)
