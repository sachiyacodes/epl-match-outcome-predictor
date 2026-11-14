"""Synthetic fixtures exercise feature chronology; they are not EPL observations."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from epl.advanced import feature_rows
from epl.features import feature_rows as baseline_rows


def frame(rows):
    return pd.DataFrame([dict(date=pd.Timestamp(day), season=season, home_team=home,
                              away_team=away, home_goals=hg, away_goals=ag)
                         for day, season, home, away, hg, ag in rows])


@pytest.fixture
def matches():
    return frame([
        ("2020-01-01", "2019-20", "Alpha", "Beta", 2, 0),
        ("2020-01-01", "2019-20", "Gamma", "Delta", 1, 1),
        ("2020-01-08", "2019-20", "Alpha", "Gamma", 0, 1),
        ("2020-01-09", "2019-20", "Beta", "Delta", 1, 1),
        ("2020-01-15", "2019-20", "Gamma", "Beta", 3, 0),
    ])


@pytest.mark.parametrize("bundle", ["elo", "rich"])
def test_own_future_and_same_day_results_do_not_leak(matches, bundle):
    before = feature_rows(matches, bundle=bundle)
    changed = matches.copy()
    changed.loc[2:, ["home_goals", "away_goals"]] = [20, 0]
    assert_frame_equal(before.iloc[:3], feature_rows(changed, bundle=bundle).iloc[:3])
    fixture = ("Alpha", "Beta", "2020-01-08")
    assert_frame_equal(feature_rows(matches, fixture, bundle=bundle),
                       feature_rows(changed, fixture, bundle=bundle))


def test_elo_bundle_preserves_baseline_features(matches):
    actual = feature_rows(matches, bundle="elo")
    expected = baseline_rows(matches)
    assert_frame_equal(actual[expected.columns], expected)
    assert actual.loc[0, "home_elo"] == 1500
    assert actual.loc[2, "home_elo"] > 1500


def test_input_alignment_and_fixture_features_match(matches):
    shuffled = matches.iloc[[4, 0, 2, 1, 3]]
    ordered = feature_rows(matches)
    actual = feature_rows(shuffled)
    assert_frame_equal(actual.sort_index(), ordered)
    fixture = tuple(matches.loc[2, ["home_team", "away_team", "date"]])
    assert_frame_equal(feature_rows(matches, fixture), ordered.iloc[[2]].reset_index(drop=True))


def test_new_season_regresses_existing_elo_and_resets_absent_team():
    matches = frame([
        ("2019-08-01", "2019-20", "Alpha", "Old", 3, 0),
        ("2020-08-01", "2020-21", "Alpha", "Beta", 3, 0),
        ("2021-08-01", "2021-22", "Alpha", "Old", 1, 1),
    ])
    rows = feature_rows(matches, k=20, home_offset=0)
    assert rows.loc[1, "home_elo"] == pytest.approx(1507.5)
    assert rows.loc[1, "away_elo"] == 1500
    assert rows.loc[1, "home_returning_or_promoted"] == 0
    assert rows.loc[1, "away_returning_or_promoted"] == 1
    assert rows.loc[2, "away_elo"] == 1500
    assert rows.loc[2, "away_returning_or_promoted"] == 1
    # A fixture-only next-season transition must match the offline path.
    assert_frame_equal(feature_rows(matches.iloc[:2], ("Alpha", "Old", "2021-08-01"), home_offset=0),
                       rows.iloc[[2]].reset_index(drop=True))


def test_rich_form_resets_after_year_absence():
    matches = frame([
        ("2019-08-01", "2019-20", "Alpha", "Beta", 5, 0),
        ("2021-08-02", "2021-22", "Alpha", "Gamma", 0, 0),
    ])
    row = feature_rows(matches).iloc[1]
    assert row.home_history == 0
    assert row.home_form_5_points == 1
    assert row.home_form_5_scored == 1.3
    assert row.home_points == 1
    assert row.home_venue_history == 0
    assert row.home_days_since_appearance == 365
    assert row.home_elo == 1500


def test_venue_shrinkage_and_opponent_rating_use_pre_match_state():
    matches = frame([
        ("2020-01-01", "2019-20", "Alpha", "Beta", 2, 0),
        ("2020-01-02", "2019-20", "Gamma", "Alpha", 1, 1),
        ("2020-01-03", "2019-20", "Alpha", "Delta", 0, 1),
    ])
    row = feature_rows(matches, home_offset=0).iloc[2]
    assert row.home_form_5_points == 2
    assert row.home_venue_points == pytest.approx((3 + 5 * 2) / 6)
    assert row.home_opponent_strength == 1500
    assert row.difference_form_5_points == 1


def test_cold_features_and_parameters(matches):
    row = feature_rows(matches, ("New", "Other", "2020-02-01")).iloc[0]
    assert row.home_form_20_points == 1
    assert row.home_form_20_scored == 1.3
    assert row.home_history == 0
    assert row.home_rest_days == 7
    assert row.home_elo == 1500
    assert row.elo_difference == 50
    assert np.isfinite(row.to_numpy()).all()
    with pytest.raises(ValueError, match="bundle"):
        feature_rows(matches, bundle="unknown")
    with pytest.raises(ValueError, match="positive"):
        feature_rows(matches, k=0)
