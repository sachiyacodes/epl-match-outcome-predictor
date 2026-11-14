"""Synthetic source records prove reconciliation behavior, not historical truth."""
import io
import json
from zipfile import ZipFile

import pandas as pd
import pytest

from epl import verify_sources as verification


def source_bytes(matches=None, teams=None):
    teams = teams or [{"wyId": 1, "name": "Alpha"}, {"wyId": 2, "name": "Beta"}]
    matches = matches or [dict(competitionId=364, seasonId=181150, status="Played", wyId=10,
                               dateutc="2017-08-11 19:00:00", teamsData={
                                   "1": {"teamId": 1, "side": "home", "score": 2},
                                   "2": {"teamId": 2, "side": "away", "score": 0}})]
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("matches_England.json", json.dumps(matches))
    return buffer.getvalue(), json.dumps(teams).encode()


@pytest.fixture
def local():
    return pd.DataFrame([dict(season="2017-18", date=pd.Timestamp("2017-08-11"),
                              home_team="Alpha FC", away_team="Beta", home_goals=2, away_goals=0)])


def test_parser_preserves_provider_identity_dates_and_full_time_score():
    frame, member_hash = verification.parse_wyscout(*source_bytes())
    assert frame.iloc[0].source_match_id == 10
    assert frame.iloc[0].date == pd.Timestamp("2017-08-11")
    assert frame.iloc[0].home_goals == 2
    assert frame.iloc[0].away_goals == 0
    assert len(member_hash) == 64


def test_parser_rejects_unknown_team_duplicate_role_and_bad_score():
    base = dict(competitionId=364, seasonId=181150, status="Played", wyId=10, dateutc="2017-08-11 19:00:00")
    for away in ({"teamId": 99, "side": "away", "score": 0},
                 {"teamId": 2, "side": "home", "score": 0},
                 {"teamId": 2, "side": "away", "score": True}):
        match = dict(base, teamsData={"1": {"teamId": 1, "side": "home", "score": 2}, "2": away})
        with pytest.raises(ValueError):
            verification.parse_wyscout(*source_bytes([match]))


def test_exact_fixture_join_normalizes_only_terminal_fc_and_does_not_modify_input(local):
    before = local.copy()
    reference, _ = verification.parse_wyscout(*source_bytes())
    result = verification.reconcile(local, reference)
    assert result["status"] == "agreement"
    assert result["agreements"] == result["paired_fixtures"] == 1
    assert result["conflicts"] == []
    pd.testing.assert_frame_equal(local, before)
    reference.loc[0, "home_team"] = "Alfa"
    mismatch = verification.reconcile(local, reference)
    assert mismatch["paired_fixtures"] == 0
    assert mismatch["conflict_count"] == 2


def test_dates_and_scores_are_compared_after_identity_join(local):
    reference, _ = verification.parse_wyscout(*source_bytes())
    reference.loc[0, "date"] = pd.Timestamp("2017-08-12")
    reference.loc[0, "away_goals"] = 1
    result = verification.reconcile(local, reference)
    assert result["paired_fixtures"] == 1
    assert result["agreements"] == 0
    assert result["conflicts"][0]["issues"] == ["date_mismatch", "away_goals_mismatch"]
    assert result["conflicts"][0]["source"]["away_goals"] == 1


def test_ambiguous_normalized_fixture_fails_closed(local):
    reference, _ = verification.parse_wyscout(*source_bytes())
    with pytest.raises(ValueError, match="Ambiguous normalized fixture"):
        verification.reconcile(pd.concat([local, local]), reference)


def test_missing_local_and_reference_rows_are_reported(local):
    reference, _ = verification.parse_wyscout(*source_bytes())
    reverse = reference.copy()
    reverse[["home_team", "away_team"]] = reverse[["away_team", "home_team"]].to_numpy()
    result = verification.reconcile(local, reverse)
    assert result["conflict_count"] == 2
    assert {entry["issues"][0] for entry in result["conflicts"]} == {"missing_independent_fixture", "missing_local_fixture"}


def test_offline_cache_requires_exact_pinned_bytes(tmp_path, monkeypatch):
    content = b"verified synthetic cache"
    monkeypatch.setattr(verification, "SOURCES", {"sample.bin": dict(url="https://example.test/data", bytes=len(content),
                                                                  sha256=verification.sha256(content))})
    with pytest.raises(ValueError, match="Missing cached source"):
        verification.cached_source(tmp_path, "sample.bin", offline=True)
    (tmp_path / "sample.bin").write_bytes(content)
    assert verification.cached_source(tmp_path, "sample.bin", offline=True) == content
    (tmp_path / "sample.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verification.cached_source(tmp_path, "sample.bin", offline=True)
    assert (tmp_path / "sample.bin").read_bytes() == b"tampered"


def test_download_is_checked_before_cache_write(tmp_path, monkeypatch):
    expected = b"expected"
    monkeypatch.setattr(verification, "SOURCES", {"sample.bin": dict(url="https://example.test/data", bytes=len(expected),
                                                                  sha256=verification.sha256(expected))})
    monkeypatch.setattr(verification, "urlopen", lambda *args, **kwargs: io.BytesIO(b"not valid"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verification.cached_source(tmp_path, "sample.bin")
    assert not list(tmp_path.iterdir())


def test_parser_rejects_duplicate_match_and_team_identifiers():
    content, teams = source_bytes()
    with ZipFile(io.BytesIO(content)) as archive:
        match = json.loads(archive.read("matches_England.json"))[0]
    with pytest.raises(ValueError, match="duplicate independent fixtures"):
        verification.parse_wyscout(*source_bytes([match, match]))
    with pytest.raises(ValueError, match="Duplicate Wyscout team identifier"):
        verification.parse_wyscout(*source_bytes(teams=[{"wyId": 1, "name": "Alpha"}, {"wyId": 1, "name": "Beta"}]))
