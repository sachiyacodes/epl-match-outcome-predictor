"""Synthetic audit fixtures test guards; historical integration uses pinned source files."""
import json
from pathlib import Path

import pandas as pd
import pytest

from epl import audit as auditing
from epl.data import ROOT, parse_season


def complete_season():
    teams = [f"Team {number}" for number in range(20)]
    rounds = []
    for _ in range(19):
        rounds.append([(teams[i], teams[19 - i]) for i in range(10)])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    rounds += [[(away, home) for home, away in fixtures] for fixtures in rounds]
    rows = []
    for week, fixtures in enumerate(rounds):
        for home, away in fixtures:
            rows.append(dict(date=pd.Timestamp("2019-08-01") + pd.Timedelta(days=7 * week),
                             home_team=home, away_team=away, home_goals=2, away_goals=0,
                             result="H", season="2019-20"))
    return pd.DataFrame(rows)


def test_array_score_is_completed_and_zero_draw_is_not_missing():
    fixture = dict(date="2025-08-16", team1="Alpha", team2="Beta", score=[0, 0])
    frame = parse_season(dict(name="English Premier League 2025/26", matches=[fixture]), "2025-26")
    assert frame.result.tolist() == ["D"]
    assert frame.home_goals.tolist() == [0]


@pytest.mark.parametrize("score", ["0-0", 0, {"ft": "0-0"}, [0, None], [0, 0, 0]])
def test_ambiguous_score_formats_are_rejected(score):
    fixture = dict(date="2025-08-16", team1="Alpha", team2="Beta", score=score)
    with pytest.raises(ValueError, match="Invalid full-time score"):
        parse_season(dict(name="English Premier League 2025/26", matches=[fixture]), "2025-26")


def test_complete_round_robin_passes_season_audit():
    report = auditing.validate_seasons(complete_season())
    assert report[0]["matches"] == 380
    assert report[0]["teams"] == 20


@pytest.mark.parametrize("day", ["2019-07-01", "2020-08-01"])
def test_out_of_season_dates_fail_even_when_syntactically_valid(day):
    frame = complete_season()
    frame.loc[0, "date"] = pd.Timestamp(day)
    with pytest.raises(ValueError, match="outside season interval"):
        auditing.validate_seasons(frame)


def test_incomplete_season_is_rejected():
    with pytest.raises(ValueError, match="380 matches"):
        auditing.validate_seasons(complete_season().iloc[:-1])


def test_duplicate_fixture_is_rejected():
    frame = complete_season()
    frame.loc[1] = frame.loc[0]
    with pytest.raises(ValueError, match="duplicate fixture"):
        auditing.validate_seasons(frame)


def test_home_away_balance_is_a_gate():
    frame = complete_season()
    frame.loc[0, ["home_team", "away_team"]] = frame.loc[0, ["away_team", "home_team"]].to_numpy()
    with pytest.raises(ValueError, match="duplicate fixture|19 home"):
        auditing.validate_seasons(frame)


def test_same_day_double_appearance_is_rejected():
    frame = complete_season()
    frame.loc[10, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="appears twice"):
        auditing.validate_seasons(frame)


def test_corrections_require_exact_fixture_identity_and_preserve_input():
    frame = complete_season()
    before = frame.copy()
    row = frame.iloc[0]
    entry = {key: row[key] for key in ["season", *auditing.KEYS]}
    entry.update(original_date=str(row.date.date()), corrected_date="2019-08-02")
    corrected = auditing.apply_corrections(frame, {"corrections": [entry]})
    assert corrected.iloc[0].date == pd.Timestamp("2019-08-02")
    pd.testing.assert_frame_equal(frame, before)
    entry["home_goals"] = 99
    with pytest.raises(ValueError, match="exactly one"):
        auditing.apply_corrections(frame, {"corrections": [entry]})


def test_altered_correction_evidence_fails_checksum():
    with pytest.raises(ValueError, match="checksum mismatch"):
        auditing.evidence_matches(b"replacement content")


def test_unreconciled_season_is_quarantined_as_a_whole(monkeypatch, tmp_path):
    older = complete_season()
    older["date"] -= pd.DateOffset(years=1)
    older["season"] = "2018-19"
    frame = pd.concat([older, complete_season()], ignore_index=True)
    monkeypatch.setattr(auditing, "load_matches", lambda _: frame)
    monkeypatch.setattr(auditing, "correction_ledger", lambda *_: (_ for _ in ()).throw(ValueError("ambiguous source")))
    (tmp_path / "2019-20.json").write_bytes(b"synthetic")
    (tmp_path / "2019-20-evidence.txt").write_bytes(b"synthetic")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="ambiguous source"):
        auditing._audit_snapshot(tmp_path)
    matches, ledger, report = auditing._audit_snapshot(tmp_path, allow_quarantine=True)
    assert set(matches.season) == {"2018-19"}
    assert ledger is None
    assert report["status"] == "passed_with_quarantine"
    assert report["quarantined"] == [{"season": "2019-20", "reason": "ambiguous source"}]


def pinned_snapshot():
    directory = ROOT / "data/audited"
    if not (directory / "audit.json").exists():
        pytest.skip("Run python -m epl.audit for pinned historical integration checks")
    return directory


def test_historical_dates_are_reconciled_against_immutable_evidence():
    directory = pinned_snapshot()
    evidence = (directory / "2019-20-evidence.txt").read_bytes()
    raw = (directory / "2019-20.json").read_bytes()
    original = parse_season(json.loads(raw), "2019-20")
    with pytest.raises(ValueError, match="outside season interval"):
        auditing.validate_seasons(original)
    ledger = auditing.correction_ledger(original, raw, evidence)
    assert len(ledger["corrections"]) == 66
    assert ledger == json.loads((directory / "corrections.json").read_text())
    corrected = auditing.apply_corrections(original, ledger)
    auditing.validate_seasons(corrected)
    fixture = corrected[(corrected.home_team == "Arsenal FC") & (corrected.away_team == "Norwich City FC")]
    assert fixture.iloc[0].date == pd.Timestamp("2020-07-01")
    assert auditing.sha256((directory / "2019-20.json").read_bytes()) == auditing.RAW_2019_SHA256


def test_audited_snapshot_and_ledger_tamper_detection(tmp_path):
    import shutil

    directory = pinned_snapshot()
    frame = auditing.load_audited(directory)
    assert len(frame) == 6080
    assert frame.date.max() == pd.Timestamp("2026-05-24")
    shutil.copytree(directory, tmp_path / "snapshot")
    path = tmp_path / "snapshot/corrections.json"
    path.write_text(path.read_text().replace("2020-07-01", "2020-07-02", 1))
    with pytest.raises(ValueError, match="ledger checksum mismatch"):
        auditing.load_audited(tmp_path / "snapshot")
