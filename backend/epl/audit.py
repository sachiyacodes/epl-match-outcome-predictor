"""Validate a separate historical snapshot and explicitly repair known source dates."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .data import ROOT, TEAM_ALIASES, ingest, load_matches, parse_season

REVISION = "e6744429ee395bc86f247348c6184bb08d4eb361"
EVIDENCE_URL = "https://raw.githubusercontent.com/openfootball/eng-england/9ee7a0c351805c99cc334478a9bf04b7d0432fec/2019-20/1-premierleague.txt"
EVIDENCE_SHA256 = "fdc8300c09e0700faa0c517ff408bc630b8b6a77b6f1aa68e5858f9adae6f72d"
RAW_2019_SHA256 = "9e1695f85ce5553f762e809dea357ec0a95838939be89569060a83c4e98a302d"
KEYS = ["home_team", "away_team", "home_goals", "away_goals"]
MONTHS = dict(zip(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), range(1, 13)))


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def evidence_matches(content):
    """Read explicit match dates from the immutable season text, not the bad JSON."""
    if sha256(content) != EVIDENCE_SHA256:
        raise ValueError("Correction evidence checksum mismatch")
    lines = content.decode("utf-8").splitlines()
    if not any("Fri Aug 9 2019 - Sun Jul 26 2020" in line for line in lines):
        raise ValueError("Unexpected evidence season interval")
    current_date = None
    records = []
    for number, line in enumerate(lines, 1):
        day = re.fullmatch(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) ([A-Z][a-z]{2}) (\d{1,2})", line.strip())
        if day:
            weekday, month, date = day.groups()
            month_number = MONTHS[month]
            current_date = pd.Timestamp(year=2019 if month_number >= 8 else 2020,
                                        month=month_number, day=int(date))
            if current_date.day_name()[:3] != weekday:
                raise ValueError(f"Evidence weekday mismatch at line {number}")
            continue
        match = re.fullmatch(r"\s+(?:\d{1,2}:\d{2}\s+)?(.+?)\s{2,}(\d+)-(\d+)(?:\s+\(\d+-\d+\))?\s{2,}(.+?)\s*", line)
        if match:
            if current_date is None:
                raise ValueError("Evidence match lacks date")
            home, hg, ag, away = match.groups()
            records.append(dict(home_team=TEAM_ALIASES.get(home, home),
                                away_team=TEAM_ALIASES.get(away, away),
                                home_goals=int(hg), away_goals=int(ag),
                                date=current_date, evidence_line=number))
    frame = pd.DataFrame(records)
    if len(frame) != 380 or frame.duplicated(["home_team", "away_team"]).any():
        raise ValueError("Expected 380 unique fixtures in correction evidence")
    return frame


def correction_ledger(frame, raw_content, evidence_content):
    """Reconcile every 2019/20 fixture, then record only the exact 66 known defects."""
    if sha256(raw_content) != RAW_2019_SHA256:
        raise ValueError("2019/20 raw file is not the audited known source revision")
    evidence = evidence_matches(evidence_content)
    paired = frame.merge(evidence, on=KEYS, how="outer", suffixes=("_raw", "_evidence"), indicator=True,
                         validate="one_to_one")
    if len(paired) != 380 or not paired["_merge"].eq("both").all():
        raise ValueError("2019/20 source fixtures/scores do not fully reconcile")
    changed = paired[paired.date_raw != paired.date_evidence]
    if len(changed) != 66 or not changed.date_raw.between("2019-07-01", "2019-07-26").all():
        raise ValueError("Unexpected 2019/20 correction set")
    corrections = []
    for row in changed.sort_values(["date_evidence", "home_team"]).itertuples():
        if row.date_evidence != row.date_raw + pd.DateOffset(years=1):
            raise ValueError("Correction differs from the corroborated source date")
        corrections.append(dict(season="2019-20", home_team=row.home_team, away_team=row.away_team,
                                home_goals=int(row.home_goals), away_goals=int(row.away_goals),
                                original_date=str(row.date_raw.date()), corrected_date=str(row.date_evidence.date()),
                                evidence_line=int(row.evidence_line)))
    return dict(version=1, raw_file="2019-20.json", raw_sha256=sha256(raw_content),
                evidence_file="2019-20-evidence.txt", evidence_url=EVIDENCE_URL,
                evidence_sha256=sha256(evidence_content), corrections=corrections)


def apply_corrections(frame, ledger):
    corrected = frame.copy()
    for entry in ledger["corrections"]:
        mask = corrected.season.eq(entry["season"])
        for field in KEYS:
            mask &= corrected[field].eq(entry[field])
        mask &= corrected.date.eq(pd.Timestamp(entry["original_date"]))
        if mask.sum() != 1:
            raise ValueError("Date correction does not match exactly one source fixture")
        corrected.loc[mask, "date"] = pd.Timestamp(entry["corrected_date"])
    return corrected


def validate_seasons(frame):
    """Gate complete EPL seasons; date syntax alone is insufficient."""
    reports = []
    for season, rows in frame.groupby("season", sort=True):
        start = int(season[:4])
        lower, upper = pd.Timestamp(start, 8, 1), pd.Timestamp(start + 1, 7, 31)
        if not rows.date.between(lower, upper).all():
            raise ValueError(f"{season}: fixtures outside season interval")
        teams = set(rows.home_team) | set(rows.away_team)
        if len(rows) != 380 or len(teams) != 20:
            raise ValueError(f"{season}: expected 380 matches and 20 teams")
        if rows.duplicated(["home_team", "away_team"]).any():
            raise ValueError(f"{season}: duplicate fixture")
        if not rows.home_team.value_counts().reindex(sorted(teams), fill_value=0).eq(19).all():
            raise ValueError(f"{season}: expected 19 home fixtures per team")
        if not rows.away_team.value_counts().reindex(sorted(teams), fill_value=0).eq(19).all():
            raise ValueError(f"{season}: expected 19 away fixtures per team")
        appearances = pd.concat([rows[["date", "home_team"]].rename(columns={"home_team": "team"}),
                                 rows[["date", "away_team"]].rename(columns={"away_team": "team"})])
        if appearances.duplicated(["date", "team"]).any():
            raise ValueError(f"{season}: team appears twice on one date")
        reports.append(dict(season=season, matches=len(rows), teams=len(teams),
                            first_date=str(rows.date.min().date()), last_date=str(rows.date.max().date())))
    if not reports:
        raise ValueError("No audited seasons remain")
    return reports


def _audit_snapshot(directory, ledger=None, allow_quarantine=False):
    directory = Path(directory)
    matches = load_matches(directory)
    quarantine = []
    generated = None
    if matches.season.eq("2019-20").any():
        try:
            generated = correction_ledger(matches[matches.season.eq("2019-20")],
                                          (directory / "2019-20.json").read_bytes(),
                                          (directory / "2019-20-evidence.txt").read_bytes())
        except (ValueError, OSError) as error:
            if not allow_quarantine:
                raise
            quarantine.append(dict(season="2019-20", reason=str(error)))
            matches = matches[~matches.season.eq("2019-20")].copy()
        else:
            if ledger is not None and ledger != generated:
                raise ValueError("Correction ledger differs from verified fixture evidence")
            matches = apply_corrections(matches, generated)
    seasons = validate_seasons(matches)
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    report = dict(audit_version=1, status="passed" if not quarantine else "passed_with_quarantine",
                  source_manifest_sha256=sha256((directory / "manifest.json").read_bytes()),
                  matches=len(matches), seasons=seasons, quarantined=quarantine,
                  correction_count=len(generated["corrections"]) if generated else 0,
                  corrections_sha256=sha256(json_bytes(generated)) if generated else None,
                  data_through=str(matches.date.max().date()),
                  official_corroboration="https://www.premierleague.com/en/news/1697162")
    return matches, generated, report


def load_audited(raw_dir=None):
    """Verify saved provenance, correction evidence and all gates on every load."""
    directory = Path(raw_dir or ROOT / "data/audited")
    report = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    ledger_path = directory / "corrections.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None
    if report["corrections_sha256"] is not None:
        if ledger is None or sha256(ledger_path.read_bytes()) != report["corrections_sha256"]:
            raise ValueError("Correction ledger checksum mismatch")
    matches, generated, verified = _audit_snapshot(directory, ledger, allow_quarantine=bool(report["quarantined"]))
    if report != verified or (generated is not None and ledger is None):
        raise ValueError("Saved audit report does not match independently verified snapshot")
    return matches


def audit(raw_dir=None, revision=REVISION):
    directory = Path(raw_dir or ROOT / "data/audited")
    ingest(2010, 2025, revision, directory)
    evidence_path = directory / "2019-20-evidence.txt"
    if not evidence_path.exists():
        with urlopen(EVIDENCE_URL, timeout=60) as response:
            content = response.read()
        if sha256(content) != EVIDENCE_SHA256:
            raise ValueError("Downloaded correction evidence checksum mismatch")
        evidence_path.write_bytes(content)
    matches, ledger, report = _audit_snapshot(directory, allow_quarantine=True)
    if ledger is not None:
        (directory / "corrections.json").write_bytes(json_bytes(ledger))
    (directory / "audit.json").write_bytes(json_bytes(report))
    load_audited(directory)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw_dir, args.revision), indent=2))
