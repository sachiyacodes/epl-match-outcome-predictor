"""Reconcile audited fixtures with an independent, openly licensed source.

Raw Wyscout downloads are retained only in the ignored external-data cache.
This module reports conflicts; it never changes serving data or model artifacts.
"""
import argparse
import io
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from .audit import load_audited, sha256
from .data import ROOT

SOURCES = {
    "matches.zip": {
        "url": "https://ndownloader.figshare.com/files/14464622",
        "sha256": "c8f92bb7533e5c127e043cee764c991b5c25b4f5e70a65be931baae0b1765ce9",
        "bytes": 645097,
        "record": "https://figshare.com/articles/dataset/Matches/7770422",
    },
    "teams.json": {
        "url": "https://ndownloader.figshare.com/files/15073697",
        "sha256": "9f7a4a3b3d92c0be33f40613ad6e6eb4316c3b9771ec74c61a22c9b8ece23a4d",
        "bytes": 27404,
        "record": "https://figshare.com/articles/dataset/Teams/7765310",
    },
}
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
FIXTURE_KEYS = ["season", "home_team", "away_team"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def cached_source(cache_dir, name, offline=False):
    """Never replace an existing source with unverified or differently pinned bytes."""
    source = SOURCES[name]
    path = Path(cache_dir) / name
    if path.exists():
        content = path.read_bytes()
    elif offline:
        raise ValueError(f"Missing cached source {name}; run without --offline to acquire it")
    else:
        with urlopen(source["url"], timeout=60) as response:
            content = response.read()
    if len(content) != source["bytes"] or sha256(content) != source["sha256"]:
        raise ValueError(f"Independent-source checksum mismatch: {name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Separate temporary names allow the shot experiment to share this cache.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{name}.verify-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return content


def canonical_team(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Independent source has an invalid team name")
    # Wyscout's short club names match after this one explicit normalization.
    return name.strip().removesuffix(" FC")


def parse_wyscout(matches_content, teams_content):
    teams = json.loads(teams_content)
    if not isinstance(teams, list):
        raise ValueError("Expected Wyscout team list")
    names = {}
    for team in teams:
        identifier = team["wyId"]
        if identifier in names:
            raise ValueError("Duplicate Wyscout team identifier")
        names[identifier] = canonical_team(team["name"])
    with ZipFile(io.BytesIO(matches_content)) as archive:
        content = archive.read("matches_England.json")
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError("Expected Wyscout match list")
    rows = []
    for match in payload:
        if match["competitionId"] != 364 or match["seasonId"] != 181150 or match["status"] != "Played":
            raise ValueError("Expected played EPL 2017/18 fixtures")
        sides = {}
        for team in match["teamsData"].values():
            side = team["side"]
            if side not in {"home", "away"} or side in sides:
                raise ValueError("Ambiguous Wyscout home/away roles")
            if team["teamId"] not in names:
                raise ValueError("Wyscout fixture references an unknown team identifier")
            score = team["score"]
            if type(score) is not int or score < 0:
                raise ValueError("Invalid Wyscout full-time score")
            sides[side] = (names[team["teamId"]], score)
        if set(sides) != {"home", "away"} or sides["home"][0] == sides["away"][0]:
            raise ValueError("Expected two distinct Wyscout contestants")
        day = pd.to_datetime(match["dateutc"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="raise")
        if pd.isna(day):
            raise ValueError("Missing Wyscout fixture date")
        rows.append(dict(season="2017-18", date=day.normalize().tz_localize(None),
                         home_team=sides["home"][0], away_team=sides["away"][0],
                         home_goals=sides["home"][1], away_goals=sides["away"][1],
                         source_match_id=match["wyId"]))
    frame = pd.DataFrame(rows)
    if frame.empty or frame.source_match_id.duplicated().any() or frame.duplicated(FIXTURE_KEYS).any():
        raise ValueError("Empty or duplicate independent fixtures")
    return frame.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True), sha256(content)


def reconcile(local, independent):
    """Join by fixture identity, so date and score disagreements remain visible."""
    if independent.empty:
        raise ValueError("Independent reference is empty")
    reference = independent.copy()
    target = local[local.season.isin(reference.season.unique())].copy()
    for frame in (reference, target):
        for field in ("home_team", "away_team"):
            frame[field] = frame[field].map(canonical_team)
        frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.normalize()
        if frame.duplicated(FIXTURE_KEYS).any():
            raise ValueError("Ambiguous normalized fixture identity")
    paired = target.merge(reference, on=FIXTURE_KEYS, how="outer", suffixes=("_local", "_source"),
                          indicator=True, validate="one_to_one")
    conflicts = []
    agreements = 0
    for row in paired.sort_values(FIXTURE_KEYS).to_dict("records"):
        issues = []
        if row["_merge"] != "both":
            issues.append("missing_independent_fixture" if row["_merge"] == "left_only" else "missing_local_fixture")
        else:
            for field in ("date", "home_goals", "away_goals"):
                if row[f"{field}_local"] != row[f"{field}_source"]:
                    issues.append(f"{field}_mismatch")
        if not issues:
            agreements += 1
            continue
        conflict = {field: row[field] for field in FIXTURE_KEYS}
        conflict["issues"] = issues
        for side in ("local", "source"):
            if row["_merge"] == ("right_only" if side == "local" else "left_only"):
                conflict[side] = None
            else:
                conflict[side] = dict(date=str(row[f"date_{side}"].date()),
                                      home_goals=int(row[f"home_goals_{side}"]),
                                      away_goals=int(row[f"away_goals_{side}"]))
        conflicts.append(conflict)
    compared = int(paired["_merge"].eq("both").sum())
    return dict(local_seasons=sorted(target.season.unique().tolist()), local_matches=len(target),
                independent_matches=len(reference), paired_fixtures=compared, agreements=agreements,
                conflicts=conflicts, conflict_count=len(conflicts),
                status="agreement" if not conflicts and len(target) == len(reference) else "conflicts")


def verify(cache_dir=None, output=None, offline=False):
    cache_dir = Path(cache_dir or ROOT / "data/external/wyscout")
    output = Path(output or ROOT / "data/verification/report.json")
    matches_content = cached_source(cache_dir, "matches.zip", offline)
    teams_content = cached_source(cache_dir, "teams.json", offline)
    independent, member_hash = parse_wyscout(matches_content, teams_content)
    if len(independent) != 380 or len(set(independent.home_team) | set(independent.away_team)) != 20:
        raise ValueError("Expected the complete 380-match, 20-team independent EPL season")
    local = load_audited()
    result = reconcile(local, independent)
    manifest = dict(version=1, source="Wyscout / Pappalardo et al. (2019)", source_paper=PAPER,
                    license="CC-BY-4.0", license_url=LICENSE_URL, files=SOURCES,
                    archive_member="matches_England.json", archive_member_sha256=member_hash,
                    normalization="Remove terminal ' FC' from team names; UTC calendar dates; exact home/away fixture join",
                    local_audit_sha256=sha256((ROOT / "data/audited/audit.json").read_bytes()),
                    local_season_sha256=sha256((ROOT / "data/audited/2017-18.json").read_bytes()))
    checked = result["paired_fixtures"]
    report = dict(version=1, independent_provider="Wyscout", total_audited_matches=len(local),
                  independently_compared_matches=checked, independently_agreeing_matches=result["agreements"],
                  unchecked_matches=len(local) - checked,
                  coverage_fraction=checked / len(local), independently_compared_seasons=result["local_seasons"],
                  result=result, provenance=manifest,
                  additional_sources=[dict(source="StatsBomb / Hudl open data", season="2015-16", status="not_used",
                      reason="Separate agreement restricts commercial use and raw redistribution and requires branding for published analysis; not bundled into this permissively licensed release",
                      agreement="https://github.com/hudl/open-data/blob/master/LICENSE.pdf")],
                  limitations=["Agreement is not proof that either provider is error-free",
                               "Only the listed season has been independently reconciled; other seasons remain unchecked",
                               "This report does not establish better prediction accuracy or change serving data/models"])
    write_json(cache_dir / "verification-manifest.json", manifest)
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = verify(args.cache_dir, args.output, args.offline)
    print(json.dumps({field: report[field] for field in ("total_audited_matches", "independently_compared_matches",
                     "independently_agreeing_matches", "unchecked_matches", "independently_compared_seasons")}, indent=2))
    if report["result"]["status"] != "agreement":
        raise SystemExit("Independent-source conflicts detected; inspect the saved report before changing data")
