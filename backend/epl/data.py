"""Acquire public-domain results; retain exact bytes and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEAM_ALIASES = {name: name + " FC" for name in (
    "Aston Villa", "Brighton & Hove Albion", "Crystal Palace", "Leicester City",
    "Manchester City", "Manchester United", "Newcastle United", "Norwich City",
    "Sheffield United", "Tottenham Hotspur", "West Bromwich Albion",
    "West Ham United", "Wolverhampton Wanderers")}


def parse_season(payload, season):
    if "Premier League" not in payload.get("name", ""):
        raise ValueError("Expected English Premier League dataset")
    rows = []
    for match in payload["matches"]:
        scores = match.get("score")
        if scores is None:
            score = None
        elif isinstance(scores, dict):
            score = scores.get("ft")
        elif isinstance(scores, list):
            score = scores
        else:
            raise ValueError("Invalid full-time score format")
        if score is None:
            continue  # Scheduled or unfinished fixtures are not training labels.
        if not isinstance(score, list) or len(score) != 2 or any(type(n) is not int or n < 0 for n in score):
            raise ValueError("Invalid full-time score")
        home, away = match["team1"], match["team2"]
        home, away = TEAM_ALIASES.get(home, home), TEAM_ALIASES.get(away, away)
        if not home or not away or home == away:
            raise ValueError("Invalid teams")
        rows.append(dict(date=match["date"], home_team=home, away_team=away,
                         home_goals=score[0], away_goals=score[1], season=season,
                         result="H" if score[0] > score[1] else "A" if score[0] < score[1] else "D"))
    if not rows:
        raise ValueError(f"No completed matches: {season}")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    return frame


def load_matches(raw_dir=None):
    raw_dir = Path(raw_dir or ROOT / "data/raw")
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    frames = []
    for entry in manifest["files"]:
        content = (raw_dir / entry["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['file']}")
        frames.append(parse_season(json.loads(content), entry["season"]))
    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "home_team", "away_team"]).reset_index(drop=True)
    if frame.duplicated(["season", "home_team", "away_team"]).any():
        raise ValueError("Duplicate fixtures")
    appearances = pd.concat([frame[["date", "home_team"]].rename(columns={"home_team": "team"}),
                             frame[["date", "away_team"]].rename(columns={"away_team": "team"})])
    if appearances.duplicated(["date", "team"]).any():
        raise ValueError("Team appears twice on one date")
    return frame


def ingest(start=2014, end=2024, revision=None, raw_dir=None):
    target = Path(raw_dir or ROOT / "data/raw")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["start"], manifest["end"]) != (start, end):
            raise ValueError("Use a separate raw directory for another season range")
        if revision and revision != manifest["revision"]:
            raise ValueError("Revision differs from cached manifest")
        for entry in manifest["files"]:
            path = target / entry["file"]
            if not path.exists():
                with urlopen(entry["url"], timeout=60) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Checksum mismatch: {entry['file']}")
                path.write_bytes(content)
        load_matches(target)
        return manifest
    if not 2010 <= start <= end <= 2025:
        raise ValueError("Use completed seasons, start years 2010 through 2025")
    if revision is None:
        with urlopen("https://api.github.com/repos/openfootball/football.json/commits/master", timeout=60) as response:
            revision = json.load(response)["sha"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Revision must be a 40-character Git commit SHA")
    entries = []
    for year in range(start, end + 1):
        season = f"{year}-{str(year + 1)[2:]}"
        url = f"https://raw.githubusercontent.com/openfootball/football.json/{revision}/{season}/en.1.json"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        frame = parse_season(json.loads(content), season)
        name = f"{season}.json"
        (target / name).write_bytes(content)
        entries.append(dict(file=name, season=season, url=url,
                            sha256=hashlib.sha256(content).hexdigest(), matches=len(frame)))
    manifest = dict(source="openfootball/football.json", license="CC0-1.0",
                    revision=revision, start=start, end=end, files=entries)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    load_matches(target)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2014)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--revision")
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(ingest(args.start, args.end, args.revision, args.raw_dir), indent=2))
