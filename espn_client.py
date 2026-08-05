"""Best-effort client for ESPN's undocumented public WNBA endpoints.

These endpoints aren't officially documented or versioned, so every call here
is defensive: a missing field, unexpected shape, or failed request degrades
to an empty/None result instead of raising, so a change on ESPN's side never
takes down the odds/prediction pipeline.
"""

import json
import re
import time
import unicodedata

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"

# TEMPORARY: dumps raw shapes of ESPN responses to stdout for one diagnostic
# run, so real (unguessed) JSON shapes can be read from GitHub Actions logs.
# Remove once espn_client.py is confirmed working against live data.
_DEBUG = True


def _debug(label, obj):
    if _DEBUG:
        print(f"  [espn-debug] {label}: {json.dumps(obj, default=str)[:1500]}")

# Bet Type (as used in wnba_historical_props.csv) -> ESPN gamelog stat
# abbreviation(s) to sum for that bet type.
STAT_KEYS_FOR_BET_TYPE = {
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
    "Threes": ["3PT"],
    "Pts + Rebs": ["PTS", "REB"],
    "Pts + Asts": ["PTS", "AST"],
    "Rebs + Asts": ["REB", "AST"],
    "P+R+A": ["PTS", "REB", "AST"],
}


def normalize_name(name):
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-zA-Z0-9 ]", "", name).lower().strip()
    return re.sub(r"\s+", " ", name)


_get_failure_debug_count = 0
_MAX_GET_FAILURE_DEBUG = 3


def _get(url, params=None, timeout=20):
    global _get_failure_debug_count

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            if _DEBUG and _get_failure_debug_count < _MAX_GET_FAILURE_DEBUG:
                _get_failure_debug_count += 1
                print(f"  [espn-debug] GET {url} -> HTTP {resp.status_code}: {resp.text[:300]}")
            return None
        return resp.json()
    except requests.RequestException as exc:
        if _DEBUG and _get_failure_debug_count < _MAX_GET_FAILURE_DEBUG:
            _get_failure_debug_count += 1
            print(f"  [espn-debug] GET {url} raised {type(exc).__name__}: {exc}")
        return None
    except ValueError as exc:
        if _DEBUG and _get_failure_debug_count < _MAX_GET_FAILURE_DEBUG:
            _get_failure_debug_count += 1
            print(f"  [espn-debug] GET {url} returned non-JSON: {exc}")
        return None


def fetch_team_ids():
    data = _get(f"{BASE}/teams")
    if not data:
        _debug("teams fetch failed", None)
        return []
    _debug("teams top-level keys", list(data.keys()))
    team_ids = []
    try:
        for league in data.get("sports", [{}])[0].get("leagues", []):
            for entry in league.get("teams", []):
                team_id = entry.get("team", {}).get("id")
                if team_id:
                    team_ids.append(team_id)
    except (AttributeError, IndexError, TypeError):
        return []
    _debug("team_ids found", team_ids)
    return team_ids


def _flatten_roster_athletes(roster_json):
    athletes = roster_json.get("athletes", [])
    flat = []
    for entry in athletes:
        if isinstance(entry, dict) and "items" in entry:
            flat.extend(entry["items"])
        elif isinstance(entry, dict):
            flat.append(entry)
    return flat


def fetch_player_id_map():
    """Returns {normalized_player_name: espn_athlete_id} across all WNBA rosters."""
    id_map = {}
    team_ids = fetch_team_ids()
    for i, team_id in enumerate(team_ids):
        roster = _get(f"{BASE}/teams/{team_id}/roster")
        if not roster:
            _debug(f"roster fetch failed for team {team_id}", None)
            continue
        if i == 0:
            _debug("first roster top-level keys", list(roster.keys()))
            raw_athletes = roster.get("athletes", [])
            _debug("first roster 'athletes' entry (raw, unflattened)", raw_athletes[0] if raw_athletes else None)
        flat = _flatten_roster_athletes(roster)
        if i == 0:
            _debug("first roster flattened athlete sample", flat[0] if flat else None)
        for athlete in flat:
            name = athlete.get("displayName") or athlete.get("fullName")
            athlete_id = athlete.get("id")
            if name and athlete_id:
                id_map[normalize_name(name)] = athlete_id
        time.sleep(0.2)
    _debug("player_id_map size", len(id_map))
    _debug("player_id_map sample keys", list(id_map.keys())[:10])
    return id_map


def fetch_injury_status_map():
    """Returns {normalized_player_name: status_string} e.g. 'Out', 'Day-To-Day'."""
    data = _get(f"{BASE}/injuries")
    if not data:
        _debug("injuries fetch failed", None)
        return {}
    _debug("injuries top-level keys", list(data.keys()))
    raw_injuries = data.get("injuries", [])
    _debug("injuries first team entry (raw)", raw_injuries[0] if raw_injuries else None)
    status_map = {}
    for team_entry in raw_injuries:
        for item in team_entry.get("injuries", []):
            athlete = item.get("athlete", {})
            name = athlete.get("displayName")
            status = item.get("status") or item.get("type", {}).get("description")
            if name and status:
                status_map[normalize_name(name)] = status
    _debug("injury_status_map", status_map)
    return status_map


def _parse_stat(raw):
    """ESPN gamelog cells are sometimes 'made-attempted' strings like '3-8'."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    raw = str(raw)
    if "-" in raw[1:]:  # keep a leading '-' (shouldn't happen for these stats) intact
        raw = raw.split("-")[0]
    try:
        return float(raw)
    except ValueError:
        return None


_gamelog_debug_printed = False


def fetch_recent_games(athlete_id, n_games=5):
    """Returns a list of {stat_name: value} dicts for an athlete's most recent games."""
    global _gamelog_debug_printed

    data = _get(f"{BASE}/athletes/{athlete_id}/gamelog")
    if not data:
        if not _gamelog_debug_printed:
            _debug(f"gamelog fetch failed for athlete {athlete_id}", None)
        return []

    if not _gamelog_debug_printed:
        _debug(f"gamelog top-level keys (athlete {athlete_id})", list(data.keys()))
        _debug("gamelog 'names'", data.get("names"))
        _debug("gamelog 'seasonTypes' (raw)", data.get("seasonTypes"))

    names = [str(n).upper() for n in data.get("names", [])]
    if not names:
        if not _gamelog_debug_printed:
            _debug("gamelog had no 'names' field", None)
            _gamelog_debug_printed = True
        return []

    raw_rows = []
    for season_type in data.get("seasonTypes", []):
        for category in season_type.get("categories", []):
            raw_rows.extend(category.get("events", []))

    if not _gamelog_debug_printed:
        _debug("gamelog raw_rows count", len(raw_rows))
        _debug("gamelog first raw row", raw_rows[0] if raw_rows else None)
        _gamelog_debug_printed = True

    games = []
    for row in raw_rows[:n_games]:
        stats = row.get("stats", [])
        game = {}
        for name, value in zip(names, stats):
            parsed = _parse_stat(value)
            if parsed is not None:
                game[name] = parsed
        if game:
            games.append(game)
    return games


def recent_form_for_bet_type(recent_games, bet_type):
    """Average of summed stat(s) over recent_games for the given Bet Type, or None."""
    keys = STAT_KEYS_FOR_BET_TYPE.get(bet_type)
    if not keys or not recent_games:
        return None

    totals = []
    for game in recent_games:
        values = [game.get(key) for key in keys]
        if any(v is None for v in values):
            continue
        totals.append(sum(values))

    if not totals:
        return None
    return sum(totals) / len(totals)
