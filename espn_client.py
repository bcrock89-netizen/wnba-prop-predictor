"""Best-effort client for ESPN's undocumented public WNBA endpoints.

These endpoints aren't officially documented or versioned, so every call here
is defensive: a missing field, unexpected shape, or failed request degrades
to an empty/None result instead of raising, so a change on ESPN's side never
takes down the odds/prediction pipeline.
"""

import re
import time
import unicodedata

import requests

# Teams/rosters/injuries.
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
# Gamelogs live on a different host/API version - site.api.espn.com's v2
# athletes/{id}/gamelog returns a clean 404 for WNBA; this v3 "common" API
# is the real one (confirmed against live data).
GAMELOG_BASE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/wnba"

# Bet Type (as used in wnba_historical_props.csv) -> ESPN gamelog stat
# abbreviation(s) to sum for that bet type. These abbreviations match the
# gamelog response's 'labels' field (e.g. "PTS", "REB", "AST", "3PT").
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


def _get(url, params=None, timeout=20):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def fetch_team_ids():
    data = _get(f"{BASE}/teams")
    if not data:
        return []
    team_ids = []
    try:
        for league in data.get("sports", [{}])[0].get("leagues", []):
            for entry in league.get("teams", []):
                team_id = entry.get("team", {}).get("id")
                if team_id:
                    team_ids.append(team_id)
    except (AttributeError, IndexError, TypeError):
        return []
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
    """Returns {normalized_player_name: {"athlete_id": ..., "team_id": ...}}
    across all WNBA rosters. team_id uses ESPN's numbering, which matches
    sportsdataverse's team_id (both are sourced from the same ESPN data)."""
    id_map = {}
    for team_id in fetch_team_ids():
        roster = _get(f"{BASE}/teams/{team_id}/roster")
        if not roster:
            continue
        for athlete in _flatten_roster_athletes(roster):
            name = athlete.get("displayName") or athlete.get("fullName")
            athlete_id = athlete.get("id")
            if name and athlete_id:
                id_map[normalize_name(name)] = {"athlete_id": athlete_id, "team_id": int(team_id)}
        time.sleep(0.2)
    return id_map


def fetch_injury_status_map():
    """Returns {normalized_player_name: status_string} e.g. 'Out', 'Day-To-Day'."""
    data = _get(f"{BASE}/injuries")
    if not data:
        return {}
    status_map = {}
    for team_entry in data.get("injuries", []):
        for item in team_entry.get("injuries", []):
            athlete = item.get("athlete", {})
            name = athlete.get("displayName")
            status = item.get("status") or item.get("type", {}).get("description")
            if name and status:
                status_map[normalize_name(name)] = status
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


def fetch_recent_games(athlete_id, n_games=5):
    """Returns a list of {stat_abbreviation: value} dicts for an athlete's most
    recent games, e.g. {"PTS": 20.0, "REB": 4.0, ...} - most recent game first."""
    data = _get(f"{GAMELOG_BASE}/athletes/{athlete_id}/gamelog")
    if not data:
        return []

    # 'labels' holds abbreviations ("PTS", "REB", ...) matching STAT_KEYS_FOR_BET_TYPE.
    # 'names' holds full words ("points", "totalRebounds", ...) - not what we key on.
    labels = [str(n).upper() for n in data.get("labels", [])]
    if not labels:
        return []

    # Events are grouped by seasonTypes -> categories (roughly by month), each
    # already newest-first; concatenating in order preserves recency overall.
    raw_rows = []
    for season_type in data.get("seasonTypes", []):
        for category in season_type.get("categories", []):
            raw_rows.extend(category.get("events", []))

    games = []
    for row in raw_rows[:n_games]:
        stats = row.get("stats", [])
        game = {}
        for label, value in zip(labels, stats):
            parsed = _parse_stat(value)
            if parsed is not None:
                game[label] = parsed
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
