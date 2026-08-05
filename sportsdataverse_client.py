"""Team-context features (pace, opponent defense, back-to-back fatigue) sourced
from sportsdataverse's pre-built WNBA box score data.

Unlike the ESPN/Odds API hosts, these are plain GitHub release downloads, so
failures here are treated the same defensive way: any problem returns None/NaN
rather than raising, since a missing feature should never break the pipeline.
"""

from io import StringIO

import pandas as pd
import requests

from espn_client import normalize_name

RELEASES_BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
REQUEST_TIMEOUT = 60


def _fetch_csv(url):
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        return pd.read_csv(StringIO(resp.text))
    except (requests.RequestException, ValueError, pd.errors.ParserError):
        return None


def fetch_team_box(season):
    return _fetch_csv(f"{RELEASES_BASE}/espn_wnba_team_boxscores/team_box_{season}.csv")


def fetch_player_box(season):
    return _fetch_csv(f"{RELEASES_BASE}/espn_wnba_player_boxscores/player_box_{season}.csv")


def add_team_game_features(team_box):
    """Adds possession/pace/points-allowed/back-to-back columns to a team_box
    frame. Two variants of pace/defense are computed:
      - *_pregame: leak-free, excludes the game in that row (for training)
      - season-to-date snapshots (see team_season_snapshot) use ALL games,
        since there's no "current game" to exclude when predicting a future one
    """
    tb = team_box.copy()
    tb["game_date"] = pd.to_datetime(tb["game_date"])
    tb = tb.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    tb["possessions"] = (
        tb["field_goals_attempted"]
        - tb["offensive_rebounds"]
        + tb["turnovers"]
        + 0.44 * tb["free_throws_attempted"]
    )

    grouped = tb.groupby("team_id")
    tb["team_pace_pregame"] = grouped["possessions"].transform(lambda s: s.shift(1).expanding().mean())
    tb["pts_allowed_pregame"] = grouped["opponent_team_score"].transform(lambda s: s.shift(1).expanding().mean())
    tb["days_since_last"] = grouped["game_date"].diff().dt.days
    tb["back_to_back"] = (tb["days_since_last"] == 1).astype(float)

    return tb


def team_season_snapshot(team_box_with_features):
    """{team_id: {pace, pts_allowed, last_game_date}} using ALL games to date -
    the team's current season-to-date form, for predicting an upcoming game."""
    snap = team_box_with_features.groupby("team_id").agg(
        pace=("possessions", "mean"),
        pts_allowed=("opponent_team_score", "mean"),
        last_game_date=("game_date", "max"),
    )
    return snap.to_dict("index")


def team_pregame_lookup(team_box_with_features):
    """{(team_id, 'YYYY-MM-DD'): (team_pace_pregame, pts_allowed_pregame, back_to_back)}"""
    lookup = {}
    for row in team_box_with_features.itertuples():
        date_str = row.game_date.strftime("%Y-%m-%d")
        lookup[(row.team_id, date_str)] = (row.team_pace_pregame, row.pts_allowed_pregame, row.back_to_back)
    return lookup


def build_player_team_lookup(player_box):
    """{(normalized_player_name, 'YYYY-MM-DD'): (team_id, opponent_team_id)}"""
    lookup = {}
    for row in player_box.itertuples():
        key = (normalize_name(row.athlete_display_name), row.game_date)
        lookup[key] = (row.team_id, row.opponent_team_id)
    return lookup


def team_name_alias_map(team_box):
    """normalized team-name alias -> team_id, covering every name-ish column,
    for resolving a live event's home/away team names to a team_id."""
    alias_cols = ["team_display_name", "team_name", "team_location", "team_short_display_name", "team_abbreviation"]
    aliases = {}
    for _, row in team_box.drop_duplicates("team_id").iterrows():
        for col in alias_cols:
            alias = normalize_name(row[col])
            if alias:
                aliases[alias] = row["team_id"]
    return aliases
