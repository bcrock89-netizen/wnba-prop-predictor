import json
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

import espn_client
import sportsdataverse_client as sdv_client

ET = ZoneInfo("America/New_York")

API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "basketball_wnba"
BASE_URL = "https://api.the-odds-api.com/v4"
REGIONS = "us"
ODDS_FORMAT = "american"
EVENT_WINDOW_HOURS = 36  # covers a full day's slate regardless of ET/UTC rollover
SEASON = datetime.now(timezone.utc).year

# Player prop market keys -> the "Bet Type" labels used in wnba_historical_props.csv
MARKET_TO_BET_TYPE = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_threes": "Threes",
    "player_points_rebounds": "Pts + Rebs",
    "player_points_assists": "Pts + Asts",
    "player_rebounds_assists": "Rebs + Asts",
    "player_points_rebounds_assists": "P+R+A",
}
MARKETS = ",".join(MARKET_TO_BET_TYPE)

HISTORY_FILE = "wnba_historical_props.csv"
PENDING_GRADES_FILE = "pending_grades.csv"
OUTPUT_DIR = os.path.join("frontend", "public")
PENDING_GRADES_COLUMNS = [
    "Date",
    "Player",
    "Matchup",
    "Commence Time",
    "Bet Type",
    "Side",
    "Line",
    "Odds",
    "BE Prob",
    "Win Probability",
    "Bookmaker",
]

# DTM's exact formula in the historical CSV can't be reconstructed from odds alone
# (it's not a consistent function of Odds/BE Prob in the sample data), so it's left
# out of the model until its true derivation is confirmed. Line/Odds/BE Prob are
# well-defined and computable identically at training time and inference time.
#
# Recent Form = average actual Stat Value over a player's last RECENT_FORM_WINDOW
# games in the same Bet Type. Computed identically for training (rolling over
# wnba_historical_props.csv, shifted so a game never sees its own result) and for
# live inference (from ESPN gamelogs), so - unlike DTM - it's an honest feature.
RECENT_FORM_WINDOW = 5

# Team Pace / Opp Def Rating / Back to Back come from sportsdataverse's WNBA box
# score data (see sportsdataverse_client.py). Same leak-free-training /
# season-to-date-inference split as Recent Form. Note: that data lags live by a
# few days, so these can be slightly stale close to "now" - documented in the README.
FEATURES = ["Line", "Odds", "BE Prob", "Recent Form", "Team Pace", "Opp Def Rating", "Back to Back"]


def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def fetch_todays_events():
    """The Odds API only serves player props per-event, so first list events
    starting within the lookahead window, then fetch odds for each one."""
    url = f"{BASE_URL}/sports/{SPORT}/events"
    resp = requests.get(url, params={"apiKey": API_KEY, "dateFormat": "iso"}, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=EVENT_WINDOW_HOURS)
    upcoming = []
    for event in events:
        commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if now <= commence <= window_end:
            upcoming.append(event)
    return upcoming


def fetch_event_props(event):
    url = f"{BASE_URL}/sports/{SPORT}/events/{event['id']}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": ODDS_FORMAT,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  ⚠️ Skipping {event.get('away_team')} @ {event.get('home_team')} ({resp.status_code})")
        return []

    data = resp.json()
    matchup = f"{event.get('away_team')} @ {event.get('home_team')}"
    rows = []
    for bookmaker in data.get("bookmakers", []):
        bookie_name = bookmaker["title"]
        for market in bookmaker.get("markets", []):
            bet_type = MARKET_TO_BET_TYPE.get(market["key"])
            if not bet_type:
                continue
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = outcome.get("name")
                line = outcome.get("point")
                odds = outcome.get("price")
                if player is None or line is None or odds is None:
                    continue

                rows.append(
                    {
                        "Player": player,
                        "Matchup": matchup,
                        "Commence Time": event.get("commence_time"),
                        "Bet Type": bet_type,
                        "Side": side,
                        "Line": float(line),
                        "Odds": int(odds),
                        "BE Prob": round(implied_probability(odds), 4),
                        "Bookmaker": bookie_name,
                    }
                )
    return rows


def fetch_todays_odds():
    if not API_KEY:
        print("❌ ODDS_API_KEY not set - skipping live odds fetch.")
        return pd.DataFrame()

    print("\U0001f4e1 Fetching today's WNBA slate...")
    try:
        events = fetch_todays_events()
    except requests.RequestException as exc:
        print(f"❌ Error fetching events: {exc}")
        return pd.DataFrame()

    if not events:
        print("ℹ️ No WNBA games in the next 36 hours.")
        return pd.DataFrame()

    print(f"\U0001f3c0 Found {len(events)} game(s). Fetching player prop odds...")
    all_rows = []
    for event in events:
        all_rows.extend(fetch_event_props(event))
        time.sleep(0.25)  # avoid hammering the API across many events

    if not all_rows:
        print("⚠️ No player prop markets returned for today's games yet.")
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def _resolve_matchup_team_ids(matchup, alias_map):
    if not matchup or " @ " not in matchup:
        return None, None
    away_name, home_name = matchup.split(" @ ", 1)
    return alias_map.get(espn_client.normalize_name(away_name)), alias_map.get(espn_client.normalize_name(home_name))


def enrich_with_live_signals(todays_slate):
    """Adds model features ('Recent Form', 'Team Pace', 'Opp Def Rating', 'Back
    to Back') and a display-only 'Injury Status' column, from ESPN and
    sportsdataverse. Best-effort throughout: any failure leaves the relevant
    columns empty rather than breaking the pipeline, since none of these
    sources are officially documented/versioned APIs."""
    todays_slate["Recent Form"] = np.nan
    todays_slate["Team Pace"] = np.nan
    todays_slate["Opp Def Rating"] = np.nan
    todays_slate["Back to Back"] = np.nan
    todays_slate["Injury Status"] = None

    player_id_map = espn_client.fetch_player_id_map()
    if not player_id_map:
        print("  ⚠️ Could not load ESPN player rosters; skipping recent form/injury/team-context enrichment.")
        return todays_slate

    injury_map = espn_client.fetch_injury_status_map()

    team_box = sdv_client.fetch_team_box(SEASON)
    team_snapshot = {}
    alias_map = {}
    if team_box is not None:
        team_box = sdv_client.add_team_game_features(team_box)
        team_snapshot = sdv_client.team_season_snapshot(team_box)
        alias_map = sdv_client.team_name_alias_map(team_box)
    else:
        print("  ⚠️ Could not load sportsdataverse team box scores; skipping team-context enrichment.")

    matchup_team_ids_cache = {}
    recent_games_cache = {}
    recent_form_values = []
    team_pace_values = []
    opp_def_values = []
    back_to_back_values = []
    injury_values = []

    for _, row in todays_slate.iterrows():
        norm_name = espn_client.normalize_name(row["Player"])
        injury_values.append(injury_map.get(norm_name))

        player_info = player_id_map.get(norm_name)
        athlete_id = player_info["athlete_id"] if player_info else None

        if not athlete_id:
            recent_form_values.append(None)
        else:
            if athlete_id not in recent_games_cache:
                recent_games_cache[athlete_id] = espn_client.fetch_recent_games(athlete_id, RECENT_FORM_WINDOW)
                time.sleep(0.15)
            recent_form_values.append(
                espn_client.recent_form_for_bet_type(recent_games_cache[athlete_id], row["Bet Type"])
            )

        team_pace = opp_def = back_to_back = None
        if player_info and team_snapshot:
            team_id = player_info["team_id"]
            team_stats = team_snapshot.get(team_id)
            if team_stats:
                team_pace = team_stats["pace"]
                commence = row.get("Commence Time")
                if commence and pd.notna(team_stats["last_game_date"]):
                    game_date = pd.to_datetime(commence).tz_localize(None).normalize()
                    gap_days = (game_date - team_stats["last_game_date"]).days
                    back_to_back = 1.0 if gap_days == 1 else 0.0

            matchup = row.get("Matchup")
            if matchup not in matchup_team_ids_cache:
                matchup_team_ids_cache[matchup] = _resolve_matchup_team_ids(matchup, alias_map)
            away_id, home_id = matchup_team_ids_cache[matchup]
            opponent_id = None
            if away_id == team_id:
                opponent_id = home_id
            elif home_id == team_id:
                opponent_id = away_id
            if opponent_id is not None:
                opp_stats = team_snapshot.get(opponent_id)
                if opp_stats:
                    opp_def = opp_stats["pts_allowed"]

        team_pace_values.append(team_pace)
        opp_def_values.append(opp_def)
        back_to_back_values.append(back_to_back)

    todays_slate["Recent Form"] = pd.to_numeric(pd.Series(recent_form_values), errors="coerce")
    todays_slate["Team Pace"] = pd.to_numeric(pd.Series(team_pace_values), errors="coerce")
    todays_slate["Opp Def Rating"] = pd.to_numeric(pd.Series(opp_def_values), errors="coerce")
    todays_slate["Back to Back"] = pd.to_numeric(pd.Series(back_to_back_values), errors="coerce")
    todays_slate["Injury Status"] = injury_values

    matched = todays_slate["Recent Form"].notna().sum()
    print(f"  \U0001fa7a Recent form resolved for {matched}/{len(todays_slate)} prop rows.")
    team_matched = todays_slate["Team Pace"].notna().sum()
    opp_matched = todays_slate["Opp Def Rating"].notna().sum()
    print(f"  \U0001f3c0 Team pace resolved for {team_matched}/{len(todays_slate)}, opp defense for {opp_matched}/{len(todays_slate)} prop rows.")
    return todays_slate


def load_history():
    try:
        history = pd.read_csv(HISTORY_FILE)
    except FileNotFoundError:
        print(f"❌ Could not find '{HISTORY_FILE}' in the root directory.")
        return pd.DataFrame()

    for col in ("BE Prob", "DTM", "Win Probability"):
        if col in history.columns:
            history[col] = history[col].astype(str).str.rstrip("%").astype(float) / 100

    history["Result"] = history["Result"].astype(str).str.upper()

    history = history.sort_values("Date").reset_index(drop=True)
    history["Recent Form"] = history.groupby(["Player", "Bet Type"])["Stat Value"].transform(
        lambda s: s.shift(1).rolling(RECENT_FORM_WINDOW, min_periods=1).mean()
    )
    return history


def enrich_history_with_team_context(history):
    """Adds 'Team Pace', 'Opp Def Rating', 'Back to Back' to historical rows by
    joining sportsdataverse's player_box (for player -> team/opponent on that
    date) and team_box (for that team's leak-free pregame pace/defense/rest).
    Best-effort: rows that can't be joined (e.g. name mismatch, or a date not
    yet covered by sportsdataverse's - typically few-day-lagged - release) get
    NaN, same as everything else in this pipeline."""
    history["Team Pace"] = np.nan
    history["Opp Def Rating"] = np.nan
    history["Back to Back"] = np.nan

    team_box = sdv_client.fetch_team_box(SEASON)
    player_box = sdv_client.fetch_player_box(SEASON)
    if team_box is None or player_box is None:
        print("  ⚠️ Could not load sportsdataverse data; skipping historical team-context features.")
        return history

    team_box = sdv_client.add_team_game_features(team_box)
    pregame = sdv_client.team_pregame_lookup(team_box)
    player_team_lookup = sdv_client.build_player_team_lookup(player_box)

    team_pace_values, opp_def_values, back_to_back_values = [], [], []
    for row in history.itertuples():
        team_opp = player_team_lookup.get((espn_client.normalize_name(row.Player), row.Date))
        if not team_opp:
            team_pace_values.append(None)
            opp_def_values.append(None)
            back_to_back_values.append(None)
            continue

        team_id, opponent_id = team_opp
        team_stats = pregame.get((team_id, row.Date))
        opp_stats = pregame.get((opponent_id, row.Date))
        team_pace_values.append(team_stats[0] if team_stats else None)
        opp_def_values.append(opp_stats[1] if opp_stats else None)
        back_to_back_values.append(team_stats[2] if team_stats else None)

    history["Team Pace"] = pd.to_numeric(pd.Series(team_pace_values), errors="coerce")
    history["Opp Def Rating"] = pd.to_numeric(pd.Series(opp_def_values), errors="coerce")
    history["Back to Back"] = pd.to_numeric(pd.Series(back_to_back_values), errors="coerce")

    matched = history["Team Pace"].notna().sum()
    print(f"  🏀 Team context resolved for {matched}/{len(history)} historical rows.")
    return history


def build_history_summary(history):
    decided = history[history["Result"].isin(["WIN", "LOSS"])]
    total = len(decided)
    wins = int((decided["Result"] == "WIN").sum())

    by_bet_type = (
        decided.groupby("Bet Type")
        .apply(
            lambda g: pd.Series(
                {
                    "bets": len(g),
                    "wins": int((g["Result"] == "WIN").sum()),
                    "win_rate": round(float((g["Result"] == "WIN").mean()), 4),
                    "profit": round(float(g["Profit"].sum()), 2),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .to_dict("records")
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_bets": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total, 4) if total else 0,
        "total_profit": round(float(history["Profit"].sum()), 2),
        "date_range": {"start": str(history["Date"].min()), "end": str(history["Date"].max())},
        "by_bet_type": by_bet_type,
    }


def train_model(history):
    clean = history.dropna(subset=FEATURES + ["Result"])
    clean = clean[clean["Result"] != "PUSH"]

    X_train = clean[FEATURES]
    y_train = (clean["Result"] == "WIN").astype(int)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


def write_predictions(output_dir, predictions_df=None, message=None):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": 0 if predictions_df is None else len(predictions_df),
        "predictions": [] if predictions_df is None else json.loads(predictions_df.to_json(orient="records")),
    }
    if message:
        payload["message"] = message

    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(payload, f, indent=2)


def game_date_et(commence_time_iso):
    """The historical CSV's Date column is a plain calendar date (no
    timezone), which for US sports data conventionally means the US/Eastern
    local date of the game - so today's picks are stamped the same way,
    rather than a UTC date that could roll over to the wrong day."""
    dt = pd.to_datetime(commence_time_iso)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.tz_convert(ET).strftime("%Y-%m-%d")


def select_best_picks(todays_slate):
    """For each (Player, Bet Type), picks the Side with positive Calculated
    Edge (if any), then the best (most bettor-favorable) American odds for
    that side across bookmakers - one row per prop. This matches how a real
    bettor would actually act (shop for the best price on the side they've
    decided on) and matches wnba_historical_props.csv's existing structure
    of one row per prop per day, rather than exploding to one row per
    bookmaker. Returns a DataFrame with PENDING_GRADES_COLUMNS."""
    picks = []
    for (player, bet_type), group in todays_slate.groupby(["Player", "Bet Type"]):
        positive = group[group["Calculated Edge"] > 0]
        if positive.empty:
            continue
        best_side = positive.groupby("Side")["Calculated Edge"].mean().idxmax()
        side_rows = positive[positive["Side"] == best_side]
        best_row = side_rows.loc[side_rows["Odds"].idxmax()]
        picks.append(best_row)

    if not picks:
        return pd.DataFrame(columns=PENDING_GRADES_COLUMNS)

    result = pd.DataFrame(picks).reset_index(drop=True)
    result["Date"] = result["Commence Time"].apply(game_date_et)
    return result[PENDING_GRADES_COLUMNS]


def append_pending_grades(picks):
    """Appends today's best-edge picks to PENDING_GRADES_FILE, so a later run
    can grade them once results are known. Dedupes on (Date, Player, Bet
    Type) so re-running the pipeline the same day doesn't create duplicates."""
    if picks.empty:
        return 0

    if os.path.exists(PENDING_GRADES_FILE):
        existing = pd.read_csv(PENDING_GRADES_FILE)
    else:
        existing = pd.DataFrame(columns=PENDING_GRADES_COLUMNS)

    combined = pd.concat([existing, picks], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Player", "Bet Type"], keep="last")
    combined.to_csv(PENDING_GRADES_FILE, index=False)
    return len(combined) - len(existing)


GRADING_STAKE = 5  # matches the flat-stake convention already used throughout wnba_historical_props.csv's Profit column


def _grade_payout(odds, result):
    if result == "PUSH":
        return 0.0
    if result == "LOSS":
        return -float(GRADING_STAKE)
    return GRADING_STAKE * (odds / 100) if odds > 0 else GRADING_STAKE * (100 / abs(odds))


def grade_pending_picks():
    """Resolves picks in PENDING_GRADES_FILE against players' actual final
    stat lines (ESPN gamelogs) and appends graded rows to HISTORY_FILE in its
    existing schema (Result/Stat Value/Profit computed the same way the rest
    of that file already is). A pick whose game hasn't concluded yet, or
    whose box score ESPN doesn't have yet, is left in PENDING_GRADES_FILE and
    retried on a later run - never dropped, never guessed at.

    DTM and Projection can't be reconstructed for these rows (same reason
    DTM is excluded from the model - see README), so they're left blank,
    same as any other missing value in this pipeline. Neither is a model
    feature, so this doesn't affect training.

    Returns (graded_count, still_pending_count)."""
    if not os.path.exists(PENDING_GRADES_FILE):
        return 0, 0

    pending = pd.read_csv(PENDING_GRADES_FILE)
    if pending.empty:
        os.remove(PENDING_GRADES_FILE)
        return 0, 0

    player_id_map = espn_client.fetch_player_id_map()
    if not player_id_map:
        print("  ⚠️ Could not load ESPN player rosters; skipping grading this run.")
        return 0, len(pending)

    graded_rows = []
    still_pending = []
    stat_line_cache = {}

    for _, row in pending.iterrows():
        athlete = player_id_map.get(espn_client.normalize_name(row["Player"]))
        if not athlete:
            still_pending.append(row)
            continue

        cache_key = (athlete["athlete_id"], row["Date"])
        if cache_key not in stat_line_cache:
            stat_line_cache[cache_key] = espn_client.fetch_stat_line_for_date(athlete["athlete_id"], row["Date"])
        game_stats = stat_line_cache[cache_key]
        if game_stats is None:
            still_pending.append(row)  # not yet played / not yet in the gamelog - retry next run
            continue

        stat_keys = espn_client.STAT_KEYS_FOR_BET_TYPE.get(row["Bet Type"])
        values = [game_stats.get(k) for k in stat_keys] if stat_keys else []
        if not stat_keys or any(v is None for v in values):
            still_pending.append(row)  # played, but this stat's missing from the box score - retry next run
            continue

        stat_value = sum(values)
        line = row["Line"]
        if stat_value == line:
            result = "PUSH"
        elif (row["Side"] == "Over") == (stat_value > line):
            result = "WIN"
        else:
            result = "LOSS"

        date = pd.to_datetime(row["Date"])
        graded_rows.append({
            "Date": row["Date"],
            "Player": row["Player"],
            "Bet Type": row["Bet Type"],
            "Side": row["Side"],
            "Line": line,
            "Odds": row["Odds"],
            "Win Probability": f"{row['Win Probability'] * 100:.1f}%",
            "DTM": np.nan,
            "Result": result,
            "Projection": np.nan,
            "Stat Value": stat_value,
            "Profit": round(_grade_payout(row["Odds"], result), 2),
            "BE Prob": f"{row['BE Prob'] * 100:.1f}%",
            "Day": date.strftime("%A"),
            "Month": date.strftime("%Y-%m"),
        })

    if graded_rows:
        history = pd.read_csv(HISTORY_FILE)
        updated = pd.concat([history, pd.DataFrame(graded_rows)], ignore_index=True)
        updated.to_csv(HISTORY_FILE, index=False)

    if still_pending:
        pd.DataFrame(still_pending).to_csv(PENDING_GRADES_FILE, index=False)
    else:
        os.remove(PENDING_GRADES_FILE)

    return len(graded_rows), len(still_pending)


def run_prediction_engine():
    print("\U0001f4c2 Loading historical tracked props...")
    history = load_history()
    if history.empty:
        return

    print("\U0001f3c0 Adding historical team-context features (pace/defense/rest)...")
    try:
        history = enrich_history_with_team_context(history)
    except Exception as exc:  # sportsdataverse's data isn't a stable/versioned API either
        print(f"  ⚠️ Historical team-context enrichment failed, continuing without it: {exc}")
        history["Team Pace"] = np.nan
        history["Opp Def Rating"] = np.nan
        history["Back to Back"] = np.nan

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = build_history_summary(history)
    with open(os.path.join(OUTPUT_DIR, "history_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ History summary: {summary['total_bets']} decided bets, {summary['win_rate']:.1%} win rate.")

    todays_slate = fetch_todays_odds()
    if todays_slate.empty:
        write_predictions(OUTPUT_DIR, message="No live odds available for today's slate yet.")
        print("✅ Wrote empty predictions.json.")
        return

    print("\U0001fa7a Fetching ESPN injury reports and recent form...")
    try:
        todays_slate = enrich_with_live_signals(todays_slate)
    except Exception as exc:  # ESPN's endpoints are undocumented; never let this break the run
        print(f"  ⚠️ Live signal enrichment failed, continuing without it: {exc}")
        todays_slate["Recent Form"] = np.nan
        todays_slate["Team Pace"] = np.nan
        todays_slate["Opp Def Rating"] = np.nan
        todays_slate["Back to Back"] = np.nan
        todays_slate["Injury Status"] = None

    print(f"\U0001f3af Training model on {len(history)} historical rows...")
    model = train_model(history)

    X_today = todays_slate[FEATURES]
    todays_slate["Win Probability"] = model.predict_proba(X_today)[:, 1]
    todays_slate["Calculated Edge"] = todays_slate["Win Probability"] - todays_slate["BE Prob"]
    todays_slate = todays_slate.sort_values(by="Calculated Edge", ascending=False)

    write_predictions(OUTPUT_DIR, predictions_df=todays_slate)
    print(f"✅ Wrote {len(todays_slate)} predictions to {OUTPUT_DIR}/predictions.json")

    picks = select_best_picks(todays_slate)
    added = append_pending_grades(picks)
    print(f"\U0001f4dd Snapshotted {len(picks)} best-edge picks for later grading ({added} new).")


if __name__ == "__main__":
    run_prediction_engine()
