import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

import espn_client

API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "basketball_wnba"
BASE_URL = "https://api.the-odds-api.com/v4"
REGIONS = "us"
ODDS_FORMAT = "american"
EVENT_WINDOW_HOURS = 36  # covers a full day's slate regardless of ET/UTC rollover

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
OUTPUT_DIR = os.path.join("frontend", "public")

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
FEATURES = ["Line", "Odds", "BE Prob", "Recent Form"]


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


def enrich_with_live_signals(todays_slate):
    """Adds 'Recent Form' (model feature) and 'Injury Status' (display-only) columns
    from ESPN. Best-effort: any failure leaves those columns empty rather than
    breaking the pipeline, since ESPN's endpoints here are undocumented."""
    todays_slate["Recent Form"] = np.nan
    todays_slate["Injury Status"] = None

    player_id_map = espn_client.fetch_player_id_map()
    if not player_id_map:
        print("  ⚠️ Could not load ESPN player rosters; skipping recent form/injury enrichment.")
        return todays_slate

    injury_map = espn_client.fetch_injury_status_map()

    recent_games_cache = {}
    recent_form_values = []
    injury_values = []
    for _, row in todays_slate.iterrows():
        norm_name = espn_client.normalize_name(row["Player"])
        injury_values.append(injury_map.get(norm_name))

        athlete_id = player_id_map.get(norm_name)
        if not athlete_id:
            recent_form_values.append(None)
            continue

        if athlete_id not in recent_games_cache:
            recent_games_cache[athlete_id] = espn_client.fetch_recent_games(athlete_id, RECENT_FORM_WINDOW)
            time.sleep(0.15)

        recent_form_values.append(
            espn_client.recent_form_for_bet_type(recent_games_cache[athlete_id], row["Bet Type"])
        )

    todays_slate["Recent Form"] = pd.to_numeric(pd.Series(recent_form_values), errors="coerce")
    todays_slate["Injury Status"] = injury_values

    matched = todays_slate["Recent Form"].notna().sum()
    print(f"  \U0001fa7a Recent form resolved for {matched}/{len(todays_slate)} prop rows.")
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


def run_prediction_engine():
    print("\U0001f4c2 Loading historical tracked props...")
    history = load_history()
    if history.empty:
        return

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
        todays_slate["Injury Status"] = None

    print(f"\U0001f3af Training model on {len(history)} historical rows...")
    model = train_model(history)

    X_today = todays_slate[FEATURES]
    todays_slate["Win Probability"] = model.predict_proba(X_today)[:, 1]
    todays_slate["Calculated Edge"] = todays_slate["Win Probability"] - todays_slate["BE Prob"]
    todays_slate = todays_slate.sort_values(by="Calculated Edge", ascending=False)

    write_predictions(OUTPUT_DIR, predictions_df=todays_slate)
    print(f"✅ Wrote {len(todays_slate)} predictions to {OUTPUT_DIR}/predictions.json")


if __name__ == "__main__":
    run_prediction_engine()
