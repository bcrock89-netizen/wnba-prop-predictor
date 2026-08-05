import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. Grab secure API key from GitHub Secrets environment
API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "basketball_wnba"
MARKETS = "player_points,player_rebounds,player_assists"

def fetch_todays_odds():
    print("📡 Fetching morning lines from The Odds API...")
    url = f"https://the-odds-api.com{SPORT}/odds-blends/"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": MARKETS,
        "oddsFormat": "american"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        return pd.DataFrame()
        
    events = response.json()
    parsed_props = []
    
    for event in events:
        for bookmaker in event.get('bookmakers', []):
            bookie_name = bookmaker['title']
            for market in bookmaker.get('markets', []):
                market_type = market['key'].replace('player_', '').capitalize() # Points, Rebounds, etc.
                for outcome in market.get('outcomes', []):
                    player = outcome.get('description')
                    side = outcome.get('name')
                    line = outcome.get('point')
                    odds = outcome.get('price')
                    
                    # Convert odds to break-even probability & DTM payouts
                    if odds > 0:
                        be_prob = 100 / (odds + 100)
                        dtm = odds / 100
                    else:
                        be_prob = abs(odds) / (abs(odds) + 100)
                        dtm = 100 / abs(odds)
                        
                    parsed_props.append({
                        "Player": player,
                        "Bet Type": market_type,
                        "Side": side,
                        "Line": float(line) if line else 0.0,
                        "Odds": int(odds) if odds else 0,
                        "BE Prob": round(be_prob, 4),
                        "DTM": round(dtm, 4),
                        "Bookmaker": bookie_name
                    })
    return pd.DataFrame(parsed_props)

def run_prediction_engine():
    # Load your uploaded 8,000-row file
    print("📂 Loading 8,000+ historical tracked records...")
    try:
        history = pd.read_csv("wnba_historical_props.csv")
    except FileNotFoundError:
        print("❌ Error: Could not find 'wnba_historical_props.csv' in the root directory.")
        return

    # Grab morning lines
    todays_slate = fetch_todays_odds()
    if todays_slate.empty:
        print("⚠️ No live lines found for today's slate yet. Exiting pipeline.")
        return

    print(f"🎯 Found {len(todays_slate)} live lines. Training predictive model...")
    
    # Define machine learning features matching your schema
    features = ['Line', 'Odds', 'BE Prob', 'DTM']
    
    # Filter historical rows where required columns have data
    clean_history = history.dropna(subset=features + ['Result'])
    
    X_train = clean_history[features]
    y_train = (clean_history['Result'] == 'Win').astype(int)
    
    # Train Gradient Boosting Classifier
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    # Make live probabilities on tonight's board
    X_today = todays_slate[features]
    todays_slate['Win Probability'] = model.predict_proba(X_today)[:, 1]
    
    # Calculate structural market edge vs sportsbooks
    todays_slate['Calculated_Edge'] = todays_slate['Win Probability'] - todays_slate['BE Prob']
    
    # Sort by the highest mathematically viable edge
    todays_slate = todays_slate.sort_values(by='Calculated_Edge', ascending=False)
    
    # Ensure frontend folder structure exists, then save output
    os.makedirs("frontend/public", exist_ok=True)
    todays_slate.to_json("frontend/public/predictions.json", orient="records", indent=2)
    print("✅ Today's mathematical projections exported to frontend/public/predictions.json")

if __name__ == "__main__":
    run_prediction_engine()
