# WNBA Prop Predictor

Daily pipeline that fetches live WNBA player-prop odds, scores them with an
XGBoost model trained on `wnba_historical_props.csv`, and publishes a
dashboard of today's highest-edge props.

## How it works

1. `pipeline.py` loads `wnba_historical_props.csv` and trains a classifier on
   `Line`, `Odds`, and `BE Prob` (break-even probability implied by the odds)
   to predict win probability.
2. It fetches today's WNBA events and player-prop odds from
   [The Odds API](https://the-odds-api.com/), scores each prop, and computes
   `Calculated Edge = Model Win Probability - BE Prob`.
3. Results are written to `frontend/public/predictions.json` and
   `frontend/public/history_summary.json`.
4. `frontend/public/index.html` is a static dashboard that reads those two
   files — no build step required.
5. A GitHub Actions workflow (`.github/workflows/main.yml`) runs the pipeline
   every day at 9:00 AM ET, then deploys `frontend/public/` to GitHub Pages.

## One-time setup

### 1. Add your Odds API key as a repo secret

The pipeline needs an API key from [the-odds-api.com](https://the-odds-api.com/)
(there's a free tier).

In the repo: **Settings → Secrets and variables → Actions → New repository
secret**
- Name: `ODDS_API_KEY`
- Value: your key

### 2. Enable GitHub Pages

**Settings → Pages → Source → GitHub Actions**

That's it — no branch to pick, the workflow deploys directly.

### 3. Run it

- It runs automatically every day at 9:00 AM ET.
- Or trigger it manually: **Actions → WNBA Predictor Engine → Run workflow**.

After a run finishes, the dashboard URL shows up on the workflow run's
summary page (and under Settings → Pages).

### 4. Run locally (optional)

```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key_here
python pipeline.py
```

This writes `frontend/public/predictions.json` and
`frontend/public/history_summary.json`; open `frontend/public/index.html`
in a browser (or `python -m http.server` from that folder) to preview the
dashboard.

## Notes / known limitations

- Player props are only returned by the Odds API close to game time —
  running the pipeline on an off day, or hours before lines are posted, will
  produce an empty (but valid) `predictions.json`, and the dashboard falls
  back to showing historical performance only.
- The historical CSV's `DTM` column could not be reverse-engineered from
  `Odds`/`BE Prob` alone (its values don't match any consistent formula
  derived from those columns), so it's currently excluded from the model
  features. If you know its intended definition, it can be added back in.

## Roadmap (phase 2)

The model currently uses only market data (line, odds, implied probability).
Planned next step: layer in free/public data sources for:
- **Injury reports** (e.g. ESPN's public injury endpoints)
- **Recent box scores / player form** (e.g. balldontlie.io)

as additional model features, once the core odds → prediction → dashboard
loop above is confirmed working end-to-end.
