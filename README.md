# WNBA Prop Predictor

Daily pipeline that fetches live WNBA player-prop odds, scores them with an
XGBoost model trained on `wnba_historical_props.csv`, and publishes a
dashboard of today's highest-edge props.

## How it works

1. `pipeline.py` loads `wnba_historical_props.csv` and trains a classifier to
   predict win probability on:
   - `Line`, `Odds`, `BE Prob` — break-even probability implied by the odds
   - `Recent Form` — a player's average actual result over their last 5 games
     in that stat category (from ESPN gamelogs, `espn_client.py`)
   - `Team Pace`, `Opp Def Rating`, `Back to Back` — the player's team's
     season-to-date pace, the opponent's points allowed, and whether it's the
     second night of a back-to-back (from sportsdataverse's WNBA box score
     data, `sportsdataverse_client.py`)
2. It fetches today's WNBA events and player-prop odds from
   [The Odds API](https://the-odds-api.com/), enriches them with the same
   live signals, scores each prop, and computes
   `Calculated Edge = Model Win Probability - BE Prob`.
3. Injury status is fetched too (ESPN) but is **display-only**, not a model
   feature — there's no historical injury data to backtest it against.
4. Results are written to `frontend/public/predictions.json` and
   `frontend/public/history_summary.json`.
5. `frontend/public/index.html` is a static dashboard that reads those two
   files — no build step required.
6. A GitHub Actions workflow (`.github/workflows/main.yml`) runs the pipeline
   every day at 9:00 AM ET, then deploys `frontend/public/` to GitHub Pages
   (deployment only happens on `main`; manual runs on other branches still
   exercise the pipeline itself).

## Data sources

| Source | Used for | Auth |
|---|---|---|
| [The Odds API](https://the-odds-api.com/) | Live player-prop odds | API key (`ODDS_API_KEY` secret) |
| ESPN's undocumented public API | Rosters, injury reports, player gamelogs | None (keyless) |
| [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) | Team box scores → pace/defense/rest | None (public GitHub release downloads) |

None of these are officially documented/versioned APIs, so every call in
`espn_client.py` and `sportsdataverse_client.py` is defensive: a bad
response degrades to `None`/`NaN` for that feature rather than crashing the
run. Every feature they provide is trained the *same way* it's computed live
(rolling/season-to-date averages, shifted to never leak a game's own
result) — the one exception is the historical CSV's `DTM` column, which was
deliberately left out because its formula can't be reconstructed from
`Odds`/`BE Prob` alone.

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
dashboard. Note: ESPN's endpoints and the-odds-api.com may be unreachable
from some sandboxed/restricted network environments; `github.com` release
downloads (sportsdataverse) generally are not.

## Notes / known limitations

- Player props are only returned by the Odds API close to game time —
  running the pipeline on an off day, or hours before lines are posted, will
  produce an empty (but valid) `predictions.json`, and the dashboard falls
  back to showing historical performance only.
- The historical CSV's `DTM` column could not be reverse-engineered from
  `Odds`/`BE Prob` alone, so it's excluded from the model. If you know its
  intended definition, it can be added back in.
- sportsdataverse's WNBA box score release lags live by roughly a few days
  (it's not updated every day). `Team Pace`/`Opp Def Rating` are season-to-date
  averages, so a few days' lag barely moves them — but `Back to Back` can be
  wrong right at that boundary (e.g. missing a game from the last day or two).
- ESPN's gamelog/roster/injury endpoints and sportsdataverse's data format
  are both unversioned; if either changes shape, the affected feature(s)
  silently fall back to missing rather than breaking the pipeline. Check the
  pipeline's own log output (`🩺`/`🏀` lines) for match-rate sanity checks
  after any change to these upstream sources.

## Model evaluation

`backtest.py` compares the current 7-feature model against the original
3-feature (Line/Odds/BE Prob) baseline on held-out historical data, using a
chronological split (train on earlier dates, test on later ones - both
models train/test on the identical rows) rather than a random split, since
that's how the model is actually used: trained once, then predicting games
it hasn't seen. Run it yourself with `python backtest.py`.

**Honest result as of this writing** (5374 train rows, 1513 held-out test
rows): both models sit at **AUC ≈ 0.49** on the test set — essentially
chance-level, meaning neither one shows a real, validated predictive edge
over the betting market on this data. The 7-feature model edges out the
baseline on AUC and accuracy and loses less money in the flat-stake betting
simulation (-7.8% ROI vs -9.7%), but every one of these deltas is small
relative to a ~1,500-row test set and should be read as noise, not proof the
new features help. Log loss actually favors the baseline slightly. This
isn't a reason to rip the features back out - none of them are theoretically
unsound, and a 3-month window of one season is a small dataset to judge them
on - but **don't treat this model's `Calculated Edge` as a validated betting
signal** until this backtest looks better with more data and/or more rigorous
validation (e.g. an expanding-window backtest across multiple seasons,
significance testing on the AUC delta).

## Roadmap

- Re-run `backtest.py` periodically as more of the season accumulates in
  `wnba_historical_props.csv`, and once (if) a full second season is
  available, extend it to an expanding-window backtest across seasons rather
  than a single train/test split.
- Further context features (e.g. rest days beyond just back-to-back,
  home/away splits) could layer onto the same sportsdataverse data already
  wired up.
