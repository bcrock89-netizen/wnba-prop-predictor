"""Compares the current 7-feature model against the original 3-feature
(Line/Odds/BE Prob) baseline on held-out historical data.

Uses a chronological split (train on earlier dates, test on later ones)
rather than a random split, since that's how the model is actually used:
trained once on everything so far, then used to predict games it hasn't
seen. Both models train/test on the exact same rows (only rows where every
feature - baseline and new - is present) so the comparison isn't skewed by
one model getting more/different training data than the other.

Run manually: `python backtest.py`. Needs network access to sportsdataverse's
GitHub release downloads (same as the live pipeline's historical enrichment);
does not need ESPN or the Odds API.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

import pipeline

BASELINE_FEATURES = ["Line", "Odds", "BE Prob"]
FULL_FEATURES = pipeline.FEATURES
TEST_FRACTION = 0.2
STAKE = 100


def payout_if_win(odds):
    """American-odds profit on a STAKE-sized bet, if it wins."""
    return odds if odds > 0 else (100 * STAKE) / abs(odds)


def prepare_dataset():
    print("Loading and enriching historical data (same as the live pipeline)...")
    history = pipeline.load_history()
    history = pipeline.enrich_history_with_team_context(history)

    clean = history.dropna(subset=FULL_FEATURES + ["Result"]).copy()
    clean = clean[clean["Result"] != "PUSH"]
    clean = clean.sort_values("Date").reset_index(drop=True)
    print(f"Usable rows (all features present, decided bets only): {len(clean)}/{len(history)}")
    return clean


def chronological_split(df, test_fraction=TEST_FRACTION):
    split_idx = int(len(df) * (1 - test_fraction))
    split_date = df.iloc[split_idx]["Date"]
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]
    return train, test, split_date


def train_and_score(train, test, features, label):
    X_train, y_train = train[features], (train["Result"] == "WIN").astype(int)
    X_test, y_test = test[features], (test["Result"] == "WIN").astype(int)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "label": label,
        "n_features": len(features),
        "log_loss": log_loss(y_test, proba),
        "auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
    }

    scored = test.copy()
    scored["model_proba"] = proba
    scored["edge"] = scored["model_proba"] - scored["BE Prob"]
    bets = scored[scored["edge"] > 0]

    if len(bets):
        profits = np.where(bets["Result"] == "WIN", bets["Odds"].apply(payout_if_win), -STAKE)
        metrics["bets_placed"] = len(bets)
        metrics["bet_win_rate"] = float((bets["Result"] == "WIN").mean())
        metrics["total_profit"] = float(np.sum(profits))
        metrics["roi_pct"] = float(metrics["total_profit"] / (len(bets) * STAKE) * 100)
    else:
        metrics.update(bets_placed=0, bet_win_rate=float("nan"), total_profit=0.0, roi_pct=float("nan"))

    return metrics, scored


def calibration_by_bucket(scored, value_col, n_buckets=5):
    """Buckets held-out predictions by value_col (e.g. model-predicted edge)
    into quantile bins and reports the ACTUAL historical win rate in each
    bucket - i.e. what really happened for bets in that confidence range,
    rather than trusting the model's stated number at face value. This is
    the basis for ranking "top picks" honestly: by empirical hit rate in
    their bucket, not by the model's raw (unvalidated) output."""
    df = scored[scored["Result"] != "PUSH"].copy()
    df["bucket"] = pd.qcut(df[value_col], q=n_buckets, duplicates="drop")
    grouped = (
        df.groupby("bucket", observed=True)
        .agg(n=("Result", "size"), avg_value=(value_col, "mean"), hit_rate=("Result", lambda s: (s == "WIN").mean()))
        .reset_index()
        .sort_values("avg_value")
    )
    return grouped


def print_calibration_report(scored, value_col, label, n_buckets=5):
    print()
    print("-" * 72)
    print(f"CALIBRATION: actual hit rate by {label} bucket (held-out test set)")
    print("-" * 72)
    grouped = calibration_by_bucket(scored, value_col, n_buckets)
    print(f"{'bucket (' + value_col + ' range)':<28}{'n':<8}{'avg ' + value_col:<16}{'actual hit rate':<16}")
    for _, row in grouped.iterrows():
        bucket_str = f"{row['bucket'].left:.3f} to {row['bucket'].right:.3f}"
        print(f"{bucket_str:<28}{int(row['n']):<8}{row['avg_value']:<16.3f}{row['hit_rate']:<16.1%}")

    hit_rates = grouped["hit_rate"].to_numpy()
    is_monotonic = np.all(np.diff(hit_rates) >= -0.02)  # small tolerance for noise
    spread = hit_rates.max() - hit_rates.min()
    print()
    if is_monotonic and spread > 0.05:
        print(f"Hit rate rises roughly monotonically across buckets (spread {spread:.1%}) - some real signal here.")
    else:
        print(
            f"Hit rate does NOT rise cleanly across buckets (spread {spread:.1%}, non-monotonic) - "
            f"a higher {value_col} here has not corresponded to actually winning more often. "
            "Ranking 'top picks' by this bucket's empirical hit rate rather than the raw model "
            "number reflects that honestly."
        )
    return grouped


def evaluate_matchup_heuristic(train, test):
    """Tests a model-free 'top picks' heuristic: historical bet-type win rate
    (computed ONLY from train, so no leakage into test) combined with a
    side-adjusted opponent-defense matchup score (from Opp Def Rating, also
    leak-free/pregame). Checks whether this actually predicts hit rate on
    held-out test data - the same scrutiny already applied to the model's
    own Edge/probability."""
    decided_train = train[train["Result"].isin(["WIN", "LOSS"])]
    decided_test = test[test["Result"].isin(["WIN", "LOSS"])]
    bet_type_train_wr = decided_train.groupby("Bet Type")["Result"].apply(lambda s: (s == "WIN").mean())
    bet_type_test_wr = decided_test.groupby("Bet Type")["Result"].apply(lambda s: (s == "WIN").mean())

    print()
    print("-" * 72)
    print("BET-TYPE WIN RATE PERSISTENCE: train win rate vs held-out test win rate")
    print("-" * 72)
    comparison = pd.DataFrame({"train_win_rate": bet_type_train_wr, "test_win_rate": bet_type_test_wr}).dropna()
    comparison = comparison.sort_values("train_win_rate", ascending=False)
    print(f"{'Bet Type':<16}{'train win rate':<18}{'test win rate':<18}")
    for bet_type, row in comparison.iterrows():
        print(f"{bet_type:<16}{row['train_win_rate']:<18.1%}{row['test_win_rate']:<18.1%}")
    correlation = comparison["train_win_rate"].corr(comparison["test_win_rate"])
    print(f"\nCorrelation between train and test win rate across bet types: {correlation:.2f}")
    if correlation > 0.3:
        print("Bet-type win rate shows some persistence out of sample - plausibly a real, usable signal.")
    else:
        print(
            "Bet-type win rate does NOT persist out of sample - a bet type that looked strong in "
            "train is not reliably strong in test. Not a signal to rank picks on."
        )

    print()
    print("-" * 72)
    print("MATCHUP HEURISTIC: side-adjusted opponent-defense favorability vs hit rate")
    print("-" * 72)
    league_avg_pts_allowed = train["Opp Def Rating"].mean()
    test_scored = test.dropna(subset=["Opp Def Rating"]).copy()
    deviation = test_scored["Opp Def Rating"] - league_avg_pts_allowed
    # Over favored by a weak (high points-allowed) opponent defense; Under favored by a strong one.
    test_scored["matchup_score"] = np.where(test_scored["Side"] == "Over", deviation, -deviation)

    matchup_calib = calibration_by_bucket(test_scored, "matchup_score", n_buckets=5)
    print(f"{'bucket (matchup_score range)':<32}{'n':<8}{'avg score':<14}{'actual hit rate':<16}")
    for _, row in matchup_calib.iterrows():
        bucket_str = f"{row['bucket'].left:.2f} to {row['bucket'].right:.2f}"
        print(f"{bucket_str:<32}{int(row['n']):<8}{row['avg_value']:<14.2f}{row['hit_rate']:<16.1%}")

    hit_rates = matchup_calib["hit_rate"].to_numpy()
    spread = hit_rates.max() - hit_rates.min()
    is_rising = np.all(np.diff(hit_rates) >= -0.02)
    print()
    if is_rising and spread > 0.05:
        print(f"Hit rate rises with matchup favorability (spread {spread:.1%}) - this looks like real signal.")
    else:
        print(
            f"Hit rate does NOT rise cleanly with matchup favorability (spread {spread:.1%}) - "
            "no clean signal here either on this test set."
        )

    return comparison, matchup_calib


def evaluate_recent_vs_season_win_rate(split_date, window=10):
    """Tests a player's own recent hit rate (win rate over their last
    `window` bets in that Bet Type, leak-free/shifted) as a 'top picks'
    signal, alongside their season-long (all prior bets, leak-free/shifted)
    win rate for direct comparison - is recent form a stronger or weaker
    signal than the simple season-to-date rate?"""
    print()
    print("=" * 72)
    print(f"RECENT (last {window} bets) vs SEASON-LONG per-player win rate")
    print("=" * 72)

    history = pipeline.load_history()  # local CSV only - no network needed for this one
    history = history.sort_values("Date").reset_index(drop=True)
    history["is_win"] = (history["Result"] == "WIN").astype(float)
    history.loc[history["Result"] == "PUSH", "is_win"] = np.nan

    grouped_is_win = history.groupby(["Player", "Bet Type"])["is_win"]
    history["recent_win_rate"] = grouped_is_win.transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    history["season_win_rate"] = grouped_is_win.transform(lambda s: s.shift(1).expanding(min_periods=3).mean())

    test_rows = history[(history["Date"] >= split_date) & (history["Result"] != "PUSH")]

    for col, label in [("recent_win_rate", f"last {window} bets"), ("season_win_rate", "season-long (all prior bets)")]:
        subset = test_rows.dropna(subset=[col])
        print(f"\n--- {label} --- (n={len(subset)}/{len(test_rows)} held-out rows with enough prior history)")
        grouped = calibration_by_bucket(subset, col, n_buckets=5)
        print(f"{'bucket':<20}{'n':<8}{'avg rate':<14}{'actual hit rate':<16}")
        for _, row in grouped.iterrows():
            bucket_str = f"{row['bucket'].left:.1%} to {row['bucket'].right:.1%}"
            print(f"{bucket_str:<20}{int(row['n']):<8}{row['avg_value']:<14.1%}{row['hit_rate']:<16.1%}")

        hit_rates = grouped["hit_rate"].to_numpy()
        spread = hit_rates.max() - hit_rates.min()
        is_rising = np.all(np.diff(hit_rates) >= -0.02)
        verdict = "RISING - plausible signal" if (is_rising and spread > 0.05) else "flat/non-monotonic - no clean signal"
        print(f"Spread: {spread:.1%}. Verdict: {verdict}.")


def print_report(baseline, full, train, test, split_date):
    print()
    print("=" * 72)
    print("BACKTEST: 3-feature baseline vs current 7-feature model")
    print("=" * 72)
    print(f"Test: {len(test)} rows, {split_date} to {test['Date'].max()} - held out, unseen by either model")
    print(f"  (trained on {len(train)} rows, {train['Date'].min()} to {train['Date'].max()})")
    print()

    header = f"{'metric':<18}{'baseline (3 feat)':<20}{'full (7 feat)':<20}{'delta':<12}"
    print(header)
    print("-" * len(header))
    for key, fmt, better_low in [
        ("log_loss", "{:.4f}", True),
        ("auc", "{:.4f}", False),
        ("accuracy", "{:.4f}", False),
    ]:
        b, f = baseline[key], full[key]
        delta = f - b
        arrow = "better" if (delta < 0) == better_low and delta != 0 else ("worse" if delta != 0 else "same")
        print(f"{key:<18}{fmt.format(b):<20}{fmt.format(f):<20}{fmt.format(delta):<8} {arrow}")

    print()
    print(f"{'Betting simulation':<18}{'baseline (3 feat)':<20}{'full (7 feat)':<20}")
    print("-" * len(header))
    print(f"{'bets placed':<18}{baseline['bets_placed']:<20}{full['bets_placed']:<20}")
    print(f"{'bet win rate':<18}{baseline['bet_win_rate']:.1%}{'':<12}{full['bet_win_rate']:.1%}")
    print(f"{'total profit':<18}${baseline['total_profit']:.2f}{'':<10}${full['total_profit']:.2f}")
    print(f"{'ROI':<18}{baseline['roi_pct']:.1f}%{'':<15}{full['roi_pct']:.1f}%")
    print()
    print(
        "Betting simulation: flat $100 stake on every held-out prop where the model's "
        "predicted win probability exceeds the book's break-even probability (Calculated "
        "Edge > 0), using each side's actual American-odds payout."
    )
    print("=" * 72)


def main():
    dataset = prepare_dataset()
    train, test, split_date = chronological_split(dataset)

    if len(test) < 30:
        print(f"⚠️ Test set is only {len(test)} rows - results below are noisy, treat directionally only.")

    baseline_metrics, _ = train_and_score(train, test, BASELINE_FEATURES, "baseline")
    full_metrics, full_scored = train_and_score(train, test, FULL_FEATURES, "full")

    print_report(baseline_metrics, full_metrics, train, test, split_date)
    print_calibration_report(full_scored, "edge", "Calculated Edge")
    print_calibration_report(full_scored, "model_proba", "model win probability")
    evaluate_matchup_heuristic(train, test)
    evaluate_recent_vs_season_win_rate(split_date)


if __name__ == "__main__":
    main()
