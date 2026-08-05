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

    return metrics


def print_report(baseline, full, train, test, split_date):
    print()
    print("=" * 72)
    print("BACKTEST: 3-feature baseline vs current 7-feature model")
    print("=" * 72)
    print(f"Train: {len(train)} rows ({train['Date'].min()} to {train['Date'].max()})")
    print(f"Test:  {len(test)} rows ({split_date} to {test['Date'].max()}) - held out, unseen by either model")
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

    baseline_metrics = train_and_score(train, test, BASELINE_FEATURES, "baseline")
    full_metrics = train_and_score(train, test, FULL_FEATURES, "full")

    print_report(baseline_metrics, full_metrics, train, test, split_date)


if __name__ == "__main__":
    main()
