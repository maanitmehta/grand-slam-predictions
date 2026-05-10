import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.tournaments import TOURNAMENTS
from config.surfaces import SURFACE_TRAINING_FILTER

warnings.filterwarnings("ignore")

BASE_DATA    = Path("data/processed")
BASE_RESULTS = Path("results")

FEATURES = [
    "winrate_diff",
    "odds_diff",
    "matches_diff",
    "rank_diff",
    "elo_diff_global",
    "elo_diff_surface",
    "surface_wr_diff",
]

# ─────────────────────────────────────────────
# 1. CALIBRATION CURVE
# ─────────────────────────────────────────────
def test_calibration(df: pd.DataFrame, tour: str, tournament: str, out_dir: Path):
    print("\n── Calibration Test ──")

    features = [f for f in FEATURES if f in df.columns]
    X = df[features]
    y = df["a_wins"]

    split = int(len(df) * 0.8)
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("model",  LogisticRegression(max_iter=1000))])
    pipeline.fit(X.iloc[:split], y.iloc[:split])
    probs = pipeline.predict_proba(X.iloc[split:])[:, 1]
    y_test = y.iloc[split:]

    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    # Mean calibration error
    mce = np.mean(np.abs(fraction_pos - mean_pred))
    print(f"  Mean calibration error: {mce:.4f}  (0 = perfect, 0.05 = good, >0.1 = poor)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="#2E86AB", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {tour.upper()} {tournament.upper()}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"calibration_{tournament}_{tour}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {path}")
    return mce


# ─────────────────────────────────────────────
# 2. PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def test_feature_importance(df: pd.DataFrame, tour: str, tournament: str,
                            out_dir: Path, n_repeats: int = 10):
    print("\n── Permutation Feature Importance ──")

    features = [f for f in FEATURES if f in df.columns]
    X = df[features].values
    y = df["a_wins"].values

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler  = StandardScaler().fit(X_train)
    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000).fit(X_tr_sc, y_train)
    base_acc = accuracy_score(y_test, model.predict(X_te_sc))

    importances = []
    for i, feat in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_te_sc.copy()
            np.random.shuffle(X_perm[:, i])
            drops.append(base_acc - accuracy_score(y_test, model.predict(X_perm)))
        importances.append({
            "feature": feat,
            "importance_mean": np.mean(drops),
            "importance_std":  np.std(drops),
        })

    imp_df = (pd.DataFrame(importances)
              .sort_values("importance_mean", ascending=True))

    print(f"  Base accuracy: {base_acc:.4f}")
    print(imp_df[["feature", "importance_mean"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2E86AB" if v >= 0 else "#E84855" for v in imp_df["importance_mean"]]
    ax.barh(imp_df["feature"], imp_df["importance_mean"],
            xerr=imp_df["importance_std"], color=colors, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy drop when feature is shuffled")
    ax.set_title(f"Feature Importance — {tour.upper()} {tournament.upper()}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"feature_importance_{tournament}_{tour}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {path}")
    return imp_df


# ─────────────────────────────────────────────
# 3. WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────
def test_walk_forward(df: pd.DataFrame, tour: str, tournament: str,
                      out_dir: Path, min_train_rows: int = 2000):
    print("\n── Walk-Forward Backtest ──")

    # We don't have a year column in the dataset so we approximate
    # by splitting into annual chunks using row ordering (data is time-sorted)
    features = [f for f in FEATURES if f in df.columns]
    X = df[features].values
    y = df["a_wins"].values
    n = len(df)

    # Split into ~yearly folds (assume ~20 rows per match-day on average)
    n_folds = 8
    fold_size = n // n_folds

    results = []
    for fold in range(2, n_folds):   # need at least 2 folds of history
        train_end = fold * fold_size
        test_end  = min((fold + 1) * fold_size, n)

        if train_end < min_train_rows:
            continue

        X_train, y_train = X[:train_end],        y[:train_end]
        X_test,  y_test  = X[train_end:test_end], y[train_end:test_end]

        scaler   = StandardScaler().fit(X_train)
        model    = LogisticRegression(max_iter=1000)
        model.fit(scaler.transform(X_train), y_train)

        preds  = model.predict(scaler.transform(X_test))
        probas = model.predict_proba(scaler.transform(X_test))

        results.append({
            "fold":         fold,
            "train_rows":   train_end,
            "test_rows":    len(y_test),
            "accuracy":     accuracy_score(y_test, preds),
            "log_loss":     log_loss(y_test, probas),
        })

    wf = pd.DataFrame(results)
    print(wf[["fold", "train_rows", "accuracy", "log_loss"]].to_string(index=False))
    print(f"\n  Mean accuracy across folds: {wf['accuracy'].mean():.4f} "
          f"± {wf['accuracy'].std():.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wf["fold"], wf["accuracy"], "o-", color="#2E86AB", label="Accuracy")
    ax.axhline(wf["accuracy"].mean(), color="#E84855", linestyle="--",
               label=f"Mean {wf['accuracy'].mean():.3f}")
    ax.fill_between(wf["fold"],
                    wf["accuracy"].mean() - wf["accuracy"].std(),
                    wf["accuracy"].mean() + wf["accuracy"].std(),
                    alpha=0.15, color="#2E86AB")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Walk-Forward Backtest — {tour.upper()} {tournament.upper()}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"walk_forward_{tournament}_{tour}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {path}")
    return wf


# ─────────────────────────────────────────────
# 4. BOOTSTRAP CONFIDENCE INTERVALS
#    on Monte Carlo title probabilities
# ─────────────────────────────────────────────
def test_bootstrap_ci(draw_path: Path, tour: str, tournament: str,
                      out_dir: Path, n_boot: int = 200, n_sim: int = 1000):
    print("\n── Bootstrap Confidence Intervals ──")

    if not draw_path.exists():
        print(f"  Skipping — no draw file at {draw_path}")
        return None

    # Import here to avoid circular issues
    from scripts.predict_match import predict_match
    from scripts.name_utils import canonical_name

    ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

    draw = pd.read_csv(draw_path)
    draw["player_A"] = draw["player_A"].apply(canonical_name)
    draw["player_B"] = draw["player_B"].apply(canonical_name)

    players_ordered = []
    for _, row in draw.iterrows():
        players_ordered.append(row["player_A"])
        players_ordered.append(row["player_B"])

    def run_sim():
        players = players_ordered.copy()
        for r in ROUNDS:
            if len(players) == 1:
                break
            next_round = []
            for i in range(0, len(players), 2):
                A, B = players[i], players[i+1]
                p = predict_match(A, B, tour=tour, tournament=tournament)
                next_round.append(A if np.random.rand() < p else B)
            players = next_round
        return players[0]

    # Bootstrap: resample simulations n_boot times
    all_boot_probs = []
    print(f"  Running {n_boot} bootstrap iterations × {n_sim} sims each...")
    for b in range(n_boot):
        counts = Counter(run_sim() for _ in range(n_sim))
        total  = sum(counts.values())
        all_boot_probs.append({p: c / total for p, c in counts.items()})

    # Aggregate across bootstraps
    all_players = sorted({p for d in all_boot_probs for p in d})
    boot_matrix = pd.DataFrame(
        [{p: d.get(p, 0) for p in all_players} for d in all_boot_probs]
    )

    summary = pd.DataFrame({
        "player":    all_players,
        "mean_prob": boot_matrix.mean(),
        "ci_low":    boot_matrix.quantile(0.05),
        "ci_high":   boot_matrix.quantile(0.95),
    }).sort_values("mean_prob", ascending=False).head(16).reset_index(drop=True)

    print(f"\n  {tournament.upper()} {tour.upper()} — Title Probabilities with 90% CI\n")
    for _, row in summary.iterrows():
        bar = "█" * int(row["mean_prob"] * 200)
        print(f"  {row['player']:<22} {row['mean_prob']:.3f}  "
              f"[{row['ci_low']:.3f} – {row['ci_high']:.3f}]  {bar}")

    path = out_dir / f"bootstrap_ci_{tournament}_{tour}.csv"
    summary.to_csv(path, index=False)
    print(f"\n  Saved → {path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(len(summary))
    ax.barh(list(y_pos), summary["mean_prob"], color="#2E86AB",
            xerr=[summary["mean_prob"] - summary["ci_low"],
                  summary["ci_high"]  - summary["mean_prob"]],
            capsize=4, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(summary["player"])
    ax.set_xlabel("Title probability (90% CI)")
    ax.set_title(f"Title Probabilities — {tour.upper()} {tournament.upper()}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    img_path = out_dir / f"bootstrap_ci_{tournament}_{tour}.png"
    fig.savefig(img_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {img_path}")
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip bootstrap CI (slow — ~5 mins)")
    args = parser.parse_args()

    cfg        = TOURNAMENTS[args.tournament]
    tour       = args.tour
    surface    = cfg["surface"]

    allowed    = SURFACE_TRAINING_FILTER[surface]
    data_path  = BASE_DATA / tour / f"ml_dataset_{args.tournament}.csv"
    draw_path  = Path(cfg["draws"][tour])
    out_dir    = BASE_RESULTS / args.tournament
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STRESS TEST — {tour.upper()} | {cfg['name']}")
    print(f"{'='*55}")

    if not data_path.exists():
        raise RuntimeError(f"Missing: {data_path} — run 04_build_ml_dataset.py first")

    df = pd.read_csv(data_path)
    print(f"\n  Dataset: {len(df):,} rows | Features: {[f for f in FEATURES if f in df.columns]}")

    # Run all four tests
    mce     = test_calibration(df, tour, args.tournament, out_dir)
    imp_df  = test_feature_importance(df, tour, args.tournament, out_dir)
    wf      = test_walk_forward(df, tour, args.tournament, out_dir)

    if not args.skip_bootstrap:
        ci = test_bootstrap_ci(draw_path, tour, args.tournament, out_dir)
    else:
        print("\n── Bootstrap CI skipped (--skip-bootstrap) ──")

    # Summary
    print(f"\n{'='*55}")
    print(f"  STRESS TEST SUMMARY — {tour.upper()} {args.tournament.upper()}")
    print(f"{'='*55}")
    print(f"  Calibration error:       {mce:.4f}  {'✓ good' if mce < 0.05 else '⚠ needs work'}")
    print(f"  Walk-forward mean acc:   {wf['accuracy'].mean():.4f} ± {wf['accuracy'].std():.4f}")
    top_feat = imp_df.sort_values("importance_mean", ascending=False).iloc[0]
    print(f"  Most important feature:  {top_feat['feature']} ({top_feat['importance_mean']:.4f})")
    print(f"\n  Charts saved to: {out_dir}/")

if __name__ == "__main__":
    main()