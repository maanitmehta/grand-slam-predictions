import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from collections import Counter
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.tournaments import TOURNAMENTS
from config.surfaces import SURFACE_TRAINING_FILTER

warnings.filterwarnings("ignore")


def _build_model(X_train, y_train, model_type="logreg"):
    """Build and return a fitted model. Falls back to LogReg when training set is tiny."""
    if len(X_train) < 100 or model_type == "logreg":
        scaler = StandardScaler().fit(X_train)
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
            scaler.transform(X_train), y_train
        )
        class _W:
            def __init__(self, sc, mo):
                self._sc, self._mo = sc, mo
            def predict(self, X):
                return self._mo.predict(self._sc.transform(X))
            def predict_proba(self, X):
                return self._mo.predict_proba(self._sc.transform(X))
        return _W(scaler, m)

    cal_split = int(len(X_train) * 0.8)
    X_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
    y_fit, y_cal = y_train[:cal_split], y_train[cal_split:]

    if len(np.unique(y_cal)) < 2 or cal_split < 40:
        scaler = StandardScaler().fit(X_train)
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
            scaler.transform(X_train), y_train
        )
        class _W:
            def __init__(self, sc, mo):
                self._sc, self._mo = sc, mo
            def predict(self, X):
                return self._mo.predict(self._sc.transform(X))
            def predict_proba(self, X):
                return self._mo.predict_proba(self._sc.transform(X))
        return _W(scaler, m)

    weights = np.linspace(0.5, 1.0, len(X_fit))
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=1,
    )
    model.fit(X_fit, y_fit, sample_weight=weights)
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


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
    "h2h_clay_diff",
    "mkt_prob_diff",
    "quality_winrate_diff",
    "elo_diff_rg",
    "clay_winrate_diff",
]


# ─────────────────────────────────────────────
# 1. CALIBRATION CURVE
# ─────────────────────────────────────────────
def test_calibration(df: pd.DataFrame, tour: str, tournament: str,
                     out_dir: Path, model_type: str):
    print("\n── Calibration Test ──")

    features = [f for f in FEATURES if f in df.columns]
    X = df[features].fillna(0.0)
    y = df["a_wins"]

    split = int(len(df) * 0.8)
    model = _build_model(X.values[:split], y.values[:split], model_type)
    probs = model.predict_proba(X.values[split:])[:, 1]
    y_test = y.iloc[split:]

    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    mce = np.mean(np.abs(fraction_pos - mean_pred))
    print(f"  Mean calibration error: {mce:.4f}  (0 = perfect, 0.05 = good, >0.1 = poor)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="#2E86AB", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {tour.upper()} {tournament.upper()} [{model_type}]")
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
                            out_dir: Path, model_type: str, n_repeats: int = 10):
    print("\n── Permutation Feature Importance ──")

    features = [f for f in FEATURES if f in df.columns]
    X = df[features].fillna(0.0).values
    y = df["a_wins"].values

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model    = _build_model(X_train, y_train, model_type)
    base_acc = accuracy_score(y_test, model.predict(X_test))

    importances = []
    for i, feat in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
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
    ax.set_title(f"Feature Importance — {tour.upper()} {tournament.upper()} [{model_type}]")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"feature_importance_{tournament}_{tour}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {path}")
    return imp_df


# ─────────────────────────────────────────────
# 3. WALK-FORWARD BACKTEST (year-based splits)
# ─────────────────────────────────────────────
def test_walk_forward(df: pd.DataFrame, tour: str, tournament: str,
                      out_dir: Path, model_type: str, min_train_rows: int = 2000):
    print("\n── Walk-Forward Backtest (year-based) ──")

    features = [f for f in FEATURES if f in df.columns]
    X_all = df[features].fillna(0.0).values
    y_all = df["a_wins"].values

    # Use calendar years if match_date available; fall back to row-count folds
    if "match_date" in df.columns:
        years = pd.to_datetime(df["match_date"]).dt.year.values
        unique_years = sorted(set(years))
        # Need enough history before first test year
        min_test_year = unique_years[0] + 4
        fold_defs = [
            (years < yr, years == yr)
            for yr in unique_years if yr >= min_test_year
        ]
        print(f"  Year-based folds: testing {min_test_year}–{unique_years[-1]}")
    else:
        n = len(df)
        n_folds   = 8
        fold_size = n // n_folds
        fold_defs = [
            (np.arange(n) < fold * fold_size,
             (np.arange(n) >= fold * fold_size) & (np.arange(n) < (fold + 1) * fold_size))
            for fold in range(2, n_folds)
        ]
        print("  Row-count folds (match_date column not found)")

    results = []
    for train_mask, test_mask in fold_defs:
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if len(X_train) < min_train_rows or len(X_test) == 0:
            continue

        model  = _build_model(X_train, y_train, model_type)
        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)

        results.append({
            "period":     int(np.max(np.where(test_mask)[0][:1])) if "match_date" not in df.columns
                          else pd.to_datetime(df.loc[test_mask, "match_date"]).dt.year.iloc[0],
            "train_rows": int(train_mask.sum()),
            "test_rows":  len(y_test),
            "accuracy":   accuracy_score(y_test, preds),
            "log_loss":   log_loss(y_test, probas),
        })

    wf = pd.DataFrame(results)
    print(wf[["period", "train_rows", "accuracy", "log_loss"]].to_string(index=False))
    print(f"\n  Mean accuracy across folds: {wf['accuracy'].mean():.4f} "
          f"± {wf['accuracy'].std():.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(wf["period"], wf["accuracy"], "o-", color="#2E86AB", label="Accuracy")
    ax.axhline(wf["accuracy"].mean(), color="#E84855", linestyle="--",
               label=f"Mean {wf['accuracy'].mean():.3f}")
    ax.fill_between(wf["period"],
                    wf["accuracy"].mean() - wf["accuracy"].std(),
                    wf["accuracy"].mean() + wf["accuracy"].std(),
                    alpha=0.15, color="#2E86AB")
    ax.set_xlabel("Year")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Walk-Forward Backtest — {tour.upper()} {tournament.upper()} [{model_type}]")
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
                p = predict_match(A, B, tour=tour, tournament=tournament, round_name=r)
                next_round.append(A if np.random.rand() < p else B)
            players = next_round
        return players[0]

    all_boot_probs = []
    print(f"  Running {n_boot} bootstrap iterations × {n_sim} sims each...")
    for b in range(n_boot):
        counts = Counter(run_sim() for _ in range(n_sim))
        total  = sum(counts.values())
        all_boot_probs.append({p: c / total for p, c in counts.items()})

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
# 5. STRATIFIED SLAM-DEPTH BACKTEST
#    Clay, R16+, both players ranked top 30
# ─────────────────────────────────────────────
SLAM_LATE_ROUNDS = {"4th Round", "Quarterfinals", "Semifinals", "The Final"}

def test_stratified_slam_depth(df: pd.DataFrame, tour: str, tournament: str,
                               out_dir: Path, model_type: str):
    print("\n── Stratified Slam-Depth Backtest (Clay · R16+ · Both Top 30) ──")

    required = {"match_round", "match_series", "max_rank"}
    if not required.issubset(df.columns):
        print("  Skipping — metadata columns missing. Re-run 04_build_ml_dataset.py first.")
        return None

    mask = (
        df["match_series"].str.contains("Grand Slam", na=False) &
        df["match_round"].isin(SLAM_LATE_ROUNDS) &
        (df["max_rank"] <= 30)
    )
    sub = df[mask].copy()
    n_matches = len(sub) // 2
    print(f"  Subgroup size: {len(sub):,} rows (~{n_matches} matches over 25 years)")

    if len(sub) < 100:
        print("  Skipping — too few rows for meaningful walk-forward.")
        return None

    features = [f for f in FEATURES if f in sub.columns]
    X = sub[features].fillna(0.0).values
    y = sub["a_wins"].values
    n = len(sub)

    n_folds   = 5
    fold_size = n // n_folds
    min_train = 60

    results = []
    for fold in range(2, n_folds):
        train_end = fold * fold_size
        test_end  = min((fold + 1) * fold_size, n)
        if train_end < min_train:
            continue

        X_train, y_train = X[:train_end],         y[:train_end]
        X_test,  y_test  = X[train_end:test_end],  y[train_end:test_end]

        model  = _build_model(X_train, y_train, model_type)
        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)
        results.append({
            "fold":       fold,
            "train_rows": train_end,
            "test_rows":  len(y_test),
            "accuracy":   accuracy_score(y_test, preds),
            "log_loss":   log_loss(y_test, probas),
        })

    if not results:
        print("  No folds passed minimum training threshold.")
        return None

    wf = pd.DataFrame(results)
    print(wf[["fold", "train_rows", "test_rows", "accuracy", "log_loss"]].to_string(index=False))
    mean_acc = wf["accuracy"].mean()
    std_acc  = wf["accuracy"].std()
    print(f"\n  Mean accuracy (slam depth): {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  NOTE: n≈{n_matches} — wide confidence intervals expected. Treat as directional only.")

    print("\n  Feature importance at slam depth:")
    importances = []
    model_full = _build_model(X, y, model_type)
    base_acc   = accuracy_score(y, model_full.predict(X))
    for i, feat in enumerate(features):
        drops = []
        for _ in range(10):
            Xp = X.copy()
            np.random.shuffle(Xp[:, i])
            drops.append(base_acc - accuracy_score(y, model_full.predict(Xp)))
        importances.append({"feature": feat, "importance": np.mean(drops)})
    imp = pd.DataFrame(importances).sort_values("importance", ascending=False)
    print(imp.to_string(index=False))

    return wf


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    parser.add_argument("--model",      choices=["logreg", "lgbm"], default="logreg",
                        help="Base learner to use in stress tests (default: logreg)")
    parser.add_argument("--standalone", action="store_true",
                        help="Exclude mkt_prob_diff — tests the simulation model feature set")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip bootstrap CI (slow — ~5 mins)")
    args = parser.parse_args()

    cfg        = TOURNAMENTS[args.tournament]
    tour       = args.tour
    model_type = args.model

    # Standalone mode: strip mkt_prob_diff so walk-forward reflects simulation model accuracy
    if args.standalone:
        global FEATURES
        FEATURES = [f for f in FEATURES if f != "mkt_prob_diff"]
        print(f"  Standalone mode — mkt_prob_diff excluded from all tests")

    data_path  = BASE_DATA / tour / f"ml_dataset_{args.tournament}.csv"
    draw_path  = Path(cfg["draws"][tour])
    out_dir    = BASE_RESULTS / args.tournament
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STRESS TEST — {tour.upper()} | {cfg['name']} | [{model_type}]")
    print(f"{'='*55}")

    if not data_path.exists():
        raise RuntimeError(f"Missing: {data_path} — run 04_build_ml_dataset.py first")

    df = pd.read_csv(data_path)
    active_features = [f for f in FEATURES if f in df.columns]
    print(f"\n  Dataset: {len(df):,} rows | Features: {active_features}")

    mce     = test_calibration(df, tour, args.tournament, out_dir, model_type)
    imp_df  = test_feature_importance(df, tour, args.tournament, out_dir, model_type)
    wf      = test_walk_forward(df, tour, args.tournament, out_dir, model_type)
    wf_slam = test_stratified_slam_depth(df, tour, args.tournament, out_dir, model_type)

    if not args.skip_bootstrap:
        ci = test_bootstrap_ci(draw_path, tour, args.tournament, out_dir)
    else:
        print("\n── Bootstrap CI skipped (--skip-bootstrap) ──")

    print(f"\n{'='*55}")
    print(f"  STRESS TEST SUMMARY — {tour.upper()} {args.tournament.upper()} [{model_type}]")
    print(f"{'='*55}")
    print(f"  Calibration error:       {mce:.4f}  {'✓ good' if mce < 0.05 else '⚠ needs work'}")
    print(f"  Walk-forward mean acc:   {wf['accuracy'].mean():.4f} ± {wf['accuracy'].std():.4f}  (all clay)")
    if wf_slam is not None:
        print(f"  Slam-depth mean acc:     {wf_slam['accuracy'].mean():.4f} ± {wf_slam['accuracy'].std():.4f}  (R16+, both top 30) ⚠ small sample")
    top_feat = imp_df.sort_values("importance_mean", ascending=False).iloc[0]
    print(f"  Most important feature:  {top_feat['feature']} ({top_feat['importance_mean']:.4f})")
    print(f"\n  Charts saved to: {out_dir}/")

if __name__ == "__main__":
    main()
