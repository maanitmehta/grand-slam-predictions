import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.tournaments import TOURNAMENTS
from scripts.model_utils import LGBMPipeline

BASE_DATA   = Path("data/processed")
BASE_MODELS = Path("models")

BASE_FEATURES = [
    "winrate_diff",
    "odds_diff",
    "matches_diff",
    "rank_diff",
    "elo_diff_global",
    "elo_diff_surface",
]

OPT_FEATURES = [
    "surface_wr_diff",
    "h2h_clay_diff",
    "mkt_prob_diff",
    "quality_winrate_diff",
    "elo_diff_rg",
    "clay_winrate_diff",
]


def build_logreg(X_train, y_train):
    """StandardScaler + LogisticRegression. More stable than LightGBM on this dataset size."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def build_lgbm(X_train, y_train):
    """LightGBM + Platt scaling with recency weighting and tighter regularisation."""
    n = len(X_train)
    cal_split = int(n * 0.8)

    X_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
    y_fit, y_cal = y_train[:cal_split], y_train[cal_split:]

    # Recency weighting — recent matches weighted up to 2× vs oldest, to reduce
    # the fold instability caused by non-stationarity in the training distribution
    weights = np.linspace(0.5, 1.0, len(X_fit))

    lgbm = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,    # raised from 20 — closes early/late fold gap
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,          # raised from 1.0
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    lgbm.fit(X_fit, y_fit, sample_weight=weights)

    calibrated = CalibratedClassifierCV(lgbm, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated, None   # no external scaler for lgbm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    parser.add_argument("--model",      choices=["logreg", "lgbm"], default="logreg",
                        help="Base learner (default: logreg — more stable on this dataset)")
    parser.add_argument("--standalone", action="store_true",
                        help="Train simulation model: excludes mkt_prob_diff, saves to sim_models path")
    args = parser.parse_args()

    cfg   = TOURNAMENTS[args.tournament]
    tour  = args.tour

    mode_label = "standalone-sim" if args.standalone else "market-inclusive"
    print(f"\n=== Training model for {tour.upper()} | {cfg['name']} | learner: {args.model} | {mode_label} ===")

    data_path = BASE_DATA / tour / f"ml_dataset_{args.tournament}.csv"
    if args.standalone:
        model_out = Path(cfg["sim_models"][tour])
    else:
        model_out = Path(cfg["models"][tour])

    if not data_path.exists():
        raise RuntimeError(f"Missing ML dataset: {data_path}. Run 04_build_ml_dataset.py first.")

    df = pd.read_csv(data_path)

    # Standalone excludes mkt_prob_diff so coefficients are market-independent
    excluded = {"mkt_prob_diff"} if args.standalone else set()

    features = BASE_FEATURES.copy()
    for opt in OPT_FEATURES:
        if opt in excluded:
            print(f"  Excluding {opt} (standalone mode)")
            continue
        if opt in df.columns and df[opt].abs().sum() > 0:
            features.append(opt)
            print(f"  Including {opt}")

    X = df[features].fillna(0.0).values
    y = df["a_wins"].values

    # 80/20 train/test split (chronological)
    n         = len(df)
    train_end = int(n * 0.8)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test  = X[train_end:]
    y_test  = y[train_end:]

    if args.model == "logreg":
        model, scaler = build_logreg(X_train, y_train)
        X_test_eval   = scaler.transform(X_test)
        y_pred         = model.predict(X_test_eval)
        y_proba        = model.predict_proba(X_test_eval)
        pipeline       = LGBMPipeline(model, features, scaler=scaler)
    else:
        model, scaler = build_lgbm(X_train, y_train)
        y_pred         = model.predict(X_test)
        y_proba        = model.predict_proba(X_test)
        pipeline       = LGBMPipeline(model, features, scaler=None)

    print("MODEL PERFORMANCE")
    print(f"  Learner:  {args.model}")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Log loss: {log_loss(y_test, y_proba):.4f}")
    print(f"  Features: {features}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Saved model → {model_out}")


if __name__ == "__main__":
    main()
