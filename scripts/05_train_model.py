import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.tournaments import TOURNAMENTS

BASE_DATA   = Path("data/processed")
BASE_MODELS = Path("models")

BASE_FEATURES = [
    "winrate_diff",
    "odds_diff",
    "matches_diff",
    "rank_diff",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    args = parser.parse_args()

    cfg = TOURNAMENTS[args.tournament]
    tour = args.tour

    print(f"\n=== Training model for {tour.upper()} | {cfg['name']} ===")

    data_path = BASE_DATA / tour / f"ml_dataset_{args.tournament}.csv"
    model_out = Path(cfg["models"][tour])

    if not data_path.exists():
        raise RuntimeError(f"Missing ML dataset: {data_path}. Run 04_build_ml_dataset.py first.")

    df = pd.read_csv(data_path)

    # Include surface feature if it exists and has signal
    features = BASE_FEATURES.copy()
    if "surface_wr_diff" in df.columns and df["surface_wr_diff"].abs().sum() > 0:
        features.append("surface_wr_diff")
        print(f"  Including surface_wr_diff feature")

    X = df[features]
    y = df["a_wins"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print("MODEL PERFORMANCE")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Log loss: {log_loss(y_test, y_proba):.4f}")
    print(f"  Features: {features}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Saved model → {model_out}")

if __name__ == "__main__":
    main()