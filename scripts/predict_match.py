import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.name_utils import canonical_name
from config.tournaments import TOURNAMENTS
from config.surfaces import SURFACE_TRAINING_FILTER

BASE_DATA = Path("data/processed")

# Reference dates per tournament — stats snapshot just before each event
TOURNAMENT_REF_DATES = {
    "ao26": pd.to_datetime("2026-01-14"),
    "fo26": pd.to_datetime("2026-05-25"),
}

# Module-level caches keyed by (tournament, tour)
_MODELS = {}
_STATS  = {}
_CACHE  = {}

def _load(tournament: str, tour: str):
    key = (tournament, tour)
    if key in _MODELS:
        return

    cfg      = TOURNAMENTS[tournament]
    surface  = cfg["surface"]
    ref_date = TOURNAMENT_REF_DATES[tournament]

    model_path = Path(cfg["models"][tour])
    if not model_path.exists():
        raise RuntimeError(
            f"Missing model: {model_path}\n"
            f"Run: python3 scripts/05_train_model.py --tournament {tournament} --tour {tour}"
        )

    _MODELS[key] = joblib.load(model_path)

    # Use all-surfaces stats for clay/grass, hard-only for hard
    if SURFACE_TRAINING_FILTER[surface] == ["hard"]:
        stats_path = BASE_DATA / tour / "rolling_player_stats.csv"
    else:
        stats_path = BASE_DATA / tour / "rolling_player_stats_all_surfaces.csv"

    if not stats_path.exists():
        raise RuntimeError(f"Missing stats: {stats_path}")

    stats = pd.read_csv(stats_path, parse_dates=["date"], low_memory=False)

    # Snapshot: latest stats per player before the tournament starts
    _STATS[key] = (
        stats[stats["date"] < ref_date]
        .sort_values("date")
        .groupby("player")
        .tail(1)
        .set_index("player")
    )

    print(f"  Loaded model + stats for {tournament.upper()} {tour.upper()} "
          f"({len(_STATS[key])} players)")

def safe_diff(x, y, default=0.0):
    if pd.isna(x) or pd.isna(y):
        return default
    return float(x) - float(y)

def predict_match(A, B, tour="atp", tournament="ao26"):
    """Return probability that player A beats player B."""
    A = canonical_name(A)
    B = canonical_name(B)

    cache_key = (A, B, tour, tournament)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    _load(tournament, tour)

    key   = (tournament, tour)
    stats = _STATS[key]
    model = _MODELS[key]

    if A not in stats.index or B not in stats.index:
        p = 0.45  # fallback for unknown players
    else:
        a = stats.loc[A]
        b = stats.loc[B]

        features = {
            "winrate_diff":  safe_diff(a["winrate_lastN"],       b["winrate_lastN"]),
            "odds_diff":     safe_diff(a["avg_odds_lastN"],       b["avg_odds_lastN"]),
            "matches_diff":  safe_diff(a["matches_played_lastN"], b["matches_played_lastN"]),
            "rank_diff":     0.0,
            "surface_wr_diff": 0.0,
        }

        # Include surface_wr_diff if the model was trained with it
        X = pd.DataFrame([features])
        model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(features.keys())

        # Only keep columns the model knows about
        named_step = model.named_steps.get("scaler") or model.named_steps.get("model")
        try:
            model_cols = model.named_steps["scaler"].feature_names_in_
            X = X[list(model_cols)]
        except (AttributeError, KeyError):
            X = X[list(features.keys())]

        p = float(model.predict_proba(X)[0, 1])
        p = min(max(p, 0.05), 0.95)

    _CACHE[cache_key] = p
    return p