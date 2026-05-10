import pandas as pd
import numpy as np
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.surfaces import SURFACE_MAP, SURFACE_TRAINING_FILTER, TOURNAMENT_SURFACE

def build_surface_features(tour: str, surface: str, window: int = 20):
    path = f"data/processed/{tour}/all_matches.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    # Normalise surface column
    df["surface_canonical"] = df["surface"].map(SURFACE_MAP).fillna("hard")

    # Filter to target surface only
    allowed = SURFACE_TRAINING_FILTER[surface]
    df_surf = df[df["surface_canonical"].isin(allowed)].copy()
    df_surf = df_surf.sort_values("date")

    # Build winner rows
    winners = df_surf[["date", "winner", "surface_canonical"]].copy()
    winners.columns = ["date", "Player", "surface"]
    winners["won"] = 1

    # Build loser rows
    losers = df_surf[["date", "loser", "surface_canonical"]].copy()
    losers.columns = ["date", "Player", "surface"]
    losers["won"] = 0

    player_matches = pd.concat([winners, losers]).sort_values("date")

    results = []
    for player, grp in player_matches.groupby("Player"):
        grp = grp.sort_values("date").copy()
        grp[f"{surface}_win_rate_{window}"] = (
            grp["won"].shift(1).rolling(window, min_periods=3).mean()
        )
        grp[f"{surface}_match_count_{window}"] = (
            grp["won"].shift(1).rolling(window, min_periods=1).count()
        )
        results.append(grp)

    out = pd.concat(results).sort_values("date")

    out_path = f"data/processed/{tour}/{surface}_surface_features.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {surface} surface features for {tour.upper()} → {out_path}")
    print(f"  Players covered: {out['Player'].nunique()}")
    print(f"  Date range: {out['date'].min().date()} to {out['date'].max().date()}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    parser.add_argument("--tournament", required=True, choices=["ao26", "fo26"])
    parser.add_argument("--window",     type=int, default=20)
    args = parser.parse_args()

    surface = TOURNAMENT_SURFACE[args.tournament]
    build_surface_features(args.tour, surface, args.window)

