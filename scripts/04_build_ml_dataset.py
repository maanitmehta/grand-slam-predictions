import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.name_utils import canonical_name
from config.tournaments import TOURNAMENTS
from config.surfaces import SURFACE_MAP, SURFACE_TRAINING_FILTER

BASE_PROCESSED = Path("data/processed")
BASE_RAW       = Path("data/raw")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    args = parser.parse_args()

    cfg             = TOURNAMENTS[args.tournament]
    surface         = cfg["surface"]
    allowed_surfaces = SURFACE_TRAINING_FILTER[surface]
    tour            = args.tour

    print(f"\n=== Building ML dataset for {tour.upper()} | {cfg['name']} | surface: {surface} ===")

    out_file      = BASE_PROCESSED / tour / f"ml_dataset_{args.tournament}.csv"
    if allowed_surfaces == ["hard"]:
        stats_path = BASE_PROCESSED / tour / "rolling_player_stats.csv"
    else:
        stats_path = BASE_PROCESSED / tour / "rolling_player_stats_all_surfaces.csv"
        
    rankings_path = BASE_RAW / f"{tour}_rankings.csv"

    for p in [stats_path, rankings_path]:
        if not p.exists():
            raise RuntimeError(f"Missing file: {p}")

    # Use all_matches for non-hard surfaces (model_base is hard-only)
    if allowed_surfaces == ["hard"]:
        matches = pd.read_csv(BASE_PROCESSED / tour / "model_base.csv",
                              parse_dates=["date"])
        # model_base already has canonical odds columns
        matches["b365w_col"] = matches.get("b365w", np.nan)
        matches["b365l_col"] = matches.get("b365l", np.nan)
    else:
        matches = pd.read_csv(BASE_PROCESSED / tour / "all_matches.csv",
                              low_memory=False, parse_dates=["date"])
        matches = matches.rename(columns={"B365W": "b365w", "B365L": "b365l"})

    # Normalise and filter surface
    matches["surface_canonical"] = matches["surface"].map(SURFACE_MAP).fillna("hard")
    matches = matches[matches["surface_canonical"].isin(allowed_surfaces)].copy()
    print(f"  Matches after surface filter {allowed_surfaces}: {len(matches):,}")

    if len(matches) == 0:
        raise RuntimeError("No matches after surface filter — check SURFACE_MAP values.")

    # Load surface-specific rolling features
    surf_feat_path = BASE_PROCESSED / tour / f"{surface}_surface_features.csv"
    surf_feats = None
    if surf_feat_path.exists():
        surf_feats = pd.read_csv(surf_feat_path, parse_dates=["date"])
        print(f"  Loaded surface features: {len(surf_feats):,} rows")

    # Rankings
    rankings = (
        pd.read_csv(rankings_path)
        .sort_values("rank")
        .drop_duplicates("player", keep="first")
        .assign(log_rank=lambda df: np.log(df["rank"]))
        .set_index("player")
    )

    stats = pd.read_csv(stats_path, parse_dates=["date"])
    stats = stats[["player", "date", "winrate_lastN", "avg_odds_lastN", "matches_played_lastN"]]

    rows = []

    for _, m in matches.iterrows():
        date = m["date"]
        w = canonical_name(m["winner"])
        l = canonical_name(m["loser"])

        w_stats = stats[(stats["player"] == w) & (stats["date"] == date)]
        l_stats = stats[(stats["player"] == l) & (stats["date"] == date)]

        if len(w_stats) != 1 or len(l_stats) != 1:
            continue

        w_stats = w_stats.iloc[0]
        l_stats = l_stats.iloc[0]

        if pd.isna(w_stats["winrate_lastN"]) or pd.isna(l_stats["winrate_lastN"]):
            continue

        rankA = float(rankings.loc[w, "log_rank"]) if w in rankings.index else np.log(200)
        rankB = float(rankings.loc[l, "log_rank"]) if l in rankings.index else np.log(200)

        # Surface win rate diff
        surf_diff = 0.0
        if surf_feats is not None:
            col = f"{surface}_win_rate_20"
            w_sf = surf_feats[(surf_feats["Player"] == w) & (surf_feats["date"] == date)]
            l_sf = surf_feats[(surf_feats["Player"] == l) & (surf_feats["date"] == date)]
            w_sr = w_sf.iloc[0][col] if len(w_sf) == 1 else np.nan
            l_sr = l_sf.iloc[0][col] if len(l_sf) == 1 else np.nan
            if not pd.isna(w_sr) and not pd.isna(l_sr):
                surf_diff = w_sr - l_sr

        rows.append({"winrate_diff": w_stats["winrate_lastN"] - l_stats["winrate_lastN"],
                     "odds_diff":    w_stats["avg_odds_lastN"] - l_stats["avg_odds_lastN"],
                     "matches_diff": w_stats["matches_played_lastN"] - l_stats["matches_played_lastN"],
                     "rank_diff":    rankB - rankA,
                     "surface_wr_diff": surf_diff,
                     "a_wins": 1})

        rows.append({"winrate_diff": l_stats["winrate_lastN"] - w_stats["winrate_lastN"],
                     "odds_diff":    l_stats["avg_odds_lastN"] - w_stats["avg_odds_lastN"],
                     "matches_diff": l_stats["matches_played_lastN"] - w_stats["matches_played_lastN"],
                     "rank_diff":    rankA - rankB,
                     "surface_wr_diff": -surf_diff,
                     "a_wins": 0})

    df = pd.DataFrame(rows).dropna()
    df["a_wins"] = df["a_wins"].astype(int)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved ML dataset → {out_file}")
    print(f"Rows: {len(df):,}")
    print("Class balance:")
    print(df["a_wins"].value_counts(normalize=True))

if __name__ == "__main__":
    main()