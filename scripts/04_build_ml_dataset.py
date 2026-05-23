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
ELO_INIT       = 1500.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    args = parser.parse_args()

    cfg              = TOURNAMENTS[args.tournament]
    surface          = cfg["surface"]
    allowed_surfaces = SURFACE_TRAINING_FILTER[surface]
    tour             = args.tour

    print(f"\n=== Building ML dataset for {tour.upper()} | {cfg['name']} | surface: {surface} ===")

    out_file      = BASE_PROCESSED / tour / f"ml_dataset_{args.tournament}.csv"
    rankings_path = BASE_RAW / f"{tour}_rankings.csv"

    # Use all-surfaces stats for clay/grass, hard-only for hard
    if allowed_surfaces == ["hard"]:
        stats_path  = BASE_PROCESSED / tour / "rolling_player_stats.csv"
        matches_src = BASE_PROCESSED / tour / "model_base.csv"
    else:
        stats_path  = BASE_PROCESSED / tour / "rolling_player_stats_all_surfaces.csv"
        matches_src = BASE_PROCESSED / tour / "all_matches.csv"

    for p in [stats_path, rankings_path, matches_src]:
        if not p.exists():
            raise RuntimeError(f"Missing file: {p}")

    # Load matches
    if allowed_surfaces == ["hard"]:
        matches = pd.read_csv(matches_src, parse_dates=["date"])
    else:
        matches = pd.read_csv(matches_src, low_memory=False, parse_dates=["date"])
        matches = matches.rename(columns={"B365W": "b365w", "B365L": "b365l"})

    # Normalise and filter by surface
    matches["surface_canonical"] = matches["surface"].map(SURFACE_MAP).fillna("hard")
    matches = matches[matches["surface_canonical"].isin(allowed_surfaces)].copy()
    print(f"  Matches after surface filter {allowed_surfaces}: {len(matches):,}")

    if len(matches) == 0:
        raise RuntimeError("No matches after surface filter — check SURFACE_MAP values.")

    # Load surface rolling features
    surf_feats = None
    surf_feat_path = BASE_PROCESSED / tour / f"{surface}_surface_features.csv"
    if surf_feat_path.exists():
        surf_feats = pd.read_csv(surf_feat_path, parse_dates=["date"])
        print(f"  Loaded surface features: {len(surf_feats):,} rows")

    # Load Elo history
    elo_hist = None
    elo_path = BASE_PROCESSED / tour / "elo_history.csv"
    if elo_path.exists():
        elo_hist = pd.read_csv(elo_path, parse_dates=["date"])
        print(f"  Loaded Elo history: {len(elo_hist):,} rows")
    else:
        print("  No Elo history found — run 08_build_elo_ratings.py to add Elo features")

    # Rankings
    rankings = (
        pd.read_csv(rankings_path)
        .sort_values("rank")
        .drop_duplicates("player", keep="first")
        .assign(log_rank=lambda df: np.log(df["rank"]))
        .set_index("player")
    )

    stats = pd.read_csv(stats_path, parse_dates=["date"], low_memory=False)
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

        # Elo diffs (pre-match ratings, winner perspective)
        elo_diff_global  = 0.0
        elo_diff_surface = 0.0
        if elo_hist is not None:
            # Look up winner's pre-match Elo
            w_row = elo_hist[(elo_hist["winner"] == w) & (elo_hist["date"] == date)]
            if len(w_row) == 0:
                w_row = elo_hist[(elo_hist["loser"] == w) & (elo_hist["date"] == date)]
                if len(w_row) > 0:
                    w_global = w_row.iloc[0]["l_elo_global"]
                    w_surf   = w_row.iloc[0].get(f"l_elo_{surface}", ELO_INIT)
                else:
                    w_global = w_surf = ELO_INIT
            else:
                w_global = w_row.iloc[0]["w_elo_global"]
                w_surf   = w_row.iloc[0].get(f"w_elo_{surface}", ELO_INIT)

            # Look up loser's pre-match Elo
            l_row = elo_hist[(elo_hist["loser"] == l) & (elo_hist["date"] == date)]
            if len(l_row) == 0:
                l_row = elo_hist[(elo_hist["winner"] == l) & (elo_hist["date"] == date)]
                if len(l_row) > 0:
                    l_global = l_row.iloc[0]["w_elo_global"]
                    l_surf   = l_row.iloc[0].get(f"w_elo_{surface}", ELO_INIT)
                else:
                    l_global = l_surf = ELO_INIT
            else:
                l_global = l_row.iloc[0]["l_elo_global"]
                l_surf   = l_row.iloc[0].get(f"l_elo_{surface}", ELO_INIT)

            elo_diff_global  = float(w_global) - float(l_global)
            elo_diff_surface = float(w_surf)   - float(l_surf)

        # Winner row
        rows.append({
            "winrate_diff":     w_stats["winrate_lastN"]       - l_stats["winrate_lastN"],
            "odds_diff":        w_stats["avg_odds_lastN"]       - l_stats["avg_odds_lastN"],
            "matches_diff":     w_stats["matches_played_lastN"] - l_stats["matches_played_lastN"],
            "rank_diff":        rankB - rankA,
            "surface_wr_diff":  surf_diff,
            "elo_diff_global":  elo_diff_global,
            "elo_diff_surface": elo_diff_surface,
            "a_wins": 1,
        })

        # Loser row (symmetric — flip all diffs)
        rows.append({
            "winrate_diff":     l_stats["winrate_lastN"]       - w_stats["winrate_lastN"],
            "odds_diff":        l_stats["avg_odds_lastN"]       - w_stats["avg_odds_lastN"],
            "matches_diff":     l_stats["matches_played_lastN"] - w_stats["matches_played_lastN"],
            "rank_diff":        rankA - rankB,
            "surface_wr_diff":  -surf_diff,
            "elo_diff_global":  -elo_diff_global,
            "elo_diff_surface": -elo_diff_surface,
            "a_wins": 0,
        })

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