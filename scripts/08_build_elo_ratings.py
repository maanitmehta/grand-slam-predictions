import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.name_utils import canonical_name
from config.surfaces import SURFACE_MAP

BASE_PROCESSED = Path("data/processed")

# Elo tuning parameters
K_BASE     = 32    # base K-factor (how fast ratings update)
K_SLAM     = 48    # higher K for Grand Slams (more signal)
ELO_INIT   = 1500  # starting rating for all players
SURFACES   = ["global", "clay", "hard", "grass"]

SLAM_NAMES = ["australian open", "roland garros", "wimbledon", "us open",
              "french open", "ao", "rg", "fo"]

def is_slam(tournament_name: str) -> bool:
    if pd.isna(tournament_name):
        return False
    return any(s in str(tournament_name).lower() for s in SLAM_NAMES)

def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

def update_elo(ra: float, rb: float, winner: str, k: float):
    ea = expected_score(ra, rb)
    eb = 1 - ea
    if winner == "A":
        return ra + k * (1 - ea), rb + k * (0 - eb)
    else:
        return ra + k * (0 - ea), rb + k * (1 - eb)

def compute_elo(tour: str):
    print(f"\n=== Computing Elo ratings for {tour.upper()} ===")

    path = BASE_PROCESSED / tour / "all_matches.csv"
    df   = pd.read_csv(path, low_memory=False, parse_dates=["date"])
    df   = df.sort_values("date").reset_index(drop=True)

    df["winner"] = df["winner"].apply(canonical_name)
    df["loser"]  = df["loser"].apply(canonical_name)
    df["surface_canonical"] = df["surface"].map(SURFACE_MAP).fillna("hard")

    # One dict of ratings per surface track
    ratings = {s: {} for s in SURFACES}

    def get_rating(surface_track: str, player: str) -> float:
        return ratings[surface_track].get(player, ELO_INIT)

    records = []

    for _, row in df.iterrows():
        w = row["winner"]
        l = row["loser"]
        surf = row["surface_canonical"]
        k = K_SLAM if is_slam(row.get("tournament", "")) else K_BASE

        # Snapshot BEFORE update (pre-match ratings as features)
        snap = {
            "date":    row["date"],
            "winner":  w,
            "loser":   l,
            "surface": surf,
        }
        for s in SURFACES:
            snap[f"w_elo_{s}"] = get_rating(s, w)
            snap[f"l_elo_{s}"] = get_rating(s, l)

        records.append(snap)

        # Update global Elo
        rw_g, rl_g = update_elo(get_rating("global", w),
                                 get_rating("global", l), "A", k)
        ratings["global"][w] = rw_g
        ratings["global"][l] = rl_g

        # Update surface-specific Elo
        rw_s, rl_s = update_elo(get_rating(surf, w),
                                 get_rating(surf, l), "A", k)
        ratings[surf][w] = rw_s
        ratings[surf][l] = rl_s

    elo_history = pd.DataFrame(records)

    # Add Elo diffs as features (winner perspective)
    for s in SURFACES:
        elo_history[f"elo_diff_{s}"] = (
            elo_history[f"w_elo_{s}"] - elo_history[f"l_elo_{s}"]
        )

    # Save full history
    hist_path = BASE_PROCESSED / tour / "elo_history.csv"
    elo_history.to_csv(hist_path, index=False)
    print(f"  Saved history → {hist_path}  ({len(elo_history):,} rows)")

    # Save final snapshot (current ratings for every player)
    final_rows = []
    all_players = set(ratings["global"].keys())
    for player in all_players:
        row = {"player": player}
        for s in SURFACES:
            row[f"elo_{s}"] = get_rating(s, player)
        final_rows.append(row)

    snapshot = pd.DataFrame(final_rows).sort_values("elo_global", ascending=False)
    snap_path = BASE_PROCESSED / tour / "elo_snapshot.csv"
    snapshot.to_csv(snap_path, index=False)
    print(f"  Saved snapshot → {snap_path}  ({len(snapshot):,} players)")
    print(f"\n  Top 10 by global Elo:")
    print(snapshot[["player", "elo_global", "elo_clay", "elo_hard"]].head(10).to_string(index=False))

if __name__ == "__main__":
    for tour in ["atp", "wta"]:
        compute_elo(tour)