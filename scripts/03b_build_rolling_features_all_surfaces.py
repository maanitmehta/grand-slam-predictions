import pandas as pd
import numpy as np
from pathlib import Path

TOURS = ["atp", "wta"]
BASE_IN  = Path("data/processed")
BASE_OUT = Path("data/processed")

def main():
    for tour in TOURS:
        print(f"\n=== Building all-surface rolling features for {tour.upper()} ===")

        inp      = BASE_IN / tour / "player_match_history_all_surfaces.csv"
        out_file = BASE_OUT / tour / "rolling_player_stats_all_surfaces.csv"

        if not inp.exists():
            raise RuntimeError(f"Missing: {inp} — run 02b first.")

        df = pd.read_csv(inp, parse_dates=["date"], low_memory=False)
        df = df.sort_values(["player", "date"])

        # Clean odds column — coerce bad values like '5..5' to NaN
        df["odds_for"] = pd.to_numeric(df["odds_for"], errors="coerce")
        df["sets_played"] = pd.to_numeric(df.get("sets_played", 0), errors="coerce").fillna(0)
        df["days_since_last"] = pd.to_numeric(df.get("days_since_last"), errors="coerce")

        # Opponent rank — higher weight for wins vs stronger opponents (lower rank = better)
        opp_rank = pd.to_numeric(df["opponent_rank"], errors="coerce").fillna(200).clip(lower=1)
        df["quality_weight"] = 1.0 / np.log1p(opp_rank)

        window = 10 if tour == "atp" else 8

        grp = df.groupby("player")

        df["matches_played_lastN"] = (
            grp["won"].shift(1).rolling(window, min_periods=1).count()
        )
        df["winrate_lastN"] = (
            grp["won"].shift(1).rolling(window, min_periods=1).mean()
        )
        df["avg_odds_lastN"] = (
            grp["odds_for"].shift(1).rolling(window, min_periods=1).mean()
        )
        df["avg_sets_lastN"] = (
            grp["sets_played"].shift(1).rolling(window, min_periods=1).mean()
        )

        # Average rest days — rolling mean of days between matches (activity pattern)
        df["avg_rest_days_lastN"] = (
            grp["days_since_last"].shift(1).rolling(window, min_periods=1).mean()
        )

        # Quality-adjusted win rate: rolling sum(won * quality_weight) / sum(quality_weight)
        # Weights wins against higher-ranked opponents more — penalises padding stats vs weak fields
        df["quality_win"] = df["won"] * df["quality_weight"]
        qw_sum = grp["quality_win"].shift(1).rolling(window, min_periods=1).sum()
        wt_sum = grp["quality_weight"].shift(1).rolling(window, min_periods=1).sum()
        df["quality_winrate_lastN"] = qw_sum / wt_sum.replace(0, np.nan)

        df = df.drop(columns=["quality_weight", "quality_win"])

        # Clay-specific win rate — rolling over clay matches only, merged back as-of
        df_valid = df.dropna(subset=["date"]).copy()
        clay = (
            df_valid[df_valid["surface"].str.strip().str.lower() == "clay"]
            .sort_values(["player", "date"])
            .copy()
        )
        clay_grp = clay.groupby("player", group_keys=False)
        clay["clay_winrate_lastN"] = clay_grp.apply(
            lambda g: g["won"].shift(1).rolling(window, min_periods=1).mean(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        # Keep last clay value per (player, date) in case of same-day matches
        clay_latest = (
            clay.sort_values("date")
            .groupby(["player", "date"])["clay_winrate_lastN"]
            .last()
            .reset_index()
        )
        df_valid = pd.merge_asof(
            df_valid.sort_values("date"),
            clay_latest.sort_values("date"),
            on="date", by="player", direction="backward",
        )
        # Rows that had null dates get NaN for clay_winrate_lastN
        null_dates = df[df["date"].isna()].copy()
        null_dates["clay_winrate_lastN"] = np.nan
        df = pd.concat([df_valid, null_dates]).sort_values(["player", "date"]).reset_index(drop=True)

        df.to_csv(out_file, index=False)
        print(f"Saved → {out_file}")
        print(f"Rows: {len(df):,}")

if __name__ == "__main__":
    main()
