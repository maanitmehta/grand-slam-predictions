import pandas as pd
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

        window = 10 if tour == "atp" else 8

        df["matches_played_lastN"] = (
            df.groupby("player")["won"]
            .shift(1).rolling(window, min_periods=1).count()
        )
        df["winrate_lastN"] = (
            df.groupby("player")["won"]
            .shift(1).rolling(window, min_periods=1).mean()
        )
        df["avg_odds_lastN"] = (
            df.groupby("player")["odds_for"]
            .shift(1).rolling(window, min_periods=1).mean()
        )

        df.to_csv(out_file, index=False)
        print(f"Saved → {out_file}")
        print(f"Rows: {len(df):,}")

if __name__ == "__main__":
    main()
