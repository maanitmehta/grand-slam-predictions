import pandas as pd
from pathlib import Path

TOURS = ["atp", "wta"]

BASE_IN = Path("data/processed")
BASE_OUT = Path("data/processed")

def main():
    for tour in TOURS:
        print(f"\n=== Building rolling features for {tour.upper()} ===")

        inp = BASE_IN / tour / "player_match_history.csv"
        out_dir = BASE_OUT / tour
        out_file = out_dir / "rolling_player_stats.csv"

        if not inp.exists():
            raise RuntimeError(f"Input file not found: {inp}")

        df = pd.read_csv(inp, parse_dates=["date"])

        df = df.sort_values(["player", "date"])

        # Slightly smaller window for WTA (optional but recommended)
        window = 10 if tour == "atp" else 8

        df["matches_played_lastN"] = (
            df.groupby("player")["won"]
            .shift(1)
            .rolling(window, min_periods=1)
            .count()
        )

        df["winrate_lastN"] = (
            df.groupby("player")["won"]
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

        df["avg_odds_lastN"] = (
            df.groupby("player")["odds_for"]
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

        print(f"Saved → {out_file}")
        print(f"Rows: {len(df):,}")

if __name__ == "__main__":
    main()
