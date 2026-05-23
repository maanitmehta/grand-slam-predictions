import pandas as pd
from pathlib import Path

TOURS = ["atp", "wta"]

BASE_IN = Path("data/processed")
BASE_OUT = Path("data/processed")

def main():
    for tour in TOURS:
        print(f"\n=== Building player history for {tour.upper()} ===")

        inp = BASE_IN / tour / "model_base.csv"
        out_dir = BASE_OUT / tour
        out_file = out_dir / "player_match_history.csv"

        if not inp.exists():
            raise RuntimeError(f"Input file not found: {inp}")

        df = pd.read_csv(inp, parse_dates=["date"])

        winners = pd.DataFrame({
            "date": df["date"],
            "season": df["season"],
            "tourney_name": df["tournament"],
            "surface": df["surface"],
            "player": df["winner"],
            "player_rank": df["winner_rank"],
            "opponent": df["loser"],
            "opponent_rank": df["loser_rank"],
            "odds_for": df.get("b365w"),
            "odds_against": df.get("b365l"),
            "won": 1
        })

        losers = pd.DataFrame({
            "date": df["date"],
            "season": df["season"],
            "tourney_name": df["tournament"],
            "surface": df["surface"],
            "player": df["loser"],
            "player_rank": df["loser_rank"],
            "opponent": df["winner"],
            "opponent_rank": df["winner_rank"],
            "odds_for": df.get("b365l"),
            "odds_against": df.get("b365w"),
            "won": 0
        })

        history = pd.concat([winners, losers], ignore_index=True)
        history = history.sort_values("date").reset_index(drop=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_file, index=False)

        print(f"Saved → {out_file}")
        print(f"Rows: {len(history):,}")

if __name__ == "__main__":
    main()
