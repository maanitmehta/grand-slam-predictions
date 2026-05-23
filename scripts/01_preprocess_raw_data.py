import pandas as pd
from pathlib import Path

TOURS = ["atp", "wta"]

BASE_IN = Path("data/processed")
BASE_OUT = Path("data/processed")

def main():
    for tour in TOURS:
        print(f"\n=== Preprocessing {tour.upper()} data ===")

        inp = BASE_IN / tour / "all_matches.csv"
        out_dir = BASE_OUT / tour
        out_file = out_dir / "model_base.csv"

        if not inp.exists():
            raise RuntimeError(f"Input file not found: {inp}")

        df = pd.read_csv(inp, low_memory=False)

        # Standardise column names
        df.columns = df.columns.str.lower().str.strip()

        # Parse dates safely
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date", "winner", "loser"])

        # Keep hard-court matches only (AO-specific assumption)
        df["surface"] = df["surface"].astype(str).str.capitalize()
        df = df[df["surface"] == "Hard"]

        # Sort chronologically
        df = df.sort_values("date")

        # Core columns
        keep = [
            "date",
            "season",
            "tournament",
            "surface",
            "winner",
            "loser",
            "winner_rank",
            "loser_rank",
        ]

        # Betting odds (if present)
        odds = [c for c in df.columns if c.startswith("b365")]

        df = df[keep + odds]

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

        print(f"Saved {len(df):,} rows → {out_file}")

if __name__ == "__main__":
    main()
