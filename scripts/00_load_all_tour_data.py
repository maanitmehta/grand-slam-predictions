import pandas as pd
from pathlib import Path

TOURS = ["atp", "wta"]

BASE_RAW = Path("data/raw/tennis_data")
BASE_OUT = Path("data/processed")

def main():
    for tour in TOURS:
        print(f"\n=== Processing {tour.upper()} data ===")

        raw_dir = BASE_RAW / tour
        out_dir = BASE_OUT / tour
        out_file = out_dir / "all_matches.csv"

        if not raw_dir.exists():
            raise RuntimeError(f"Raw directory not found: {raw_dir}")

        dfs = []

        files = sorted(
            list(raw_dir.glob("*.xls")) +
            list(raw_dir.glob("*.xlsx")) +
            list(raw_dir.glob("*.csv"))
        )

        if not files:
            raise RuntimeError(f"No tour files found for {tour}")

        print(f"Found {len(files)} files for tour {tour}")

        for f in files:
            print(f"Loading {f.name}")

            if f.suffix == ".csv":
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)

            df.columns = df.columns.str.strip()

            rename = {
                "Winner": "winner",
                "Loser": "loser",
                "Surface": "surface",
                "Date": "date",
                "Tournament": "tournament",
                "WRank": "winner_rank",
                "LRank": "loser_rank"
            }

            df = df.rename(
                columns={k: v for k, v in rename.items() if k in df.columns}
            )

            # Extract year from filename (robustly)
            year_digits = "".join(filter(str.isdigit, f.stem))
            if len(year_digits) >= 4:
                year = int(year_digits[:4])
                df["season"] = year
            else:
                df["season"] = None

            dfs.append(df)

        out = pd.concat(dfs, ignore_index=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_file, index=False)

        print(f"Saved {len(out):,} rows → {out_file}")

if __name__ == "__main__":
    main()
