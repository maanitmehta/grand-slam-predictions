import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

from scripts.predict_match import predict_match
from scripts.name_utils import canonical_name

TOURS = ["atp", "wta"]

BASE_PROCESSED = Path("data/processed")
BASE_RESULTS = Path("results")

N_SIM = 10_000
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

def simulate_round(players, round_name, tour, verbose=False):
    winners = []

    for i in range(0, len(players), 2):
        A = players[i]
        B = players[i + 1]

        p = predict_match(A, B, tour=tour)
        winner = A if np.random.rand() < p else B
        winners.append(winner)

        if verbose:
            print(f"{round_name}: {A} vs {B} → {winner} (p={p:.2f})")

    return winners

def simulate_tournament(draw, tour, verbose=False):
    players = []
    for _, row in draw.iterrows():
        players.append(row["player_A"])
        players.append(row["player_B"])

    for r in ROUNDS:
        players = simulate_round(players, r, tour, verbose)
        if verbose:
            print(f"\nWinners after {r}: {players}\n")

    return players[0]

def main():
    for tour in TOURS:
        print(f"\n==============================")
        print(f" Simulating AO 2026 — {tour.upper()}")
        print(f"==============================\n")

        draw_path = BASE_PROCESSED / tour / "ao_2026_draw.csv"
        results_dir = BASE_RESULTS / tour
        results_file = results_dir / "ao_2026_title_probabilities.csv"

        if not draw_path.exists():
            raise RuntimeError(f"Missing draw file: {draw_path}")

        draw = pd.read_csv(draw_path, engine="python")

        draw["player_A"] = draw["player_A"].apply(canonical_name)
        draw["player_B"] = draw["player_B"].apply(canonical_name)

        # -----------------------------
        # Debug: one full tournament
        # -----------------------------
        print("\n DEBUG: ONE FULL TOURNAMENT RUN\n")
        champ = simulate_tournament(draw, tour, verbose=True)
        print(f"\n Champion (single run): {champ}\n")

        # -----------------------------
        # Monte Carlo simulation
        # -----------------------------
        counts = Counter()

        for _ in range(N_SIM):
            champ = simulate_tournament(draw, tour, verbose=False)
            counts[champ] += 1

        results = (
            pd.DataFrame.from_dict(counts, orient="index", columns=["wins"])
            .assign(prob=lambda df: df["wins"] / N_SIM)
            .sort_values("prob", ascending=False)
        )

        print(f"\n AUSTRALIAN OPEN 2026 — {tour.upper()} TITLE PROBABILITIES\n")
        print(results.head(15))

        results_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_file)

        print(f"\nSaved results → {results_file}")

if __name__ == "__main__":
    main()
