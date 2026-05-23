import pandas as pd
import numpy as np
import argparse
from collections import Counter
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.predict_match import predict_match
from scripts.name_utils import canonical_name
from config.tournaments import TOURNAMENTS

N_SIM  = 10_000
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

def simulate_round(players, round_name, tour, tournament, verbose=False):
    winners = []
    for i in range(0, len(players), 2):
        A, B = players[i], players[i + 1]
        p = predict_match(A, B, tour=tour, tournament=tournament)
        winner = A if np.random.rand() < p else B
        winners.append(winner)
        if verbose:
            print(f"  {round_name}: {A} vs {B} → {winner} (p={p:.2f})")
    return winners

def simulate_tournament(draw, tour, tournament, verbose=False):
    players = []
    for _, row in draw.iterrows():
        players.append(row["player_A"])
        players.append(row["player_B"])

    for r in ROUNDS:
        if len(players) == 1:
            break
        players = simulate_round(players, r, tour, tournament, verbose)
        if verbose:
            print(f"\n  After {r}: {players}\n")

    return players[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    parser.add_argument("--sims",       type=int, default=N_SIM)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    cfg        = TOURNAMENTS[args.tournament]
    tour       = args.tour
    draw_path  = Path(cfg["draws"][tour])
    result_dir = Path(cfg["results"][tour]).parent
    result_file = Path(cfg["results"][tour])

    print(f"\n{'='*50}")
    print(f"  {cfg['name']} — {tour.upper()}")
    print(f"  Surface: {cfg['surface']}  |  Simulations: {args.sims:,}")
    print(f"{'='*50}\n")

    if not draw_path.exists():
        raise RuntimeError(
            f"Draw file not found: {draw_path}\n"
            f"Add your draw CSV to {draw_path} and re-run."
        )

    draw = pd.read_csv(draw_path, engine="python")
    draw["player_A"] = draw["player_A"].apply(canonical_name)
    draw["player_B"] = draw["player_B"].apply(canonical_name)

    print(f"Draw loaded: {len(draw)} first-round matches\n")

    # One verbose debug run
    if args.verbose:
        print("--- DEBUG: single full run ---")
        champ = simulate_tournament(draw, tour, args.tournament, verbose=True)
        print(f"\nChampion (single run): {champ}\n")
        print("--- Monte Carlo ---\n")

    # Monte Carlo
    counts = Counter()
    for _ in range(args.sims):
        counts[simulate_tournament(draw, tour, args.tournament)] += 1

    results = (
        pd.DataFrame.from_dict(counts, orient="index", columns=["wins"])
        .assign(title_prob=lambda df: df["wins"] / args.sims)
        .sort_values("title_prob", ascending=False)
        .reset_index()
        .rename(columns={"index": "player"})
    )

    print(f"{cfg['name']} — {tour.upper()} TITLE PROBABILITIES\n")
    print(results.head(16).to_string(index=False))

    result_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(result_file, index=False)
    print(f"\nSaved → {result_file}")

if __name__ == "__main__":
    main()