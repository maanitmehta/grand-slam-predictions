"""
Pre-compute head-to-head records for every pair of players who appear in the FO26 draw.
Outputs data/h2h_atp.json and data/h2h_wta.json.
Run from the repo root: python3 scripts/build_h2h.py
"""
import pandas as pd
import json
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.name_utils import canonical_name
from config.tournaments import TOURNAMENTS


def _readable(name: str) -> str:
    if name and '.' in name and ' ' in name:
        last, init = name.split(' ', 1)
        return f"{init.replace('.', '')}. {last}"
    return name


def build_h2h(tour: str):
    cfg = TOURNAMENTS['fo26']

    # Draw players (canonical names)
    draw = pd.read_csv(cfg['draws'][tour], engine='python')
    draw['player_A'] = draw['player_A'].apply(canonical_name)
    draw['player_B'] = draw['player_B'].apply(canonical_name)
    draw_players = set(draw['player_A'].tolist() + draw['player_B'].tolist())

    # Full match history
    match_path = Path(f'data/processed/{tour}/all_matches.csv')
    matches = pd.read_csv(match_path, low_memory=False,
                          usecols=['date', 'surface', 'winner', 'loser', 'tournament'])
    matches['winner'] = matches['winner'].apply(canonical_name)
    matches['loser']  = matches['loser'].apply(canonical_name)
    matches['surface'] = matches['surface'].str.strip().str.capitalize()
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')

    # Only keep matches between draw players
    in_draw = matches['winner'].isin(draw_players) & matches['loser'].isin(draw_players)
    relevant = matches[in_draw].sort_values('date', ascending=False).reset_index(drop=True)

    h2h = {}
    players = sorted(draw_players)

    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            mask = (
                ((relevant['winner'] == p1) & (relevant['loser'] == p2)) |
                ((relevant['winner'] == p2) & (relevant['loser'] == p1))
            )
            head = relevant[mask]
            if head.empty:
                continue

            clay = head[head['surface'] == 'Clay']

            p1_overall = int((head['winner'] == p1).sum())
            p2_overall = int((head['winner'] == p2).sum())
            p1_clay    = int((clay['winner'] == p1).sum())
            p2_clay    = int((clay['winner'] == p2).sum())

            last_matches = []
            for _, row in head.head(5).iterrows():
                last_matches.append({
                    'date':       str(row['date'])[:10],
                    'surface':    str(row['surface']),
                    'winner':     _readable(str(row['winner'])),
                    'tournament': str(row['tournament']) if pd.notna(row['tournament']) else '',
                })

            key = f"{p1}|{p2}"
            h2h[key] = {
                'p1': _readable(p1), 'p2': _readable(p2),
                'overall': {'p1': p1_overall, 'p2': p2_overall},
                'clay':    {'p1': p1_clay,    'p2': p2_clay},
                'last':    last_matches,
            }

    print(f"  {tour.upper()}: {len(h2h)} H2H pairs found from {len(relevant)} relevant matches")
    return h2h


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    Path('data').mkdir(exist_ok=True)

    for tour in ('atp', 'wta'):
        print(f"Building {tour.upper()} H2H…")
        data = build_h2h(tour)
        out = Path(f'data/h2h_{tour}.json')
        with open(out, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        size = out.stat().st_size / 1024
        print(f"  Saved → {out} ({size:.0f} KB, {len(data)} pairs)\n")
