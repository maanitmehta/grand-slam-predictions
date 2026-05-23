import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.name_utils import canonical_name
from config.tournaments import TOURNAMENTS
from config.surfaces import SURFACE_MAP, SURFACE_TRAINING_FILTER

BASE_PROCESSED = Path("data/processed")
BASE_RAW       = Path("data/raw")
ELO_INIT       = 1500.0

# Ordinal round depth — later rounds carry more signal (stronger field, higher stakes)
ROUND_DEPTH_MAP = {
    "R128": 1, "Round 1": 1, "1st Round": 1,
    "R64":  2, "Round 2": 2, "2nd Round": 2,
    "R32":  3, "Round 3": 3, "3rd Round": 3,
    "R16":  4, "Round 4": 4, "4th Round": 4,
    "QF":   5, "Quarterfinals": 5,
    "SF":   6, "Semifinals": 6,
    "F":    7, "The Final": 7, "Final": 7,
    "RR":   2,
}

RG_NAMES = ["roland garros", "french open", "roland-garros"]

def _is_rg(tournament_name) -> bool:
    if pd.isna(tournament_name):
        return False
    return any(s in str(tournament_name).lower() for s in RG_NAMES)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", required=True, choices=TOURNAMENTS.keys())
    parser.add_argument("--tour",       required=True, choices=["atp", "wta"])
    args = parser.parse_args()

    cfg              = TOURNAMENTS[args.tournament]
    surface          = cfg["surface"]
    allowed_surfaces = SURFACE_TRAINING_FILTER[surface]
    tour             = args.tour

    print(f"\n=== Building ML dataset for {tour.upper()} | {cfg['name']} | surface: {surface} ===")

    out_file      = BASE_PROCESSED / tour / f"ml_dataset_{args.tournament}.csv"
    rankings_path = BASE_RAW / f"{tour}_rankings.csv"

    # Use all-surfaces stats for clay/grass, hard-only for hard
    if allowed_surfaces == ["hard"]:
        stats_path  = BASE_PROCESSED / tour / "rolling_player_stats.csv"
        matches_src = BASE_PROCESSED / tour / "model_base.csv"
    else:
        stats_path  = BASE_PROCESSED / tour / "rolling_player_stats_all_surfaces.csv"
        matches_src = BASE_PROCESSED / tour / "all_matches.csv"

    for p in [stats_path, rankings_path, matches_src]:
        if not p.exists():
            raise RuntimeError(f"Missing file: {p}")

    # Load matches
    if allowed_surfaces == ["hard"]:
        matches = pd.read_csv(matches_src, parse_dates=["date"])
    else:
        matches = pd.read_csv(matches_src, low_memory=False, parse_dates=["date"])
        matches = matches.rename(columns={"B365W": "b365w", "B365L": "b365l"})

    # Normalise and filter by surface
    matches["surface_canonical"] = matches["surface"].map(SURFACE_MAP).fillna("hard")
    matches = matches[matches["surface_canonical"].isin(allowed_surfaces)].copy()
    print(f"  Matches after surface filter {allowed_surfaces}: {len(matches):,}")

    if len(matches) == 0:
        raise RuntimeError("No matches after surface filter — check SURFACE_MAP values.")

    # Load surface rolling features
    surf_feats = None
    surf_feat_path = BASE_PROCESSED / tour / f"{surface}_surface_features.csv"
    if surf_feat_path.exists():
        surf_feats = pd.read_csv(surf_feat_path, parse_dates=["date"])
        print(f"  Loaded surface features: {len(surf_feats):,} rows")

    # Load Elo history
    elo_hist = None
    elo_path = BASE_PROCESSED / tour / "elo_history.csv"
    if elo_path.exists():
        elo_hist = pd.read_csv(elo_path, parse_dates=["date"])
        print(f"  Loaded Elo history: {len(elo_hist):,} rows")
    else:
        print("  No Elo history found — run 08_build_elo_ratings.py to add Elo features")

    # Rankings
    rankings = (
        pd.read_csv(rankings_path)
        .sort_values("rank")
        .drop_duplicates("player", keep="first")
        .assign(log_rank=lambda df: np.log(df["rank"]))
        .set_index("player")
    )

    stats = pd.read_csv(stats_path, parse_dates=["date"], low_memory=False)
    stat_cols = ["player", "date", "winrate_lastN", "avg_odds_lastN", "matches_played_lastN"]
    for opt_col in ("avg_sets_lastN", "avg_rest_days_lastN", "quality_winrate_lastN", "clay_winrate_lastN"):
        if opt_col in stats.columns:
            stat_cols.append(opt_col)
    stats = stats[stat_cols]

    # Sort by date for running H2H (no lookahead)
    matches = matches.sort_values("date").reset_index(drop=True)

    # Running clay H2H: keyed by (min(p1,p2), max(p1,p2)) → {p1: wins, p2: wins}
    h2h_clay: dict = {}

    def _h2h_diff(w: str, l: str) -> float:
        key = (min(w, l), max(w, l))
        if key not in h2h_clay:
            return 0.0
        rec = h2h_clay[key]
        return float(rec.get(w, 0) - rec.get(l, 0))

    def _h2h_update(w: str, l: str) -> None:
        key = (min(w, l), max(w, l))
        if key not in h2h_clay:
            h2h_clay[key] = {}
        h2h_clay[key][w] = h2h_clay[key].get(w, 0) + 1
        h2h_clay[key].setdefault(l, 0)

    # Determine whether this tournament is Roland Garros (for RG-specific Elo)
    is_rg_tournament = args.tournament.startswith("fo") or "roland" in cfg.get("name", "").lower()

    rows = []

    for _, m in matches.iterrows():
        date = m["date"]
        w = canonical_name(m["winner"])
        l = canonical_name(m["loser"])

        w_stats = stats[(stats["player"] == w) & (stats["date"] == date)]
        l_stats = stats[(stats["player"] == l) & (stats["date"] == date)]

        if len(w_stats) != 1 or len(l_stats) != 1:
            continue

        w_stats = w_stats.iloc[0]
        l_stats = l_stats.iloc[0]

        if pd.isna(w_stats["winrate_lastN"]) or pd.isna(l_stats["winrate_lastN"]):
            continue

        rankA = float(rankings.loc[w, "log_rank"]) if w in rankings.index else np.log(200)
        rankB = float(rankings.loc[l, "log_rank"]) if l in rankings.index else np.log(200)

        # Surface win rate diff
        surf_diff = 0.0
        if surf_feats is not None:
            col = f"{surface}_win_rate_20"
            w_sf = surf_feats[(surf_feats["Player"] == w) & (surf_feats["date"] == date)]
            l_sf = surf_feats[(surf_feats["Player"] == l) & (surf_feats["date"] == date)]
            w_sr = w_sf.iloc[0][col] if len(w_sf) == 1 else np.nan
            l_sr = l_sf.iloc[0][col] if len(l_sf) == 1 else np.nan
            if not pd.isna(w_sr) and not pd.isna(l_sr):
                surf_diff = w_sr - l_sr

        # Elo diffs (pre-match ratings, winner perspective)
        elo_diff_global  = 0.0
        elo_diff_surface = 0.0
        elo_diff_rg      = 0.0
        if elo_hist is not None:
            # Look up winner's pre-match Elo
            w_row = elo_hist[(elo_hist["winner"] == w) & (elo_hist["date"] == date)]
            if len(w_row) == 0:
                w_row = elo_hist[(elo_hist["loser"] == w) & (elo_hist["date"] == date)]
                if len(w_row) > 0:
                    w_global = w_row.iloc[0]["l_elo_global"]
                    w_surf   = w_row.iloc[0].get(f"l_elo_{surface}", ELO_INIT)
                    w_rg     = w_row.iloc[0].get("l_elo_roland_garros", ELO_INIT)
                else:
                    w_global = w_surf = w_rg = ELO_INIT
            else:
                w_global = w_row.iloc[0]["w_elo_global"]
                w_surf   = w_row.iloc[0].get(f"w_elo_{surface}", ELO_INIT)
                w_rg     = w_row.iloc[0].get("w_elo_roland_garros", ELO_INIT)

            # Look up loser's pre-match Elo
            l_row = elo_hist[(elo_hist["loser"] == l) & (elo_hist["date"] == date)]
            if len(l_row) == 0:
                l_row = elo_hist[(elo_hist["winner"] == l) & (elo_hist["date"] == date)]
                if len(l_row) > 0:
                    l_global = l_row.iloc[0]["w_elo_global"]
                    l_surf   = l_row.iloc[0].get(f"w_elo_{surface}", ELO_INIT)
                    l_rg     = l_row.iloc[0].get("w_elo_roland_garros", ELO_INIT)
                else:
                    l_global = l_surf = l_rg = ELO_INIT
            else:
                l_global = l_row.iloc[0]["l_elo_global"]
                l_surf   = l_row.iloc[0].get(f"l_elo_{surface}", ELO_INIT)
                l_rg     = l_row.iloc[0].get("l_elo_roland_garros", ELO_INIT)

            elo_diff_global  = float(w_global) - float(l_global)
            elo_diff_surface = float(w_surf)   - float(l_surf)
            if is_rg_tournament:
                elo_diff_rg = float(w_rg) - float(l_rg)

        # H2H on clay (running — only previous matches, no lookahead)
        h2h_diff = _h2h_diff(w, l)
        _h2h_update(w, l)

        # Rest days diff (positive = winner had more recent rest)
        rest_days_diff = 0.0
        if "avg_rest_days_lastN" in w_stats.index and "avg_rest_days_lastN" in l_stats.index:
            w_rest = w_stats.get("avg_rest_days_lastN", np.nan)
            l_rest = l_stats.get("avg_rest_days_lastN", np.nan)
            if pd.notna(w_rest) and pd.notna(l_rest):
                rest_days_diff = float(w_rest) - float(l_rest)

        # Quality-adjusted win rate diff
        quality_wr_diff = 0.0
        if "quality_winrate_lastN" in w_stats.index and "quality_winrate_lastN" in l_stats.index:
            w_qwr = w_stats.get("quality_winrate_lastN", np.nan)
            l_qwr = l_stats.get("quality_winrate_lastN", np.nan)
            if pd.notna(w_qwr) and pd.notna(l_qwr):
                quality_wr_diff = float(w_qwr) - float(l_qwr)

        # Clay-specific rolling win rate diff (recent clay form only)
        clay_wr_diff = 0.0
        if "clay_winrate_lastN" in w_stats.index and "clay_winrate_lastN" in l_stats.index:
            w_cwr = w_stats.get("clay_winrate_lastN", np.nan)
            l_cwr = l_stats.get("clay_winrate_lastN", np.nan)
            if pd.notna(w_cwr) and pd.notna(l_cwr):
                clay_wr_diff = float(w_cwr) - float(l_cwr)

        # Market probability diff — current match pre-match odds (0.0 when unavailable)
        b365w = pd.to_numeric(m.get("b365w", np.nan), errors="coerce")
        b365l = pd.to_numeric(m.get("b365l", np.nan), errors="coerce")
        mkt_prob_diff = 0.0
        if pd.notna(b365w) and pd.notna(b365l) and b365w > 1.0 and b365l > 1.0:
            inv_w = 1.0 / b365w
            inv_l = 1.0 / b365l
            mkt_prob_diff = (inv_w / (inv_w + inv_l)) - (inv_l / (inv_w + inv_l))

        # Round depth — later rounds = better opponents = stronger signal from rating features
        match_round  = m.get("Round", np.nan)
        round_depth  = float(ROUND_DEPTH_MAP.get(str(match_round), 3))

        match_series = m.get("Series", None) or m.get("Tier", np.nan)
        w_rank_raw   = pd.to_numeric(m.get("winner_rank", np.nan), errors="coerce")
        l_rank_raw   = pd.to_numeric(m.get("loser_rank",  np.nan), errors="coerce")
        max_rank     = max(w_rank_raw, l_rank_raw) if pd.notna(w_rank_raw) and pd.notna(l_rank_raw) else np.nan

        base = {
            "winrate_diff":        w_stats["winrate_lastN"]       - l_stats["winrate_lastN"],
            "odds_diff":           w_stats["avg_odds_lastN"]       - l_stats["avg_odds_lastN"],
            "matches_diff":        w_stats["matches_played_lastN"] - l_stats["matches_played_lastN"],
            "rank_diff":           rankB - rankA,
            "surface_wr_diff":     surf_diff,
            "elo_diff_global":     elo_diff_global,
            "elo_diff_surface":    elo_diff_surface,
            "h2h_clay_diff":       h2h_diff,
            "mkt_prob_diff":       mkt_prob_diff,
            "round_depth":         round_depth,
            "rest_days_diff":      rest_days_diff,
            "quality_winrate_diff": quality_wr_diff,
            "elo_diff_rg":         elo_diff_rg,
            "clay_winrate_diff":   clay_wr_diff,
            # metadata (not features)
            "match_round":         match_round,
            "match_series":        match_series,
            "max_rank":            max_rank,
            "match_winner":        w,
            "match_loser":         l,
            "match_date":          str(date)[:10],
            "b365w":               b365w,
            "b365l":               b365l,
        }

        # Winner row
        rows.append({**base, "a_wins": 1})

        # Loser row (symmetric — flip all diffs)
        rows.append({**base,
            "winrate_diff":        -base["winrate_diff"],
            "odds_diff":           -base["odds_diff"],
            "matches_diff":        -base["matches_diff"],
            "rank_diff":           -base["rank_diff"],
            "surface_wr_diff":     -base["surface_wr_diff"],
            "elo_diff_global":     -base["elo_diff_global"],
            "elo_diff_surface":    -base["elo_diff_surface"],
            "h2h_clay_diff":       -base["h2h_clay_diff"],
            "mkt_prob_diff":       -base["mkt_prob_diff"],
            "round_depth":          base["round_depth"],   # same for both players
            "rest_days_diff":      -base["rest_days_diff"],
            "quality_winrate_diff": -base["quality_winrate_diff"],
            "elo_diff_rg":         -base["elo_diff_rg"],
            "clay_winrate_diff":   -base["clay_winrate_diff"],
            "a_wins": 0,
        })

    # Core features required for a row to be kept
    core_feature_cols = [
        "winrate_diff", "odds_diff", "matches_diff", "rank_diff",
        "surface_wr_diff", "elo_diff_global", "elo_diff_surface",
        "h2h_clay_diff",
    ]
    df = pd.DataFrame(rows).dropna(subset=core_feature_cols)
    df["a_wins"] = df["a_wins"].astype(int)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved ML dataset → {out_file}")
    print(f"Rows: {len(df):,}")
    print("Class balance:")
    print(df["a_wins"].value_counts(normalize=True))
    mkt_coverage = df["mkt_prob_diff"].abs().gt(0).mean()
    print(f"mkt_prob_diff coverage (non-zero rows): {mkt_coverage:.1%}")

if __name__ == "__main__":
    main()
