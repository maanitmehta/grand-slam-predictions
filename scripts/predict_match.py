import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.name_utils import canonical_name
from scripts.model_utils import LGBMPipeline  # noqa: F401 — needed for joblib deserialization
from config.tournaments import TOURNAMENTS
from config.surfaces import SURFACE_TRAINING_FILTER

BASE_MODELS = Path("models")
BASE_DATA   = Path("data/processed")
BASE_RAW    = Path("data/raw")

TOURNAMENT_REF_DATES = {
    "ao26": pd.to_datetime("2026-01-14"),
    "fo26": pd.to_datetime("2026-05-25"),
}

_LOG_RANK_DEFAULT = np.log(200)

# Ordinal round encoding — must match 04_build_ml_dataset.py
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

# Module-level caches keyed by (tournament, tour)
_MODELS  = {}
_STATS   = {}
_ELO     = {}
_SURF_WR = {}
_RANKS   = {}
_H2H     = {}
_CACHE   = {}


def _load(tournament: str, tour: str):
    key = (tournament, tour)
    if key in _MODELS:
        return

    cfg      = TOURNAMENTS[tournament]
    surface  = cfg["surface"]
    ref_date = TOURNAMENT_REF_DATES[tournament]

    # Model
    model_path = Path(cfg["models"][tour])
    if not model_path.exists():
        raise RuntimeError(
            f"Missing model: {model_path}\n"
            f"Run: python3 scripts/05_train_model.py --tournament {tournament} --tour {tour}"
        )
    _MODELS[key] = joblib.load(model_path)

    # Rolling stats (all-surfaces for clay/grass, hard-only for hard)
    stats_path = (
        BASE_DATA / tour / "rolling_player_stats.csv"
        if SURFACE_TRAINING_FILTER[surface] == ["hard"]
        else BASE_DATA / tour / "rolling_player_stats_all_surfaces.csv"
    )
    if not stats_path.exists():
        raise RuntimeError(f"Missing stats: {stats_path}")
    stats = pd.read_csv(stats_path, parse_dates=["date"], low_memory=False)
    stats["player"] = stats["player"].apply(canonical_name)
    _STATS[key] = (
        stats[stats["date"] < ref_date]
        .sort_values("date")
        .groupby("player")
        .tail(1)
        .set_index("player")
    )

    # Elo snapshot
    elo_path = BASE_DATA / tour / "elo_snapshot.csv"
    if elo_path.exists():
        elo = pd.read_csv(elo_path)
        elo["player"] = elo["player"].apply(canonical_name)
        _ELO[key] = elo.set_index("player")
    else:
        _ELO[key] = None

    # Surface win rates (most recent before ref_date)
    surf_wr_path = BASE_DATA / tour / f"{surface}_surface_features.csv"
    if surf_wr_path.exists():
        sf  = pd.read_csv(surf_wr_path, parse_dates=["date"])
        sf["Player"] = sf["Player"].apply(canonical_name)
        col = f"{surface}_win_rate_20"
        if col in sf.columns:
            recent = (
                sf[sf["date"] < ref_date]
                .sort_values("date")
                .groupby("Player")
                .last()
            )
            _SURF_WR[key] = recent[col].dropna().to_dict()
        else:
            _SURF_WR[key] = {}
    else:
        _SURF_WR[key] = {}

    # Current rankings (log scale, consistent with training)
    ranks_path = BASE_RAW / f"{tour}_rankings.csv"
    if ranks_path.exists():
        rk = pd.read_csv(ranks_path)
        rk["player"] = rk["player"].apply(canonical_name)
        _RANKS[key] = {row["player"]: np.log(row["rank"]) for _, row in rk.iterrows()}
    else:
        _RANKS[key] = {}

    # H2H clay records (keyed as "min_player|max_player" alphabetically)
    h2h_path = Path(f"data/h2h_{tour}.json")
    if h2h_path.exists():
        with open(h2h_path) as f:
            _H2H[key] = json.load(f)
    else:
        _H2H[key] = {}

    print(f"  Loaded model + stats for {tournament.upper()} {tour.upper()} "
          f"({len(_STATS[key])} players)")


_FALLBACK_ELO = 1500.0


def _elo_prob(a_elo: float, b_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(a_elo - b_elo) / 400.0))


def _get_elo(player, elo_df, col) -> float:
    if elo_df is not None and player in elo_df.index:
        v = elo_df.loc[player, col] if col in elo_df.columns else _FALLBACK_ELO
        return float(v) if pd.notna(v) else _FALLBACK_ELO
    return _FALLBACK_ELO


def safe_diff(x, y, default=0.0):
    try:
        if pd.isna(x) or pd.isna(y):
            return default
        return float(x) - float(y)
    except Exception:
        return default


def predict_match(A, B, tour="atp", tournament="ao26", **kwargs):
    """Return probability that player A beats player B.

    Optional kwargs:
      round_name  str   — e.g. "R64", "QF" — used for round_depth feature
      b365w       float — pre-match B365 odds for player A (winner perspective)
      b365l       float — pre-match B365 odds for player B
    """
    A = canonical_name(A)
    B = canonical_name(B)

    cache_key = (A, B, tour, tournament, kwargs.get("round_name"))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    _load(tournament, tour)

    key     = (tournament, tour)
    stats   = _STATS[key]
    model   = _MODELS[key]
    elo     = _ELO[key]
    surf_wr = _SURF_WR[key]
    ranks   = _RANKS[key]
    h2h     = _H2H[key]
    surface = TOURNAMENTS[tournament]["surface"]

    a_surf   = _get_elo(A, elo, f"elo_{surface}")
    b_surf   = _get_elo(B, elo, f"elo_{surface}")
    a_global = _get_elo(A, elo, "elo_global")
    b_global = _get_elo(B, elo, "elo_global")

    # Fallback when no rolling stats available
    if A not in stats.index or B not in stats.index:
        p = 0.5 * _elo_prob(a_global, b_global) + 0.5 * _elo_prob(a_surf, b_surf)
        p = min(max(p, 0.05), 0.95)
        _CACHE[cache_key] = p
        return p

    a = stats.loc[A]
    b = stats.loc[B]

    rank_a = ranks.get(A, _LOG_RANK_DEFAULT)
    rank_b = ranks.get(B, _LOG_RANK_DEFAULT)

    swr_a = surf_wr.get(A, float("nan"))
    swr_b = surf_wr.get(B, float("nan"))

    # H2H clay diff (A wins on clay minus B wins on clay, all history)
    h2h_clay_diff = 0.0
    h2h_key = f"{min(A, B)}|{max(A, B)}"
    if h2h_key in h2h:
        rec = h2h[h2h_key]
        p1_is_A = rec["p1"] == A or canonical_name(rec["p1"]) == A
        a_clay = rec["clay"]["p1"] if p1_is_A else rec["clay"]["p2"]
        b_clay = rec["clay"]["p2"] if p1_is_A else rec["clay"]["p1"]
        h2h_clay_diff = float(a_clay - b_clay)

    # Rest days diff (positive = A had more rest on average)
    rest_days_diff = safe_diff(
        a.get("avg_rest_days_lastN", float("nan")),
        b.get("avg_rest_days_lastN", float("nan")),
    )

    # Quality-adjusted win rate diff
    quality_winrate_diff = safe_diff(
        a.get("quality_winrate_lastN", float("nan")),
        b.get("quality_winrate_lastN", float("nan")),
    )

    # Market probability diff — use passed odds when available, else neutral 0.0
    mkt_prob_diff = 0.0
    b365w = kwargs.get("b365w")
    b365l = kwargs.get("b365l")
    if b365w is not None and b365l is not None:
        try:
            bw, bl = float(b365w), float(b365l)
            if bw > 1.0 and bl > 1.0:
                inv_a = 1.0 / bw
                inv_b = 1.0 / bl
                mkt_prob_diff = (inv_a / (inv_a + inv_b)) - (inv_b / (inv_a + inv_b))
        except (TypeError, ValueError):
            pass

    # Round depth (default to mid-draw 3 when unspecified)
    round_name  = kwargs.get("round_name", "R32")
    round_depth = float(ROUND_DEPTH_MAP.get(str(round_name), 3))

    # Roland Garros-specific Elo diff (only for RG tournaments)
    is_rg = tournament.startswith("fo") or "roland" in TOURNAMENTS[tournament].get("name", "").lower()
    elo_diff_rg = 0.0
    if is_rg:
        a_rg = _get_elo(A, elo, "elo_roland_garros")
        b_rg = _get_elo(B, elo, "elo_roland_garros")
        elo_diff_rg = a_rg - b_rg

    features = {
        "winrate_diff":          safe_diff(a["winrate_lastN"],       b["winrate_lastN"]),
        "odds_diff":             safe_diff(a["avg_odds_lastN"],       b["avg_odds_lastN"]),
        "matches_diff":          safe_diff(a["matches_played_lastN"], b["matches_played_lastN"]),
        "rank_diff":             rank_b - rank_a,
        "surface_wr_diff":       safe_diff(swr_a, swr_b),
        "elo_diff_global":       a_global - b_global,
        "elo_diff_surface":      a_surf   - b_surf,
        "h2h_clay_diff":         h2h_clay_diff,
        "mkt_prob_diff":         mkt_prob_diff,
        "quality_winrate_diff":  quality_winrate_diff,
        "elo_diff_rg":           elo_diff_rg,
        "clay_winrate_diff":     safe_diff(
            a.get("clay_winrate_lastN", float("nan")),
            b.get("clay_winrate_lastN", float("nan")),
        ),
    }

    X = pd.DataFrame([features])
    # Model selects only the features it was trained on
    try:
        model_cols = list(model.named_steps["scaler"].feature_names_in_)
        X = X[model_cols]
    except (AttributeError, KeyError):
        pass

    p = float(model.predict_proba(X)[0, 1])
    p = min(max(p, 0.05), 0.95)
    _CACHE[cache_key] = p
    return p
