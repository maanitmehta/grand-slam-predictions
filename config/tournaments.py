# config/tournaments.py

TOURNAMENTS = {
    "ao26": {
        "name": "Australian Open 2026",
        "surface": "hard",
        "draw_size": 128,
        "draws":   {"atp": "data/draws/ao26/atp_draw.csv",
                    "wta": "data/draws/ao26/wta_draw.csv"},
        "models":  {"atp": "models/ao26/atp_model.pkl",
                    "wta": "models/ao26/wta_model.pkl"},
        "results": {"atp": "results/ao26/atp_probs.csv",
                    "wta": "results/ao26/wta_probs.csv"},
    },
    "fo26": {
        "name": "French Open 2026",
        "surface": "clay",
        "draw_size": 128,
        "draws":   {"atp": "data/draws/fo26/atp_draw.csv",
                    "wta": "data/draws/fo26/wta_draw.csv"},
        "models":  {"atp": "models/fo26/atp_model.pkl",
                    "wta": "models/fo26/wta_model.pkl"},
        "results": {"atp": "results/fo26/atp_probs.csv",
                    "wta": "results/fo26/wta_probs.csv"},
    },
}