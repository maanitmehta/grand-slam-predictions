# config/surfaces.py

SURFACE_MAP = {
    "Hard": "hard",
    "Clay": "clay", 
    "Grass": "grass",
    "Carpet": "hard",  # treat carpet as hard
}

SURFACE_TRAINING_FILTER = {
    "hard":  ["hard"],
    "clay":  ["clay"],
    "grass": ["grass"],
}

TOURNAMENT_SURFACE = {
    "ao26": "hard",
    "fo26": "clay",
    "wimbledon": "grass",
}
