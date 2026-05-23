import pandas as pd

stats = pd.read_csv(
    "data/processed/rolling_player_stats.csv",
    engine="python"
)

# Long-run average winrate per player
strength = (
    stats.groupby("player")["winrate_last10"]
    .mean()
    .clip(0.35, 0.75)   # prevent extremes
)

strength.to_csv("data/processed/player_strength.csv")
print("Saved player_strength.csv")
