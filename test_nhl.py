import pandas as pd
df = pd.read_csv("analysis/nhl/data/nhl_player_summary_2014_2024.csv")
print(df["residual"].describe())
