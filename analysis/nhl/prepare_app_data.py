from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 100000
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "data" / "shots_2007-2024.csv"
OUTPUT_FILE = SCRIPT_DIR / "data" / "app_data" / "nhl_shots_2024.csv.gz"

print("Starting NHL data extraction...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Keep only necessary columns to save memory
cols_to_keep = [
    'shotID', 'homeTeamCode', 'awayTeamCode', 'season', 'game_id', 'team',
    'teamCode', 'event', 'goal', 'shotGoalieFroze', 'shotRebound', 'xCord', 
    'yCord', 'shotAngle', 'shotDistance', 'shotType', 'shooterName', 'xGoal',
    'shotWasOnGoal'
]

# Read data in chunks
filtered_chunks = []
for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)):
    if i % 5 == 0:
        print(f"Processed chunk {i}")
    
    # Check if necessary columns exist in the chunk, keep only those that do
    current_cols = [c for c in cols_to_keep if c in chunk.columns]

    # Filter for 'season' == 2024
    if 'season' in chunk.columns:
        chunk_filtered = chunk[chunk['season'] == 2024][current_cols].copy()
        if not chunk_filtered.empty:
            filtered_chunks.append(chunk_filtered)

print("Concatenating chunks...")
nhl_2024 = pd.concat(filtered_chunks, ignore_index=True)

print(f"Loaded {len(nhl_2024)} rows for season 2024.")

print("Calculating SDI...")
# Max values for normalization
nhl_2024['shotDistance'] = pd.to_numeric(nhl_2024['shotDistance'], errors='coerce')
nhl_2024['shotAngle'] = pd.to_numeric(nhl_2024['shotAngle'], errors='coerce')
nhl_2024['shotRebound'] = pd.to_numeric(nhl_2024['shotRebound'], errors='coerce').fillna(0)
nhl_2024['shotGoalieFroze'] = pd.to_numeric(nhl_2024['shotGoalieFroze'], errors='coerce').fillna(0)

max_dist = nhl_2024['shotDistance'].max()
nhl_2024['difficulty_distance'] = (nhl_2024['shotDistance'] / max_dist) * 100

max_angle = nhl_2024['shotAngle'].abs().max()
nhl_2024['difficulty_angle'] = (nhl_2024['shotAngle'].abs() / max_angle) * 100

nhl_2024['difficulty_rebound'] = np.where(nhl_2024['shotRebound'] == 1, 30, 0)
nhl_2024['difficulty_goalie_froze'] = np.where(nhl_2024['shotGoalieFroze'] == 1, 20, 0)

nhl_2024['SDI'] = (nhl_2024['difficulty_distance'] * 0.4 + 
                   nhl_2024['difficulty_angle'] * 0.3 + 
                   nhl_2024['difficulty_rebound'] * 0.2 + 
                   nhl_2024['difficulty_goalie_froze'] * 0.1)

print("Saving to compressed CSV...")
nhl_2024.to_csv(OUTPUT_FILE, index=False, compression='gzip')
print(f"Successfully saved to {OUTPUT_FILE}")
