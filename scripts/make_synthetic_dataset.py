from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1600
kickoff = pd.date_range("2022-01-01", periods=n, freq="6h", tz="UTC")
raw = rng.dirichlet([2.5, 1.5, 2.0], size=n)
margin = 1.06
implied = raw * margin
odds = 1 / implied
actual = ["HOME", "DRAW", "AWAY"]
y = np.array([rng.choice(3, p=p) for p in raw])

frame = pd.DataFrame({
    "match_id": [f"m{i:05d}" for i in range(n)],
    "kickoff_utc": kickoff,
    "snapshot_utc": kickoff - pd.Timedelta(hours=3),
    "home_team": [f"H{i%30}" for i in range(n)],
    "away_team": [f"A{i%30}" for i in range(n)],
    "odds_home": odds[:,0],
    "odds_draw": odds[:,1],
    "odds_away": odds[:,2],
    "source_a_home": np.clip(raw[:,0] + rng.normal(0,.025,n), .01,.98),
    "source_a_draw": np.clip(raw[:,1] + rng.normal(0,.025,n), .01,.98),
    "source_a_away": np.clip(raw[:,2] + rng.normal(0,.025,n), .01,.98),
    "actual": np.array(actual)[y],
})
frame.to_parquet("synthetic_history.parquet", index=False)
print("wrote synthetic_history.parquet", frame.shape)
