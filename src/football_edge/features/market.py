from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.models.market_baseline import devig_array


def add_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    probs = np.vstack([
        devig_array(row)
        for row in out[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    ])
    out[["market_home", "market_draw", "market_away"]] = probs
    return out


def add_market_logit_features(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    out = add_market_probabilities(df)
    p = out[["market_home", "market_draw", "market_away"]].to_numpy(dtype=float)
    p = np.clip(p, eps, 1 - eps)
    out["market_log_home_draw"] = np.log(p[:, 0] / p[:, 1])
    out["market_log_away_draw"] = np.log(p[:, 2] / p[:, 1])
    out["market_top_prob"] = p.max(axis=1)
    out["market_gap"] = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
    out["market_entropy"] = -(p * np.log(p)).sum(axis=1) / np.log(3)
    return out


def build_movement_features(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Build first/last market movement features for each match.

    Input must contain multiple pre-match snapshots per match. No post-kickoff data is used.
    """
    x = add_market_probabilities(snapshot_df)
    x["snapshot_utc"] = pd.to_datetime(x["snapshot_utc"], utc=True)
    x = x.sort_values(["match_id", "snapshot_utc"])

    rows = []
    for match_id, g in x.groupby("match_id", sort=False):
        first = g.iloc[0]
        last = g.iloc[-1]
        rows.append({
            "match_id": match_id,
            "market_move_home": last.market_home - first.market_home,
            "market_move_draw": last.market_draw - first.market_draw,
            "market_move_away": last.market_away - first.market_away,
            "market_odds_move_home_log": np.log(last.odds_home / first.odds_home),
            "market_odds_move_draw_log": np.log(last.odds_draw / first.odds_draw),
            "market_odds_move_away_log": np.log(last.odds_away / first.odds_away),
            "snapshot_count": int(len(g)),
        })
    return pd.DataFrame(rows)
