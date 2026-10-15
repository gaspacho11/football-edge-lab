from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "match_id",
    "kickoff_utc",
    "snapshot_utc",
    "home_team",
    "away_team",
    "odds_home",
    "odds_draw",
    "odds_away",
}

RESULT_COLUMNS = {"actual", "home_goals", "away_goals", "result", "final_score"}


def validate_model_frame(df: pd.DataFrame, prediction_mode: bool = True) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    kickoff = pd.to_datetime(df["kickoff_utc"], utc=True)
    snap = pd.to_datetime(df["snapshot_utc"], utc=True)
    if not (snap < kickoff).all():
        bad = df.index[~(snap < kickoff)].tolist()[:10]
        raise ValueError(f"non-prematch snapshots at rows: {bad}")

    if prediction_mode:
        leaked = RESULT_COLUMNS & set(df.columns)
        if leaked:
            raise ValueError(f"result columns forbidden in prediction frame: {sorted(leaked)}")
