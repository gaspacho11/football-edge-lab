from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_idx: pd.Index
    test_idx: pd.Index
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def expanding_time_folds(
    df: pd.DataFrame,
    *,
    time_col: str = "kickoff_utc",
    min_train_matches: int = 1000,
    test_days: int = 60,
    step_days: int = 60,
) -> Iterator[WalkForwardFold]:
    if df.empty:
        return

    times = pd.to_datetime(df[time_col], utc=True)
    ordered = df.assign(_time=times).sort_values("_time")
    start_time = ordered.iloc[min_train_matches - 1]["_time"]
    test_start = start_time + pd.Timedelta(microseconds=1)
    max_time = ordered["_time"].max()

    while test_start <= max_time:
        test_end = test_start + pd.Timedelta(days=test_days)
        train_mask = ordered["_time"] < test_start
        test_mask = (ordered["_time"] >= test_start) & (ordered["_time"] < test_end)
        train = ordered.index[train_mask]
        test = ordered.index[test_mask]
        if len(train) >= min_train_matches and len(test) > 0:
            yield WalkForwardFold(
                train_idx=train,
                test_idx=test,
                train_end=ordered.loc[train, "_time"].max(),
                test_start=test_start,
                test_end=test_end,
            )
        test_start = test_start + pd.Timedelta(days=step_days)
