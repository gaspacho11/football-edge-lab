from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_edge.features.market import add_market_logit_features

LABEL_TO_INT = {"HOME": 0, "DRAW": 1, "AWAY": 2}


@dataclass
class MarketResidualClassifier:
    """Regularized model that learns a correction around the market baseline.

    Market transformed features are always present; extra columns represent incremental
    information. It is intentionally simple so that any improvement is easier to audit.
    """

    extra_features: list[str]
    C: float = 0.5
    random_state: int = 20261015

    def __post_init__(self) -> None:
        self.market_features = [
            "market_log_home_draw",
            "market_log_away_draw",
            "market_top_prob",
            "market_gap",
            "market_entropy",
        ]
        self.feature_names = self.market_features + list(self.extra_features)
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=self.C,
                max_iter=2000,
                random_state=self.random_state,
            )),
        ])

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = add_market_logit_features(df)
        missing = [c for c in self.feature_names if c not in x.columns]
        if missing:
            raise ValueError(f"missing model features: {missing}")
        return x[self.feature_names].astype(float)

    def fit(self, df: pd.DataFrame, target_col: str = "actual") -> "MarketResidualClassifier":
        if target_col not in df.columns:
            raise ValueError(f"missing target column: {target_col}")
        y = df[target_col].map(LABEL_TO_INT)
        if y.isna().any():
            raise ValueError("target contains invalid outcomes")
        self.pipeline.fit(self._x(df), y.astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.pipeline.predict_proba(self._x(df))
        classes = self.pipeline.named_steps["model"].classes_
        if list(classes) != [0, 1, 2]:
            raise RuntimeError(f"training fold missing outcome class; classes={classes}")
        return p
