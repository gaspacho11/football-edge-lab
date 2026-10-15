from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

LABEL_TO_INT = {"HOME": 0, "DRAW": 1, "AWAY": 2}


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.eye(3)[y_true.astype(int)]
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def score_predictions(actual: pd.Series, probs: np.ndarray) -> dict[str, float]:
    y = actual.map(LABEL_TO_INT).to_numpy(dtype=int)
    pred = np.argmax(probs, axis=1)
    return {
        "n": float(len(y)),
        "logloss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, probs),
        "accuracy": float(np.mean(pred == y)),
        "mean_confidence": float(np.mean(np.max(probs, axis=1))),
    }


def market_prob_matrix(df: pd.DataFrame) -> np.ndarray:
    from football_edge.features.market import add_market_probabilities

    x = add_market_probabilities(df)
    return x[["market_home", "market_draw", "market_away"]].to_numpy(dtype=float)
