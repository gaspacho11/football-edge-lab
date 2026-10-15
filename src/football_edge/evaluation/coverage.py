from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_TO_INT = {"HOME": 0, "DRAW": 1, "AWAY": 2}


def accuracy_at_coverage(actual: pd.Series, probs: np.ndarray, coverages=None) -> pd.DataFrame:
    if coverages is None:
        coverages = np.arange(0.1, 1.01, 0.1)
    y = actual.map(LABEL_TO_INT).to_numpy(dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    order = np.argsort(-conf)

    rows = []
    n = len(y)
    for c in coverages:
        k = max(1, int(np.ceil(float(c) * n)))
        idx = order[:k]
        rows.append({
            "coverage": k / n,
            "n": k,
            "accuracy": float(np.mean(pred[idx] == y[idx])),
            "mean_confidence": float(np.mean(conf[idx])),
            "min_confidence": float(np.min(conf[idx])),
        })
    return pd.DataFrame(rows)
