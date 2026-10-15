from __future__ import annotations

import numpy as np


def devig_array(decimal_odds: np.ndarray | list[float]) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.shape != (3,):
        raise ValueError("expected exactly three decimal odds: HOME, DRAW, AWAY")
    if not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("decimal odds must be finite and > 1.0")
    inv = 1.0 / odds
    return inv / inv.sum()


def devig_decimal_odds(home: float, draw: float, away: float) -> np.ndarray:
    return devig_array([home, draw, away])
