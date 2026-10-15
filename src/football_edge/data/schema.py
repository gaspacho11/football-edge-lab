from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

Outcome = Literal["HOME", "DRAW", "AWAY"]


class ProbabilityVector(BaseModel):
    home: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    away: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def sums_to_one(self) -> "ProbabilityVector":
        total = self.home + self.draw + self.away
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"probabilities must sum to 1, got {total}")
        return self

    def as_array(self) -> np.ndarray:
        return np.array([self.home, self.draw, self.away], dtype=float)

    @property
    def argmax(self) -> Outcome:
        return ("HOME", "DRAW", "AWAY")[int(np.argmax(self.as_array()))]  # type: ignore[return-value]


class SnapshotRecord(BaseModel):
    match_id: str
    kickoff_utc: datetime
    snapshot_utc: datetime
    home_team: str
    away_team: str
    market: ProbabilityVector

    @model_validator(mode="after")
    def snapshot_must_be_prematch(self) -> "SnapshotRecord":
        if self.snapshot_utc >= self.kickoff_utc:
            raise ValueError("snapshot_utc must be strictly before kickoff_utc")
        return self
