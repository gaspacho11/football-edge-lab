from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _canonical_csv_bytes(df: pd.DataFrame) -> bytes:
    ordered = df.copy()
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    if "match_id" in ordered.columns:
        ordered = ordered.sort_values("match_id")
    return ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")


def freeze_dataframe(df: pd.DataFrame, output_dir: str | Path, protocol: str) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = _canonical_csv_bytes(df)
    digest = hashlib.sha256(payload).hexdigest()
    csv_path = output / "frozen_predictions.csv"
    csv_path.write_bytes(payload)

    manifest = {
        "protocol": protocol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "sha256": digest,
        "file": csv_path.name,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_freeze(output_dir: str | Path) -> bool:
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    payload = (output / manifest["file"]).read_bytes()
    return hashlib.sha256(payload).hexdigest() == manifest["sha256"]
