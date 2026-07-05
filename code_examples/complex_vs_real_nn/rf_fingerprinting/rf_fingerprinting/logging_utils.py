"""Logging and export utilities for RF fingerprinting experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def save_json(path: str, payload: Dict):
    """Save a dictionary to a JSON file.

    Args:
        path: Output JSON file path.
        payload: JSON-serializable dictionary.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_scorecard_csv(path: str, rows: Iterable[Dict[str, object]]):
    """Save experiment scorecard rows to CSV.

    Args:
        path: Output CSV path.
        rows: Iterable of dictionaries with identical keys.
    """
    rows = list(rows)
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
