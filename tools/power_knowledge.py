"""Static descriptions for public combat Powers.

CommunicationMod supplies the active Power's ID, display name, amount, and
type. This module only supplements its human-readable effect text; live combat
values always remain authoritative.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWER_PATH = PROJECT_ROOT / "data" / "knowledge" / "sts1_powers.json"
CARD_POWER_PATH = PROJECT_ROOT / "data" / "knowledge" / "sts1_card_powers.json"
_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None


def normalize_power_key(value: Any) -> str:
    """Match protocol camel-case IDs, spaced names, and archive snake-case IDs."""
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").strip().upper())


def register(index: dict[str, dict[str, Any]], entry: dict[str, Any]) -> None:
    """Index a static effect under both its ID and display name."""
    if not isinstance(entry.get("description"), str) or not entry["description"].strip():
        return
    for value in (entry.get("id"), entry.get("name")):
        key = normalize_power_key(value)
        if key:
            index.setdefault(key, entry)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load() -> dict[str, dict[str, dict[str, Any]]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    cache = {"powers": {}, "card_powers": {}}
    powers = load_json(POWER_PATH)
    if isinstance(powers, dict):
        powers = powers.get("powers", powers)
    if isinstance(powers, list):
        for entry in powers:
            if isinstance(entry, dict):
                register(cache["powers"], entry)

    card_powers = load_json(CARD_POWER_PATH)
    if isinstance(card_powers, dict):
        card_powers = card_powers.get("card_powers", card_powers)
    if isinstance(card_powers, dict):
        for source_id, entries in card_powers.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                register(cache["card_powers"], entry)
                source_key = normalize_power_key(source_id)
                if source_key and isinstance(entry.get("description"), str) and entry["description"].strip():
                    cache["card_powers"].setdefault(source_key, entry)

    _CACHE = cache
    return _CACHE


def power_description(power: dict[str, Any]) -> str | None:
    """Look up one active Power, preferring canonical Power definitions."""
    cache = _load()
    keys = (normalize_power_key(power.get("id")), normalize_power_key(power.get("name")))
    for source in (cache["powers"], cache["card_powers"]):
        for key in keys:
            entry = source.get(key)
            if entry is not None:
                return str(entry["description"])
    return None
