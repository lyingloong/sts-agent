"""Static StS1 relic and potion knowledge fallbacks.

Only enriches entities already exposed by CommunicationMod. It never infers a
hidden reward from its type, event text, or the game's random-state internals.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATHS = {
    "relic": PROJECT_ROOT / "data" / "knowledge" / "sts1_relics.json",
    "potion": PROJECT_ROOT / "data" / "knowledge" / "sts1_potions.json",
}
_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None


def normalize_item_key(value: Any) -> str:
    """Normalize CommunicationMod's camel-case IDs and source IDs alike."""
    text = str(value or "").strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def _load() -> dict[str, dict[str, dict[str, Any]]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {kind: {} for kind in KNOWLEDGE_PATHS}
    for kind, path in KNOWLEDGE_PATHS.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entities = payload.get(f"{kind}s", payload) if isinstance(payload, dict) else payload
        if isinstance(entities, dict):
            entities = list(entities.values())
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            for value in (entity.get("id"), entity.get("name")):
                key = normalize_item_key(value)
                if key:
                    _CACHE[kind][key] = entity
    return _CACHE


def _effect(kind: str, item: dict[str, Any]) -> dict[str, Any] | None:
    source = _load()[kind]
    entry = source.get(normalize_item_key(item.get("id"))) or source.get(normalize_item_key(item.get("name")))
    if entry is None:
        return None
    if kind == "relic":
        fields = ("id", "name", "description", "tier", "color")
    else:
        fields = ("id", "name", "description", "rarity", "color", "is_thrown", "target")
    result = {key: entry[key] for key in fields if entry.get(key) is not None}
    return {"source": "static_knowledge"} | result


def relic_effect(relic: dict[str, Any]) -> dict[str, Any] | None:
    return _effect("relic", relic)


def potion_effect(potion: dict[str, Any]) -> dict[str, Any] | None:
    return _effect("potion", potion)
