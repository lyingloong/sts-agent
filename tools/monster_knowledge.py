"""Static, public StS1 monster knowledge used for future-boss planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from item_knowledge import normalize_item_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "sts1_monsters.json"
_CACHE: dict[str, dict[str, Any]] | None = None


def _load() -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {}
    try:
        payload = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _CACHE
    entities = payload.get("monsters", payload) if isinstance(payload, dict) else payload
    if not isinstance(entities, list):
        return _CACHE
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        for value in (entity.get("id"), entity.get("name")):
            key = normalize_item_key(value)
            if key:
                _CACHE[key] = entity
    return _CACHE


def boss_profile(value: Any) -> dict[str, Any] | None:
    """Return a concise public mechanics preview for one known Act boss.

    Damage fields are static reference values, not a prediction of the next
    intent. Current combat intent remains the responsibility of the live game
    state and is intentionally kept separate.
    """
    key = normalize_item_key(value)
    entry = _load().get(key)
    if not isinstance(entry, dict) or str(entry.get("type") or "").casefold() != "boss":
        return None
    moves: list[dict[str, Any]] = []
    for move in entry.get("moves", []):
        if not isinstance(move, dict):
            continue
        item = {
            field: move.get(field)
            for field in ("id", "name", "intent", "damage", "damage_ascension", "hits", "block")
            if move.get(field) is not None
        }
        if item:
            moves.append(item)
    return {
        field: entry.get(field)
        for field in ("id", "name", "act", "min_hp", "max_hp", "min_hp_ascension", "max_hp_ascension")
        if entry.get(field) is not None
    } | {"moves": moves}


def monster_profile(value: Any) -> dict[str, Any] | None:
    """Return static reference mechanics for one known combat enemy.

    This is deliberately separate from the live enemy fields. ``moves`` lists
    possibilities from the knowledge snapshot; the current intent, damage,
    HP, block, and Powers remain sourced from CommunicationMod.
    """
    key = normalize_item_key(value)
    entry = _load().get(key)
    if not isinstance(entry, dict):
        return None
    moves: list[dict[str, Any]] = []
    for move in entry.get("moves", []):
        if not isinstance(move, dict):
            continue
        item = {
            field: move.get(field)
            for field in ("id", "name", "intent", "damage", "damage_ascension", "hits", "block")
            if move.get(field) is not None
        }
        if item:
            moves.append(item)
    result = {
        field: entry.get(field)
        for field in (
            "id",
            "name",
            "type",
            "act",
            "min_hp",
            "max_hp",
            "min_hp_ascension",
            "max_hp_ascension",
        )
        if entry.get(field) is not None
    }
    result["moves"] = moves
    result["source"] = "static_knowledge"
    return result
