"""Static StS1 card knowledge fallback.

Runtime fields from CommunicationMod remain authoritative. This module supplies
public identity, cost, type, target, numeric values, flags, tags, keywords, and
base/upgraded descriptions when the protocol does not serialize card fields.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sts_agent.paths import PROJECT_ROOT

KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "sts1_cards.json"
_CACHE: dict[str, dict[str, Any]] | None = None


def normalize_card_key(value: Any) -> str:
    """Normalize card IDs across protocol and knowledge-source spellings."""
    text = str(value or "").strip().upper()
    # CommunicationMod and the card snapshot differ mainly on spaces versus
    # underscores, but accepting punctuation makes the lookup robust to older
    # snapshots and modded-card naming conventions too.
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")


def _load() -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {}
    if not KNOWLEDGE_PATH.exists():
        return _CACHE
    try:
        payload = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _CACHE
    cards = payload.get("cards", payload) if isinstance(payload, dict) else payload
    if isinstance(cards, dict):
        for card_id, card in cards.items():
            if isinstance(card, dict):
                key = normalize_card_key(card.get("id", card_id))
                if key:
                    _CACHE[key] = card
    elif isinstance(cards, list):
        for card in cards:
            if isinstance(card, dict) and card.get("id"):
                key = normalize_card_key(card["id"])
                if key:
                    _CACHE[key] = card
    return _CACHE


def card_effect(card: dict[str, Any]) -> dict[str, Any] | None:
    """Return a compact, explicitly static effect description for one card."""
    entry = _load().get(normalize_card_key(card.get("id", "")))
    if entry is None:
        return None
    upgrade = entry.get("upgrade") if isinstance(entry.get("upgrade"), dict) else {}
    upgraded = int(card.get("upgrades", 0) or 0) > 0
    # Preserve the key fields from the static source. They are nested under
    # ``effect`` so static base values cannot be mistaken for live combat
    # values exposed by CommunicationMod.
    effect: dict[str, Any] = {
        "source": "static_knowledge",
        "id": entry.get("id"),
        "name": entry.get("name"),
        "cost": entry.get("cost"),
        "type": entry.get("type"),
        "rarity": entry.get("rarity"),
        "target": entry.get("target"),
        "color": entry.get("color"),
        "description": upgrade.get("description") if upgraded and upgrade.get("description") else entry.get("description"),
        "base_description": entry.get("description"),
        "upgraded_description": upgrade.get("description"),
        "base_damage": entry.get("damage"),
        "base_block": entry.get("block"),
        "base_magic_number": entry.get("magic_number"),
        "base_hit_count": entry.get("hit_count"),
        "exhaust": entry.get("exhaust"),
        "ethereal": entry.get("ethereal"),
        "innate": entry.get("innate"),
        "retain": entry.get("retain"),
        "self_retain": entry.get("self_retain"),
        "purge_on_use": entry.get("purge_on_use"),
        "tags": entry.get("tags", []),
        "keywords": entry.get("keywords", []),
        "upgrade": {
            "cost": upgrade.get("cost"),
            "damage": upgrade.get("damage"),
            "block": upgrade.get("block"),
            "magic_number": upgrade.get("magic_number"),
            "description": upgrade.get("description"),
        },
        "upgrade_damage": upgrade.get("damage"),
        "upgrade_block": upgrade.get("block"),
        "upgrade_magic_number": upgrade.get("magic_number"),
        "experience": entry.get("experience", []),
    }
    effect["upgrade"] = {key: value for key, value in effect["upgrade"].items() if value is not None}
    if not isinstance(effect.get("experience"), list) or not effect["experience"]:
        effect.pop("experience", None)
    return {key: value for key, value in effect.items() if value is not None}
