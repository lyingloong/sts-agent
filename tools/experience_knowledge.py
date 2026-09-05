"""Human-audited, entity-scoped experience for the StS static knowledge base.

Experience is intentionally a small replacement list rather than an append-only
memory.  A missing field has the same meaning as ``experience: []``; accepted
updates materialize the field in the tracked JSON snapshot.  This keeps the
initial knowledge diff small while preserving the user-maintained knowledge.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATHS = {
    "card": PROJECT_ROOT / "data" / "knowledge" / "sts1_cards.json",
    "relic": PROJECT_ROOT / "data" / "knowledge" / "sts1_relics.json",
    "potion": PROJECT_ROOT / "data" / "knowledge" / "sts1_potions.json",
    "monster": PROJECT_ROOT / "data" / "knowledge" / "sts1_monsters.json",
    "event": PROJECT_ROOT / "data" / "knowledge" / "sts1_events.json",
    "power": PROJECT_ROOT / "data" / "knowledge" / "sts1_powers.json",
}
AUDIT_LOG_PATH = PROJECT_ROOT / "data" / "knowledge" / "experience_audit.jsonl"
KIND_ALIASES = {"cards": "card", "relics": "relic", "potions": "potion", "monsters": "monster", "events": "event", "powers": "power"}
_KEY_RE = re.compile(r"[^A-Z0-9]+")


class ExperienceError(ValueError):
    """Raised when an entity or proposed experience list is invalid."""


def normalize_kind(value: Any) -> str:
    kind = str(value or "").strip().casefold()
    return KIND_ALIASES.get(kind, kind)


def normalize_entity_key(value: Any) -> str:
    return _KEY_RE.sub("", str(value or "").strip().upper())


def validate_experience(values: Any, max_entries: int = 3) -> list[str]:
    """Normalize a teacher replacement list and enforce the hard size bound."""
    if values is None:
        return []
    if not isinstance(values, list):
        raise ExperienceError("experience must be a JSON array")
    if len(values) > max_entries:
        raise ExperienceError(f"experience contains {len(values)} entries; maximum is {max_entries}")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ExperienceError("each experience entry must be a non-empty string")
        text = " ".join(value.split()).strip()
        if text and text not in result:
            result.append(text)
    return result


def _entities(payload: Any, kind: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get(f"{kind}s", payload)
    else:
        value = payload
    if isinstance(value, dict):
        value = list(value.values())
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _load_payload(kind: str) -> tuple[Path, Any, list[dict[str, Any]]]:
    normalized = normalize_kind(kind)
    path = KNOWLEDGE_PATHS.get(normalized)
    if path is None:
        raise ExperienceError(f"unsupported entity type: {kind}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperienceError(f"cannot read knowledge file: {path}") from exc
    return path, payload, _entities(payload, normalized)


def find_entity(kind: Any, entity_id: Any) -> dict[str, Any] | None:
    """Find by ID or display name, returning a detached copy."""
    normalized_kind = normalize_kind(kind)
    try:
        _path, _payload, entities = _load_payload(normalized_kind)
    except ExperienceError:
        return None
    key = normalize_entity_key(entity_id)
    if not key:
        return None
    for entity in entities:
        if any(normalize_entity_key(entity.get(field)) == key for field in ("id", "name")):
            return copy.deepcopy(entity)
    return None


def get_experience(entity: dict[str, Any] | None) -> list[str]:
    if not isinstance(entity, dict):
        return []
    try:
        return validate_experience(entity.get("experience", []))
    except ExperienceError:
        return []


def public_entity(entity: dict[str, Any], max_chars: int = 12000) -> dict[str, Any]:
    """Keep useful static facts while avoiding move tables and asset metadata."""
    omitted = {"experience", "moves", "flavor", "image_url", "icon", "triggers"}
    result = {key: value for key, value in entity.items() if key not in omitted}
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(encoded) <= max_chars:
        return result
    # Descriptions are the most useful fallback when a rare entity is large.
    return {key: result.get(key) for key in ("id", "name", "type", "description") if result.get(key) is not None}


def entity_profile(kind: Any, entity_id: Any) -> dict[str, Any] | None:
    """Return a compact directly-injectable static profile with experience."""
    entity = find_entity(kind, entity_id)
    if entity is None:
        return None
    result = public_entity(entity)
    values = get_experience(entity)
    if values:
        result["experience"] = values
    result["source"] = "static_knowledge"
    return result


def replace_experience(kind: Any, entity_id: Any, values: Any, max_entries: int = 3) -> list[str]:
    """Atomically replace one entity's experience after human approval."""
    normalized_kind = normalize_kind(kind)
    path, payload, entities = _load_payload(normalized_kind)
    replacement = validate_experience(values, max_entries)
    key = normalize_entity_key(entity_id)
    target = next(
        (entity for entity in entities if any(normalize_entity_key(entity.get(field)) == key for field in ("id", "name"))),
        None,
    )
    if target is None:
        raise ExperienceError(f"unknown {normalized_kind} entity: {entity_id}")
    if replacement:
        target["experience"] = replacement
    else:
        target.pop("experience", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    # Runtime enrichers cache snapshots for fast per-step lookup. Refresh all
    # of them immediately so the next action sees the accepted experience.
    try:
        import card_knowledge
        import item_knowledge
        import monster_knowledge
        import power_knowledge
        card_knowledge._CACHE = None
        item_knowledge._CACHE = None
        monster_knowledge._CACHE = None
        power_knowledge._CACHE = None
    except ImportError:
        pass
    return replacement


def append_audit(record: dict[str, Any]) -> None:
    """Record the approved replacement separately from the compact entity field."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("at", datetime.now(timezone.utc).isoformat())
    with AUDIT_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
