"""Public room-local decision history, shared by live play and recording.

Never store full observations here: they contain the history itself, the map,
and many repeated entities. Outcomes describe observed snapshots, not a claim
that the animation has finished or that all changes were caused by one action.
"""

from __future__ import annotations

import copy
import re
from typing import Any


ROOM_HISTORY_LIMIT = 30


def room_identity(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in ("act", "floor", "room_type")}


def _room_key(room: dict[str, Any]) -> tuple[Any, Any]:
    return room.get("act"), room.get("floor")


def _entity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    # Stable public identity/effect, not transient choice aliases or Java UUIDs.
    fields = ("id", "name", "upgrades", "type", "cost", "cost_for_turn", "description", "price",
              "final_single_hit_damage", "final_block", "damage", "block", "hit_count",
              "reward_id", "reward_type", "gold", "amount", "x", "y")
    result = {key: value[key] for key in fields if key in value}
    for key in ("card", "relic", "potion"):
        child = _entity(value.get(key))
        if child:
            result[key] = child
    return result or None


def _index(action: dict[str, Any], field: str, parameter: str) -> int | None:
    value = action.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(rf"(?:^|:){parameter}=(-?\d+)(?:$|:)", str(action.get("id") or ""))
    return int(match.group(1)) if match else None


def _at(items: Any, index: int | None) -> Any:
    return items[index] if isinstance(items, list) and index is not None and 0 <= index < len(items) else None


def action_details(pre: dict[str, Any], action: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve objects using the pre-action screen's indices, never the next hand."""
    details: dict[str, Any] = {}
    screen = str(pre.get("screen_type") or "").upper()
    content = pre.get("screen_state") or {}
    combat = pre.get("combat") or {}
    kind = action.get("kind")
    index = _index(action, "index", "index")
    entity = None
    if kind == "play":
        card_index = _index(action, "card_index", "card")
        entity = _at(combat.get("hand"), card_index - 1 if card_index is not None else None)
    elif kind in {"potion_use", "potion_discard"}:
        entity = _at(pre.get("potions"), _index(action, "slot", "slot"))
    elif kind == "choose":
        choice = _at(pre.get("choices"), index)
        if choice is not None:
            details["choice_text"] = choice
        candidates = {
            "GRID": "cards", "CARD_REWARD": "cards", "HAND_SELECT": "hand",
            "COMBAT_REWARD": "rewards", "BOSS_REWARD": "relics",
        }
        if screen in candidates:
            entity = _at(content.get(candidates[screen]), index)
        elif screen == "SHOP_SCREEN":
            # Choice indices do not necessarily match the combined shop lists.
            # Only use an exact, unambiguous public name/ID match.
            matches = [item for key in ("cards", "relics", "potions")
                       for item in content.get(key, []) if isinstance(item, dict)
                       and choice in (item.get("name"), item.get("id"))]
            entity = matches[0] if len(matches) == 1 else None
    if (resolved := _entity(entity)) is not None:
        details["object"] = resolved
    target = _index(action, "target", "target")
    if target is not None:
        monster = _entity(_at(combat.get("monsters"), target)) or {}
        details["target"] = {"index": target, **monster}
    for key in ("selection_mode", "selection_instruction", "selection_reason", "source_reward_id"):
        if content.get(key) is not None:
            details[key] = copy.deepcopy(content[key])
    if isinstance(pre.get("selection_context"), dict):
        details["selection_context"] = copy.deepcopy(pre["selection_context"])
    if kind in {"confirm", "return", "cancel"}:
        selected = content.get("selected_cards", content.get("selected"))
        if isinstance(selected, list):
            details["selected_cards"] = [_entity(card) for card in selected]
        if isinstance(content.get("confirmation_card"), dict):
            details["confirmation_card"] = _entity(content["confirmation_card"])
    # Recorder's semantic flip marker has a stable board position; CHOOSE's
    # index is not necessarily that position (choices shrink during the game).
    if isinstance(raw.get("board_position"), int):
        details["board_position"] = raw["board_position"]
    elif screen == "EVENT":
        choice = _at(pre.get("choices"), index)
        match = re.fullmatch(r"card\s*(\d+)", str(choice or ""), re.IGNORECASE)
        if match:
            details["board_position"] = int(match.group(1))
    return details


def _resources(state: dict[str, Any]) -> dict[str, Any]:
    result = {key: state.get(key) for key in ("current_hp", "max_hp", "gold", "keys")}
    combat = state.get("combat")
    if isinstance(combat, dict) and state.get("room_phase") == "COMBAT":
        result.update({key: combat.get("player", {}).get(key) for key in ("energy", "block", "stance", "powers")})
        result["turn"] = combat.get("turn")
        result["monsters"] = [
            {key: monster.get(key) for key in ("id", "name", "current_hp", "block", "powers", "is_gone")}
            for monster in combat.get("monsters", []) or [] if isinstance(monster, dict)
        ]
    return result


def _entity_counts(state: dict[str, Any], key: str) -> dict[tuple[Any, Any, Any], tuple[dict[str, Any], int]]:
    result: dict[tuple[Any, Any, Any], tuple[dict[str, Any], int]] = {}
    for item in state.get(key, []) or []:
        if key == "potions" and isinstance(item, dict) and str(item.get("id", "")).casefold() == "potion slot":
            continue
        entity = _entity(item)
        if not entity:
            continue
        identity = entity.get("id"), entity.get("name"), entity.get("upgrades")
        count = item.get("count", 1)
        count = count if isinstance(count, int) else 1
        result[identity] = (entity, result.get(identity, ({}, 0))[1] + count)
    return result


def enrich_history_item(item: dict[str, Any], pre: dict[str, Any], post: dict[str, Any],
                        action: dict[str, Any], raw: dict[str, Any] | None = None) -> None:
    item["room_before"] = room_identity(pre)
    item["room_after"] = room_identity(post)
    item["turn_before"] = (pre.get("combat") or {}).get("turn") if pre.get("room_phase") == "COMBAT" else None
    item["turn_after"] = (post.get("combat") or {}).get("turn") if post.get("room_phase") == "COMBAT" else None
    item["details"] = action_details(pre, action, raw if isinstance(raw, dict) else {})
    before, after = _resources(pre), _resources(post)
    changes = {key: {"before": value, "observed_after": after[key]}
               for key, value in before.items()
               if key in after and value != after[key] and value is not None and after[key] is not None}
    # A new room's draw/monster list is not a result of damaging the old enemy.
    if _room_key(item["room_before"]) != _room_key(item["room_after"]):
        changes = {key: value for key, value in changes.items() if key in {"current_hp", "max_hp", "gold", "keys"}}
    item["observed_changes"] = copy.deepcopy(changes)
    if post.get("act") is None or post.get("floor") is None:
        # Returning to a menu has no inventory snapshot, not an empty deck.
        return
    for key in ("deck", "relics", "potions"):
        before_counts, after_counts = _entity_counts(pre, key), _entity_counts(post, key)
        differences = []
        for identity in sorted(before_counts.keys() | after_counts.keys(), key=str):
            old = before_counts.get(identity, ({}, 0))
            new = after_counts.get(identity, ({}, 0))
            if new[1] != old[1]:
                differences.append({"entity": new[0] or old[0], "count_delta": new[1] - old[1]})
        if differences:
            item["observed_changes"][key] = differences


def _event_changes(before: Any, after: Any, details: dict[str, Any]) -> dict[str, Any]:
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    changes = {key: copy.deepcopy(after[key]) for key in ("body_text", "choices", "options")
               if key in after and after[key] != before.get(key)}
    previous = before.get("match_game") or {}
    current = after.get("match_game") or {}
    board_before = {card.get("position"): card for card in previous.get("board", []) or []}
    revealed = []
    position = details.get("board_position")
    for card in current.get("board", []) or []:
        old = board_before.get(card.get("position"), {})
        if card.get("revealed") is True and isinstance(card.get("card"), dict) and (
            card.get("position") == position or old.get("revealed") is not True
            or (card.get("face_up") is True and old.get("face_up") is not True)
        ):
            revealed.append({"position": card.get("position"), "card": _entity(card["card"])})
    if revealed:
        changes["revealed_cards"] = revealed
    for key in ("remaining_attempts", "matched_pairs", "selected_positions", "game_done", "awaiting_resolution"):
        if key in current and current[key] != previous.get(key):
            changes[key] = copy.deepcopy(current[key])
    return changes


def room_history_summary(state: dict[str, Any], history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """One floor is one room even when its UI changes; show its last 30 steps."""
    room = room_identity(state)
    key = _room_key(room)
    entries = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        before, after = item.get("room_before"), item.get("room_after")
        # Legacy records without room metadata remain usable, but cannot
        # establish a reliable room boundary. New live/recorded records can.
        if isinstance(before, dict) and isinstance(after, dict) and key not in {_room_key(before), _room_key(after)}:
            continue
        entries.append(item)
    limit = ROOM_HISTORY_LIMIT
    shown = entries[-limit:]
    steps = []
    current_board = ((state.get("screen_state") or {}).get("match_game") or {}).get("board") or []
    known_cards = {card.get("position"): _entity(card.get("card")) for card in current_board
                   if isinstance(card, dict) and card.get("revealed") is True and isinstance(card.get("card"), dict)}
    opening = None
    for item in entries:
        for phase in ("before", "after"):
            snapshot = item.get(f"event_{phase}")
            if isinstance(snapshot, dict) and _room_key(item.get(f"room_{phase}") or room) == key:
                opening = {field: copy.deepcopy(snapshot.get(field))
                           for field in ("event_id", "event_name", "body_text", "choices", "options")}
                break
        if opening is not None:
            break
    for item in shown:
        entry = {key: copy.deepcopy(item[key]) for key in (
            "step_id", "screen_before", "screen_after", "turn_before", "turn_after", "action_kind", "action_id",
            "action", "source", "execution_status", "details", "observed_changes",
        ) if key in item}
        if isinstance(item.get("room_before"), dict) and _room_key(item["room_before"]) != key:
            entry["role"] = "进入当前房间"
        position = (entry.get("details") or {}).get("board_position")
        if position in known_cards:
            # Recover a slow flip's known face from durable public board memory
            # without pretending that its original post-snapshot was settled.
            entry.setdefault("details", {})["known_card_at_position"] = known_cards[position]
        transitions = _event_changes(item.get("event_before"), item.get("event_after"), item.get("details") or {})
        if transitions:
            entry["event_changes"] = transitions
        steps.append(entry)
    result = {"room": room, "total_steps": len(entries), "shown_steps": len(steps),
              "omitted_steps": len(entries) - len(steps), "limit": limit, "steps": steps}
    if opening:
        result["event_opening"] = opening
    return result
