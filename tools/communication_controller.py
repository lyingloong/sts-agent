"""Human controller for CommunicationMod.

CommunicationMod owns this process' stdin/stdout, so terminal input cannot be
read from input().  The companion human_terminal.py connects over localhost
and supplies semantic actions.  This process keeps stdout strictly reserved
for the CommunicationMod protocol.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from agent_config import ConfigError, load_agent_config
from card_knowledge import card_effect
from item_knowledge import potion_effect, relic_effect


ACTION_HISTORY_LIMIT = 6


PLAYER_CLASSES = {
    "IRONCLAD": "IRONCLAD",
    "SILENT": "THE_SILENT",
    "THE_SILENT": "THE_SILENT",
    "DEFECT": "DEFECT",
    "WATCHER": "WATCHER",
}
MAX_ASCENSION = 20


def normalize_action_logprobs(items: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Normalize raw action log probabilities over the current legal set."""
    parsed: list[dict[str, Any]] = []
    for item in items:
        try:
            value = float(item.get("logprob"))
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        parsed.append(dict(item) | {"logprob": value})
    if not parsed:
        return None
    maximum = max(item["logprob"] for item in parsed)
    weights = [math.exp(item["logprob"] - maximum) for item in parsed]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        return None
    result = [item | {"probability": weight / total} for item, weight in zip(parsed, weights)]
    return sorted(result, key=lambda item: float(item["probability"]), reverse=True)


def confidence_summary(
    action_scores: list[dict[str, Any]],
    min_probability: float,
) -> dict[str, Any]:
    """Return reproducible routing statistics for an already sorted policy."""
    top_probability = float(action_scores[0]["probability"])
    second_probability = float(action_scores[1]["probability"]) if len(action_scores) > 1 else 0.0
    margin = top_probability - second_probability
    entropy = -sum(
        float(item["probability"]) * math.log(float(item["probability"]))
        for item in action_scores
        if float(item["probability"]) > 0
    )
    # Scores are softmax-normalized over the legal action set, so once p1
    # reaches the configured threshold, a separate p1-p2 floor is redundant.
    route = "auto_execute" if top_probability >= min_probability else "human_confirm"
    return {
        "top_probability": top_probability,
        "second_probability": second_probability,
        "margin": margin,
        "entropy": entropy,
        "min_probability": min_probability,
        "route": route,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_id(value: Any) -> str:
    return str(value or "")


def is_slot(potion: dict[str, Any]) -> bool:
    return str(potion.get("id", "")).lower() == "potion slot"


def potion_inventory_full(state: dict[str, Any]) -> bool:
    """Return whether the three visible potion slots are all occupied."""
    potions = state.get("potions")
    if not isinstance(potions, list) or len(potions) < 3:
        return False
    return not any(isinstance(item, dict) and is_slot(item) for item in potions)


def blocked_potion_reward_indices(message: dict[str, Any]) -> set[int]:
    """Find reward/shop choices that would silently stall with full potions."""
    state = game_state(message)
    if not potion_inventory_full(state):
        return set()
    screen = str(state.get("screen_type") or "").upper()
    choices = state.get("choice_list")
    if not isinstance(choices, list):
        return set()
    screen_state = state.get("screen_state")
    if not isinstance(screen_state, dict):
        screen_state = {}

    blocked: set[int] = set()
    if screen == "COMBAT_REWARD":
        rewards = screen_state.get("rewards")
        if isinstance(rewards, list):
            for index, reward in enumerate(rewards):
                if isinstance(reward, dict) and str(reward.get("reward_type", "")).upper() == "POTION":
                    blocked.add(index)
        # Older CommunicationMod versions only expose choice_list.
        blocked.update(
            index
            for index, label in enumerate(choices)
            if str(label).strip().lower() in {"potion", "药水"}
        )
    elif screen == "SHOP_SCREEN":
        potion_names = {
            str(item.get("name"))
            for item in screen_state.get("potions", [])
            if isinstance(item, dict) and item.get("name")
        }
        blocked.update(index for index, label in enumerate(choices) if str(label) in potion_names)
    return blocked


def blocked_potion_choice_error(message: dict[str, Any], index: int) -> str | None:
    if index in blocked_potion_reward_indices(message):
        return "药水栏已满，不能拾取或购买药水；请改选其他奖励，或先丢弃一瓶药水。"
    return None


def command_set(message: dict[str, Any]) -> set[str]:
    return {str(item).lower() for item in message.get("available_commands", [])}


def game_state(message: dict[str, Any]) -> dict[str, Any]:
    value = message.get("game_state")
    return value if isinstance(value, dict) else {}


def visible_card(card: dict[str, Any]) -> dict[str, Any]:
    """Keep card facts visible to the player; drop per-run UUIDs."""
    # CommunicationMod versions differ slightly. Keep effect fields only when
    # the protocol actually exposes them; never copy UUIDs or hidden objects.
    keys = (
        "name",
        "id",
        "type",
        "cost",
        "upgrades",
        "ethereal",
        "exhausts",
        "rarity",
        "is_playable",
        "has_target",
        "damage",
        "block",
        "magic_number",
        "base_damage",
        "base_block",
        "base_magic_number",
        "description",
        "raw_description",
    )
    result = {key: card.get(key) for key in keys if key in card}
    runtime_keys = {
        "damage",
        "block",
        "magic_number",
        "base_damage",
        "base_block",
        "base_magic_number",
        "hit_count",
        "description",
        "raw_description",
    }
    # A protocol key with a null value is not useful evidence that the runtime
    # knows the effect; keep the static value in that case.
    runtime_present = any(card.get(key) is not None for key in runtime_keys)
    static_effect = card_effect(card)
    if static_effect is not None:
        if runtime_present:
            static_effect = dict(static_effect)
            static_effect["source"] = "runtime_over_static"
            for key in runtime_keys:
                if card.get(key) is not None:
                    static_effect[key] = card[key]
        result["effect"] = static_effect
    return result


def card_effect_summary(card: dict[str, Any]) -> str:
    """Render the structured card knowledge compactly for the human terminal."""
    effect = card.get("effect") if isinstance(card.get("effect"), dict) else visible_card(card).get("effect")
    if not isinstance(effect, dict):
        return ""
    fields = []
    for key, label in (
        ("type", "type"),
        ("rarity", "rarity"),
        ("target", "target"),
        ("color", "color"),
    ):
        if effect.get(key) is not None:
            fields.append(f"{label}={effect[key]}")
    for key, label in (
        ("damage", "dmg"),
        ("block", "block"),
        ("magic_number", "magic"),
        ("hit_count", "hits"),
    ):
        value = effect.get(key)
        if value is None:
            value = effect.get(f"base_{key}")
        if value is not None:
            fields.append(f"{label}={value}")
    for key in ("exhaust", "ethereal", "innate", "retain", "self_retain", "purge_on_use"):
        if effect.get(key):
            fields.append(key)
    if card.get("upgrades"):
        fields.append(f"upgraded={card['upgrades']}")
    description = effect.get("description") or effect.get("raw_description")
    text = " | " + " ".join(fields) if fields else ""
    if description:
        text += f" | {str(description).replace(chr(10), ' / ')}"
    return text


def visible_potion(potion: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "id", "description", "can_use", "can_discard", "requires_target")
    result = {key: potion.get(key) for key in keys if key in potion}
    effect = potion_effect(potion)
    if effect is not None:
        if potion.get("description") is not None:
            effect = dict(effect) | {"source": "runtime_over_static", "description": potion["description"]}
        result["effect"] = effect
    return result


def visible_relic(relic: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "id", "description", "counter")
    result = {key: relic.get(key) for key in keys if key in relic}
    effect = relic_effect(relic)
    if effect is not None:
        if relic.get("description") is not None:
            effect = dict(effect) | {"source": "runtime_over_static", "description": relic["description"]}
        result["effect"] = effect
    return result


def powers_summary(powers: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(powers, list):
        return result
    for power in powers:
        if not isinstance(power, dict):
            continue
        item: dict[str, Any] = {}
        for key in ("name", "id", "amount", "type"):
            if key in power:
                item[key] = power[key]
        result.append(item)
    return result


def visible_map_node(node: dict[str, Any]) -> dict[str, Any]:
    """Keep the publicly rendered map graph without internal Java objects."""
    result: dict[str, Any] = {}
    for key in ("x", "y", "symbol", "room_type"):
        if key in node and isinstance(node[key], (str, int, float, bool)):
            result[key] = node[key]
    for relation in ("children", "parents"):
        links = node.get(relation)
        if isinstance(links, list):
            result[relation] = [
                {key: item[key] for key in ("x", "y") if key in item and isinstance(item[key], (int, float))}
                for item in links
                if isinstance(item, dict)
            ]
    return result


def visible_map(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [visible_map_node(node) for node in value if isinstance(node, dict)]


def is_map_room_transition(state: dict[str, Any]) -> bool:
    """Detect a map overlay exposed before the current room is complete."""
    return state.get("screen_type") == "MAP" and state.get("room_phase") != "COMPLETE"


def is_settings_overlay(state: dict[str, Any]) -> bool:
    """Detect the pause/settings overlay that temporarily hides combat input."""
    screen_name = str(state.get("screen_name") or "").upper()
    return bool(state.get("is_screen_up")) and screen_name in {"SETTINGS", "PAUSE"}


def shop_visit_key(message: dict[str, Any]) -> tuple[int, int] | None:
    """Identify a shop by Act and floor for per-run re-entry guards."""
    state = game_state(message)
    try:
        act = int(state.get("act"))
        floor = int(state.get("floor"))
    except (TypeError, ValueError):
        return None
    return act, floor


def is_shop_room(state: dict[str, Any]) -> bool:
    return str(state.get("screen_type") or "").upper() == "SHOP_ROOM"


def shop_reentry_blocked(message: dict[str, Any], visited_shop_rooms: set[tuple[int, int]] | None) -> bool:
    if not visited_shop_rooms or not is_shop_room(game_state(message)):
        return False
    key = shop_visit_key(message)
    return key in visited_shop_rooms if key is not None else False


def run_memory_summary(visited_shop_rooms: set[tuple[int, int]] | None) -> dict[str, Any]:
    """Expose explicit environment constraints without replaying history."""
    rooms = sorted(visited_shop_rooms or set())
    return {"visited_shop_rooms": [{"act": act, "floor": floor} for act, floor in rooms]}


def recent_action_summary(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep a small, stable action context for the next model decision."""
    if not isinstance(history, list):
        return []
    result: list[dict[str, Any]] = []
    for item in history[-ACTION_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        compact: dict[str, Any] = {}
        for key in ("step_id", "screen_before", "action_kind", "action_id", "action", "screen_after"):
            value = item.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                compact[key] = value
        if compact:
            result.append(compact)
    return result


def accepted_action_history_item(pending: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """Convert one stable controller acknowledgement into compact model context."""
    action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
    state = game_state(message)
    return {
        "step_id": pending.get("step_id"),
        "screen_before": pending.get("screen_type"),
        "action_kind": action.get("kind"),
        "action_id": action.get("id"),
        "action": action.get("label") or action.get("command"),
        "screen_after": state.get("screen_type"),
    }


def candidate_assist_allowed(message: dict[str, Any], screens: tuple[str, ...] = ("ALL",)) -> bool:
    """Return whether this in-game screen should receive model candidates."""
    configured = {token.strip().upper() for token in screens if token.strip()}
    if not configured:
        return False
    if "ALL" in configured:
        return bool(message.get("in_game"))
    state = game_state(message)
    return (
        "COMBAT" in configured
        and state.get("screen_type") == "NONE"
        and state.get("room_phase") == "COMBAT"
        and isinstance(state.get("combat_state"), dict)
        and not is_settings_overlay(state)
    )


def normalize_state(message: dict[str, Any]) -> dict[str, Any]:
    """Create the model-facing state without hidden draw order or RNG seed."""
    state = game_state(message)
    map_transition = is_map_room_transition(state)
    result: dict[str, Any] = {
        "screen_type": "TRANSITION" if map_transition else state.get("screen_type"),
        "screen_name": state.get("screen_name"),
        "screen_up": state.get("is_screen_up"),
        "room_phase": state.get("room_phase"),
        "action_phase": state.get("action_phase"),
        "room_type": state.get("room_type"),
        "floor": state.get("floor"),
        "act": state.get("act"),
        "class": state.get("class"),
        "ascension_level": state.get("ascension_level"),
        "current_hp": state.get("current_hp"),
        "max_hp": state.get("max_hp"),
        "gold": state.get("gold"),
        "act_boss": state.get("act_boss"),
        "relics": [visible_relic(item) for item in state.get("relics", []) if isinstance(item, dict)],
        "potions": [visible_potion(item) for item in state.get("potions", []) if isinstance(item, dict)],
    }

    # The generated act map is public in the normal UI. Keep the full graph so
    # route decisions can use future branches, not only the next layer.
    full_map = visible_map(state.get("map"))
    if full_map:
        screen_state = state.get("screen_state")
        map_summary: dict[str, Any] = {"nodes": full_map}
        if isinstance(screen_state, dict):
            map_summary.update(
                {
                    "current_node": screen_state.get("current_node"),
                    "next_nodes": screen_state.get("next_nodes", []),
                    "boss_available": screen_state.get("boss_available"),
                }
            )
        result["map"] = map_summary

    result["deck"] = deck_summary(state.get("deck", []))
    result["choices"] = [] if map_transition else list(state.get("choice_list", []))

    screen_state = state.get("screen_state")
    if isinstance(screen_state, dict):
        result["screen_state"] = (
            {"transition": "room_not_complete"}
            if map_transition
            else normalize_screen_state(state.get("screen_type"), screen_state)
        )

    # CommunicationMod can briefly expose the map overlay while the current
    # event room is still active. It is not safe to route to the next node in
    # that state: doing so can replay the event and desynchronize the UI.
    if map_transition:
        result["state_warning"] = "map_visible_before_room_complete"

    combat = state.get("combat_state")
    if isinstance(combat, dict):
        result["combat"] = normalize_combat(combat)
    return result


def policy_observation(
    message: dict[str, Any],
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    action_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Route only the public context relevant to the active screen."""
    normalized = normalize_state(message)
    normalized["recent_actions"] = recent_action_summary(action_history)
    # Full route information belongs only to the actual map selection screen.
    # A transient map overlay is normalized as TRANSITION and must not leak
    # route detail into event/reward/combat decisions.
    if normalized.get("screen_type") == "MAP":
        normalized["run_memory"] = run_memory_summary(visited_shop_rooms)
        return normalized

    normalized.pop("map", None)
    normalized["run_memory"] = run_memory_summary(visited_shop_rooms)
    if normalized.get("screen_type") != "NONE" or normalized.get("room_phase") != "COMBAT":
        return normalized

    # Combat runs every turn, so keep this prompt compact even though map
    # context has already been removed above.
    return {
        key: normalized.get(key)
        for key in (
            "screen_type",
            "room_phase",
            "action_phase",
            "floor",
            "act",
            "class",
            "ascension_level",
            "current_hp",
            "max_hp",
            "gold",
            "relics",
            "potions",
            "combat",
            "recent_actions",
        )
    }


def deck_summary(deck: Any) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, int]] = Counter()
    if not isinstance(deck, list):
        return []
    for card in deck:
        if not isinstance(card, dict):
            continue
        counts[(str(card.get("id", card.get("name", ""))), int(card.get("upgrades", 0) or 0))] += 1
    return [
        {"id": card_id, "upgrades": upgrades, "count": count}
        for (card_id, upgrades), count in sorted(counts.items())
    ]


def overlay_deck_summary(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Group the currently visible deck for the compact state overlay."""
    deck = game_state(message).get("deck")
    counts: Counter[tuple[str, int]] = Counter()
    if not isinstance(deck, list):
        return []
    for card in deck:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or card.get("id") or "未知卡牌")
        try:
            upgrades = int(card.get("upgrades", 0) or 0)
        except (TypeError, ValueError):
            upgrades = 0
        counts[(name, upgrades)] += 1
    return [
        {"name": name, "upgrades": upgrades, "count": count}
        for (name, upgrades), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


PUBLIC_ITEM_SCALAR_FIELDS = {
    "reward_type",
    "name",
    "id",
    "price",
    "gold",
    "amount",
    "text",
    "label",
    "description",
    "option",
    "disabled",
    "available",
    "type",
    "rarity",
    "tier",
    "color",
    "counter",
}


def _public_scalars(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if (key in PUBLIC_ITEM_SCALAR_FIELDS or key.startswith("requires_"))
        and (isinstance(item, (str, int, float, bool)) or item is None)
    }


def _enrich_public_item_fields(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize only entities CommunicationMod explicitly exposes."""
    result = _public_scalars(value)
    for key, normalizer in (("card", visible_card), ("relic", visible_relic), ("potion", visible_potion)):
        item = value.get(key)
        if isinstance(item, dict):
            result[key] = normalizer(item)
    for key, normalizer in (("cards", visible_card), ("relics", visible_relic), ("potions", visible_potion)):
        items = value.get(key)
        if isinstance(items, list):
            result[key] = [normalizer(item) for item in items if isinstance(item, dict)]
    return result


def normalize_reward(value: Any) -> Any:
    """Keep publicly revealed reward contents while excluding internal objects."""
    if not isinstance(value, dict):
        return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    result = _enrich_public_item_fields(value)
    reward_type = str(value.get("reward_type", "")).upper()
    # Some protocol versions put the entity fields directly on a reward rather
    # than under ``relic``/``potion``. This still only uses already visible ID
    # or name fields; a bare {"reward_type": "RELIC"} stays bare.
    if reward_type == "RELIC" and (value.get("id") or value.get("name")):
        result["relic"] = visible_relic(value)
    elif reward_type == "POTION" and (value.get("id") or value.get("name")):
        result["potion"] = visible_potion(value)
    elif reward_type == "CARD" and (value.get("id") or value.get("name")):
        result["card"] = visible_card(value)
    return result


def normalize_event_option(value: Any) -> Any:
    """Preserve option wording and enrich only directly attached item objects."""
    if not isinstance(value, dict):
        return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    return _enrich_public_item_fields(value)


def normalize_screen_state(screen_type: Any, value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if screen_type in {"CARD_REWARD", "GRID"}:
        cards = value.get("cards", [])
        result["cards"] = [visible_card(card) for card in cards if isinstance(card, dict)]
        if screen_type == "GRID":
            mode_fields = (
                ("for_purge", "PURGE", "选择 1 张牌移除"),
                ("for_transform", "TRANSFORM", "选择 1 张牌进行变化"),
                ("for_upgrade", "UPGRADE", "选择 1 张牌升级"),
            )
            for flag, mode, instruction in mode_fields:
                if value.get(flag):
                    result["selection_mode"] = mode
                    result["selection_instruction"] = instruction
                    break
            else:
                if value.get("any_number"):
                    result["selection_mode"] = "SELECT"
                    result["selection_instruction"] = "选择符合当前界面要求的牌"
        for key in ("selected_cards", "num_cards", "for_transform", "for_upgrade", "for_purge", "any_number", "confirm_up", "skip_available"):
            if key in value:
                if key == "selected_cards":
                    result[key] = [visible_card(card) for card in value[key] if isinstance(card, dict)]
                else:
                    result[key] = value[key]
    elif screen_type == "EVENT":
        result["event_id"] = value.get("event_id")
        result["event_name"] = value.get("event_name")
        result["body_text"] = value.get("body_text")
        result["options"] = [normalize_event_option(item) for item in value.get("options", [])]
    elif screen_type == "COMBAT_REWARD":
        result["rewards"] = [normalize_reward(item) for item in value.get("rewards", [])]
    elif screen_type == "REST":
        result["rest_options"] = value.get("rest_options", [])
        result["has_rested"] = value.get("has_rested")
    elif screen_type == "SHOP_SCREEN":
        result["cards"] = [visible_card(item) | {"price": item.get("price")} for item in value.get("cards", []) if isinstance(item, dict)]
        result["relics"] = [visible_relic(item) | {"price": item.get("price")} for item in value.get("relics", []) if isinstance(item, dict)]
        result["potions"] = [visible_potion(item) | {"price": item.get("price")} for item in value.get("potions", []) if isinstance(item, dict)]
        result["purge_available"] = value.get("purge_available")
        result["purge_cost"] = value.get("purge_cost")
    elif screen_type == "BOSS_REWARD":
        result["relics"] = [visible_relic(item) for item in value.get("relics", []) if isinstance(item, dict)]
    elif screen_type == "HAND_SELECT":
        result["hand"] = [visible_card(item) for item in value.get("hand", []) if isinstance(item, dict)]
        result["selected"] = [visible_card(item) for item in value.get("selected", []) if isinstance(item, dict)]
        for key in ("max_cards", "can_pick_zero"):
            if key in value:
                result[key] = value[key]
    elif screen_type == "MAP":
        for key in ("current_node", "next_nodes", "boss_available", "first_node_chosen"):
            if key in value:
                result[key] = value[key]
    else:
        # Keep small scalar fields for unhandled screens, but avoid copying
        # arbitrary internal objects into the training prompt.
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                result[key] = item
    return result


def normalize_combat(combat: dict[str, Any]) -> dict[str, Any]:
    player = combat.get("player") if isinstance(combat.get("player"), dict) else {}
    monsters: list[dict[str, Any]] = []
    for monster in combat.get("monsters", []):
        if not isinstance(monster, dict):
            continue
        monsters.append(
            {
                "id": monster.get("id"),
                "name": monster.get("name"),
                "current_hp": monster.get("current_hp"),
                "max_hp": monster.get("max_hp"),
                "block": monster.get("block"),
                "intent": monster.get("intent"),
                "intent_base_damage": monster.get("move_base_damage"),
                "intent_damage": monster.get("move_adjusted_damage"),
                "intent_hits": monster.get("move_hits"),
                "is_gone": monster.get("is_gone"),
                "powers": powers_summary(monster.get("powers")),
            }
        )
    return {
        "turn": combat.get("turn"),
        "hand": [visible_card(card) for card in combat.get("hand", []) if isinstance(card, dict)],
        "player": {
            "current_hp": player.get("current_hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
            "powers": powers_summary(player.get("powers")),
        },
        "monsters": monsters,
        "draw_count": len(combat.get("draw_pile", [])) if isinstance(combat.get("draw_pile"), list) else None,
        "discard_count": len(combat.get("discard_pile", [])) if isinstance(combat.get("discard_pile"), list) else None,
        "exhaust_count": len(combat.get("exhaust_pile", [])) if isinstance(combat.get("exhaust_pile"), list) else None,
    }


def semantic_actions(
    message: dict[str, Any],
    visited_shop_rooms: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Build action candidates using stable semantic commands, not clicks."""
    commands = command_set(message)
    state = game_state(message)
    result: list[dict[str, Any]] = []
    if is_settings_overlay(state):
        # CommunicationMod exposes the overlay through the generic KEY
        # command rather than RETURN/CANCEL. Do not expose underlying combat
        # actions while the settings screen is on top of the game.
        if "key" in commands:
            result.append(
                {
                    "id": "KEY:CANCEL",
                    "command": "KEY Cancel",
                    "kind": "key",
                    "label": "关闭设置界面",
                }
            )
        return result
    combat = state.get("combat_state")
    if state.get("screen_type") == "NONE" and state.get("room_phase") == "COMBAT" and isinstance(combat, dict):
        hand = combat.get("hand", [])
        monsters = combat.get("monsters", [])
        live_targets = [
            index for index, monster in enumerate(monsters)
            if isinstance(monster, dict) and not monster.get("is_gone", False)
        ]
        if "play" in commands:
            for index, card in enumerate(hand, start=1):
                if not isinstance(card, dict) or not card.get("is_playable", False):
                    continue
                if card.get("has_target", False):
                    for target in live_targets:
                        result.append(
                            {
                                "id": f"PLAY:card={index}:target={target}",
                                "command": f"PLAY {index} {target}",
                                "kind": "play",
                                "card_index": index,
                                "target": target,
                                "label": f"出牌 {card.get('name', index)} -> 敌人 {target}",
                            }
                        )
                else:
                    result.append(
                        {
                            "id": f"PLAY:card={index}",
                            "command": f"PLAY {index}",
                            "kind": "play",
                            "card_index": index,
                            "label": f"出牌 {card.get('name', index)}",
                        }
                    )
        if "end" in commands:
            result.append({"id": "END_TURN", "command": "END", "kind": "end_turn", "label": "结束回合"})
        if "potion" in commands:
            for index, potion in enumerate(state.get("potions", [])):
                if not isinstance(potion, dict) or is_slot(potion):
                    continue
                if potion.get("can_use", False):
                    if potion.get("requires_target", False):
                        for target in live_targets:
                            result.append(
                                {
                                    "id": f"POTION_USE:slot={index}:target={target}",
                                    "command": f"POTION USE {index} {target}",
                                "kind": "potion_use",
                                "slot": index,
                                "target": target,
                                "label": f"使用药水 {potion.get('name', index)} -> 敌人 {target}",
                            }
                            )
                    else:
                        result.append(
                            {
                                "id": f"POTION_USE:slot={index}",
                                "command": f"POTION USE {index}",
                                "kind": "potion_use",
                                "slot": index,
                                "label": f"使用药水 {potion.get('name', index)}",
                            }
                        )
                if potion.get("can_discard", False):
                    result.append(
                        {
                            "id": f"POTION_DISCARD:slot={index}",
                        "command": f"POTION DISCARD {index}",
                        "kind": "potion_discard",
                        "slot": index,
                        "label": f"丢弃药水 {potion.get('name', index)}",
                    }
                    )
        return result

    choices = state.get("choice_list", [])
    if is_map_room_transition(state):
        if "return" in commands:
            result.append({"id": "RETURN", "command": "RETURN", "kind": "return"})
        return result
    if "choose" in commands and isinstance(choices, list) and not shop_reentry_blocked(message, visited_shop_rooms):
        blocked_potions = blocked_potion_reward_indices(message)
        for index, label in enumerate(choices):
            if index in blocked_potions:
                continue
            result.append(
                {
                    "id": f"CHOOSE:index={index}",
                    "command": f"CHOOSE {index}",
                    "kind": "choose",
                    "index": index,
                    "label": str(label),
                }
            )
    if "confirm" in commands or "proceed" in commands:
        command = "CONFIRM" if "confirm" in commands else "PROCEED"
        result.append({"id": command, "command": command, "kind": "confirm"})
    for command, action_id, kind in (
        ("cancel", "CANCEL", "cancel"),
        ("return", "RETURN", "return"),
        ("skip", "SKIP", "skip"),
        ("leave", "LEAVE", "leave"),
    ):
        if command in commands:
            result.append({"id": action_id, "command": command.upper(), "kind": kind})
    return result


def action_for_command(actions: list[dict[str, Any]], command: str) -> dict[str, Any] | None:
    upper = command.upper()
    for action in actions:
        if action["command"].upper() == upper:
            return action
    return None


def valid_start_seed(seed: str) -> bool:
    """Validate a seed token accepted by CommunicationMod's START command."""
    return bool(seed) and seed.isalnum()


def start_seed_error(seed: str) -> str:
    """Explain the distinction between START seeds and observed runtime seeds."""
    if seed.startswith("-"):
        return (
            "CommunicationMod 的 START 只接受字母数字种子；"
            "游戏状态中的 game_state.seed 可能为负数，但不能直接作为 START 参数。"
        )
    return "种子只能包含字母和数字。"


def parse_start_arguments(parts: list[str]) -> tuple[str, int, str | None, str | None]:
    """Parse the human-friendly START syntax into protocol arguments."""
    tokens = parts[1:]
    character = "IRONCLAD"
    ascension = 0
    seed: str | None = None

    if not tokens:
        return character, ascension, seed, None

    first = tokens[0].upper()
    if first in PLAYER_CLASSES:
        character = PLAYER_CLASSES[first]
        tokens = tokens[1:]
    else:
        # Keep the old ``s SEED`` shorthand as Ironclad A0.
        if len(tokens) == 1:
            seed = tokens[0]
            if not valid_start_seed(seed):
                return character, ascension, None, start_seed_error(seed)
            return character, ascension, seed, None
        return character, ascension, None, "角色必须是 IRONCLAD、SILENT、DEFECT 或 WATCHER。"

    if tokens:
        if tokens[0].startswith("-"):
            return character, ascension, None, start_seed_error(tokens[0])
        try:
            ascension = int(tokens[0])
        except ValueError:
            # ``s SILENT SEED`` is a convenient A0 shorthand.
            seed = tokens[0]
            if not valid_start_seed(seed):
                return character, ascension, None, start_seed_error(seed)
            tokens = tokens[1:]
        else:
            tokens = tokens[1:]
            if not 0 <= ascension <= MAX_ASCENSION:
                return character, ascension, None, "进阶必须在 0 到 20 之间。"

    if tokens:
        if len(tokens) != 1 or not valid_start_seed(tokens[0]):
            invalid_seed = tokens[0] if tokens else ""
            return character, ascension, None, start_seed_error(invalid_seed)
        seed = tokens[0]
    return character, ascension, seed, None


def candidate_action_for_input(
    parts: list[str],
    message: dict[str, Any],
    decision: dict[str, Any] | None,
    visited_shop_rooms: set[tuple[int, int]] | None = None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Resolve explicit Candidate Assist shortcuts before game choices."""
    if not decision:
        return None, None, None
    candidates = decision.get("model_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None, None, None
    actions = semantic_actions(message, visited_shop_rooms)
    action_by_id = {str(action.get("id")): action for action in actions}
    op = parts[0].lower()
    candidate_index: int | None = None
    if op in {"a", "accept"} and len(parts) == 1:
        candidate_index = 1
    elif op in {"pick", "candidate", "k"} and len(parts) == 2 and parts[1].isdigit():
        candidate_index = int(parts[1])
    if candidate_index is None:
        return None, None, None
    if not 1 <= candidate_index <= len(candidates):
        return None, None, "候选序号超出范围；请输入 a 或 pick <候选序号>。"
    item = candidates[candidate_index - 1]
    candidate_id = str(item.get("action_id", "")) if isinstance(item, dict) else ""
    action = action_by_id.get(candidate_id)
    if action is None:
        return None, None, "模型候选已过期，请输入 s 刷新状态。"
    return action["command"], action, None


def candidate_source_for_input(text: str, decision: dict[str, Any] | None) -> str:
    if not decision:
        return "human_only"
    if not decision.get("model_candidates"):
        return "human_after_model_fallback"
    try:
        parts = shlex.split(text.strip())
    except ValueError:
        return "human_override"
    if parts and parts[0].lower() in {"a", "accept"}:
        return "model_accept"
    if parts and parts[0].lower() in {"pick", "candidate", "k"}:
        return "human_choice"
    return "human_override"


def finalize_decision(
    decision: dict[str, Any] | None,
    action: dict[str, Any],
    source: str,
) -> dict[str, Any] | None:
    if not decision:
        return None
    result = dict(decision)
    executed = str(action.get("id", ""))
    result["executed_action"] = executed
    result["source"] = source
    result["overridden"] = executed != str(decision.get("model_action", ""))
    return result


def parse_human_input(
    text: str,
    message: dict[str, Any],
    decision: dict[str, Any] | None = None,
    visited_shop_rooms: set[tuple[int, int]] | None = None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return (game command, action metadata, error)."""
    stripped = text.strip()
    if not stripped:
        return None, None, "请输入动作。"
    try:
        parts = shlex.split(stripped)
    except ValueError as exc:
        return None, None, f"无法解析输入：{exc}"
    if not parts:
        return None, None, "请输入动作。"
    op = parts[0].lower()
    actions = semantic_actions(message, visited_shop_rooms)

    if op in {"q", "quit", "exit"}:
        return "__QUIT__", None, None
    if op in {"h", "help"}:
        return "__HELP__", None, None
    candidate_command, candidate_action, candidate_error = candidate_action_for_input(
        parts,
        message,
        decision,
        visited_shop_rooms,
    )
    if candidate_command is not None or candidate_error is not None:
        return candidate_command, candidate_action, candidate_error
    if op == "state" or (op == "s" and message.get("in_game")):
        return "STATE", None, None
    if op in {"p", "play"}:
        if len(parts) not in {2, 3}:
            return None, None, "出牌格式：p <手牌序号> [敌人序号]，手牌从 1 开始，敌人从 0 开始。"
        try:
            card_index = int(parts[1])
            target = int(parts[2]) if len(parts) == 3 else None
        except ValueError:
            return None, None, "手牌和目标必须是整数。"
        command = f"PLAY {card_index}" + (f" {target}" if target is not None else "")
        action = action_for_command(actions, command)
        # CommunicationMod requires an explicit target for enemy-target cards.
        # For a single live enemy, accept the shorter human input ``p <card>``
        # and resolve it to that enemy before sending the game command.
        if action is None and target is None and len(parts) == 2:
            state = game_state(message)
            combat = state.get("combat_state")
            if isinstance(combat, dict):
                monsters = combat.get("monsters", [])
                live_targets = [
                    index for index, monster in enumerate(monsters)
                    if isinstance(monster, dict) and not monster.get("is_gone", False)
                ]
                if len(live_targets) == 1:
                    resolved_command = f"PLAY {card_index} {live_targets[0]}"
                    action = action_for_command(actions, resolved_command)
                    if action is not None:
                        return resolved_command, action, None
        if action is None:
            return None, None, "该出牌动作当前不在合法动作列表中。"
        return command, action, None
    if op in {"e", "end", "end_turn"}:
        action = action_for_command(actions, "END")
        return ("END", action, None) if action else (None, None, "当前不能结束回合。")
    if op in {"u", "use", "potion"}:
        if len(parts) not in {2, 3}:
            return None, None, "使用药水格式：u <药水槽位> [敌人序号]。"
        try:
            slot = int(parts[1])
            target = int(parts[2]) if len(parts) == 3 else None
        except ValueError:
            return None, None, "药水槽位和目标必须是整数。"
        command = f"POTION USE {slot}" + (f" {target}" if target is not None else "")
        action = action_for_command(actions, command)
        if action is None and target is None and len(parts) == 2:
            state = game_state(message)
            combat = state.get("combat_state")
            if isinstance(combat, dict):
                monsters = combat.get("monsters", [])
                live_targets = [
                    index for index, monster in enumerate(monsters)
                    if isinstance(monster, dict) and not monster.get("is_gone", False)
                ]
                if len(live_targets) == 1:
                    resolved_command = f"POTION USE {slot} {live_targets[0]}"
                    action = action_for_command(actions, resolved_command)
                    if action is not None:
                        return resolved_command, action, None
        return (command, action, None) if action else (None, None, "该药水动作当前不合法。")
    if op in {"d", "discard"}:
        if len(parts) != 2:
            return None, None, "丢弃药水格式：d <药水槽位>。"
        try:
            slot = int(parts[1])
        except ValueError:
            return None, None, "药水槽位必须是整数。"
        command = f"POTION DISCARD {slot}"
        action = action_for_command(actions, command)
        return (command, action, None) if action else (None, None, "该药水当前不能丢弃。")
    if op in {"m", "g", "choose", "o"} or op.isdigit():
        token = parts[1] if op in {"m", "g", "choose", "o"} and len(parts) > 1 else op
        try:
            index = int(token)
        except ValueError:
            return None, None, "选项序号必须是整数。"
        if shop_reentry_blocked(message, visited_shop_rooms):
            return None, None, "该商店入口本层已访问，不能再次进入；请输入 c 继续离开。"
        blocked_error = blocked_potion_choice_error(message, index)
        if blocked_error:
            return None, None, blocked_error
        command = f"CHOOSE {index}"
        action = action_for_command(actions, command)
        return (command, action, None) if action else (None, None, "该选项当前不可选择。")
    if op in {"c", "confirm", "proceed", "n", "next"}:
        for name in ("CONFIRM", "PROCEED"):
            action = action_for_command(actions, name)
            if action:
                return action["command"], action, None
        return None, None, "当前没有确认或继续动作。"
    if op in {"x", "esc", "escape", "cancel", "return", "skip", "leave"}:
        preferred = {
            "x": ("CANCEL", "RETURN", "SKIP", "LEAVE", "KEY CANCEL"),
            "esc": ("KEY CANCEL", "CANCEL", "RETURN", "SKIP", "LEAVE"),
            "escape": ("KEY CANCEL", "CANCEL", "RETURN", "SKIP", "LEAVE"),
            "skip": ("SKIP",),
            "leave": ("LEAVE",),
        }.get(op, (op.upper(),))
        for name in preferred:
            action = action_for_command(actions, name)
            if action:
                return action["command"], action, None
        return None, None, "当前没有取消、返回或跳过动作。"
    if op in {"start", "s"}:
        if message.get("in_game"):
            return None, None, "当前已经在游戏中。"
        character, ascension, seed, error = parse_start_arguments(parts)
        if error:
            return None, None, error
        command = f"START {character} {ascension}" + (f" {seed}" if seed else "")
        return command, {
            "id": "START_RUN",
            "kind": "start_run",
            "command": command,
            "character": character,
            "ascension": ascension,
            "seed": seed,
        }, None
    return None, None, "未知输入。输入 h 查看帮助。"


def overlay_input_hint(action: dict[str, Any]) -> str:
    """Return the shortest human-terminal command that resolves to this action."""
    kind = str(action.get("kind", ""))
    if kind == "play":
        command = f"p {action.get('card_index')}"
        return f"{command} {action['target']}" if isinstance(action.get("target"), int) else command
    if kind == "potion_use":
        command = f"u {action.get('slot')}"
        return f"{command} {action['target']}" if isinstance(action.get("target"), int) else command
    if kind == "potion_discard":
        return f"d {action.get('slot')}"
    if kind == "choose" and isinstance(action.get("index"), int):
        return str(action["index"])
    if kind == "end_turn":
        return "e"
    if kind == "confirm":
        return "c"
    if kind in {"cancel", "return", "skip", "leave", "key"}:
        return "x"
    return str(action.get("command") or "")


def overlay_action_row(message: dict[str, Any], action: dict[str, Any]) -> dict[str, str]:
    """Build the four compact columns shown by the Overlay action table."""
    state = game_state(message)
    kind = str(action.get("kind", ""))
    selection = ""
    target = ""

    if kind == "play":
        combat = state.get("combat_state") if isinstance(state.get("combat_state"), dict) else {}
        hand = combat.get("hand", [])
        card_index = action.get("card_index")
        if isinstance(hand, list) and isinstance(card_index, int) and 0 < card_index <= len(hand):
            card = hand[card_index - 1]
            if isinstance(card, dict):
                selection = str(card.get("name") or f"手牌 {card_index}")
        if not selection:
            selection = f"手牌 {card_index}" if card_index is not None else ""
        action_type = "出牌"
    elif kind in {"potion_use", "potion_discard"}:
        potions = state.get("potions", [])
        slot = action.get("slot")
        if isinstance(potions, list) and isinstance(slot, int) and 0 <= slot < len(potions):
            potion = potions[slot]
            if isinstance(potion, dict):
                selection = str(potion.get("name") or f"药水 {slot}")
        if not selection:
            selection = f"药水 {slot}" if slot is not None else ""
        action_type = "使用药水" if kind == "potion_use" else "丢弃药水"
    elif kind == "choose":
        action_type = "选择"
        selection = str(action.get("label") or "")
    else:
        action_type = {
            "end_turn": "结束回合",
            "confirm": "确认/继续",
            "return": "返回",
            "cancel": "取消",
            "skip": "跳过",
            "leave": "离开",
            "key": "按键",
        }.get(kind, str(action.get("label") or action.get("id") or "动作"))
        if kind == "key":
            selection = str(action.get("label") or "")

    target_index = action.get("target")
    if isinstance(target_index, int):
        target = f"敌人 {target_index}"
    return {
        "type": action_type,
        "selection": selection,
        "target": target,
        "command": overlay_input_hint(action),
    }


def overlay_view_data(
    message: dict[str, Any],
    run_id: str | None,
    actions: list[dict[str, Any]],
    action_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return only the state information useful beside the game window."""
    if not message.get("in_game"):
        return {"summary": "状态：主菜单", "legal_actions": []}

    state = game_state(message)
    map_transition = is_map_room_transition(state)
    screen = str(state.get("screen_type", ""))
    display_screen = "TRANSITION" if map_transition else screen
    summary = "\n".join(
        (
            f"RUN {run_id or '-'} | {state.get('class', '?')} A{state.get('ascension_level', '?')} "
            f"| Act {state.get('act', '?')} Floor {state.get('floor', '?')} | {display_screen}",
            f"HP {state.get('current_hp', '?')}/{state.get('max_hp', '?')} "
            f"| Gold {state.get('gold', '?')} | Room {state.get('room_phase', '')}",
        )
    )
    return {
        "summary": summary,
        "deck": overlay_deck_summary(message),
        "recent_actions": recent_action_summary(action_history),
        "legal_actions": [overlay_action_row(message, action) for action in actions],
    }


def format_view(
    message: dict[str, Any],
    run_id: str | None,
    error: str | None = None,
    decision: dict[str, Any] | None = None,
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> str:
    state = game_state(message)
    lines: list[str] = ["STS AGENT", "=" * 78]
    if error:
        lines.extend(("[NOTICE]", error, "-" * 78))
    if not message.get("in_game"):
        lines.extend(
            (
                "状态：主菜单",
                "  s                         开始 Ironclad A0，随机种子",
                "  s <seed>                 Ironclad A0，使用指定种子",
                "  s <角色> <进阶> [种子]   角色：IRONCLAD/SILENT/DEFECT/WATCHER，进阶 0-20",
                "",
                "输入：h 帮助 | q 断开终端",
            )
        )
        return "\n".join(lines)

    map_transition = is_map_room_transition(state)
    screen = str(state.get("screen_type", ""))
    display_screen = "TRANSITION" if map_transition else screen
    lines.append(
        f"RUN {run_id or '-'} | {state.get('class', '?')} A{state.get('ascension_level', '?')} "
        f"| Act {state.get('act', '?')} Floor {state.get('floor', '?')} | {display_screen}"
    )
    lines.append(
        f"HP {state.get('current_hp', '?')}/{state.get('max_hp', '?')} "
        f"| Gold {state.get('gold', '?')} | Room {state.get('room_phase', '')}"
    )
    confidence = decision.get("confidence") if isinstance(decision, dict) else None
    if isinstance(confidence, dict):
        route = "将自动执行" if confidence.get("route") == "auto_execute" else "等待人工确认"
        lines.append(
            "模型置信度："
            f"p1={float(confidence.get('top_probability', 0.0)):.1%} "
            f"margin={float(confidence.get('margin', 0.0)):.1%} "
            f"entropy={float(confidence.get('entropy', 0.0)):.3f}；{route}"
        )
    if decision and decision.get("model_candidates"):
        lines.append("模型候选（输入 a 接受第一项，或输入 pick 1/2/3 选择；也可直接手动输入动作）：")
        for item in decision["model_candidates"]:
            score = item.get("score")
            score_text = f" p={score:.1%}" if isinstance(score, (int, float)) else ""
            lines.append(f"  [{item.get('rank')}] {item.get('action_id')} {item.get('label', '')}{score_text}")
    elif decision and decision.get("assist_status"):
        reason = str(decision.get("assist_reason") or "未提供原因")
        lines.append(f"模型状态：{decision['assist_status']}（{reason}）；当前为人工操作。")
    lines.append("-" * 78)
    combat = state.get("combat_state")
    settings_overlay = is_settings_overlay(state)
    if settings_overlay:
        lines.append("界面覆盖层：设置（战斗仍在进行）")
        lines.append("请先输入 x 关闭设置界面；关闭后会恢复出牌和结束回合动作。")
    elif screen == "NONE" and state.get("room_phase") == "COMBAT" and isinstance(combat, dict):
        player = combat.get("player", {}) if isinstance(combat.get("player"), dict) else {}
        lines.append(f"战斗 | Turn {combat.get('turn', '?')} | Energy {player.get('energy', '?')} | Block {player.get('block', '?')}")
        lines.append("敌人")
        for index, monster in enumerate(combat.get("monsters", [])):
            if not isinstance(monster, dict):
                continue
            intent_damage = monster.get("move_adjusted_damage")
            intent_hits = monster.get("move_hits")
            intent_detail = ""
            if intent_damage is not None and intent_damage != -1:
                intent_detail = f" dmg={intent_damage}"
                if intent_hits not in (None, 1):
                    intent_detail += f"x{intent_hits}"
            lines.append(
                f"  [{index}] {monster.get('name')} HP {monster.get('current_hp')}/{monster.get('max_hp')} "
                f"Intent={monster.get('intent')}{intent_detail} Block={monster.get('block', 0)} "
                f"{'GONE' if monster.get('is_gone') else ''}"
            )
        lines.append("手牌")
        for index, card in enumerate(combat.get("hand", []), start=1):
            if not isinstance(card, dict):
                continue
            playable = "可用" if card.get("is_playable") else "不可用"
            target = "需目标" if card.get("has_target") else "无目标"
            if card.get("has_target"):
                live_targets = [
                    monster for monster in combat.get("monsters", [])
                    if isinstance(monster, dict) and not monster.get("is_gone", False)
                ]
                if len(live_targets) == 1:
                    target += "（单敌人可省略目标）"
            lines.append(
                f"  [{index}] {card.get('name')} cost={card.get('cost')} {playable} {target}"
                f"{card_effect_summary(card)}"
            )
        lines.append("药水")
        for index, potion in enumerate(state.get("potions", [])):
            if not isinstance(potion, dict) or is_slot(potion):
                continue
            lines.append(f"  [{index}] {potion.get('name')} use={potion.get('can_use')} discard={potion.get('can_discard')}")
    else:
        screen_state = state.get("screen_state")
        if isinstance(screen_state, dict):
            if screen == "EVENT":
                lines.append(f"事件：{screen_state.get('event_name')} - {screen_state.get('body_text', '')}")
                for index, option in enumerate(screen_state.get("options", [])):
                    if isinstance(option, dict):
                        lines.append(f"  [{index}] {option.get('text', option.get('label', ''))}")
            elif screen in {"CARD_REWARD", "GRID"}:
                for index, card in enumerate(screen_state.get("cards", [])):
                    if isinstance(card, dict):
                        lines.append(f"  [{index}] {card.get('name')} ({card.get('id')}) cost={card.get('cost')}")
                selected = screen_state.get("selected_cards", [])
                if selected:
                    lines.append(f"已选：{', '.join(str(card.get('name')) for card in selected if isinstance(card, dict))}")
            elif screen == "COMBAT_REWARD":
                lines.append(f"奖励：{screen_state.get('rewards', [])}")
            elif screen == "SHOP_SCREEN":
                lines.append(f"商店卡牌：")
                for index, item in enumerate(screen_state.get("cards", [])):
                    lines.append(f"  card[{index}] {item.get('name')} price={item.get('price')}")
                lines.append("商店遗物：")
                for index, item in enumerate(screen_state.get("relics", [])):
                    lines.append(f"  relic[{index}] {item.get('name')} price={item.get('price')}")
                lines.append("商店药水：")
                for index, item in enumerate(screen_state.get("potions", [])):
                    lines.append(f"  potion[{index}] {item.get('name')} price={item.get('price')}")
                if screen_state.get("purge_available"):
                    lines.append(f"  purge price={screen_state.get('purge_cost')}")
            elif screen == "BOSS_REWARD":
                for index, relic in enumerate(screen_state.get("relics", [])):
                    lines.append(f"  relic[{index}] {relic.get('name')}")
            elif screen == "HAND_SELECT":
                for index, card in enumerate(screen_state.get("hand", [])):
                    lines.append(f"  card[{index}] {card.get('name')}")
                lines.append(f"已选 {len(screen_state.get('selected', []))}/{screen_state.get('max_cards', '?')}")
            elif screen == "REST":
                lines.append(f"营火选项：{screen_state.get('rest_options', [])}")
    if map_transition:
        lines.append("地图暂不可选择：当前房间尚未完成，请输入 x 返回房间或输入 s 刷新状态。")
    elif screen == "MAP":
        if isinstance(screen_state, dict):
            lines.append(
                f"地图 | 当前：{screen_state.get('current_node')} | "
                f"下一节点：{screen_state.get('next_nodes', [])} | "
                f"Boss：{screen_state.get('boss_available')}"
            )
        raw_map = state.get("map", [])
        if isinstance(raw_map, list):
            rows: dict[Any, list[str]] = {}
            for node in raw_map:
                if not isinstance(node, dict):
                    continue
                rows.setdefault(node.get("y", "?"), []).append(
                    f"{node.get('symbol', '?')}@{node.get('x', '?')}"
                )
            lines.append("地图概览（按层）：")
            for y in sorted(rows, key=lambda value: str(value)):
                lines.append(f"  y={y}: " + " ".join(rows[y]))
    choices = state.get("choice_list", [])
    if choices and not map_transition:
        lines.append("选项：" + " | ".join(f"{i}:{choice}" for i, choice in enumerate(choices)))
        blocked_potions = blocked_potion_reward_indices(message)
        if blocked_potions:
            labels = ", ".join(str(choices[index]) for index in sorted(blocked_potions))
            lines.append(f"不可选：{labels}（药水栏已满；请改选其他选项或先丢弃药水）")
    if shop_reentry_blocked(message, visited_shop_rooms):
        lines.append("商店入口已在本层访问；不能再次进入。请输入 c 继续离开商店房间。")
    lines.append("合法动作")
    for action in actions if actions is not None else semantic_actions(message, visited_shop_rooms):
        label = action["id"]
        if action.get("label"):
            label += f" ({action['label']})"
        lines.append(f"  {label}")
    lines.append("-" * 78)
    lines.append("输入：h 帮助 | s 刷新 | q 断开终端")
    return "\n".join(lines)


class TerminalBridge:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.connection: socket.socket | None = None
        self.reader: TextIO | None = None
        self.writer: TextIO | None = None

    def accept(self) -> None:
        if self.connection is not None:
            return
        print(f"waiting for terminal client on {self.host}:{self.port}", file=sys.stderr, flush=True)
        connection, address = self.server.accept()
        self.connection = connection
        self.reader = connection.makefile("r", encoding="utf-8", newline="\n")
        self.writer = connection.makefile("w", encoding="utf-8", newline="\n")
        self.send({"type": "hello", "message": f"connected to {address[0]}:{address[1]}"})

    def send(self, packet: dict[str, Any]) -> None:
        self.accept()
        assert self.writer is not None
        self.writer.write(json.dumps(packet, ensure_ascii=False) + "\n")
        self.writer.flush()

    def receive(self) -> dict[str, Any] | None:
        self.accept()
        assert self.reader is not None
        line = self.reader.readline()
        if not line:
            self.close_client()
            return None
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return {"type": "input", "text": line.strip()}
        return value if isinstance(value, dict) else {"type": "input", "text": str(value)}

    def close_client(self) -> None:
        for item in (self.reader, self.writer, self.connection):
            try:
                if item is not None:
                    item.close()
            except OSError:
                pass
        self.reader = None
        self.writer = None
        self.connection = None

    def close(self) -> None:
        self.close_client()
        self.server.close()


class PolicyClient:
    """Optional local policy subprocess used by Candidate Assist.

    CommunicationMod and the terminal remain independent from Transformers.
    A missing model, invalid configuration, or response timeout disables this client for
    the current controller session and leaves human-only input available.
    """

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root
        policy_config = config["policy"]
        confidence_config = policy_config["confidence_auto"]
        self.enabled = policy_config["enabled"]
        self.model = policy_config["model"]
        self.endpoint = policy_config["endpoint"]
        self.top_k = policy_config["top_k"]
        self.screens = tuple(item.strip().upper() for item in policy_config["screens"])
        self.confidence_auto = confidence_config["enabled"]
        self.auto_min_probability = confidence_config["min_probability"]
        self.timeout = float(policy_config["timeout_seconds"])
        self.start_timeout = float(policy_config["start_timeout_seconds"])
        self.python = policy_config["python"]
        self.mock = policy_config["mock"]
        self.debug = policy_config["debug"]
        self.process: subprocess.Popen[str] | None = None
        self.request_id = 0
        self.disabled_reason: str | None = None

    def _python_executable(self) -> str:
        if self.python != "auto":
            return self.python
        if os.name == "nt":
            candidate = self.project_root / ".venv" / "Scripts" / "python.exe"
        else:
            candidate = self.project_root / ".venv" / "bin" / "python"
        return str(candidate) if candidate.exists() else sys.executable

    @staticmethod
    def _readline_with_timeout(stream: TextIO, timeout: float) -> str | None:
        result: list[str | None] = [None]

        def read() -> None:
            try:
                result[0] = stream.readline()
            except (OSError, ValueError):
                result[0] = None

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        return result[0] if not thread.is_alive() else None

    def _disable(self, reason: str) -> None:
        self.disabled_reason = reason
        self.enabled = False
        self.close()
        print(f"candidate assist disabled: {reason}", file=sys.stderr, flush=True)

    def _fallback_decision(self, status: str, reason: str) -> dict[str, Any]:
        """Return an observable human-fallback state for the terminal/trace."""
        return {
            "mode": "confidence_auto" if self.confidence_auto else "candidate_assist",
            "model_id": self.model,
            "candidate_source": "policy_server",
            "model_action": None,
            "model_candidates": [],
            "model_confidence": None,
            "assist_status": status,
            "assist_reason": reason,
            "executed_action": None,
            "source": None,
            "overridden": None,
        }

    def status_decision(self) -> dict[str, Any]:
        """Expose startup/readiness state even on screens without ranking."""
        if self.enabled and self.process is not None and self.process.poll() is None:
            mode = (
                f"置信度自动执行已启用（p1>={self.auto_min_probability:.0%}）"
                if self.confidence_auto
                else "本局每个游戏界面均会请求候选"
            )
            return self._fallback_decision("就绪", f"{self.model} 已预热，{mode}")
        return self._fallback_decision("不可用", self.disabled_reason or "模型辅助已关闭")

    def prewarm(self) -> dict[str, Any]:
        """Load the local policy before CommunicationMod receives readiness."""
        if not self.enabled:
            return self.status_decision()
        self._start()
        return self.status_decision()

    def _start(self) -> bool:
        if not self.enabled:
            return False
        if self.process is not None and self.process.poll() is None:
            return True
        server = self.project_root / "tools" / "policy_server.py"
        if not server.exists():
            self._disable(f"missing {server}")
            return False
        command = [
            self._python_executable(),
            str(server),
            "--model",
            self.model,
            "--endpoint",
            self.endpoint,
            "--top-k",
            str(self.top_k),
        ]
        # Keep game-time startup deterministic. Downloading model weights is
        # an explicit preparation step handled by download_qwen.py.
        command.append("--local-files-only")
        if self.mock:
            command.append("--mock")
        if self.debug:
            command.append("--debug")
        child_env = os.environ.copy()
        child_env["HF_ENDPOINT"] = self.endpoint
        # The policy protocol carries Chinese card and event text. Force the
        # child interpreter to decode its JSONL pipe as UTF-8 on Windows.
        child_env["PYTHONUTF8"] = "1"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                env=child_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            self._disable(f"cannot start policy server: {exc}")
            return False
        assert self.process.stdout is not None
        line = self._readline_with_timeout(self.process.stdout, self.start_timeout)
        if not line:
            try:
                exit_code = self.process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                exit_code = self.process.poll()
            self._disable(f"policy server did not become ready (exit={exit_code})")
            return False
        try:
            ready = json.loads(line)
        except json.JSONDecodeError as exc:
            self._disable(f"invalid policy handshake: {exc}")
            return False
        if ready.get("type") != "ready":
            self._disable(f"unexpected policy handshake: {ready}")
            return False
        print(
            f"candidate assist ready: {ready.get('model_id', self.model)} endpoint={ready.get('endpoint', self.endpoint)}",
            file=sys.stderr,
            flush=True,
        )
        return True

    def rank(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not legal_actions:
            return None
        if not self.enabled:
            return self._fallback_decision("不可用", self.disabled_reason or "模型辅助已关闭")
        if not self._start():
            return self._fallback_decision("不可用", self.disabled_reason or "模型服务未启动")
        assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
        self.request_id += 1
        # Qwen 0.6B often fails to copy long semantic IDs such as
        # PLAY:card=2:target=1. Give the model short aliases and resolve them
        # back to the canonical, currently legal action IDs here.
        alias_to_action: dict[str, dict[str, Any]] = {}
        model_actions: list[dict[str, Any]] = []
        for index, action in enumerate(legal_actions, start=1):
            alias = f"A{index}"
            alias_to_action[alias] = action
            model_actions.append({"id": alias, "label": str(action.get("label") or action.get("command") or alias)})
        request = {
            "type": "score" if self.confidence_auto else "rank",
            "request_id": self.request_id,
            "observation": observation,
            "legal_actions": model_actions,
            "top_k": self.top_k,
        }
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            self._disable(f"policy request failed: {exc}")
            return self._fallback_decision("不可用", self.disabled_reason or "模型请求失败")
        line = self._readline_with_timeout(self.process.stdout, self.timeout)
        if not line:
            self._disable("policy response timeout or closed pipe")
            return self._fallback_decision("超时", self.disabled_reason or "模型未在时限内返回")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self._disable(f"invalid policy response: {exc}")
            return self._fallback_decision("不可用", self.disabled_reason or "模型返回格式无效")
        if response.get("type") == "error":
            self._disable(str(response.get("error", "policy error")))
            return self._fallback_decision("不可用", self.disabled_reason or "模型服务报错")
        expected_type = "scoring" if self.confidence_auto else "ranking"
        if response.get("type") != expected_type:
            self._disable(f"unexpected policy response: {response}")
            return self._fallback_decision("不可用", self.disabled_reason or "模型服务返回未知消息")
        if self.confidence_auto:
            return self._scoring_decision(response, alias_to_action)
        return self._ranking_decision(response, alias_to_action)

    def _ranking_decision(
        self,
        response: dict[str, Any],
        alias_to_action: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for item in response.get("candidates", []):
            if not isinstance(item, dict):
                continue
            alias = str(item.get("action_id", ""))
            action = alias_to_action.get(alias)
            if action is None:
                continue
            candidate_id = str(action.get("id", ""))
            if not candidate_id or any(existing["action_id"] == candidate_id for existing in candidates):
                continue
            candidates.append(
                {
                    "rank": len(candidates) + 1,
                    "action_id": candidate_id,
                    "label": action.get("label") or action.get("command"),
                    "score": item.get("score"),
                    "log_score": item.get("log_score"),
                }
            )
        if not candidates:
            return self._fallback_decision("无候选", "模型没有返回可解析的动作；可以直接手动操作")
        return {
            "mode": "candidate_assist",
            "model_id": response.get("model_id", self.model),
            "candidate_source": "policy_server",
            "model_action": candidates[0]["action_id"],
            "model_candidates": candidates,
            "model_confidence": candidates[0].get("score"),
            "executed_action": None,
            "source": None,
            "overridden": None,
        }

    def _scoring_decision(
        self,
        response: dict[str, Any],
        alias_to_action: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        raw_scores = response.get("action_scores")
        if not isinstance(raw_scores, list):
            return self._fallback_decision("分数无效", "模型没有返回全体合法动作的分数")
        by_alias: dict[str, dict[str, Any]] = {}
        for item in raw_scores:
            if not isinstance(item, dict):
                return self._fallback_decision("分数无效", "模型动作分数格式无效")
            alias = str(item.get("action_id", ""))
            if alias not in alias_to_action or alias in by_alias:
                return self._fallback_decision("分数无效", "模型动作分数包含未知或重复别名")
            by_alias[alias] = item
        if set(by_alias) != set(alias_to_action):
            return self._fallback_decision("分数不完整", "模型未对每个合法动作评分；已退回人工确认")
        normalized = normalize_action_logprobs(list(by_alias.values()))
        if normalized is None:
            return self._fallback_decision("分数无效", "模型动作分数不是有限数值")
        action_scores: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for rank, item in enumerate(normalized, start=1):
            alias = str(item["action_id"])
            action = alias_to_action[alias]
            canonical_id = str(action.get("id", ""))
            score = {
                "alias": alias,
                "action_id": canonical_id,
                "logprob": item["logprob"],
                "probability": item["probability"],
            }
            action_scores.append(score)
            if rank <= self.top_k:
                candidates.append(
                    {
                        "rank": rank,
                        "action_id": canonical_id,
                        "label": action.get("label") or action.get("command"),
                        "score": item["probability"],
                        "log_score": item["logprob"],
                    }
                )
        confidence = confidence_summary(action_scores, self.auto_min_probability)
        return {
            "mode": "confidence_auto",
            "model_id": response.get("model_id", self.model),
            "candidate_source": "policy_server_scoring",
            "model_action": candidates[0]["action_id"],
            "model_candidates": candidates,
            "action_scores": action_scores,
            "model_confidence": confidence["top_probability"],
            "confidence": confidence,
            "executed_action": None,
            "source": None,
            "overridden": None,
        }

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                process.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            process.terminate()
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass


def write_jsonl(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def help_text() -> str:
    return "\n".join(
        (
            "终端动作：",
            "  p <手牌> [敌人]          出牌；手牌从 1 开始，单敌人可省略目标",
            "  e                        结束回合",
            "  u <药水槽> [敌人]        使用药水",
            "  d <药水槽>               丢弃药水",
            "  <序号> / m <序号>        选择事件、奖励或商店选项",
            "  c                        确认/继续",
            "  x                        取消/返回/跳过",
            "  s                        刷新状态；主菜单中开始游戏",
            "  a                        接受模型第一候选（Candidate Assist）",
            "  pick <序号>               选择显示的模型候选",
            "  1 / 2 / 3                保留给事件、奖励或商店中的游戏选项",
            "  h                        显示帮助",
            "  q                        断开终端，controller 继续运行",
        )
    )


def filename_token(value: Any, fallback: str) -> str:
    """Keep run metadata readable while preventing path separators."""
    text = str(value) if value is not None else fallback
    text = "".join(char if char.isalnum() or char in "-_." else "_" for char in text)
    return text or fallback


def display_seed_from_internal(value: Any) -> str | None:
    """Decode CommunicationMod's signed long seed to StS's display seed.

    StS displays seeds using a base-35 alphabet of digits and uppercase
    letters (the letter ``O`` is omitted), while the
    protocol exposes the underlying signed 64-bit value in ``game_state.seed``.
    Negative Java longs are treated as unsigned for this conversion.
    """
    try:
        if isinstance(value, bool):
            return None
        seed = int(value)
    except (TypeError, ValueError):
        return None
    if not -(1 << 63) <= seed < (1 << 64):
        return None
    unsigned_seed = seed + (1 << 64) if seed < 0 else seed
    # StS omits O to avoid confusing it with the digit 0.
    alphabet = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
    if unsigned_seed == 0:
        return "0"
    encoded: list[str] = []
    while unsigned_seed:
        unsigned_seed, remainder = divmod(unsigned_seed, len(alphabet))
        encoded.append(alphabet[remainder])
    return "".join(reversed(encoded))


def run_output_paths(
    trace_dir: Path,
    session_id: str,
    run_index: int,
    run_config: dict[str, Any],
) -> tuple[Path, Path]:
    character = filename_token(run_config.get("character"), "UNKNOWN")
    ascension = filename_token(run_config.get("ascension"), "0")
    seed = filename_token(run_config.get("seed") or "random", "random")
    stem = f"{session_id}-run-{run_index:03d}-{character}-A{ascension}-seed-{seed}"
    return trace_dir / f"action-trace-{stem}.jsonl", trace_dir / f"run-events-{stem}.jsonl"


def auto_action_for_decision(
    decision: dict[str, Any] | None,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve only a controller-approved high-confidence model action."""
    if not isinstance(decision, dict):
        return None
    confidence = decision.get("confidence")
    if not isinstance(confidence, dict) or confidence.get("route") != "auto_execute":
        return None
    action_id = str(decision.get("model_action", ""))
    for action in actions:
        if str(action.get("id", "")) == action_id:
            return action
    return None


def ask_action(
    bridge: TerminalBridge,
    message: dict[str, Any],
    run_id: str | None,
    error: str | None = None,
    policy: PolicyClient | None = None,
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    action_history: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
    actions = semantic_actions(message, visited_shop_rooms)
    if policy is None:
        decision = None
    elif candidate_assist_allowed(message, policy.screens):
        decision = policy.rank(policy_observation(message, visited_shop_rooms, action_history), actions)
    else:
        decision = policy.status_decision()
    # Error responses are a hard human handoff. Repeating an otherwise
    # high-confidence action after the protocol rejects it can create a tight
    # automatic loop and should never be treated as a new policy decision.
    automatic = None if error else auto_action_for_decision(decision, actions)
    if automatic is not None:
        automatic_decision = finalize_decision(decision, automatic, "model_auto")
        confidence = automatic_decision.get("confidence", {}) if automatic_decision else {}
        print(
            f"confidence auto: {automatic.get('id')} "
            f"p={float(confidence.get('top_probability', 0.0)):.3f} "
            f"margin={float(confidence.get('margin', 0.0)):.3f}",
            file=sys.stderr,
            flush=True,
        )
        return str(automatic["command"]), automatic, automatic_decision
    while True:
        bridge.send(
            {
                "type": "view",
                "text": format_view(message, run_id, error, decision, visited_shop_rooms, actions),
                "in_game": bool(message.get("in_game")),
                "decision": decision,
                "overlay": overlay_view_data(message, run_id, actions, action_history),
            }
        )
        response = bridge.receive()
        if response is None:
            continue
        text = str(response.get("text", ""))
        command, action, parse_error = parse_human_input(text, message, decision, visited_shop_rooms)
        if command == "__QUIT__":
            bridge.send({"type": "shutdown", "message": "terminal disconnected; controller is still running"})
            # Keep the CommunicationMod process and TCP listener alive. The
            # terminal client can be restarted and will receive this same view.
            bridge.close_client()
            error = None
            continue
        if command == "__HELP__":
            error = help_text()
            continue
        if parse_error or command is None or action is None:
            error = parse_error or "无效动作。"
            continue
        if command == "STATE":
            print(command, flush=True)
            return command, {"id": "STATE", "kind": "state_request", "command": command}, None
        source = candidate_source_for_input(text, decision)
        return command, action, finalize_decision(decision, action, source)


def main() -> int:
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = Path(__file__).resolve().parents[1]
    try:
        config = load_agent_config(project_root)
    except ConfigError as exc:
        print(f"invalid agent configuration: {exc}", file=sys.stderr, flush=True)
        return 2
    raw_dir = project_root / "data" / "raw" / "communication_controller"
    trace_dir = project_root / "data" / "trajectories"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"messages-{session_id}.jsonl"

    bridge = TerminalBridge(config["server"]["host"], config["server"]["port"])
    policy = PolicyClient(project_root, config)
    pending: dict[str, Any] | None = None
    current_message: dict[str, Any] | None = None
    run_id: str | None = None
    run_config: dict[str, Any] | None = None
    run_index = 0
    step_id = 0
    message_index = 0
    visited_shop_rooms: set[tuple[int, int]] = set()
    action_history: list[dict[str, Any]] = []
    trace_output: TextIO | None = None
    event_output: TextIO | None = None
    trace_path: Path | None = None
    event_path: Path | None = None

    def close_run_outputs() -> None:
        nonlocal trace_output, event_output
        for handle in (trace_output, event_output):
            if handle is not None:
                handle.flush()
                handle.close()
        trace_output = None
        event_output = None

    def open_run_outputs() -> None:
        nonlocal trace_output, event_output, trace_path, event_path
        close_run_outputs()
        assert run_config is not None
        trace_path, event_path = run_output_paths(trace_dir, session_id, run_index, run_config)
        trace_output = trace_path.open("w", encoding="utf-8")
        event_output = event_path.open("w", encoding="utf-8")

    def rebind_run_outputs_after_seed() -> None:
        nonlocal trace_output, event_output, trace_path, event_path
        if run_config is None or run_config.get("seed") is None or trace_path is None or event_path is None:
            return
        final_trace_path, final_event_path = run_output_paths(trace_dir, session_id, run_index, run_config)
        if final_trace_path == trace_path and final_event_path == event_path:
            return
        close_run_outputs()
        os.replace(trace_path, final_trace_path)
        os.replace(event_path, final_event_path)
        trace_path, event_path = final_trace_path, final_event_path
        trace_output = trace_path.open("a", encoding="utf-8")
        event_output = event_path.open("a", encoding="utf-8")

    def write_trace(value: dict[str, Any]) -> None:
        if trace_output is not None:
            write_jsonl(trace_output, value)

    def write_event(value: dict[str, Any]) -> None:
        if event_output is not None:
            write_jsonl(event_output, value)

    def trace_state(message: dict[str, Any]) -> dict[str, Any]:
        if not message.get("in_game"):
            return {"in_game": False}
        value = normalize_state(message)
        value["run_memory"] = run_memory_summary(visited_shop_rooms)
        value["recent_actions"] = recent_action_summary(action_history)
        return value

    # The game must not expose a start action before the policy has finished
    # loading. CommunicationMod waits for this stdout handshake.
    policy.prewarm()

    # Required handshake. No diagnostics may go to stdout.
    print("ready", flush=True)
    print(f"controller session={session_id}", file=sys.stderr, flush=True)

    try:
        with raw_path.open("w", encoding="utf-8") as raw_output:
            for line in sys.stdin:
                raw = line.rstrip("\r\n")
                if not raw:
                    continue
                message_index += 1
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError as exc:
                    write_jsonl(raw_output, {"message_index": message_index, "received_at": now_iso(), "raw": raw, "parse_error": str(exc)})
                    continue
                write_jsonl(raw_output, {"message_index": message_index, "received_at": now_iso(), "raw": raw, "message": message})

                if not isinstance(message, dict):
                    continue
                if "error" in message:
                    if pending is not None:
                        write_trace(
                            {
                                "schema_version": "0.2" if pending.get("decision") else "0.1",
                                "session_id": session_id,
                                "run_id": pending.get("run_id"),
                                "run_config": pending.get("run_config"),
                                "step_id": pending.get("step_id"),
                                "pre_state": pending.get("pre_state"),
                                "legal_actions": pending.get("legal_actions", []),
                                "decision": pending.get("decision"),
                                "action": pending.get("action"),
                                "command": pending.get("command"),
                                "post_state": None,
                                "accepted": False,
                                "error": message.get("error"),
                            },
                        )
                        pending = None
                    if current_message is not None:
                        result = ask_action(
                            bridge,
                            current_message,
                            run_id,
                            str(message.get("error")),
                            policy,
                            visited_shop_rooms,
                            action_history,
                        )
                        if result is None:
                            return 0
                        command, action, decision = result
                        if command != "STATE":
                            pending = make_pending(
                                current_message,
                                run_id,
                                run_config,
                                step_id,
                                action,
                                decision,
                                visited_shop_rooms,
                                action_history,
                            )
                            step_id += 1
                        else:
                            pending = None
                    continue

                if not message.get("ready_for_command"):
                    continue

                if run_config is not None and message.get("in_game") and str(game_state(message).get("screen_type")) == "SHOP_SCREEN":
                    key = shop_visit_key(message)
                    if key is not None and key not in visited_shop_rooms:
                        visited_shop_rooms.add(key)
                        write_event(
                            {
                                "event": "shop_entered",
                                "run_id": run_id,
                                "act": key[0],
                                "floor": key[1],
                                "at": now_iso(),
                            }
                        )

                if pending is not None:
                    action_history.append(accepted_action_history_item(pending, message))
                    del action_history[:-ACTION_HISTORY_LIMIT]
                    write_trace(
                        {
                            "schema_version": "0.2" if pending.get("decision") else "0.1",
                            "session_id": session_id,
                            "run_id": pending.get("run_id"),
                            "run_config": pending.get("run_config"),
                            "step_id": pending.get("step_id"),
                            "screen_type": pending.get("screen_type"),
                            "pre_state": pending.get("pre_state"),
                            "legal_actions": pending.get("legal_actions", []),
                            "decision": pending.get("decision"),
                            "action": pending.get("action"),
                            "command": pending.get("command"),
                            "post_state": trace_state(message),
                            "accepted": True,
                            "stable_after_action": True,
                            "terminal": game_state(message).get("screen_type") in {"GAME_OVER", "VICTORY"},
                        },
                    )
                    pending = None

                current_message = message
                if not message.get("in_game"):
                    # A main-menu observation starts a fresh run context; do
                    # not leak the previous run's action history into it.
                    action_history.clear()
                if run_config is not None and run_config.get("internal_seed") is None and message.get("in_game"):
                    observed_seed = game_state(message).get("seed")
                    if observed_seed is not None:
                        display_seed = display_seed_from_internal(observed_seed)
                        run_config["internal_seed"] = observed_seed
                        if run_config.get("seed") is None and display_seed is not None:
                            run_config["seed"] = display_seed
                            run_config["seed_source"] = "game_state_derived"
                        write_event(
                            {
                                "event": "run_seed_observed",
                                "run_id": run_id,
                                "seed": run_config["seed"],
                                "display_seed": display_seed,
                                "internal_seed": observed_seed,
                                "at": now_iso(),
                            },
                        )
                        rebind_run_outputs_after_seed()
                result = ask_action(
                    bridge,
                    message,
                    run_id,
                    policy=policy,
                    visited_shop_rooms=visited_shop_rooms,
                    action_history=action_history,
                )
                if result is None:
                    return 0
                command, action, decision = result
                if action.get("kind") == "start_run":
                    run_index += 1
                    run_id = f"{session_id}-run-{run_index:03d}"
                    step_id = 0
                    visited_shop_rooms.clear()
                    action_history.clear()
                    run_config = {
                        "character": action.get("character", "IRONCLAD"),
                        "ascension": action.get("ascension", 0),
                        "seed": action.get("seed"),
                        "seed_source": "requested" if action.get("seed") else "random",
                        "internal_seed": None,
                    }
                    open_run_outputs()
                    write_event(
                        {
                            "event": "run_start",
                            "run_id": run_id,
                            "run_config": run_config,
                            "command": command,
                            "at": now_iso(),
                        },
                    )
                    print(command, flush=True)
                    pending = None
                elif command == "STATE":
                    pending = None
                else:
                    pending = make_pending(
                        message,
                        run_id,
                        run_config,
                        step_id,
                        action,
                        decision,
                        visited_shop_rooms,
                        action_history,
                    )
                    step_id += 1
                    print(command, flush=True)
    except KeyboardInterrupt:
        print("controller interrupted", file=sys.stderr, flush=True)
        return 0
    finally:
        close_run_outputs()
        policy.close()
        bridge.close()
    return 0


def make_pending(
    message: dict[str, Any],
    run_id: str | None,
    run_config: dict[str, Any] | None,
    step_id: int,
    action: dict[str, Any],
    decision: dict[str, Any] | None = None,
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    action_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = normalize_state(message)
    normalized["run_memory"] = run_memory_summary(visited_shop_rooms)
    normalized["recent_actions"] = recent_action_summary(action_history)
    return {
        "run_id": run_id,
        "run_config": run_config.copy() if run_config is not None else None,
        "step_id": step_id,
        "screen_type": normalized.get("screen_type"),
        "pre_state": normalized,
        "legal_actions": semantic_actions(message, visited_shop_rooms),
        "decision": decision,
        "action": action,
        "command": action.get("command"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
