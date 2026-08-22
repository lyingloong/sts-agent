"""Human controller for CommunicationMod.

CommunicationMod owns this process' stdin/stdout, so terminal input cannot be
read from input().  The companion human_terminal.py connects over localhost
and supplies semantic actions.  This process keeps stdout strictly reserved
for the CommunicationMod protocol.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from card_knowledge import card_effect


HOST = os.environ.get("STS_AGENT_HOST", "127.0.0.1")
PORT = int(os.environ.get("STS_AGENT_PORT", "8765"))

PLAYER_CLASSES = {
    "IRONCLAD": "IRONCLAD",
    "SILENT": "THE_SILENT",
    "THE_SILENT": "THE_SILENT",
    "DEFECT": "DEFECT",
    "WATCHER": "WATCHER",
}
MAX_ASCENSION = 20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_id(value: Any) -> str:
    return str(value or "")


def is_slot(potion: dict[str, Any]) -> bool:
    return str(potion.get("id", "")).lower() == "potion slot"


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
    keys = ("name", "id", "can_use", "can_discard", "requires_target")
    return {key: potion.get(key) for key in keys if key in potion}


def visible_relic(relic: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "id", "counter")
    return {key: relic.get(key) for key in keys if key in relic}


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


def normalize_screen_state(screen_type: Any, value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if screen_type in {"CARD_REWARD", "GRID"}:
        cards = value.get("cards", [])
        result["cards"] = [visible_card(card) for card in cards if isinstance(card, dict)]
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
        result["options"] = value.get("options", [])
    elif screen_type == "COMBAT_REWARD":
        result["rewards"] = value.get("rewards", [])
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


def semantic_actions(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Build action candidates using stable semantic commands, not clicks."""
    commands = command_set(message)
    state = game_state(message)
    result: list[dict[str, Any]] = []
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
                            }
                        )
                else:
                    result.append(
                        {
                            "id": f"PLAY:card={index}",
                            "command": f"PLAY {index}",
                            "kind": "play",
                            "card_index": index,
                        }
                    )
        if "end" in commands:
            result.append({"id": "END_TURN", "command": "END", "kind": "end_turn"})
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
                                }
                            )
                    else:
                        result.append(
                            {
                                "id": f"POTION_USE:slot={index}",
                                "command": f"POTION USE {index}",
                                "kind": "potion_use",
                                "slot": index,
                            }
                        )
                if potion.get("can_discard", False):
                    result.append(
                        {
                            "id": f"POTION_DISCARD:slot={index}",
                            "command": f"POTION DISCARD {index}",
                            "kind": "potion_discard",
                            "slot": index,
                        }
                    )
        return result

    choices = state.get("choice_list", [])
    if is_map_room_transition(state):
        if "return" in commands:
            result.append({"id": "RETURN", "command": "RETURN", "kind": "return"})
        return result
    if "choose" in commands and isinstance(choices, list):
        for index, label in enumerate(choices):
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


def parse_human_input(text: str, message: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
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
    actions = semantic_actions(message)

    if op in {"q", "quit", "exit"}:
        return "__QUIT__", None, None
    if op in {"h", "help"}:
        return "__HELP__", None, None
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
        command = f"CHOOSE {index}"
        action = action_for_command(actions, command)
        return (command, action, None) if action else (None, None, "该选项当前不可选择。")
    if op in {"c", "confirm", "proceed", "n", "next"}:
        for name in ("CONFIRM", "PROCEED"):
            action = action_for_command(actions, name)
            if action:
                return action["command"], action, None
        return None, None, "当前没有确认或继续动作。"
    if op in {"x", "cancel", "return", "skip", "leave"}:
        preferred = {"x": ("CANCEL", "RETURN", "SKIP", "LEAVE"), "skip": ("SKIP",), "leave": ("LEAVE",)}.get(op, (op.upper(),))
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


def format_view(message: dict[str, Any], run_id: str | None, error: str | None = None) -> str:
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
    lines.append("-" * 78)
    combat = state.get("combat_state")
    if screen == "NONE" and state.get("room_phase") == "COMBAT" and isinstance(combat, dict):
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
    lines.append("合法动作")
    for action in semantic_actions(message):
        label = action["id"]
        if action.get("label"):
            label += f" ({action['label']})"
        lines.append(f"  {label}")
    lines.append("-" * 78)
    lines.append("输入：h 帮助 | s 刷新 | q 断开终端")
    return "\n".join(lines)


class TerminalBridge:
    def __init__(self, host: str, port: int) -> None:
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
        print(f"waiting for terminal client on {HOST}:{PORT}", file=sys.stderr, flush=True)
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


def ask_action(bridge: TerminalBridge, message: dict[str, Any], run_id: str | None, error: str | None = None) -> tuple[str, dict[str, Any]] | None:
    while True:
        bridge.send(
            {
                "type": "view",
                "text": format_view(message, run_id, error),
                "in_game": bool(message.get("in_game")),
            }
        )
        response = bridge.receive()
        if response is None:
            continue
        text = str(response.get("text", ""))
        command, action, parse_error = parse_human_input(text, message)
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
            return command, {"id": "STATE", "kind": "state_request", "command": command}
        return command, action


def main() -> int:
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "communication_controller"
    trace_dir = project_root / "data" / "trajectories"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"messages-{session_id}.jsonl"

    bridge = TerminalBridge(HOST, PORT)
    pending: dict[str, Any] | None = None
    current_message: dict[str, Any] | None = None
    run_id: str | None = None
    run_config: dict[str, Any] | None = None
    run_index = 0
    step_id = 0
    message_index = 0
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
                                "schema_version": "0.1",
                                "session_id": session_id,
                                "run_id": pending.get("run_id"),
                                "run_config": pending.get("run_config"),
                                "step_id": pending.get("step_id"),
                                "pre_state": pending.get("pre_state"),
                                "legal_actions": pending.get("legal_actions", []),
                                "action": pending.get("action"),
                                "command": pending.get("command"),
                                "post_state": None,
                                "accepted": False,
                                "error": message.get("error"),
                            },
                        )
                        pending = None
                    if current_message is not None:
                        result = ask_action(bridge, current_message, run_id, str(message.get("error")))
                        if result is None:
                            return 0
                        command, action = result
                        if command != "STATE":
                            pending = make_pending(current_message, run_id, run_config, step_id, action)
                            step_id += 1
                        else:
                            pending = None
                    continue

                if not message.get("ready_for_command"):
                    continue

                if pending is not None:
                    write_trace(
                        {
                            "schema_version": "0.1",
                            "session_id": session_id,
                            "run_id": pending.get("run_id"),
                            "run_config": pending.get("run_config"),
                            "step_id": pending.get("step_id"),
                            "screen_type": pending.get("screen_type"),
                            "pre_state": pending.get("pre_state"),
                            "legal_actions": pending.get("legal_actions", []),
                            "action": pending.get("action"),
                            "command": pending.get("command"),
                            "post_state": normalize_state(message) if message.get("in_game") else {"in_game": False},
                            "accepted": True,
                            "stable_after_action": True,
                            "terminal": game_state(message).get("screen_type") in {"GAME_OVER", "VICTORY"},
                        },
                    )
                    pending = None

                current_message = message
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
                result = ask_action(bridge, message, run_id)
                if result is None:
                    return 0
                command, action = result
                if action.get("kind") == "start_run":
                    run_index += 1
                    run_id = f"{session_id}-run-{run_index:03d}"
                    step_id = 0
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
                    pending = make_pending(message, run_id, run_config, step_id, action)
                    step_id += 1
                    print(command, flush=True)
    except KeyboardInterrupt:
        print("controller interrupted", file=sys.stderr, flush=True)
        return 0
    finally:
        close_run_outputs()
        bridge.close()
    return 0


def make_pending(
    message: dict[str, Any],
    run_id: str | None,
    run_config: dict[str, Any] | None,
    step_id: int,
    action: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_state(message)
    return {
        "run_id": run_id,
        "run_config": run_config.copy() if run_config is not None else None,
        "step_id": step_id,
        "screen_type": normalized.get("screen_type"),
        "pre_state": normalized,
        "legal_actions": semantic_actions(message),
        "action": action,
        "command": action.get("command"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
