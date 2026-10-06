"""Human controller for CommunicationMod.

CommunicationMod owns this process' stdin/stdout, so terminal input cannot be
read from input().  The companion terminal client connects over localhost
and supplies semantic actions.  This process keeps stdout strictly reserved
for the CommunicationMod protocol.
"""

from __future__ import annotations

import copy
import argparse
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from sts_agent.config import ConfigError, load_agent_config
from sts_agent.learning.posthoc_feedback import (
    SCHEMA_VERSION as POSTHOC_SCHEMA_VERSION,
    PosthocFeedbackClient,
    PosthocFeedbackError,
    action_review_from_trace,
    now_utc as posthoc_now_utc,
    terminal_summary as posthoc_terminal_summary,
    write_artifact as write_posthoc_artifact,
)
from sts_agent.learning.prompt_evolution import PromptEvolutionError, PromptEvolutionManager
from sts_agent.policy.teacher import RemoteTeacherClient, TeacherError
from sts_agent.paths import PROJECT_ROOT


ACTION_HISTORY_LIMIT = 6
# Keep a longer private window so multi-step events can retain their opening
# text and state transitions, while ordinary prompts still receive six recent
# actions only.
ACTION_HISTORY_STORAGE_LIMIT = 16
EVENT_HISTORY_LIMIT = 16
# CommunicationMod can report a ready combat state before Smoke Bomb's escape
# animation has switched the room to its actual post-escape screen.
SMOKE_BOMB_ANIMATION_DELAY_SECONDS = 5.0
SMOKE_BOMB_STATE_POLL_SECONDS = 1.0
SMOKE_BOMB_TRANSITION_TIMEOUT_SECONDS = 15.0
# Give the overlay a brief frame to show an automatic decision before the
# command is sent to the game.
# Give the game UI enough time to visibly settle before a controller-owned
# action is sent. This affects only environment automation, not model calls or
# human/Codex actions.
AUTOMATIC_ACTION_DELAY_SECONDS = 0.25


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
    """Find reward/shop choices unavailable from potion capacity or Sozu."""
    state = game_state(message)
    if not potion_inventory_full(state) and not has_relic(state, "SOZU"):
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
            str(item.get("name")).strip().casefold()
            for item in screen_state.get("potions", [])
            if isinstance(item, dict) and item.get("name")
        }
        blocked.update(
            index
            for index, label in enumerate(choices)
            if any(name and name in str(label).strip().casefold() for name in potion_names)
        )
    return blocked


def blocked_potion_choice_error(message: dict[str, Any], index: int) -> str | None:
    if index in blocked_potion_reward_indices(message):
        if has_relic(game_state(message), "SOZU"):
            return "持有苏祖（Sozu），不能再获得或购买药水；请改选其他选项。"
        return "药水栏已满，不能拾取或购买药水；请改选其他奖励，或先丢弃一瓶药水。"
    return None


def blocked_gold_reward_indices(message: dict[str, Any]) -> set[int]:
    """Find CombatReward gold choices made invalid by Ectoplasm."""
    state = game_state(message)
    if not has_relic(state, "ECTOPLASM") or str(state.get("screen_type") or "").upper() != "COMBAT_REWARD":
        return set()
    choices = state.get("choice_list")
    if not isinstance(choices, list):
        return set()
    screen_state = state.get("screen_state")
    rewards = screen_state.get("rewards") if isinstance(screen_state, dict) else None
    blocked = {
        index
        for index, reward in enumerate(rewards)
        if isinstance(reward, dict) and str(reward.get("reward_type") or "").upper() == "GOLD"
    } if isinstance(rewards, list) else set()
    # Older CommunicationMod versions sometimes expose only a generic label.
    blocked.update(
        index
        for index, label in enumerate(choices)
        if str(label).strip().casefold() in {"gold", "金币"}
    )
    return blocked


def blocked_gold_choice_error(message: dict[str, Any], index: int) -> str | None:
    if index in blocked_gold_reward_indices(message):
        return "持有灵体外质（Ectoplasm），不能再获得金币；请改选其他奖励。"
    return None


def command_set(message: dict[str, Any]) -> set[str]:
    return {str(item).lower() for item in message.get("available_commands", [])}


def game_state(message: dict[str, Any]) -> dict[str, Any]:
    value = message.get("game_state")
    return value if isinstance(value, dict) else {}


KEY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "ruby": ("ruby", "ruby_key", "red", "red_key", "has_ruby_key", "hasRubyKey"),
    "emerald": ("emerald", "emerald_key", "green", "green_key", "has_emerald_key", "hasEmeraldKey"),
    "sapphire": ("sapphire", "sapphire_key", "blue", "blue_key", "has_sapphire_key", "hasSapphireKey"),
}


def normalize_keys(state: dict[str, Any]) -> dict[str, bool] | None:
    """Normalize heart-key flags from CommunicationMod or ActionRecorder.

    CommunicationMod versions before the ActionRecorder bridge did not expose
    keys at all. Return ``None`` in that case rather than inventing three false
    values, which would incorrectly tell a policy that the keys were checked.
    """
    raw = state.get("keys")
    source = raw if isinstance(raw, dict) else state
    result: dict[str, bool] = {}
    for canonical, aliases in KEY_FIELD_ALIASES.items():
        for alias in aliases:
            if alias not in source:
                continue
            value = source.get(alias)
            if isinstance(value, (bool, int, float)):
                result[canonical] = bool(value)
                break
    if result:
        return result
    if isinstance(raw, list):
        normalized = {_status_key(item) for item in raw}
        result = {
            canonical: any(_status_key(alias) in normalized for alias in aliases)
            for canonical, aliases in KEY_FIELD_ALIASES.items()
        }
        return result
    return None


# Relic rewards remain policy decisions: a relic can compete with a key or
# another mutually exclusive reward. Only deterministic, non-strategic reward
# handling is automated here.
REWARD_AUTOMATION_ORDER = ("GOLD", "POTION", "CARD")
# In StS1 these are the only potions that can be deliberately consumed
# outside combat. CommunicationMod may report ``can_use=true`` globally for
# other potions, so do not treat that flag as context-independent.
NON_COMBAT_USABLE_POTIONS = frozenset({"BLOODPOTION", "FRUITJUICE"})


def combat_reward_entries(message: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """Return visible CombatRewardScreen entries with their choice indices.

    CommunicationMod exposes one parent-screen choice per entry in ``rewards``.
    Do not infer reward types from arbitrary text: a missing or malformed entry
    simply remains a model decision instead of risking a wrong automatic pick.
    """
    state = game_state(message)
    if str(state.get("screen_type") or "").upper() != "COMBAT_REWARD":
        return []
    choices = state.get("choice_list")
    screen_state = state.get("screen_state")
    rewards = screen_state.get("rewards") if isinstance(screen_state, dict) else None
    if not isinstance(choices, list) or not isinstance(rewards, list):
        return []
    result: list[tuple[int, str, dict[str, Any]]] = []
    for index, reward in enumerate(rewards):
        if index >= len(choices) or not isinstance(reward, dict):
            continue
        reward_type = str(reward.get("reward_type") or "").upper()
        if reward_type in REWARD_AUTOMATION_ORDER:
            result.append((index, reward_type, reward))
    return result


def reward_automation_label(reward_type: str, reward: dict[str, Any]) -> str:
    """Describe a controller-owned parent reward action for history and trace."""
    if reward_type == "RELIC":
        relic = reward.get("relic") if isinstance(reward.get("relic"), dict) else reward
        return f"自动拾取遗物 {relic.get('name') or relic.get('id') or ''}".strip()
    if reward_type == "GOLD":
        gold = reward.get("gold")
        return f"自动拾取金币 {gold}" if gold is not None else "自动拾取金币"
    if reward_type == "POTION":
        potion = reward.get("potion") if isinstance(reward.get("potion"), dict) else reward
        return f"自动拾取药水 {potion.get('name') or potion.get('id') or ''}".strip()
    return "自动查看卡牌奖励"


def environment_auto_action(
    message: dict[str, Any],
    actions: list[dict[str, Any]],
    visited_shop_rooms: set[tuple[int, int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Choose one non-strategic progress action before calling a policy.

    The controller owns only actions with no strategic trade-off: visible
    combat reward gold, a potion when an inventory slot is free, opening a
    visible card reward, and entering a just-reached shop. Relic rewards stay
    as policy decisions because they may compete with a key or another reward.
    It also executes
    a sole legal action, because there is no remaining choice for a policy to
    make (for example, ``PROCEED`` after leaving a shop or ``END_TURN`` when a
    combat hand has no playable cards or usable potions). Card selection, boss
    relics, purchases, purge decisions, and full-inventory potion choices
    remain policy actions.
    """
    state = game_state(message)
    screen_type = str(state.get("screen_type") or "").upper()
    actions_by_id = {str(action.get("id")): action for action in actions}

    if screen_type == "COMBAT_REWARD":
        for reward_type in REWARD_AUTOMATION_ORDER:
            for index, entry_type, reward in combat_reward_entries(message):
                if entry_type != reward_type:
                    continue
                if reward_type == "POTION" and (potion_inventory_full(state) or has_relic(state, "SOZU")):
                    continue
                if reward_type == "GOLD" and has_relic(state, "ECTOPLASM"):
                    continue
                action = actions_by_id.get(f"CHOOSE:index={index}")
                if action is None:
                    # A full inventory masks the potion action; continue with
                    # later safe entries such as CARD.
                    continue
                action = dict(action)
                action["label"] = reward_automation_label(reward_type, reward)
                return action, {
                    "rule": f"combat_reward_{reward_type.lower()}",
                    "reason": action["label"],
                    "reward_type": reward_type,
                }

    if screen_type == "SHOP_ROOM":
        # Remember visits for automation only. Orrery rewards can close the
        # merchant screen; revisiting remains a real, legal player decision.
        already_visited = shop_visit_key(message) in (visited_shop_rooms or set())
        choices = state.get("choice_list")
        if isinstance(choices, list) and not already_visited:
            for index, choice in enumerate(choices):
                normalized = "".join(str(choice).casefold().split())
                if normalized not in {"merchant", "shop", "entershop", "商人", "商店", "进入商店"}:
                    continue
                action = actions_by_id.get(f"CHOOSE:index={index}")
                if action is None:
                    continue
                action = dict(action)
                action["label"] = "自动打开商人界面"
                return action, {
                    "rule": "shop_open_merchant",
                    "reason": "自动打开商人界面",
                }

    # Map navigation is safe to automate only after semantic action
    # generation has resolved all route choices.  In particular, do not use
    # ``screen_state.next_nodes`` directly: relics such as Winged Greaves can
    # alter the route action space, and stale protocol fields may coexist with
    # the current map view.  Requiring exactly one final CHOOSE action keeps
    # this conservative and makes potion operations irrelevant here (they are
    # filtered by semantic_actions before this function is called).
    if is_map_screen(state):
        route_actions = [
            action for action in actions
            if str(action.get("kind") or "") == "choose"
            and str(action.get("command") or "").upper().startswith("CHOOSE ")
        ]
        if len(actions) == 1 and len(route_actions) == 1:
            action = dict(route_actions[0])
            label = str(action.get("label") or action.get("id") or "地图节点")
            action["label"] = f"自动选择唯一可达节点：{label}"
            return action, {
                "rule": "map_single_reachable_node",
                "reason": label,
                "screen_type": screen_type,
            }
        # Multiple route actions are meaningful choices (including routes
        # exposed by Winged Greaves), so leave them to the policy.
        return None

    # Do this last so explicit reward/shop rules retain their useful labels.
    # The semantic action compiler has already filtered unavailable cards,
    # targets, potions, and blocked choices, so one remaining action is safe to
    # execute without an inference request.
    if (
        len(actions) == 1
        and not is_map_screen(state)
        and actions[0].get("kind") not in {"potion_use", "potion_discard"}
    ):
        action = dict(actions[0])
        label = str(action.get("label") or action.get("command") or action.get("id") or "唯一动作")
        action["label"] = f"自动执行 {label}"
        return action, {
            "rule": "single_legal_action",
            "reason": f"当前仅有一个合法动作：{label}",
            "screen_type": screen_type,
        }
    return None


def environment_auto_decision(action: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    """Make controller progress actions durable and distinct from policy actions."""
    action_id = str(action.get("id") or "")
    return {
        "mode": "environment_automation",
        "model_id": "controller",
        "candidate_source": "environment_guard",
        "model_action": action_id,
        "model_candidates": [],
        "environment": detail,
        "executed_action": action_id,
        "source": "environment_auto",
        "overridden": False,
    }


CARD_MODEL_FIELDS = (
    "id",
    "name",
    "type",
    "cost",
    "cost_for_turn",
    "upgrades",
    "is_playable",
    "description",
    "raw_description",
    "description_source",
    "description_complete",
    "unresolved_description_variables",
    "target",
    "damage",
    "base_damage",
    "block",
    "base_block",
    "magic_number",
    "hit_count",
    "multi_damage",
    "rarity",
    "color",
    "exhaust",
    "ethereal",
    "innate",
    "retain",
    "self_retain",
    "purge_on_use",
)
POTION_MODEL_FIELDS = (
    "id",
    "name",
    "description",
    "raw_description",
    "description_source",
    "tooltips",
    "target",
    "can_use",
)
RELIC_MODEL_FIELDS = ("id", "name", "description", "raw_description", "description_source", "counter")
_MISSING = object()


def _numeric_value(value: Any) -> float | int | None:
    """Parse a protocol value that represents one number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*", value)
        if match:
            number = float(match.group(1))
            if math.isfinite(number):
                return int(number) if number.is_integer() else number
    return None


def _integer_value(value: Any) -> int | None:
    number = _numeric_value(value)
    if number is None:
        return None
    return int(number)


def _status_key(value: Any) -> str:
    """Normalize English and localized Power names without dropping Unicode."""
    text = str(value or "").strip().upper()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def _power_amount(powers: Any, aliases: tuple[str, ...]) -> int:
    """Sum an effect amount for the named Power IDs/names."""
    wanted = {_status_key(alias) for alias in aliases}
    total = 0
    if not isinstance(powers, list):
        return total
    for power in powers:
        if not isinstance(power, dict):
            continue
        keys = {_status_key(power.get(field)) for field in ("id", "name")}
        keys.discard("")
        # CommunicationMod has used both ``Strength`` and ``StrengthPower``
        # spellings across versions. Do not use substring matching: powers
        # such as Strength Down must not be mistaken for Strength.
        matched = any(
            key in wanted or key in {f"{alias}POWER" for alias in wanted}
            for key in keys
        )
        if matched:
            amount = _integer_value(power.get("amount"))
            if amount is not None:
                total += amount
    return total


def _calculated_card_values(
    card: dict[str, Any],
    *,
    player_powers: Any = None,
) -> dict[str, int]:
    """Use the live hand preview as-is; cached values already include Powers."""
    result: dict[str, int] = {}
    card_type = _status_key(card.get("type"))
    if "ATTACK" in card_type or "攻击" in card_type:
        damage = _integer_value(card.get("final_single_hit_damage"))
        if damage is None:
            damage = _integer_value(card.get("damage"))
        if damage is not None:
            result["final_single_hit_damage"] = damage
    block = _integer_value(card.get("final_block"))
    if block is None:
        block = _integer_value(card.get("block"))
    if block is not None:
        result["final_block"] = block
    return result


# These relics can cause follow-up selection screens after pickup.
# a player-directed card selection. Effects such as Pandora's Box, War Paint,
# and Whetstone are deliberately absent because their card changes are random
# or automatic rather than a decision screen.
RELIC_PICKUP_CARD_SELECTIONS: dict[str, dict[str, Any]] = {
    "ASTROLABE": {
        "screen_types": ("GRID",),
        "selection_type": "ASTROLABE_TRANSFORM_UPGRADE",
        "selection_mode": "TRANSFORM_UPGRADE",
        "required_cards": 3,
        "instruction": "星盘：必须选择 3 张不同的当前牌；它们会被变化为随机牌并升级。优先变化重复、低价值或妨碍构筑的牌，避免变化关键牌。",
    },
    "EMPTYCAGE": {
        "screen_types": ("GRID",),
        "selection_type": "EMPTY_CAGE_PURGE",
        "selection_mode": "PURGE",
        "required_cards": 2,
        "instruction": "空鸟笼：必须选择 2 张不同的当前牌并永久移除。优先移除重复、低价值或妨碍构筑的牌，避免移除关键牌。",
    },
    "BOTTLEDFLAME": {
        "screen_types": ("GRID",),
        "selection_type": "BOTTLED_FLAME",
        "selection_mode": "BOTTLE",
        "required_cards": 1,
        "instruction": "火焰瓶：选择 1 张攻击牌；之后每场战斗开始时它会进入手牌。优先选择开局就有价值、且不会妨碍起手的攻击牌。",
    },
    "BOTTLEDLIGHTNING": {
        "screen_types": ("GRID",),
        "selection_type": "BOTTLED_LIGHTNING",
        "selection_mode": "BOTTLE",
        "required_cards": 1,
        "instruction": "闪电瓶：选择 1 张技能牌；之后每场战斗开始时它会进入手牌。优先选择开局就有价值、且不会妨碍起手的技能牌。",
    },
    "BOTTLEDTORNADO": {
        "screen_types": ("GRID",),
        "selection_type": "BOTTLED_TORNADO",
        "selection_mode": "BOTTLE",
        "required_cards": 1,
        "instruction": "龙卷瓶：选择 1 张能力牌；之后每场战斗开始时它会进入手牌。优先选择能尽早发挥长期价值的能力牌。",
    },
    "DOLLYSMIRROR": {
        "screen_types": ("GRID",),
        "selection_type": "DOLLYS_MIRROR_COPY",
        "selection_mode": "COPY",
        "required_cards": 1,
        "instruction": "多莉的镜子：选择 1 张当前牌，牌组会获得它的一张额外复制。优先复制当前构筑的核心牌，不要无目的地复制基础牌。",
    },
    "ORRERY": {
        "screen_types": ("CARD_REWARD",),
        "selection_type": "ORRERY_CARD_REWARDS",
        "selection_mode": "RELIC_CARD_REWARD",
        "required_cards": 5,
        "instruction": "奥雷里：这是获得遗物后触发的连续 5 次卡牌奖励。每一组都依据当前牌组缺口独立决定拿牌或跳过，不要把它当成普通的一次奖励。",
    },
}


def public_value(source: dict[str, Any], field: str, aliases: tuple[str, ...] = ()) -> Any:
    """Return the first non-null public value for a canonical schema field."""
    for key in (field, *aliases):
        value = source.get(key)
        if value is not None:
            return value
    return _MISSING


def merge_public_fields(
    runtime: dict[str, Any],
    fields: tuple[str, ...],
    runtime_aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Project only fields present in the current game observation."""
    result: dict[str, Any] = {}
    for field in fields:
        value = public_value(runtime, field, (runtime_aliases or {}).get(field, ()))
        if value is not _MISSING:
            result[field] = value
    if not result.get("description") and result.get("raw_description"):
        result["description"] = result["raw_description"]
    return result


def visible_card(
    card: dict[str, Any],
    *,
    player_powers: Any = None,
) -> dict[str, Any]:
    """Expose game-reported card facts without a local entity database."""
    result = merge_public_fields(card, CARD_MODEL_FIELDS)
    for field in ("id", "name", "type", "cost", "upgrades", "is_playable", "description"):
        result.setdefault(field, None)
    if player_powers is not None:
        result.update(_calculated_card_values(card))
    return result


def card_effect_summary(card: dict[str, Any], *, player_powers: Any = None) -> str:
    """Render live card text and values compactly for the controller UI."""
    values = visible_card(card, player_powers=player_powers)
    fields = []
    for key, label in (("type", "type"), ("target", "target"), ("cost", "cost"), ("cost_for_turn", "cost now")):
        if values.get(key) is not None:
            fields.append(f"{label}={values[key]}")
    for key, label in (("damage", "dmg"), ("block", "block"), ("magic_number", "magic"), ("hit_count", "hits")):
        if values.get(key) is not None:
            fields.append(f"{label}={values[key]}")
    description = values.get("description")
    result = " | " + " ".join(fields) if fields else ""
    if description:
        result += f" | {str(description).replace(chr(10), ' / ')}"
    return result


def visible_potion(potion: dict[str, Any]) -> dict[str, Any]:
    result = merge_public_fields(potion, POTION_MODEL_FIELDS)
    for field in ("id", "name", "description", "target", "can_use"):
        result.setdefault(field, None)
    return result


def visible_relic(relic: dict[str, Any]) -> dict[str, Any]:
    result = merge_public_fields(relic, RELIC_MODEL_FIELDS)
    for field in ("id", "name", "description"):
        result.setdefault(field, None)
    if isinstance(result.get("counter"), (int, float)) and not isinstance(result["counter"], bool) and result["counter"] < 0:
        result.pop("counter")
    return result


def relic_identity(relic: dict[str, Any]) -> str:
    """Return a normalized identifier for a live relic instance."""
    return _status_key(relic.get("id")) or _status_key(relic.get("name"))


RELIC_DECISION_RULES = {
    "SOZU": {
        "kind": "NO_POTION_GAIN",
        "instruction": "持有苏祖（Sozu）：不能再获得或购买药水。",
    },
    "ECTOPLASM": {
        "kind": "NO_GOLD_GAIN",
        "instruction": "持有灵体外质（Ectoplasm）：不能再获得金币。",
    },
    "RUNICDOME": {
        "kind": "HIDDEN_ENEMY_INTENTS",
        "instruction": "持有符文圆顶（Runic Dome）：敌人意图在战斗中不可见。",
    },
    "VELVETCHOKER": {
        "kind": "CARD_PLAY_LIMIT",
        "instruction": "持有添水（Velvet Choker）：每回合最多打出 6 张牌。",
    },
    "CURSEDKEY": {
        "kind": "CHEST_ADDS_CURSE",
        "instruction": "持有诅咒钥匙（Cursed Key）：开启宝箱会获得一张诅咒。",
    },
    "COFFEEDRIPPER": {
        "kind": "NO_REST",
        "instruction": "持有咖啡滴滤器（Coffee Dripper）：不能在营火休息。",
    },
    "FUSIONHAMMER": {
        "kind": "NO_SMITH",
        "instruction": "持有融合之锤（Fusion Hammer）：不能在营火锻造。",
    },
}


def active_relic_keys(state: dict[str, Any]) -> set[str]:
    relics = state.get("relics")
    if not isinstance(relics, list):
        return set()
    return {
        relic_identity(relic)
        for relic in relics
        if isinstance(relic, dict) and relic_identity(relic)
    }


def has_relic(state: dict[str, Any], relic_key: str) -> bool:
    return _status_key(relic_key) in active_relic_keys(state)


def card_plays_this_turn(action_history: list[dict[str, Any]] | None) -> int:
    """Count accepted card plays after the most recent end-turn boundary."""
    count = 0
    for item in reversed(action_history or []):
        if not isinstance(item, dict):
            continue
        if item.get("action_kind") == "end_turn":
            break
        if item.get("action_kind") == "play":
            count += 1
    return count


def relic_decision_constraints(
    relics: Any,
    action_history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Expose public, persistent relic rules that change future choices."""
    result: list[dict[str, Any]] = []
    if not isinstance(relics, list):
        return result
    plays_this_turn = card_plays_this_turn(action_history)
    for relic in relics:
        if not isinstance(relic, dict):
            continue
        key = relic_identity(relic)
        rule = RELIC_DECISION_RULES.get(key)
        if rule is None:
            continue
        item = {
            "kind": rule["kind"],
            "relic": {field: relic.get(field) for field in ("id", "name") if relic.get(field) is not None},
            "instruction": rule["instruction"],
        }
        if key == "VELVETCHOKER":
            item["plays_this_turn"] = plays_this_turn
            item["plays_remaining_this_turn"] = max(0, 6 - plays_this_turn)
        result.append(item)
    return result


def newly_gained_relics(before: Any, after: Any) -> list[dict[str, Any]]:
    """Preserve relic additions across one accepted action, including swaps."""
    before_counts = Counter(
        relic_identity(relic)
        for relic in before if isinstance(relic, dict) and relic_identity(relic)
    ) if isinstance(before, list) else Counter()
    result: list[dict[str, Any]] = []
    if not isinstance(after, list):
        return result
    for relic in after:
        if not isinstance(relic, dict):
            continue
        key = relic_identity(relic)
        if key and before_counts[key] > 0:
            before_counts[key] -= 1
            continue
        result.append(visible_relic(relic))
    return result


def powers_summary(powers: Any) -> list[dict[str, Any]] | None:
    result: list[dict[str, Any]] = []
    if not isinstance(powers, list):
        return None
    for power in powers:
        if not isinstance(power, dict):
            continue
        item: dict[str, Any] = {}
        for key in (
            "name", "id", "amount", "type", "description", "raw_description",
            "description_source", "description_complete", "unresolved_description_variables",
        ):
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
    """Identify a shop by Act and floor for visit memory and automation."""
    state = game_state(message)
    try:
        act = int(state.get("act"))
        floor = int(state.get("floor"))
    except (TypeError, ValueError):
        return None
    return act, floor


def is_shop_room(state: dict[str, Any]) -> bool:
    return str(state.get("screen_type") or "").upper() == "SHOP_ROOM"


def run_memory_summary(
    visited_shop_rooms: set[tuple[int, int]] | None,
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Expose visit/reward memory; a shop visit is not a re-entry constraint."""
    rooms = sorted(visited_shop_rooms or set())
    declined = sorted(declined_card_reward_keys or set())
    return {
        "visited_shop_rooms": [{"act": act, "floor": floor} for act, floor in rooms],
        "declined_card_rewards": [{"act": act, "floor": floor} for act, floor in declined],
    }


def recent_action_summary(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep a small, stable action context for the next model decision."""
    if not isinstance(history, list):
        return []
    result: list[dict[str, Any]] = []
    for item in history[-ACTION_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        compact: dict[str, Any] = {}
        for key in ("step_id", "screen_before", "action_kind", "action_id", "action", "screen_after", "source"):
            value = item.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                compact[key] = value
        if compact:
            result.append(compact)
    return result


def event_snapshot(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Capture the public, changing part of one event screen."""
    if not isinstance(state, dict) or str(state.get("screen_type") or "").upper() != "EVENT":
        return None
    screen_state = state.get("screen_state")
    if not isinstance(screen_state, dict):
        return None
    return {
        "event_id": screen_state.get("event_id"),
        "event_name": screen_state.get("event_name"),
        "body_text": screen_state.get("body_text"),
        "choices": list(state.get("choices", [])),
        "options": screen_state.get("options", []),
    }


def same_event(snapshot: dict[str, Any], event_id: Any, event_name: Any) -> bool:
    """Match snapshots by stable ID, falling back to the visible event name."""
    snapshot_id = snapshot.get("event_id")
    if snapshot_id is not None and event_id is not None:
        return str(snapshot_id) == str(event_id)
    snapshot_name = snapshot.get("event_name")
    return snapshot_name is not None and event_name is not None and str(snapshot_name) == str(event_name)


def event_history_summary(
    history: list[dict[str, Any]] | None,
    current_event: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return a bounded transition history for the currently open event."""
    if not isinstance(history, list) or not isinstance(current_event, dict):
        return []
    event_id = current_event.get("event_id")
    event_name = current_event.get("event_name")
    result: list[dict[str, Any]] = []
    for item in history[-EVENT_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        before = item.get("event_before")
        after = item.get("event_after")
        before_matches = isinstance(before, dict) and same_event(before, event_id, event_name)
        after_matches = isinstance(after, dict) and same_event(after, event_id, event_name)
        if not before_matches and not after_matches:
            continue
        entry: dict[str, Any] = {}
        for key in ("step_id", "action_kind", "action_id", "action"):
            value = item.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                entry[key] = value
        if before_matches:
            entry["before"] = before
        if after_matches:
            entry["after"] = after
        result.append(entry)
    return result


def action_parameter(action_id: Any, parameter: str) -> int | None:
    """Read one integer parameter from a canonical semantic action ID."""
    marker = f"{parameter}="
    text = str(action_id or "")
    if marker not in text:
        return None
    raw = text.split(marker, 1)[1].split(":", 1)[0]
    try:
        return int(raw)
    except ValueError:
        return None


def is_smoke_bomb_potion_use(pending: dict[str, Any]) -> bool:
    """Return whether a pending action used Smoke Bomb.

    Use the pre-action inventory because the acknowledgement usually arrives
    after the potion has already disappeared from the live game state.
    """
    action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
    if action.get("kind") != "potion_use":
        return False
    slot = action.get("slot")
    if not isinstance(slot, int):
        slot = action_parameter(action.get("id"), "slot")
    pre_state = pending.get("pre_state") if isinstance(pending.get("pre_state"), dict) else {}
    potions = pre_state.get("potions")
    if not isinstance(potions, list) or slot is None or not 0 <= slot < len(potions):
        return False
    potion = potions[slot]
    if not isinstance(potion, dict):
        return False
    potion_id = str(potion.get("id") or "").replace(" ", "").replace("_", "").casefold()
    potion_name = str(potion.get("name") or "").strip().casefold()
    return potion_id == "smokebomb" or potion_name in {"smoke bomb", "烟雾弹"}


def smoke_bomb_escape_in_progress(message: dict[str, Any]) -> bool:
    """Return whether a Smoke Bomb transition still exposes the old combat."""
    if not message.get("in_game"):
        return False
    state = game_state(message)
    return (
        str(state.get("screen_type") or "").upper() == "NONE"
        and str(state.get("room_phase") or "").upper() == "COMBAT"
    )


def generated_card_choice_source(pending: dict[str, Any]) -> dict[str, Any] | None:
    """Describe a potion/card action that opened a combat-time card choice."""
    action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
    pre_state = pending.get("pre_state") if isinstance(pending.get("pre_state"), dict) else {}
    kind = str(action.get("kind") or "")
    source: dict[str, Any] | None = None
    source_type: str | None = None
    if kind == "potion_use":
        slot = action_parameter(action.get("id"), "slot")
        potions = pre_state.get("potions")
        if isinstance(potions, list) and slot is not None and 0 <= slot < len(potions):
            potion = potions[slot]
            if isinstance(potion, dict):
                source = potion
                source_type = "POTION"
    elif kind == "play":
        card_index = action_parameter(action.get("id"), "card")
        combat = pre_state.get("combat")
        hand = combat.get("hand") if isinstance(combat, dict) else None
        # Semantic card indices are one-based, matching CommunicationMod.
        if isinstance(hand, list) and card_index is not None and 1 <= card_index <= len(hand):
            card = hand[card_index - 1]
            if isinstance(card, dict):
                source = card
                source_type = "CARD"
    if source is None or source_type is None:
        return None
    fields = ("id", "name", "description", "type", "cost", "upgrades")
    return {"source_type": source_type} | {key: source.get(key) for key in fields if key in source}


def generated_card_choice_context(
    normalized: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Annotate card choices created by a just-used potion or just-played card."""
    if (
        str(normalized.get("screen_type") or "").upper() != "CARD_REWARD"
        or str(normalized.get("room_phase") or "").upper() != "COMBAT"
        or not isinstance(history, list)
        or not history
    ):
        return None
    previous = history[-1]
    if not isinstance(previous, dict) or previous.get("screen_after") != "CARD_REWARD":
        return None
    source = previous.get("card_choice_source")
    if not isinstance(source, dict):
        return None
    return {
        "kind": "COMBAT_GENERATED_CARD_CHOICE",
        "source": source,
        "guidance": "这是刚才药水或出牌生成的战斗内选牌，不是普通战后永久卡牌奖励；以 source.description 说明的持续时间、费用和效果为准。",
    }


def relic_pickup_card_selection_context(
    normalized: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Identify the first continuous card-selection flow caused by a relic.

    The relic persists in the inventory after its pickup effect finishes, so
    checking the current relic list alone would mislabel unrelated future
    events. Instead, walk backward through only the current continuous
    selection screen until the action that added the relic is reached.
    """
    screen_type = str(normalized.get("screen_type") or "").upper()
    if screen_type not in {"GRID", "CARD_REWARD"} or not isinstance(history, list):
        return None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        gained = item.get("relics_gained")
        if isinstance(gained, list):
            for relic in gained:
                if not isinstance(relic, dict):
                    continue
                specification = RELIC_PICKUP_CARD_SELECTIONS.get(relic_identity(relic))
                if specification is None or screen_type not in specification["screen_types"]:
                    continue
                screen_state = normalized.get("screen_state")
                selected = []
                if isinstance(screen_state, dict):
                    selected = screen_state.get("selected_cards")
                    if not isinstance(selected, list):
                        selected = screen_state.get("selected")
                return {
                    "kind": "RELIC_PICKUP_CARD_SELECTION",
                    "selection_type": specification["selection_type"],
                    "selection_mode": specification["selection_mode"],
                    "selection_instruction": specification["instruction"],
                    "required_cards": specification["required_cards"],
                    "selected_count": len(selected) if isinstance(selected, list) else 0,
                    "source": visible_relic(relic),
                }
            return None
        before = str(item.get("screen_before") or "").upper()
        after = str(item.get("screen_after") or "").upper()
        if before != screen_type or after != screen_type:
            return None
    return None


def attach_runtime_context(
    normalized: dict[str, Any],
    visited_shop_rooms: set[tuple[int, int]] | None,
    action_history: list[dict[str, Any]] | None,
    declined_card_reward_keys: set[tuple[int, int]] | None,
) -> dict[str, Any]:
    """Add bounded cross-step context shared by live prompts and trace records."""
    normalized["recent_actions"] = recent_action_summary(action_history)
    normalized["run_memory"] = run_memory_summary(visited_shop_rooms, declined_card_reward_keys)
    constraints = relic_decision_constraints(normalized.get("relics"), action_history)
    if constraints:
        normalized["decision_constraints"] = constraints
    selection_context = relic_pickup_card_selection_context(normalized, action_history)
    if selection_context is None:
        selection_context = generated_card_choice_context(normalized, action_history)
    if selection_context is not None:
        normalized["selection_context"] = selection_context
        screen_state = normalized.get("screen_state")
        if isinstance(screen_state, dict) and selection_context.get("kind") == "RELIC_PICKUP_CARD_SELECTION":
            screen_state["selection_mode"] = selection_context["selection_mode"]
            screen_state["selection_instruction"] = selection_context["selection_instruction"]
    screen_state = normalized.get("screen_state")
    if str(normalized.get("screen_type") or "").upper() == "EVENT" and isinstance(screen_state, dict):
        event_history = event_history_summary(action_history, screen_state)
        if event_history:
            normalized["event_history"] = event_history
    return normalized


def accepted_action_history_item(pending: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """Convert one stable controller acknowledgement into compact model context."""
    action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
    state = game_state(message)
    result = {
        "step_id": pending.get("step_id"),
        "screen_before": pending.get("screen_type"),
        "action_kind": action.get("kind"),
        "action_id": action.get("id"),
        "action": action.get("label") or action.get("command"),
        "screen_after": state.get("screen_type"),
    }
    decision = pending.get("decision")
    if isinstance(decision, dict) and isinstance(decision.get("source"), str):
        result["source"] = decision["source"]
    post_state = normalize_state(message)
    gained_relics = newly_gained_relics(
        pending.get("pre_state", {}).get("relics") if isinstance(pending.get("pre_state"), dict) else None,
        state.get("relics"),
    )
    if gained_relics:
        result["relics_gained"] = gained_relics
    before = event_snapshot(pending.get("pre_state"))
    after = event_snapshot(post_state)
    if before is not None:
        result["event_before"] = before
    if after is not None:
        result["event_after"] = after
    if str(state.get("screen_type") or "").upper() == "CARD_REWARD":
        source = generated_card_choice_source(pending)
        if source is not None:
            result["card_choice_source"] = source
    return result


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
    keys = normalize_keys(state)
    if keys is not None:
        result["keys"] = keys

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
        result["combat"] = normalize_combat(combat, hide_enemy_intents=has_relic(state, "RUNICDOME"))
    return result


def policy_observation(
    message: dict[str, Any],
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    action_history: list[dict[str, Any]] | None = None,
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Route only the public context relevant to the active screen."""
    normalized = normalize_state(message)
    attach_runtime_context(normalized, visited_shop_rooms, action_history, declined_card_reward_keys)
    return route_policy_observation(normalized)


def route_policy_observation(normalized: dict[str, Any]) -> dict[str, Any]:
    """Apply the inference context budget to an already-normalized state.

    Trace files retain full normalized state for auditability. Both live policy
    inference and offline SFT must route that state through this function so a
    model is never trained on map information that it cannot see at runtime.
    The function mutates and returns ``normalized`` for the same efficient
    ownership pattern used by ``policy_observation``.
    """
    def remove_local_knowledge(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("knowledge", "experience", "act_boss_profile"):
                value.pop(key, None)
            if value.get("source") == "static_knowledge":
                value.pop("source", None)
            for child in value.values():
                remove_local_knowledge(child)
        elif isinstance(value, list):
            for child in value:
                remove_local_knowledge(child)

    # Re-exported traces may predate runtime descriptions and contain cached
    # local-database fields. Do not train on context that inference no longer
    # receives.
    remove_local_knowledge(normalized)

    # Full route information belongs only to the actual map selection screen.
    # A transient map overlay is normalized as TRANSITION and must not leak
    # route detail into event/reward/combat decisions.
    screen = str(normalized.get("screen_type") or "").upper()
    if screen == "MAP":
        normalized.pop("combat", None)
        return normalized

    normalized.pop("map", None)
    active_combat = (
        str(normalized.get("room_phase") or "").upper() == "COMBAT"
        and screen in {"NONE", "COMBAT", "GRID", "HAND_SELECT", "CARD_REWARD"}
    )
    if not active_combat:
        # CommunicationMod may briefly retain the previous combat payload on
        # a non-combat screen. Never expose that stale state to a prompt.
        normalized.pop("combat", None)
        return normalized

    # Combat-time choices still require energy, hand, intents and powers.
    # Keep their selection requirements/source as well as the live battle.
    if screen in {"GRID", "HAND_SELECT", "CARD_REWARD"}:
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
            "act_boss",
            "keys",
            "relics",
            "potions",
            "deck",
            "decision_constraints",
            "combat",
            "recent_actions",
        )
    }


def deck_summary(deck: Any) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    entries: dict[str, dict[str, Any]] = {}
    if not isinstance(deck, list):
        return []
    for raw in deck:
        if not isinstance(raw, dict):
            continue
        card = visible_card(raw)
        entry = {
            key: card[key]
            for key in ("id", "name", "type", "cost", "upgrades", "description")
            if key in card
        }
        signature = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        counts[signature] += 1
        entries[signature] = entry
    return [entries[key] | {"count": counts[key]} for key in sorted(entries)]


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
    "reward_id", "reward_index", "linked_reward_id",
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


def normalize_match_game(value: Any) -> dict[str, Any] | None:
    """Project public board memory without revealing unturned card faces."""
    if not isinstance(value, dict):
        return None
    result = {
        key: value[key] for key in (
            "phase", "remaining_attempts", "matched_pairs", "game_done",
            "awaiting_resolution", "selected_positions",
        ) if key in value
    }
    board = value.get("board")
    if board is None:
        result["board"] = None
    elif isinstance(board, list):
        result["board"] = []
        for raw in board:
            if not isinstance(raw, dict):
                continue
            item = {key: raw[key] for key in (
                "position", "row", "column", "face_up", "revealed", "matched",
            ) if key in raw}
            card = raw.get("card")
            # 'revealed' is durable public memory. Merely matched/face-up
            # is not sufficient evidence in an incomplete source snapshot.
            item["card"] = visible_card(card) if raw.get("revealed") is True and isinstance(card, dict) else None
            result["board"].append(item)
    return result


def normalize_screen_state(screen_type: Any, value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "source_reward_id" in value:
        result["source_reward_id"] = value["source_reward_id"]
    if screen_type in {"CARD_REWARD", "GRID"}:
        cards = value.get("cards", [])
        result["cards"] = [visible_card(card) for card in cards if isinstance(card, dict)]
        if screen_type == "GRID":
            selection_count = value.get("num_cards")
            if not isinstance(selection_count, int) or selection_count < 1:
                selection_count = 1
            mode_fields = (
                ("for_purge", "PURGE", f"选择 {selection_count} 张牌移除"),
                ("for_transform", "TRANSFORM", f"选择 {selection_count} 张牌进行变化"),
                ("for_upgrade", "UPGRADE", f"选择 {selection_count} 张牌升级"),
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
        for key in ("selected_cards", "num_cards", "for_transform", "for_upgrade", "for_purge", "any_number", "confirm_up", "confirm_screen_up", "selection_reason", "for_clarity", "skip_available"):
            if key in value:
                if key == "selected_cards":
                    result[key] = [visible_card(card) for card in value[key] if isinstance(card, dict)]
                else:
                    result[key] = value[key]
        if value.get("selection_reason"):
            result["selection_instruction"] = value["selection_reason"]
        if isinstance(value.get("confirmation_card"), dict):
            result["confirmation_card"] = visible_card(value["confirmation_card"])
            if value["confirmation_card"].get("uuid") is not None:
                result["confirmation_card"]["uuid"] = value["confirmation_card"]["uuid"]
    elif screen_type == "EVENT":
        result["event_id"] = value.get("event_id")
        result["event_name"] = value.get("event_name")
        result["body_text"] = value.get("body_text")
        result["options"] = [normalize_event_option(item) for item in value.get("options", [])]
        if "match_game" in value:
            result["match_game"] = normalize_match_game(value["match_game"])
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
        result["relics"] = [visible_relic(item) | {key: item[key] for key in ("reward_id", "reward_index") if key in item}
                            for item in value.get("relics", []) if isinstance(item, dict)]
    elif screen_type == "HAND_SELECT":
        result["hand"] = [visible_card(item) for item in value.get("hand", []) if isinstance(item, dict)]
        result["selected"] = [visible_card(item) for item in value.get("selected", []) if isinstance(item, dict)]
        for key in ("max_cards", "can_pick_zero", "up_to", "any_number", "for_upgrade", "for_transform", "selection_reason"):
            if key in value:
                result[key] = value[key]
        if value.get("selection_reason"):
            result["selection_instruction"] = value["selection_reason"]
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


def normalize_combat(combat: dict[str, Any], *, hide_enemy_intents: bool = False) -> dict[str, Any]:
    player = combat.get("player") if isinstance(combat.get("player"), dict) else {}
    raw_stance = player.get("stance")
    stance = (
        {key: raw_stance.get(key) for key in ("id", "name", "description")}
        if isinstance(raw_stance, dict) else None
    )
    player_powers = player.get("powers") if isinstance(player.get("powers"), list) else None
    raw_monsters = combat.get("monsters")
    monster_values = raw_monsters if isinstance(raw_monsters, list) else []
    monsters: list[dict[str, Any]] = []
    for monster in monster_values:
        if not isinstance(monster, dict):
            continue
        item = {
            "id": monster.get("id"),
            "name": monster.get("name"),
            "current_hp": monster.get("current_hp"),
            "max_hp": monster.get("max_hp"),
            "block": monster.get("block"),
            "is_gone": monster.get("is_gone"),
            "powers": powers_summary(monster.get("powers")),
        }
        if not hide_enemy_intents:
            per_hit = _integer_value(monster.get("move_adjusted_damage"))
            if per_hit is not None and per_hit < 0:
                per_hit = None
            hits = _integer_value(monster.get("move_hits"))
            item.update(
                {
                    "intent": monster.get("intent"),
                    "intent_base_damage": monster.get("move_base_damage") if _integer_value(monster.get("move_base_damage")) is not None and _integer_value(monster.get("move_base_damage")) >= 0 else None,
                    "intent_damage": per_hit,
                    "intent_hits": hits,
                    "intent_total_damage": per_hit * max(1, hits or 1) if per_hit is not None else None,
                }
            )
        monsters.append(item)
    return {
        "turn": combat.get("turn"),
        "hand": [
            visible_card(card, player_powers=player_powers if player_powers is not None else [])
            for card in combat.get("hand", [])
            if isinstance(card, dict)
        ] if isinstance(combat.get("hand"), list) else None,
        "player": {
            "current_hp": player.get("current_hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
            "stance": stance,
            "powers": powers_summary(player.get("powers")),
        },
        "monsters": monsters if isinstance(raw_monsters, list) else None,
        "draw_count": len(combat.get("draw_pile", [])) if isinstance(combat.get("draw_pile"), list) else None,
        "discard_count": len(combat.get("discard_pile", [])) if isinstance(combat.get("discard_pile"), list) else None,
        "exhaust_count": len(combat.get("exhaust_pile", [])) if isinstance(combat.get("exhaust_pile"), list) else None,
    }


def semantic_actions(
    message: dict[str, Any],
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
    *,
    fine_grained: bool = True,
    include_recorded_choices: bool = False,
) -> list[dict[str, Any]]:
    """Build fine-grained actions; raw recorded choices bypass policy masks only.

    Macro expansion remains opt-in with fine_grained=False. Record restoration
    uses include_recorded_choices=True so cancelled/deselected attempts survive.
    """
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
        result.extend(potion_actions(state, commands, live_targets, in_combat=True))
        return result

    choices = state.get("choice_list", [])
    if is_map_room_transition(state) and not include_recorded_choices:
        if "return" in commands:
            result.append({"id": "RETURN", "command": "RETURN", "kind": "return"})
        return result

    # Room-level decisions are exposed as semantic actions.  CommunicationMod
    # still receives the original CHOOSE index, but the policy never has to
    # infer that (for example) option 1 means "smith" and then repeat the
    # card-selection protocol itself.  The execution metadata is consumed by
    # the controller after the first command and is deliberately ignored by
    # policy prompt builders.
    screen_type = str(state.get("screen_type") or "").upper()
    if not fine_grained and screen_type == "REST" and "choose" in commands and isinstance(choices, list):
        for index, choice in enumerate(choices):
            normalized = "".join(str(choice).casefold().split())
            if normalized in {"rest", "休息"}:
                result.append({"id": "CAMPFIRE:REST", "command": f"CHOOSE {index}", "kind": "campfire_rest", "label": "休息"})
            elif normalized in {"smith", "upgrade", "锻造", "升级"}:
                # One visible action per card is easier to train than a
                # generic "smith" action followed by an unlabelled GRID.
                for card in state.get("deck", []):
                    if not isinstance(card, dict) or card.get("uuid") is None:
                        continue
                    card_id = str(card.get("uuid"))
                    name = card.get("name") or card.get("id") or card_id
                    result.append({
                        "id": f"CAMPFIRE:SMITH:card={card_id}",
                        "command": f"CHOOSE {index}",
                        "kind": "campfire_smith",
                        "label": f"锻造 {name}",
                        "card_uuid": card_id,
                        "card_name": name,
                        "execution": {"type": "campfire_smith", "card_uuid": card_id, "phase": "await_grid"},
                    })
            elif normalized in {"recall", "回忆"}:
                result.append({"id": "CAMPFIRE:RECALL", "command": f"CHOOSE {index}", "kind": "campfire_recall", "label": "回忆"})
            elif normalized in {"lift", "举重"}:
                result.append({"id": "CAMPFIRE:LIFT", "command": f"CHOOSE {index}", "kind": "campfire_lift", "label": "举重"})
            elif normalized in {"dig", "挖掘"}:
                result.append({"id": "CAMPFIRE:DIG", "command": f"CHOOSE {index}", "kind": "campfire_dig", "label": "挖掘"})
        # A used campfire exposes only a continue/proceed control, which is a
        # deterministic transition and remains handled below.
        if result:
            return result

    if not fine_grained and screen_type == "SHOP_SCREEN" and "choose" in commands:
        screen_state = state.get("screen_state") if isinstance(state.get("screen_state"), dict) else {}
        structured = False
        mapped_indices: set[int] = set()
        cards = screen_state.get("cards") if isinstance(screen_state.get("cards"), list) else []
        potions = screen_state.get("potions") if isinstance(screen_state.get("potions"), list) else []
        relics = screen_state.get("relics") if isinstance(screen_state.get("relics"), list) else []
        blocked_indices = blocked_potion_reward_indices(message) | blocked_gold_reward_indices(message)
        for index, choice in enumerate(choices if isinstance(choices, list) else []):
            if index in blocked_indices:
                continue
            label = str(choice)
            # The final purge entry is represented separately from shop cards.
            if label.casefold() in {"purge", "删牌", "移除卡牌"} or label.startswith("purge"):
                for card in state.get("deck", []):
                    if not isinstance(card, dict) or card.get("uuid") is None:
                        continue
                    structured = True
                    uuid = str(card["uuid"])
                    name = card.get("name") or card.get("id") or uuid
                    result.append({
                        "id": f"SHOP:PURGE:card={uuid}",
                        "command": f"CHOOSE {index}",
                        "kind": "shop_purge",
                        "label": f"删除 {name}",
                        "card_uuid": uuid,
                        "card_name": name,
                        "execution": {"type": "shop_purge", "card_uuid": uuid, "phase": "await_grid"},
                    })
                if any(isinstance(card, dict) and card.get("uuid") is not None for card in state.get("deck", [])):
                    mapped_indices.add(index)
                continue
            # Match the visible shop entries by stable order.  If a mod
            # version does not expose structured entities, retain the generic
            # CHOOSE fallback below for compatibility.
            entity = None
            kind = None
            # Choice-list order is not stable across CommunicationMod
            # versions (and may omit unaffordable items), so match by the
            # visible entity name instead of assuming cards/potions/relics
            # occupy contiguous ranges.
            for candidate, candidate_kind in (
                *((item, "shop_buy_card") for item in cards),
                *((item, "shop_buy_potion") for item in potions),
                *((item, "shop_buy_relic") for item in relics),
            ):
                if not isinstance(candidate, dict):
                    continue
                candidate_name = str(candidate.get("name") or candidate.get("id") or "")
                if candidate_name and (candidate_name in label or label in candidate_name):
                    entity, kind = candidate, candidate_kind
                    break
            if isinstance(entity, dict):
                structured = True
                entity_id = str(entity.get("uuid") or entity.get("id") or entity.get("name") or index)
                name = entity.get("name") or entity.get("id") or label
                price = entity.get("price")
                price_text = f" ({price} 金币)" if price is not None else ""
                prefix = {"shop_buy_card": "SHOP:BUY_CARD", "shop_buy_potion": "SHOP:BUY_POTION", "shop_buy_relic": "SHOP:BUY_RELIC"}[kind]
                result.append({"id": f"{prefix}:{entity_id}", "command": f"CHOOSE {index}", "kind": kind, "index": index, "label": f"购买 {name}{price_text}"})
                mapped_indices.add(index)
        visible_indices = set(range(len(choices))) - blocked_indices if isinstance(choices, list) else set()
        if structured and isinstance(choices, list) and mapped_indices == visible_indices:
            return result
    # Potions can be used before a non-map screen's primary choice, such as
    # freeing a full slot before taking a reward. Enemy-target potions are not
    # emitted here because there is no combat target list on these screens.
    if not is_map_screen(state):
        result.extend(potion_actions(state, commands))
    if "choose" in commands and isinstance(choices, list):
        blocked_indices = blocked_potion_reward_indices(message) | blocked_gold_reward_indices(message) | blocked_card_reward_indices(
            message,
            declined_card_reward_keys,
        ) | selected_card_choice_indices(message)
        if include_recorded_choices:
            # Human recording mirrors CommunicationMod's raw choices. Do not
            # mask inert rewards or hide deselection. Macro granularity is separate.
            blocked_indices = set()
        for index, label in enumerate(choices):
            if index in blocked_indices:
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

    # A map is a route-selection screen, never a potion-use screen. Some
    # CommunicationMod versions briefly retain commands from the preceding
    # combat while the map overlay is opening; filter those stale actions at
    # the final boundary so they cannot reach the model or terminal.
    if is_map_screen(state):
        result = [action for action in result if not is_map_potion_action(action)]
    return result


def is_map_potion_action(action: dict[str, Any]) -> bool:
    """Return whether an action is a potion operation leaked into the map."""
    kind = str(action.get("kind") or "").casefold()
    action_id = str(action.get("id") or "").casefold()
    text = " ".join(
        str(action.get(field) or "")
        for field in ("command", "label", "selection")
    ).casefold()
    return (
        kind in {"potion_use", "potion_discard"}
        or action_id.startswith("potion_")
        or "potion" in text
        or "药水" in text
    )


def is_map_screen(state: dict[str, Any]) -> bool:
    """Return whether the current state represents route selection."""
    return (
        str(state.get("screen_type") or "").upper() == "MAP"
        or str(state.get("room_phase") or "").upper() == "MAP"
    )


def potion_actions(
    state: dict[str, Any],
    commands: set[str],
    live_targets: list[int] | None = None,
    *,
    in_combat: bool = False,
) -> list[dict[str, Any]]:
    """Build legal potion operations for the current screen.

    CommunicationMod advertises potion support on screens where a potion can
    be used before making the screen's primary choice (for example, a full
    reward inventory). Enemy-target operations are meaningful only in combat.
    Outside combat, only Blood Potion and Fruit Juice are manually usable;
    other potions are still exposed when they can be discarded to free a slot.
    """
    if "potion" not in commands:
        return []
    targets = list(live_targets or [])
    result: list[dict[str, Any]] = []
    for index, potion in enumerate(state.get("potions", [])):
        if not isinstance(potion, dict) or is_slot(potion):
            continue
        name = potion.get("name", index)
        potion_key = _status_key(potion.get("id")) or _status_key(potion.get("name"))
        can_use = bool(potion.get("can_use", False)) and (
            in_combat or potion_key in NON_COMBAT_USABLE_POTIONS
        )
        if can_use:
            if potion.get("requires_target", False):
                for target in targets:
                    result.append(
                        {
                            "id": f"POTION_USE:slot={index}:target={target}",
                            "command": f"POTION USE {index} {target}",
                            "kind": "potion_use",
                            "slot": index,
                            "target": target,
                            "label": f"使用药水 {name} -> 敌人 {target}",
                        }
                    )
            else:
                result.append(
                    {
                        "id": f"POTION_USE:slot={index}",
                        "command": f"POTION USE {index}",
                        "kind": "potion_use",
                        "slot": index,
                        "label": f"使用药水 {name}",
                    }
                )
        if potion.get("can_discard", False):
            result.append(
                {
                    "id": f"POTION_DISCARD:slot={index}",
                    "command": f"POTION DISCARD {index}",
                    "kind": "potion_discard",
                    "slot": index,
                    "label": f"丢弃药水 {name}",
                }
            )
    return result


def action_for_command(actions: list[dict[str, Any]], command: str) -> dict[str, Any] | None:
    upper = command.upper()
    for action in actions:
        if action["command"].upper() == upper:
            return action
    return None


def make_sts1_game_adapter() -> Any:
    """Expose the CommunicationMod implementation through the framework API.

    The controller remains the compatibility entrypoint, but new integrations
    can depend on this adapter instead of importing the controller's private
    helpers. ``execute_action`` resolves a semantic command against the
    current action list; the live controller still owns the actual socket send.
    """

    from sts_agent.game.adapter import Sts1Adapter

    def execute_action(message: dict[str, Any], action: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
        del kwargs
        return action_for_command(semantic_actions(message), str(action.get("command") or ""))

    return Sts1Adapter(
        normalize=normalize_state,
        observe=policy_observation,
        actions=semantic_actions,
        execute=execute_action,
    )


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
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Resolve explicit Candidate Assist shortcuts before game choices."""
    if not decision:
        return None, None, None
    candidates = decision.get("model_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None, None, None
    actions = semantic_actions(message, visited_shop_rooms, declined_card_reward_keys)
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
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
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
    actions = semantic_actions(message, visited_shop_rooms, declined_card_reward_keys)

    # Semantic actions may share one underlying CHOOSE command (for example,
    # each possible card to smith at a campfire).  Accept the stable action ID
    # itself so terminal and accessibility clients can select the intended
    # entity without relying on a duplicated protocol index.
    if len(parts) == 1:
        normalized_input = stripped.casefold()
        for candidate in actions:
            if normalized_input == str(candidate.get("id") or "").casefold():
                return str(candidate["command"]), candidate, None

    if op in {"q", "quit", "exit"}:
        return "__QUIT__", None, None
    if op in {"h", "help"}:
        return "__HELP__", None, None
    candidate_command, candidate_action, candidate_error = candidate_action_for_input(
        parts,
        message,
        decision,
        visited_shop_rooms,
        declined_card_reward_keys,
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
        blocked_error = blocked_potion_choice_error(message, index) or blocked_gold_choice_error(message, index)
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
    if kind.startswith("campfire_") or kind.startswith("shop_"):
        return str(action.get("id") or action.get("command") or "")
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
    elif kind.startswith("campfire_"):
        action_type = "营火"
        selection = str(action.get("label") or "")
    elif kind.startswith("shop_"):
        action_type = "商店"
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
    keys = normalize_keys(state)
    if keys is not None:
        key_text = " | ".join(
            f"{label}{'✓' if keys.get(name) else '·'}"
            for name, label in (("ruby", "红"), ("emerald", "绿"), ("sapphire", "蓝"))
        )
        summary += f"\n钥匙 {key_text}"
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
    if state.get("act_boss"):
        lines.append(f"本层首领：{state.get('act_boss')}")
    confidence = decision.get("confidence") if isinstance(decision, dict) else None
    if isinstance(confidence, dict):
        route = "将自动执行" if confidence.get("route") == "auto_execute" else "等待人工确认"
        lines.append(
            "模型置信度："
            f"p1={float(confidence.get('top_probability', 0.0)):.1%} "
            f"margin={float(confidence.get('margin', 0.0)):.1%} "
            f"entropy={float(confidence.get('entropy', 0.0)):.3f}；{route}"
        )
    environment = decision.get("environment") if isinstance(decision, dict) else None
    if isinstance(environment, dict):
        lines.append(f"环境自动执行：{environment.get('reason') or environment.get('rule') or '推进界面'}")
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
        hide_enemy_intents = has_relic(state, "RUNICDOME")
        lines.append(f"战斗 | Turn {combat.get('turn', '?')} | Energy {player.get('energy', '?')} | Block {player.get('block', '?')}")
        lines.append("敌人")
        for index, monster in enumerate(combat.get("monsters", [])):
            if not isinstance(monster, dict):
                continue
            intent_damage = None if hide_enemy_intents else monster.get("move_adjusted_damage")
            intent_hits = None if hide_enemy_intents else monster.get("move_hits")
            intent_detail = ""
            if intent_damage is not None and intent_damage != -1:
                intent_detail = f" dmg={intent_damage}"
                if intent_hits not in (None, 1):
                    intent_detail += f"x{intent_hits}"
                total_damage = (
                    intent_damage * intent_hits
                    if isinstance(intent_damage, (int, float)) and isinstance(intent_hits, int)
                    else None
                )
                if total_damage is not None and intent_hits not in (None, 1):
                    intent_detail += f" total={total_damage}"
            intent = "HIDDEN" if hide_enemy_intents else monster.get("intent")
            lines.append(
                f"  [{index}] {monster.get('name')} HP {monster.get('current_hp')}/{monster.get('max_hp')} "
                f"Intent={intent}{intent_detail} Block={monster.get('block', 0)} "
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
                f"{card_effect_summary(card, player_powers=player.get('powers', []))}"
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
            reason = "持有苏祖，不能再获得药水" if has_relic(state, "SOZU") else "药水栏已满；请改选其他选项或先丢弃药水"
            lines.append(f"不可选：{labels}（{reason}）")
        blocked_gold = blocked_gold_reward_indices(message)
        if blocked_gold:
            labels = ", ".join(str(choices[index]) for index in sorted(blocked_gold))
            lines.append(f"不可选：{labels}（持有灵体外质，不能再获得金币）")
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
        try:
            self.writer.write(json.dumps(packet, ensure_ascii=False) + "\n")
            self.writer.flush()
        except OSError:
            self.close_client()
            raise

    def receive(self) -> dict[str, Any] | None:
        self.accept()
        assert self.reader is not None
        try:
            line = self.reader.readline()
        except OSError:
            self.close_client()
            return None
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
        model_config = policy_config["model"]
        request_config = policy_config["request"]
        local_config = policy_config["local"]
        development_config = policy_config["development"]
        confidence_config = policy_config["confidence_auto"]
        self.enabled = policy_config["enabled"]
        self.runtime = policy_config["runtime"]
        self.model = model_config["base"]
        self.initial_adapter = model_config["initial_adapter"]
        self.initial_policy_version = model_config["initial_policy_version"]
        self.endpoint = model_config["hub_endpoint"]
        self.top_k = request_config["top_k"]
        self.screens = tuple(item.strip().upper() for item in request_config["screens"])
        self.confidence_auto = confidence_config["enabled"]
        self.auto_min_probability = confidence_config["min_probability"]
        self.timeout = float(request_config["response_timeout_seconds"])
        self.start_timeout = float(request_config["startup_timeout_seconds"])
        self.python = local_config["python"]
        self.mock = development_config["mock"]
        self.debug = development_config["debug"]
        self.remote = policy_config["remote"]
        self.process: subprocess.Popen[str] | None = None
        self.request_id = 0
        self.disabled_reason: str | None = None
        self.remote_model_id = self.model
        self.online_rl_status: dict[str, Any] = {}
        self.online_rl_attribution_enabled = False
        self.api = None
        if self.runtime == "remote_api":
            from sts_agent.policy.api import ApiPolicyClient

            self.api = ApiPolicyClient(project_root, config)

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
        if self.runtime == "remote_online_rl":
            self.online_rl_status.update({"stage": "远端在线策略不可用", "error": reason})
        self.close()
        print(f"candidate assist disabled: {reason}", file=sys.stderr, flush=True)

    def _set_online_rl_status(self, stage: str | None = None, **values: Any) -> dict[str, Any]:
        """Maintain the display-only lifecycle snapshot for Online RL."""
        if stage is not None:
            self.online_rl_status["stage"] = stage
        for key, value in values.items():
            if value is None:
                if key == "error":
                    self.online_rl_status.pop(key, None)
                continue
            self.online_rl_status[key] = value
        return dict(self.online_rl_status)

    def online_rl_status_decision(self, stage: str | None = None, **values: Any) -> dict[str, Any]:
        """Build a read-only decision packet for the Overlay's Online RL window."""
        return {
            "mode": "remote_online_rl",
            "model_id": self.remote_model_id,
            "candidate_source": "online_rl_worker",
            "model_action": None,
            "model_candidates": [],
            "online_rl": self._set_online_rl_status(stage, **values),
            "executed_action": None,
            "source": None,
            "overridden": None,
        }

    def record_online_episode_result(self, response: dict[str, Any]) -> dict[str, Any]:
        """Project a terminal worker response into stable, user-facing status."""
        common = {
            "group_id": response.get("group_id"),
            "policy_version": response.get("policy_version"),
            "iteration": response.get("iteration"),
            "group_size": response.get("group_size"),
            "episodes_collected": response.get("episodes_collected"),
        }
        if response.get("type") == "policy_updated":
            return self.online_rl_status_decision(
                "参数更新完成；下一组 rollout 将使用新策略",
                **common,
                rollout_index=response.get("group_size"),
                update=response.get("update"),
                error=None,
            )
        if response.get("type") == "episode_queued":
            return self.online_rl_status_decision(
                "本局轨迹已入队；等待下一次 rollout",
                **common,
                rollout_index=response.get("episodes_collected"),
                error=None,
            )
        if response.get("type") == "group_completed_no_update":
            return self.online_rl_status_decision(
                "本组归因不足，未更新参数；下一组继续使用当前策略",
                **common,
                rollout_index=response.get("group_size"),
                update=response.get("update"),
                error=None,
            )
        return self.online_rl_status_decision(
            "本局轨迹上传或参数更新失败",
            **common,
            error=str(response.get("error") or "远端 worker 返回未知结果"),
        )

    def _fallback_decision(self, status: str, reason: str) -> dict[str, Any]:
        """Return an observable human-fallback state for the terminal/trace."""
        result = {
            "mode": "remote_online_rl" if self.runtime == "remote_online_rl" else ("confidence_auto" if self.confidence_auto else "candidate_assist"),
            "model_id": self.remote_model_id,
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
        if self.runtime == "remote_online_rl":
            result["candidate_source"] = "online_rl_worker"
            result["online_rl"] = self._set_online_rl_status(
                "远程在线策略请求中" if status == "请求中" else "远程在线策略需要人工处理",
                error=reason if status in {"不可用", "超时"} else None,
            )
        return result

    def status_decision(self) -> dict[str, Any]:
        """Expose startup/readiness state even on screens without ranking."""
        if self.api is not None:
            return self.api.status_decision()
        if self.enabled and self.process is not None and self.process.poll() is None:
            if self.runtime == "remote_online_rl":
                return self.online_rl_status_decision("在线策略就绪；等待下一次 rollout")
            mode = (
                f"置信度自动执行已启用（p1>={self.auto_min_probability:.0%}）"
                if self.confidence_auto
                else "本局每个游戏界面均会请求候选"
            )
            return self._fallback_decision("就绪", f"{self.model} 已预热，{mode}")
        return self._fallback_decision("不可用", self.disabled_reason or "模型辅助已关闭")

    def requesting_decision(self) -> dict[str, Any]:
        """Expose an immediate UI state while a policy request is in flight."""
        if self.api is not None:
            return self.api.requesting_decision()
        if not self.enabled:
            return self.status_decision()
        if self.runtime == "remote_online_rl":
            group_size = self.online_rl_status.get("group_size")
            episodes_collected = self.online_rl_status.get("episodes_collected")
            stage = "正在建立 rollout 组并采样"
            if isinstance(group_size, int) and group_size > 0:
                completed = episodes_collected if isinstance(episodes_collected, int) else 0
                next_rollout = 1 if completed >= group_size else completed + 1
                stage = f"正在为第 {next_rollout}/{group_size} 次 rollout 采样"
            self._set_online_rl_status(stage)
            return self._fallback_decision("请求中", "远程在线策略正在采样动作")
        location = "远程推理服务" if self.runtime != "local" else "本机模型"
        return self._fallback_decision("请求中", f"{location}正在计算候选动作")

    def prewarm(self) -> dict[str, Any]:
        """Load the local policy before CommunicationMod receives readiness."""
        if self.api is not None:
            return self.api.prewarm()
        if not self.enabled:
            return self.status_decision()
        self._start()
        return self.status_decision()

    def _start(self) -> bool:
        if not self.enabled:
            return False
        if self.process is not None and self.process.poll() is None:
            return True
        if self.runtime == "local":
            command = [
                self._python_executable(),
                "-m",
                "sts_agent.policy.local_qwen",
                "--model",
                self.model,
                "--endpoint",
                self.endpoint,
                "--top-k",
                str(self.top_k),
                # Keep game-time startup deterministic. Downloading model
                # weights is an explicit preparation step handled elsewhere.
                "--local-files-only",
            ]
            if self.initial_adapter:
                command.extend(["--adapter", self.initial_adapter])
            if self.mock:
                command.append("--mock")
            if self.debug:
                command.append("--debug")
        else:
            remote_root = str(self.remote["project_root"])
            if self.runtime == "remote_inference":
                inference_gpu = self.remote["inference_gpu"]
                remote_command = [
                    str(self.remote["python"]),
                    "-m",
                    "sts_agent.policy.local_qwen",
                    "--model",
                    self.model,
                    "--endpoint",
                    self.endpoint,
                    "--top-k",
                    str(self.top_k),
                    "--local-files-only",
                    "--gpu-device",
                    str(inference_gpu["device"]),
                    "--minimum-free-memory-mib",
                    str(inference_gpu["minimum_free_memory_mib"]),
                ]
                if self.initial_adapter:
                    remote_command.extend(["--adapter", self.initial_adapter])
                if self.mock:
                    remote_command.append("--mock")
                if self.debug:
                    remote_command.append("--debug")
            else:
                remote_command = [
                    str(self.remote["python"]),
                    "-m",
                    "sts_agent.learning.online_rl.worker",
                    "--config",
                    str(self.remote["online_rl_config"]),
                    "--model",
                    self.model,
                    "--initial-adapter",
                    self.initial_adapter,
                    "--initial-policy-version",
                    self.initial_policy_version,
                ]
                if self.mock:
                    remote_command.append("--mock")
                if self.debug:
                    remote_command.append("--debug")
            remote_shell = f"cd {shlex.quote(remote_root)} && exec {shlex.join(remote_command)}"
            command = [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={int(float(self.remote['connect_timeout_seconds']))}",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                str(self.remote["ssh_host"]),
                remote_shell,
            ]
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
        self.remote_model_id = str(ready.get("model_id") or self.model)
        if self.runtime == "remote_online_rl":
            self.online_rl_attribution_enabled = bool(ready.get("attribution_enabled"))
            self._set_online_rl_status(
                "在线策略就绪；等待开始 rollout",
                policy_version=ready.get("policy_version"),
                iteration=ready.get("iteration"),
                group_size=ready.get("group_size"),
                gpu=ready.get("gpu"),
                attribution_enabled=self.online_rl_attribution_enabled,
                error=None,
            )
        print(
            f"policy ready: {ready.get('model_id', self.model)} runtime={self.runtime} "
            f"version={ready.get('policy_version', '-')}",
            file=sys.stderr,
            flush=True,
        )
        return True

    def rank(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        run_id: str | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.api is not None:
            return self.api.rank(observation, legal_actions, run_id=run_id, run_config=run_config)
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
            "type": "sample" if self.runtime == "remote_online_rl" else ("score" if self.confidence_auto else "rank"),
            "request_id": self.request_id,
            "observation": observation,
            "legal_actions": model_actions,
            "top_k": self.top_k,
        }
        if self.runtime == "remote_online_rl":
            if not run_id or not isinstance(run_config, dict):
                return self._fallback_decision("不可用", "在线策略缺少当前对局标识或起始配置")
            request.update({"episode_id": run_id, "run_config": run_config})
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
        expected_type = "sampled_action" if self.runtime == "remote_online_rl" else ("scoring" if self.confidence_auto else "ranking")
        if response.get("type") != expected_type:
            self._disable(f"unexpected policy response: {response}")
            return self._fallback_decision("不可用", self.disabled_reason or "模型服务返回未知消息")
        if self.runtime == "remote_online_rl":
            return self._sampled_decision(response, alias_to_action)
        if self.confidence_auto:
            return self._scoring_decision(response, alias_to_action)
        return self._ranking_decision(response, alias_to_action)

    def _sampled_decision(
        self,
        response: dict[str, Any],
        alias_to_action: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        raw_scores = response.get("action_scores")
        selected_alias = str(response.get("action_id", ""))
        if not isinstance(raw_scores, list) or selected_alias not in alias_to_action:
            return self._fallback_decision("分数无效", "在线策略返回了未知动作或缺少完整分数")
        by_alias: dict[str, dict[str, Any]] = {}
        for item in raw_scores:
            if not isinstance(item, dict):
                return self._fallback_decision("分数无效", "在线策略动作分数格式无效")
            alias = str(item.get("action_id", ""))
            if alias not in alias_to_action or alias in by_alias:
                return self._fallback_decision("分数无效", "在线策略动作分数包含未知或重复别名")
            try:
                probability = float(item.get("probability"))
                normalized_logprob = float(item.get("normalized_logprob"))
            except (TypeError, ValueError):
                return self._fallback_decision("分数无效", "在线策略概率不是数值")
            if not math.isfinite(probability) or not math.isfinite(normalized_logprob) or probability < 0:
                return self._fallback_decision("分数无效", "在线策略概率无效")
            by_alias[alias] = dict(item) | {"probability": probability, "normalized_logprob": normalized_logprob}
        if set(by_alias) != set(alias_to_action) or not math.isclose(sum(item["probability"] for item in by_alias.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            return self._fallback_decision("分数不完整", "在线策略没有返回规范化的全动作概率")
        normalized = sorted(by_alias.values(), key=lambda item: float(item["probability"]), reverse=True)
        action_scores: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for rank, item in enumerate(normalized, start=1):
            alias = str(item["action_id"])
            action = alias_to_action[alias]
            canonical_id = str(action.get("id", ""))
            action_scores.append(
                {
                    "alias": alias,
                    "action_id": canonical_id,
                    "logprob": float(item["logprob"]),
                    "probability": float(item["probability"]),
                }
            )
            if rank <= self.top_k:
                candidates.append(
                    {
                        "rank": rank,
                        "action_id": canonical_id,
                        "label": action.get("label") or action.get("command"),
                        "score": float(item["probability"]),
                        "log_score": float(item["logprob"]),
                    }
                )
        selected = by_alias[selected_alias]
        selected_action = alias_to_action[selected_alias]
        self._set_online_rl_status(
            f"正在执行第 {response.get('rollout_index', '?')}/{response.get('group_size', '?')} 次 rollout",
            group_id=response.get("group_id"),
            policy_version=response.get("policy_version"),
            iteration=response.get("iteration"),
            group_size=response.get("group_size"),
            rollout_index=response.get("rollout_index"),
            episodes_collected=response.get("episodes_collected"),
            error=None,
        )
        return {
            "mode": "remote_online_rl",
            "model_id": response.get("model_id", self.remote_model_id),
            "candidate_source": "online_rl_worker",
            "model_action": str(selected_action.get("id", "")),
            "model_action_label": selected_action.get("label") or selected_action.get("command"),
            "model_action_probability": float(selected["probability"]),
            "model_candidates": candidates,
            "action_scores": action_scores,
            "model_confidence": float(selected["probability"]),
            "online_rl": dict(self.online_rl_status) | {
                "selected_alias": selected_alias,
                "selected_logprob": float(selected["normalized_logprob"]),
                "sampling_draw": response.get("sampling_draw"),
            },
            "executed_action": None,
            "source": None,
            "overridden": None,
        }
    def submit_episode(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Upload one prepared, validated episode to the resident online worker."""

        if self.runtime != "remote_online_rl" or not self.enabled:
            return None
        if not self._start() or self.process is None or self.process.stdin is None or self.process.stdout is None:
            return {"type": "error", "error": self.disabled_reason or "online worker is unavailable"}
        self.request_id += 1
        request = {"type": "episode", "request_id": self.request_id, "episode": payload}
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            self._disable(f"online episode upload failed: {exc}")
            return {"type": "error", "error": self.disabled_reason}
        line = self._readline_with_timeout(self.process.stdout, float(self.remote["update_timeout_seconds"]))
        if not line:
            self._disable("online episode upload/update timed out")
            return {"type": "error", "error": self.disabled_reason}
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self._disable(f"invalid online worker response: {exc}")
            return {"type": "error", "error": self.disabled_reason}
        if response.get("type") == "error":
            return response
        if response.get("type") not in {"episode_queued", "policy_updated", "group_completed_no_update"}:
            self._disable(f"unexpected online worker response: {response}")
            return {"type": "error", "error": self.disabled_reason}
        return response

    def submit_trace(
        self,
        trace_path: Path,
        run_id: str,
        run_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Upload one durable terminal episode to the online worker.

        The file is written locally before this call. A broken SSH connection
        therefore pauses the next group but never loses the game trajectory.
        """

        if self.runtime != "remote_online_rl" or not self.enabled:
            return None
        try:
            payload = online_episode_from_trace(trace_path, run_id, run_config)
        except (OSError, ValueError) as exc:
            return {"type": "error", "error": f"cannot prepare online episode: {exc}"}
        return self.submit_episode(payload)

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
        if getattr(self, "api", None) is not None:
            self.api.close()
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


def online_episode_from_trace(
    trace_path: Path,
    run_id: str,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a replayable online-RL episode from accepted local trace records."""

    steps: list[dict[str, Any]] = []
    group_id: str | None = None
    policy_version: str | None = None
    terminal_state: dict[str, Any] | None = None
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trace JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                continue
            if record.get("terminal") and isinstance(record.get("post_state"), dict):
                terminal_state = record["post_state"]
            decision = record.get("decision")
            if not isinstance(decision, dict) or decision.get("mode") != "remote_online_rl":
                continue
            if record.get("accepted") is not True or decision.get("source") != "online_policy_sample":
                raise ValueError("online rollout contains an unaccepted or human-overridden action")
            online = decision.get("online_rl")
            pre_state = record.get("pre_state")
            actions = record.get("legal_actions")
            if not isinstance(online, dict) or not isinstance(pre_state, dict) or not isinstance(actions, list):
                raise ValueError("online rollout trace is missing decision context")
            current_group_id = str(online.get("group_id") or "")
            current_version = str(online.get("policy_version") or "")
            selected_alias = str(online.get("selected_alias") or "")
            if not current_group_id or not current_version or not selected_alias:
                raise ValueError("online rollout trace is missing group, version, or selected alias")
            if group_id is None:
                group_id = current_group_id
                policy_version = current_version
            elif group_id != current_group_id or policy_version != current_version:
                raise ValueError("one episode cannot mix online RL groups or policy versions")
            model_actions = [
                {"id": f"A{index}", "label": str(action.get("label") or action.get("command") or f"A{index}")}
                for index, action in enumerate(actions, start=1)
                if isinstance(action, dict)
            ]
            aliases = {str(action["id"]) for action in model_actions}
            if selected_alias not in aliases:
                raise ValueError("selected alias is absent from reconstructed legal actions")
            try:
                old_logprob = float(online.get("selected_logprob"))
            except (TypeError, ValueError) as exc:
                raise ValueError("selected online action has no numeric old log probability") from exc
            steps.append(
                {
                    "step_id": int(record.get("step_id")),
                    "observation": route_policy_observation(copy.deepcopy(pre_state)),
                    "legal_actions": model_actions,
                    "selected_alias": selected_alias,
                    "old_logprob": old_logprob,
                }
            )
    if not steps:
        raise ValueError("trace contains no online RL actions")
    if group_id is None or policy_version is None or terminal_state is None:
        raise ValueError("online rollout must end in a stable GAME_OVER or VICTORY state")
    return {
        "episode_id": run_id,
        "group_id": group_id,
        "policy_version": policy_version,
        "run_config": run_config,
        "steps": steps,
        "terminal_state": terminal_state,
    }


def request_posthoc_reflection(
    bridge: TerminalBridge,
    run_id: str,
    terminal_state: dict[str, Any],
    action_review: list[dict[str, Any]],
) -> dict[str, Any]:
    """Request short positive/negative feedback after a complete RL rollout.

    This is a separate terminal packet rather than a game action.  A closed
    client is treated as an explicit skip so a disconnected Overlay cannot
    leave CommunicationMod blocked on the victory/death screen.
    """

    try:
        bridge.send(
            {
                "type": "posthoc_reflection_request",
                "run_id": run_id,
                "terminal_state": posthoc_terminal_summary(terminal_state),
                "action_review": action_review,
            }
        )
    except OSError:
        return {"reinforce": "", "weaken": "", "skipped": True, "reason": "terminal_unavailable"}
    response = bridge.receive()
    if response is None:
        return {"reinforce": "", "weaken": "", "skipped": True, "reason": "terminal_disconnected"}
    if response.get("type") != "posthoc_reflection":
        return {"reinforce": "", "weaken": "", "skipped": True, "reason": "invalid_terminal_response"}
    if response.get("skip") is True:
        return {"reinforce": "", "weaken": "", "skipped": True, "reason": "human_skipped"}
    return {
        "reinforce": str(response.get("reinforce") or ""),
        "weaken": str(response.get("weaken") or ""),
        "skipped": False,
        "reason": "submitted",
    }


def request_prompt_evolution_review(
    bridge: TerminalBridge,
    review: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    """Block at a terminal state until the human accepts, rejects, or abandons.

    Prompt candidates are never promoted merely because their scalar score is
    higher. A disconnected UI abandons the current candidate so it cannot
    silently alter the active policy.
    """

    try:
        bridge.send(
            {
                "type": "prompt_evolution_review_request",
                "review": review,
                "phase": review.get("phase", "validation") if isinstance(review, dict) else "validation",
                "error": error,
            }
        )
    except OSError:
        return {"verdict": "abandon", "reason": "review UI unavailable"}
    response = bridge.receive()
    if response is None:
        return {"verdict": "abandon", "reason": "review UI disconnected"}
    if response.get("type") != "prompt_evolution_review":
        return {"verdict": "abandon", "reason": "invalid review response"}
    return {
        "verdict": str(response.get("verdict") or "abandon"),
        "reason": str(response.get("reason") or ""),
    }




class TeacherClient:
    """Remote decision helper constrained to controller-provided actions."""

    def __init__(
        self,
        project_root: Path,
        config: dict[str, Any],
        prompt_evolution: PromptEvolutionManager | None = None,
    ) -> None:
        teacher_config = config["teacher"]
        self.remote = RemoteTeacherClient(project_root, teacher_config)
        self.prompt_evolution = prompt_evolution
        self.enabled = self.remote.enabled
        self.model = self.remote.model
        self.thinking = self.remote.thinking
        self.screens = self.remote.screens

    @property
    def mode(self) -> str:
        if self.prompt_evolution is not None and self.prompt_evolution.enabled:
            return "prompt_evolution"
        return "teacher_collection"

    def _fallback_decision(self, status: str, reason: str) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model_id": self.model,
            "candidate_source": "remote_teacher",
            "model_action": None,
            "model_candidates": [],
            "teacher": {"status": status, "reason": reason},
            "assist_status": status,
            "assist_reason": reason,
            "executed_action": None,
            "source": None,
            "overridden": None,
        }

    def status_decision(self) -> dict[str, Any]:
        if not self.enabled:
            return self._fallback_decision("已关闭", "远程教师采集未启用")
        if self.remote.last_error:
            return self._fallback_decision("不可用", self.remote.last_error)
        return self._fallback_decision("就绪", f"{self.model} 将自动选择每个合法动作")

    def prewarm(self) -> dict[str, Any]:
        """Check only the ignored local credential file before game startup."""
        if not self.enabled:
            return self.status_decision()
        error = self.remote.preflight()
        if error:
            print(f"teacher collection unavailable: {error}", file=sys.stderr, flush=True)
        else:
            print(f"teacher collection ready: {self.model}", file=sys.stderr, flush=True)
        return self.status_decision()

    def rank(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not legal_actions:
            return self._fallback_decision("无合法动作", "当前状态没有可选择动作")
        if not self.enabled:
            return self.status_decision()
        alias_to_action: dict[str, dict[str, Any]] = {}
        model_actions: list[dict[str, Any]] = []
        for index, action in enumerate(legal_actions, start=1):
            alias = f"A{index}"
            alias_to_action[alias] = action
            model_actions.append({"id": alias, "label": str(action.get("label") or action.get("command") or alias)})
        def forward_update(update: dict[str, Any]) -> None:
            if on_update is None:
                return
            on_update(
                {
                    "mode": self.mode,
                    "model_id": self.model,
                    "candidate_source": "remote_teacher",
                    "model_action": None,
                    "model_candidates": [],
                    "teacher": {
                        "provider": "openai_compatible",
                        "model_id": self.model,
                        "thinking_enabled": self.thinking,
                        "thinking": update.get("thinking", ""),
                        "output": update.get("output", ""),
                        "retrying": bool(update.get("retrying")),
                        "retry_attempt": update.get("retry_attempt"),
                        "retry_delay_seconds": update.get("retry_delay_seconds"),
                        "retry_reason": update.get("retry_reason"),
                    },
                    "assist_status": "网络重试中" if update.get("retrying") else "思考中",
                    "executed_action": None,
                    "source": None,
                    "overridden": None,
                }
            )

        system_prompt: str | None = None
        prompt_version: str | None = None
        if self.prompt_evolution is not None and self.prompt_evolution.enabled:
            system_prompt, prompt_version = self.prompt_evolution.system_prompt(observation)
        try:
            selection = self.remote.select(
                observation,
                model_actions,
                on_update=forward_update,
                system_prompt=system_prompt,
                prompt_version=prompt_version,
                temperature=0.0 if self.prompt_evolution is not None and self.prompt_evolution.enabled else None,
            )
        except TeacherError as exc:
            return self._fallback_decision("不可用", str(exc))
        alias = str(selection.get("action_id", ""))
        action = alias_to_action.get(alias)
        if action is None:
            return self._fallback_decision("动作无效", "教师未返回合法动作")
        teacher_metadata = {
            "provider": "openai_compatible",
            "model_id": selection.get("model_id", self.model),
            "prompt_version": selection.get("prompt_version"),
            "thinking_enabled": bool(selection.get("thinking_enabled", self.thinking)),
            "request_id": selection.get("request_id"),
            "latency_ms": selection.get("latency_ms"),
            "attempt": selection.get("attempt"),
            "finish_reason": selection.get("finish_reason"),
            "truncated": bool(selection.get("truncated")),
            "output": selection.get("output", ""),
            "thinking": selection.get("thinking", ""),
            "usage": selection.get("usage"),
        }
        return {
            "mode": self.mode,
            "model_id": selection.get("model_id", self.model),
            "candidate_source": "remote_teacher",
            "model_action": action.get("id"),
            "model_candidates": [
                {
                    "rank": 1,
                    "action_id": action.get("id"),
                    "label": action.get("label") or action.get("command"),
                    "score": None,
                }
            ],
            "teacher": teacher_metadata,
            "executed_action": None,
            "source": None,
            "overridden": None,
        }


def write_jsonl(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def trace_decision(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip verbose remote reasoning from durable action traces.

    The live Overlay receives the full bounded teacher panel payload. SFT and
    replay files retain the selected action and request metadata, not hidden
    chain-of-thought text or the teacher's full response.
    """
    if not isinstance(decision, dict):
        return decision
    result = dict(decision)
    teacher = result.get("teacher")
    if isinstance(teacher, dict):
        teacher_result = dict(teacher)
        teacher_result.pop("thinking", None)
        teacher_result.pop("output", None)
        result["teacher"] = teacher_result
    return result


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
    if decision.get("mode") == "baseline_inference":
        if not decision.get("api", {}).get("revalidated"):
            return None
        action_id = str(decision.get("model_action", ""))
        return next((action for action in actions if str(action.get("id", "")) == action_id), None)
    if decision.get("mode") == "remote_online_rl":
        action_id = str(decision.get("model_action", ""))
        return next((action for action in actions if str(action.get("id", "")) == action_id), None)
    confidence = decision.get("confidence")
    if not isinstance(confidence, dict) or confidence.get("route") != "auto_execute":
        return None
    action_id = str(decision.get("model_action", ""))
    for action in actions:
        if str(action.get("id", "")) == action_id:
            return action
    return None


def combat_reward_key(message: dict[str, Any]) -> tuple[int, int] | None:
    """Identify the current combat reward screen within a run."""
    state = game_state(message)
    if str(state.get("screen_type") or "").upper() != "COMBAT_REWARD":
        return None
    try:
        return int(state.get("act")), int(state.get("floor"))
    except (TypeError, ValueError):
        return None


def blocked_card_reward_indices(
    message: dict[str, Any],
    declined_card_reward_keys: set[tuple[int, int]] | None,
) -> set[int]:
    """Hide a combat card reward after the player/model has skipped it once.

    CommunicationMod returns from ``CARD_REWARD -> SKIP`` to the parent
    ``COMBAT_REWARD`` and leaves the same card entry selectable. Reopening it
    only reproduces the same offer, so it is not a meaningful action.
    """
    key = combat_reward_key(message)
    if key is None or key not in (declined_card_reward_keys or set()):
        return set()
    state = game_state(message)
    choices = state.get("choice_list")
    screen_state = state.get("screen_state")
    rewards = screen_state.get("rewards") if isinstance(screen_state, dict) else None
    blocked = {
        index
        for index, reward in enumerate(rewards)
        if isinstance(reward, dict) and str(reward.get("reward_type", "")).upper() == "CARD"
    } if isinstance(rewards, list) else set()
    if blocked or not isinstance(choices, list):
        return blocked
    return {
        index
        for index, label in enumerate(choices)
        if str(label).strip().casefold() in {"card", "cards", "card reward", "卡牌", "卡牌奖励"}
    }


def selected_card_choice_indices(message: dict[str, Any]) -> set[int]:
    """Return selected card positions that must not be selected again.

    GRID and HAND_SELECT screens keep every selectable card in a list and
    highlighted cards in a separate list. Their public UUIDs identify an
    exact instance, so five same-named Strikes remain distinct. Older protocol
    variants without UUIDs fall back to one-to-one matching by visible fields.
    """
    state = game_state(message)
    screen_type = str(state.get("screen_type") or "").upper()
    fields_by_screen = {
        "GRID": ("cards", "selected_cards"),
        "HAND_SELECT": ("hand", "selected"),
    }
    selection_fields = fields_by_screen.get(screen_type)
    if selection_fields is None:
        return set()
    screen_state = state.get("screen_state")
    choices = state.get("choice_list")
    if not isinstance(screen_state, dict) or not isinstance(choices, list):
        return set()
    cards = screen_state.get(selection_fields[0])
    selected_cards = screen_state.get(selection_fields[1])
    if not isinstance(cards, list) or not isinstance(selected_cards, list):
        return set()

    available_indices = {
        index
        for index, card in enumerate(cards)
        if index < len(choices) and isinstance(card, dict)
    }
    uuid_to_index = {
        str(card["uuid"]): index
        for index, card in enumerate(cards)
        if index in available_indices and isinstance(card, dict) and card.get("uuid") is not None
    }
    blocked: set[int] = set()
    unmatched_selected: list[dict[str, Any]] = []
    for selected in selected_cards:
        if not isinstance(selected, dict):
            continue
        uuid = selected.get("uuid")
        index = uuid_to_index.get(str(uuid)) if uuid is not None else None
        if index is None:
            unmatched_selected.append(selected)
        else:
            blocked.add(index)

    for selected in unmatched_selected:
        for index in sorted(available_indices - blocked):
            card = cards[index]
            if not isinstance(card, dict):
                continue
            fields = tuple(
                key
                for key in ("id", "upgrades", "name", "cost")
                if selected.get(key) is not None and card.get(key) is not None
            )
            if fields and all(selected[key] == card[key] for key in fields):
                blocked.add(index)
                break
    return blocked


def remember_declined_combat_card_reward(
    pending: dict[str, Any],
    message: dict[str, Any],
    declined_card_reward_keys: set[tuple[int, int]],
) -> bool:
    """Record a confirmed card-reward skip that returned to combat rewards."""
    action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
    if pending.get("screen_type") != "CARD_REWARD" or action.get("kind") != "skip":
        return False
    key = combat_reward_key(message)
    if key is None:
        return False
    declined_card_reward_keys.add(key)
    return True


def teacher_action_for_decision(
    decision: dict[str, Any] | None,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve one remote-teacher action only after the local legality check."""
    if not isinstance(decision, dict) or decision.get("mode") not in {"teacher_collection", "prompt_evolution"}:
        return None
    if not decision.get("model_candidates"):
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
    teacher: TeacherClient | None = None,
    run_config: dict[str, Any] | None = None,
    visited_shop_rooms: set[tuple[int, int]] | None = None,
    action_history: list[dict[str, Any]] | None = None,
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
    actions = semantic_actions(message, visited_shop_rooms, declined_card_reward_keys)

    def send_model_update(
        decision_update: dict[str, Any],
        *,
        model_pending: bool = False,
        auto_execute: bool = False,
    ) -> None:
        """Push a transient teacher or policy status without changing game state."""
        if getattr(bridge, "connection", None) is None:
            return
        packet = {
                    "type": "model_update",
                    "text": format_view(message, run_id, error, decision_update, visited_shop_rooms, actions),
                    "in_game": bool(message.get("in_game")),
                    "decision": decision_update,
                    "overlay": overlay_view_data(message, run_id, actions, action_history),
                    "model_pending": model_pending,
                    "auto_execute": auto_execute,
                }
        evolution = getattr(teacher, "prompt_evolution", None)
        if evolution is not None and evolution.enabled:
            packet["prompt_evolution"] = evolution.status()
        try:
            bridge.send(packet)
        except OSError:
            # The game action can still be executed if the optional UI closes
            # during a long streaming request.
            return

    # Do this before contacting a teacher or policy. It keeps reward handling
    # deterministic and avoids spending an inference call on a non-decision.
    if not error:
        environment_action = environment_auto_action(message, actions, visited_shop_rooms)
        if environment_action is not None:
            action, detail = environment_action
            decision = environment_auto_decision(action, detail)
            send_model_update(decision, auto_execute=True)
            time.sleep(AUTOMATIC_ACTION_DELAY_SECONDS)
            print(
                f"environment auto: {action.get('id')} rule={detail.get('rule')}",
                file=sys.stderr,
                flush=True,
            )
            return str(action["command"]), action, decision

    teacher_active = (
        teacher is not None
        and bool(getattr(teacher, "enabled", False))
        and candidate_assist_allowed(message, teacher.screens)
    )

    def send_teacher_update(
        decision_update: dict[str, Any],
        *,
        lightweight: bool = True,
    ) -> None:
        """Send live teacher output without repainting the current game view.

        A streaming completion can yield many small reasoning fragments.  The
        regular ``model_update`` packet includes a formatted terminal view and
        a full overlay state, which made each fragment rebuild the status
        table.  Only the first/terminal status needs that complete packet.
        """
        if getattr(bridge, "connection", None) is None:
            return
        if not lightweight:
            send_model_update(decision_update, model_pending=True)
            return
        packet: dict[str, Any] = {
            "type": "teacher_update",
            "decision": decision_update,
            "model_pending": True,
        }
        evolution = getattr(teacher, "prompt_evolution", None)
        if evolution is not None and evolution.enabled:
            packet["prompt_evolution"] = evolution.status()
        try:
            bridge.send(packet)
        except OSError:
            # The game action can still be executed if the optional UI closes
            # during a long streaming request.
            return

    if teacher_active:
        send_teacher_update(
            {
                "mode": getattr(teacher, "mode", "teacher_collection"),
                "model_id": teacher.model,
                "candidate_source": "remote_teacher",
                "model_action": None,
                "model_candidates": [],
                "teacher": {
                    "provider": "openai_compatible",
                    "model_id": teacher.model,
                    "thinking_enabled": bool(getattr(teacher, "thinking", True)),
                    "thinking": "正在请求教师模型...",
                    "output": "",
                },
                "assist_status": "请求中",
                "executed_action": None,
                "source": None,
                "overridden": None,
            },
            lightweight=False,
        )
        decision = teacher.rank(
            policy_observation(message, visited_shop_rooms, action_history, declined_card_reward_keys),
            actions,
            on_update=send_teacher_update,
        )
    elif policy is None:
        decision = None
    elif error and getattr(policy, "runtime", "local") == "remote_api":
        policy.api.pending = None
        decision = {**policy.status_decision(), "assist_status": "暂停，需人工处理", "assist_reason": error}
    elif candidate_assist_allowed(message, policy.screens):
        observation = policy_observation(message, visited_shop_rooms, action_history, declined_card_reward_keys)
        requesting_decision = getattr(policy, "requesting_decision", None)
        if callable(requesting_decision):
            send_model_update(requesting_decision(), model_pending=True)
        if getattr(policy, "runtime", "local") in {"remote_online_rl", "remote_api"}:
            decision = policy.rank(observation, actions, run_id=run_id, run_config=run_config)
        else:
            decision = policy.rank(observation, actions)
    else:
        decision = policy.status_decision()
    if (not error and isinstance(decision, dict) and decision.get("mode") == "baseline_inference"
            and decision.get("model_action") and not decision.get("api", {}).get("revalidated")):
        send_model_update(decision, model_pending=True)
        # STATE is a protocol query, not a game action or training example.
        print("STATE", flush=True)
        return "STATE", {"id": "STATE", "kind": "state_request", "command": "STATE"}, None
    # Error responses are a hard human handoff. Repeating an otherwise
    # high-confidence action after the protocol rejects it can create a tight
    # automatic loop and should never be treated as a new policy decision.
    automatic = None
    automatic_source = "model_auto"
    if not error:
        if teacher_active:
            automatic = teacher_action_for_decision(decision, actions)
            automatic_source = "teacher_auto"
        else:
            automatic = auto_action_for_decision(decision, actions)
            if isinstance(decision, dict) and decision.get("mode") == "remote_online_rl":
                automatic_source = "online_policy_sample"
            elif isinstance(decision, dict) and decision.get("mode") == "baseline_inference":
                automatic_source = "baseline_model"
    if automatic is not None:
        automatic_decision = finalize_decision(decision, automatic, automatic_source)
        confidence = automatic_decision.get("confidence", {}) if automatic_decision else {}
        if automatic_source == "teacher_auto":
            teacher_info = automatic_decision.get("teacher", {}) if automatic_decision else {}
            print(
                f"teacher auto: {automatic.get('id')} model={teacher_info.get('model_id', '?')} "
                f"latency_ms={teacher_info.get('latency_ms', '?')}",
                file=sys.stderr,
                flush=True,
            )
        elif automatic_source == "online_policy_sample":
            online = automatic_decision.get("online_rl", {}) if automatic_decision else {}
            print(
                f"online policy: {automatic.get('id')} group={online.get('group_id', '?')} "
                f"version={online.get('policy_version', '?')}",
                file=sys.stderr,
                flush=True,
            )
        elif automatic_source == "baseline_model":
            api_info = automatic_decision.get("api", {})
            print(f"baseline auto: {automatic.get('id')} latency_ms={api_info.get('latency_ms')} "
                  f"input_tokens={api_info.get('input_tokens')}", file=sys.stderr, flush=True)
        else:
            print(
                f"confidence auto: {automatic.get('id')} "
                f"p={float(confidence.get('top_probability', 0.0)):.3f} "
                f"margin={float(confidence.get('margin', 0.0)):.3f}",
                file=sys.stderr,
                flush=True,
            )
        if automatic_source in {"teacher_auto", "online_policy_sample", "baseline_model"}:
            # Both automatic modes bypass terminal input. Push their resolved
            # result first so the Overlay can show the actual selected action
            # instead of leaving the Model/Action panel blank.
            send_model_update(automatic_decision, auto_execute=True)
        time.sleep(AUTOMATIC_ACTION_DELAY_SECONDS)
        return str(automatic["command"]), automatic, automatic_decision
    while True:
        packet = {
                "type": "view",
                "text": format_view(message, run_id, error, decision, visited_shop_rooms, actions),
                "in_game": bool(message.get("in_game")),
                "decision": decision,
                "overlay": overlay_view_data(message, run_id, actions, action_history),
            }
        if config is not None and config["framework"]["control_mode"] == "codex":
            packet["state_id"] = uuid.uuid4().hex
            # START is a session command, not an in-run semantic action.
            # Preserve the protocol capabilities so MCP can identify a menu
            # even when CommunicationMod supplies no game_state at all.
            packet["available_commands"] = sorted(command_set(message))
            packet["ready_for_command"] = message.get("ready_for_command")
            packet["error"] = error or message.get("error")
            packet["observation"] = policy_observation(
                message,
                visited_shop_rooms,
                action_history,
                declined_card_reward_keys,
            )
            packet["legal_actions"] = actions
        evolution = getattr(teacher, "prompt_evolution", None)
        if evolution is not None and evolution.enabled:
            packet["prompt_evolution"] = evolution.status()
        try:
            bridge.send(packet)
            response = bridge.receive()
        except OSError:
            # Re-publish this decision after a client reconnects. Never replay
            # a game command just because its client connection was lost.
            bridge.close_client()
            continue
        if response is None:
            continue
        text = str(response.get("text", ""))
        command, action, parse_error = parse_human_input(
            text,
            message,
            decision,
            visited_shop_rooms,
            declined_card_reward_keys,
        )
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
        codex_control = config is not None and config["framework"]["control_mode"] == "codex"
        source = "codex" if codex_control else candidate_source_for_input(text, decision)
        if codex_control:
            decision = {
                "mode": "codex_collection",
                "candidate_source": "codex",
                "model_action": str(action.get("id") or ""),
                "model_candidates": [],
                "source": "codex",
                "overridden": False,
            }
        return command, action, finalize_decision(decision, action, source)


def main() -> int:
    parser = argparse.ArgumentParser(description="CommunicationMod controller for the StS agent")
    parser.add_argument("--config", default="config/agent.json", help="项目根目录下的配置文件路径")
    args = parser.parse_args()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = PROJECT_ROOT
    try:
        config = load_agent_config(project_root, args.config)
    except ConfigError as exc:
        print(f"invalid agent configuration: {exc}", file=sys.stderr, flush=True)
        return 2
    if config["framework"]["learning_method"] == "collect_human":
        # Keep CommunicationMod's command stable when switching modes. Human
        # gameplay requires the passive state/action bridge, not TerminalBridge.
        from sts_agent.recording.communication_bridge import main as recording_main

        return recording_main()
    raw_dir = project_root / "data" / "raw" / "communication_controller"
    trace_dir = project_root / "data" / "trajectories"
    reflection_dir = project_root / "data" / "reflections"
    attribution_dir = project_root / "data" / "attributions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"messages-{session_id}.jsonl"

    bridge = TerminalBridge(config["server"]["host"], config["server"]["port"])
    try:
        prompt_evolution = PromptEvolutionManager(
            project_root,
            config["prompt_evolution"],
            config["teacher"],
            session_id,
        )
    except PromptEvolutionError as exc:
        print(f"prompt evolution unavailable: {exc}", file=sys.stderr, flush=True)
        return 2
    policy = PolicyClient(project_root, config)
    if policy.runtime == "remote_api":
        # AI evaluation records never enter the human collection directory.
        trace_dir = policy.api.output_dir
        raw_path = trace_dir / f"messages-{session_id}.jsonl"
    teacher = TeacherClient(project_root, config, prompt_evolution)
    posthoc = PosthocFeedbackClient(project_root, config["posthoc_feedback"])
    pending: dict[str, Any] | None = None
    current_message: dict[str, Any] | None = None
    run_id: str | None = None
    run_config: dict[str, Any] | None = None
    run_index = 0
    step_id = 0
    message_index = 0
    visited_shop_rooms: set[tuple[int, int]] = set()
    declined_card_reward_keys: set[tuple[int, int]] = set()
    action_history: list[dict[str, Any]] = []
    smoke_bomb_transition: dict[str, Any] | None = None
    online_restart_config: dict[str, Any] | None = None
    prompt_evolution_restart_config: dict[str, Any] | None = None
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
        output_dir = run_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path, event_path = run_output_paths(output_dir, session_id, run_index, run_config)
        trace_output = trace_path.open("w", encoding="utf-8")
        event_output = event_path.open("w", encoding="utf-8")

    def rebind_run_outputs_after_seed() -> None:
        nonlocal trace_output, event_output, trace_path, event_path
        if run_config is None or run_config.get("seed") is None or trace_path is None or event_path is None:
            return
        final_trace_path, final_event_path = run_output_paths(run_output_dir(), session_id, run_index, run_config)
        if final_trace_path == trace_path and final_event_path == event_path:
            return
        close_run_outputs()
        os.replace(trace_path, final_trace_path)
        os.replace(event_path, final_event_path)
        trace_path, event_path = final_trace_path, final_event_path
        trace_output = trace_path.open("a", encoding="utf-8")
        event_output = event_path.open("a", encoding="utf-8")

    def run_output_dir() -> Path:
        if run_config is not None and run_config.get("collection_mode") == "codex":
            return project_root / config["codex"]["output_dir"]
        return trace_dir

    def write_trace(value: dict[str, Any]) -> None:
        if trace_output is not None:
            write_jsonl(trace_output, value)
            if policy.runtime == "remote_api":
                policy.api.record_result(value)

    def write_event(value: dict[str, Any]) -> None:
        if event_output is not None:
            write_jsonl(event_output, value)

    def trace_state(message: dict[str, Any]) -> dict[str, Any]:
        if not message.get("in_game"):
            return {"in_game": False}
        value = normalize_state(message)
        return attach_runtime_context(value, visited_shop_rooms, action_history, declined_card_reward_keys)

    def begin_run(action: dict[str, Any], collection_mode: str) -> str:
        """Start a new durable run context for a human or online-RL START."""

        nonlocal run_index, run_id, step_id, run_config
        run_index += 1
        run_id = f"{session_id}-run-{run_index:03d}"
        step_id = 0
        visited_shop_rooms.clear()
        declined_card_reward_keys.clear()
        action_history.clear()
        run_config = {
            "character": action.get("character", "IRONCLAD"),
            "ascension": action.get("ascension", 0),
            "seed": action.get("seed"),
            "seed_source": "requested" if action.get("seed") else "random",
            "internal_seed": None,
            "collection_mode": collection_mode,
            "actor": config["codex"]["actor_id"] if collection_mode == "codex" else (
                "human" if collection_mode in {"interactive", "collect_human"} else collection_mode
            ),
        }
        if prompt_evolution.enabled:
            run_config["prompt_evolution"] = prompt_evolution.begin_run(run_config)
        open_run_outputs()
        write_event(
            {
                "event": "run_start",
                "run_id": run_id,
                "run_config": run_config,
                "command": action["command"],
                "at": now_iso(),
            }
        )
        return str(action["command"])

    # The game must not expose a start action before local setup has finished.
    # CommunicationMod waits for this stdout handshake.
    teacher.prewarm()
    prompt_evolution_preflight_error = prompt_evolution.preflight()
    if prompt_evolution.enabled:
        print(
            f"prompt evolution ready: character selected at START; active skill version will be loaded per run"
            if prompt_evolution_preflight_error is None
            else f"prompt evolution unavailable: {prompt_evolution_preflight_error}",
            file=sys.stderr,
            flush=True,
        )
    posthoc_preflight_error = posthoc.preflight()
    policy.prewarm()
    if policy.runtime == "remote_online_rl" and policy.enabled and posthoc.enabled != policy.online_rl_attribution_enabled:
        # The two switches must move together. Either mismatch would silently
        # change update semantics: old workers would train every step, while a
        # masked worker would receive no selected steps at all.
        policy._disable("posthoc_feedback.enabled 与远端 attribution.enabled 必须同时启用或关闭")

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
                                "decision": trace_decision(pending.get("decision")),
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
                            bridge=bridge,
                            message=current_message,
                            run_id=run_id,
                            error=str(message.get("error")),
                            policy=policy,
                            teacher=teacher,
                            run_config=run_config,
                            visited_shop_rooms=visited_shop_rooms,
                            action_history=action_history,
                            declined_card_reward_keys=declined_card_reward_keys,
                            config=config,
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
                                declined_card_reward_keys,
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

                # A semantic room action may own several protocol-level
                # commands.  Consume those continuation packets before the
                # ordinary pending-step acknowledgement is recorded; the
                # whole sequence remains one policy decision in the trace.
                if pending is not None and pending.get("action", {}).get("execution"):
                    continuation = advance_semantic_execution(pending, message)
                    if continuation is not None:
                        print(continuation, flush=True)
                        continue

                if pending is not None:
                    smoke_bomb_used = is_smoke_bomb_potion_use(pending)
                    if remember_declined_combat_card_reward(pending, message, declined_card_reward_keys):
                        key = combat_reward_key(message)
                        if key is not None:
                            write_event(
                                {
                                    "event": "combat_card_reward_declined",
                                    "run_id": run_id,
                                    "act": key[0],
                                    "floor": key[1],
                                    "at": now_iso(),
                                }
                            )
                    action_history.append(accepted_action_history_item(pending, message))
                    del action_history[:-ACTION_HISTORY_STORAGE_LIMIT]
                    completed_record = {
                        "schema_version": "0.2" if pending.get("decision") else "0.1",
                        "session_id": session_id,
                        "run_id": pending.get("run_id"),
                        "run_config": pending.get("run_config"),
                        "step_id": pending.get("step_id"),
                        "screen_type": pending.get("screen_type"),
                        "pre_state": pending.get("pre_state"),
                        "legal_actions": pending.get("legal_actions", []),
                        "decision": trace_decision(pending.get("decision")),
                        "action": pending.get("action"),
                        "command": pending.get("command"),
                        "post_state": trace_state(message),
                        "accepted": True,
                        "stable_after_action": not smoke_bomb_used,
                        "terminal": game_state(message).get("screen_type") in {"GAME_OVER", "VICTORY"},
                    }
                    write_trace(completed_record)
                    if (
                        completed_record["terminal"]
                        and policy.runtime == "remote_online_rl"
                        and run_id is not None
                        and run_config is not None
                        and trace_path is not None
                        and trace_output is not None
                    ):
                        trace_output.flush()
                        try:
                            online_episode = online_episode_from_trace(trace_path, run_id, run_config)
                        except (OSError, ValueError) as exc:
                            online_result: dict[str, Any] | None = {
                                "type": "error",
                                "error": f"cannot prepare online episode: {exc}",
                            }
                        else:
                            if posthoc.enabled:
                                reflection_stage = policy.online_rl_status_decision(
                                    "教师正在自动复盘本局" if posthoc.mode == "auto" else "等待本局终局人工反思"
                                )
                                try:
                                    bridge.send(
                                        {
                                            "type": "online_rl_update",
                                            "model_id": reflection_stage["model_id"],
                                            "online_rl": reflection_stage["online_rl"],
                                        }
                                    )
                                except OSError:
                                    pass
                                if posthoc.mode == "auto":
                                    reflection = posthoc.automatic_reflection()
                                else:
                                    raw_reflection = request_posthoc_reflection(
                                        bridge,
                                        run_id,
                                        online_episode["terminal_state"],
                                        action_review_from_trace(trace_path),
                                    )
                                    reflection = posthoc.normalize_reflection(
                                        raw_reflection.get("reinforce"),
                                        raw_reflection.get("weaken"),
                                    ) | {"reason": raw_reflection.get("reason")}
                                reflection_path = reflection_dir / f"reflection-{run_id}.json"
                                write_posthoc_artifact(
                                    reflection_path,
                                    {
                                        "schema_version": POSTHOC_SCHEMA_VERSION,
                                        "kind": "teacher_auto_review" if posthoc.mode == "auto" else "human_reflection",
                                        "run_id": run_id,
                                        "run_config": run_config,
                                        "terminal_state": posthoc_terminal_summary(online_episode["terminal_state"]),
                                        "reflection": reflection,
                                        "created_at_utc": posthoc_now_utc(),
                                    },
                                )
                                annotation: dict[str, Any]
                                if posthoc_preflight_error:
                                    annotation = {
                                        "status": "teacher_unavailable",
                                        "error": posthoc_preflight_error,
                                        "applied": [],
                                        "rejected": [],
                                    }
                                else:
                                    annotation_stage = policy.online_rl_status_decision(
                                        "教师正在自动定位可强化和应弱化的关键动作"
                                        if posthoc.mode == "auto"
                                        else "教师正在定位可强化和应弱化的关键动作"
                                    )
                                    try:
                                        bridge.send(
                                            {
                                                "type": "online_rl_update",
                                                "model_id": annotation_stage["model_id"],
                                                "online_rl": annotation_stage["online_rl"],
                                            }
                                        )
                                    except OSError:
                                        pass
                                    try:
                                        annotation = posthoc.annotate(reflection, online_episode)
                                    except PosthocFeedbackError as exc:
                                        annotation = {
                                            "status": "teacher_error",
                                            "error": str(exc),
                                            "applied": [],
                                            "rejected": [],
                                        }
                                attribution_path = attribution_dir / f"attribution-{run_id}.json"
                                write_posthoc_artifact(
                                    attribution_path,
                                    {
                                        "schema_version": POSTHOC_SCHEMA_VERSION,
                                        "kind": "teacher_attribution",
                                        "run_id": run_id,
                                        "run_config": run_config,
                                        "terminal_state": posthoc_terminal_summary(online_episode["terminal_state"]),
                                        "reflection_path": str(reflection_path.relative_to(project_root)),
                                        "annotation": annotation,
                                        "created_at_utc": posthoc_now_utc(),
                                    },
                                )
                                try:
                                    bridge.send(
                                        {
                                            "type": "posthoc_attribution_result",
                                            "run_id": run_id,
                                            "terminal_state": posthoc_terminal_summary(online_episode["terminal_state"]),
                                            "annotation": annotation,
                                        }
                                    )
                                except OSError:
                                    pass
                                online_episode["attributions"] = list(annotation.get("applied") or [])
                                write_event(
                                    {
                                        "event": "posthoc_feedback_complete",
                                        "run_id": run_id,
                                        "reflection_path": str(reflection_path.relative_to(project_root)),
                                        "attribution_path": str(attribution_path.relative_to(project_root)),
                                        "status": annotation.get("status"),
                                        "selected_steps": len(online_episode["attributions"]),
                                        "at": now_iso(),
                                    }
                                )

                            current_online = policy.online_rl_status
                            rollout_index = current_online.get("rollout_index")
                            group_size = current_online.get("group_size")
                            is_group_final = (
                                isinstance(rollout_index, int)
                                and isinstance(group_size, int)
                                and rollout_index == group_size
                            )
                            upload_stage = (
                                f"正在更新参数：第 {rollout_index}/{group_size} 次 rollout 已结束"
                                if is_group_final
                                else f"正在上传第 {rollout_index or '?'} / {group_size or '?'} 次 rollout 轨迹"
                            )
                            upload_decision = policy.online_rl_status_decision(upload_stage)
                            try:
                                bridge.send(
                                    {
                                        "type": "online_rl_update",
                                        "model_id": upload_decision["model_id"],
                                        "online_rl": upload_decision["online_rl"],
                                    }
                                )
                            except OSError:
                                pass
                            online_result = policy.submit_episode(online_episode)
                        if online_result is not None:
                            online_decision = policy.record_online_episode_result(online_result)
                            try:
                                bridge.send(
                                    {
                                        "type": "online_rl_update",
                                        "model_id": online_decision["model_id"],
                                        "online_rl": online_decision["online_rl"],
                                    }
                                )
                            except OSError:
                                pass
                            write_event(
                                {
                                    "event": "online_rl_episode_result",
                                    "run_id": run_id,
                                    "result": online_result,
                                    "at": now_iso(),
                                }
                            )
                            print(
                                f"online RL episode: {online_result.get('type')} "
                                f"group={online_result.get('group_id', '?')} "
                                f"policy={online_result.get('policy_version', '?')}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if online_result.get("type") in {"episode_queued", "policy_updated", "group_completed_no_update"}:
                                online_restart_config = run_config.copy()
                    if (
                        completed_record["terminal"]
                        and prompt_evolution.enabled
                        and run_id is not None
                        and run_config is not None
                        and trace_path is not None
                        and trace_output is not None
                    ):
                        trace_output.flush()
                        if prompt_evolution_preflight_error:
                            evolution_result: dict[str, Any] = {
                                "kind": "proposal_error",
                                "error": prompt_evolution_preflight_error,
                            }
                        else:
                            evolution_result = prompt_evolution.complete_run(
                                run_id,
                                run_config,
                                trace_path,
                                completed_record["post_state"],
                            )
                        if evolution_result.get("kind") in {"candidate_review", "review", "result_review"}:
                            review_error: str | None = None
                            while evolution_result.get("kind") in {"candidate_review", "review", "result_review"}:
                                review = evolution_result.get("review")
                                try:
                                    bridge.send(
                                        {
                                            "type": "prompt_evolution_update",
                                            "model_id": teacher.model,
                                            "prompt_evolution": prompt_evolution.status(),
                                        }
                                    )
                                except OSError:
                                    pass
                                review_response = request_prompt_evolution_review(bridge, review, review_error)
                                evolution_result = prompt_evolution.handle_review(review_response)
                                if evolution_result.get("kind") == "error":
                                    review_error = str(evolution_result.get("error") or "审核结果无效")
                                else:
                                    review_error = None
                        if evolution_result.get("kind") == "start_candidate":
                            prompt_evolution_restart_config = dict(evolution_result["run_config"])
                        try:
                            bridge.send(
                                {
                                    "type": "prompt_evolution_update",
                                    "model_id": teacher.model,
                                    "prompt_evolution": prompt_evolution.status(),
                                }
                            )
                        except OSError:
                            pass
                        write_event(
                            {
                                "event": "prompt_evolution_terminal",
                                "run_id": run_id,
                                "result": evolution_result,
                                "at": now_iso(),
                            }
                        )
                        print(
                            f"prompt evolution: {evolution_result.get('kind')} "
                            f"policy={prompt_evolution.active_version}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if smoke_bomb_used:
                        transition_started_at = time.monotonic()
                        smoke_bomb_transition = {
                            "resume_at": transition_started_at + SMOKE_BOMB_ANIMATION_DELAY_SECONDS,
                            "deadline": transition_started_at + SMOKE_BOMB_TRANSITION_TIMEOUT_SECONDS,
                            "run_id": pending.get("run_id"),
                            "step_id": pending.get("step_id"),
                        }
                        write_event(
                            {
                                "event": "smoke_bomb_animation_wait",
                                "run_id": pending.get("run_id"),
                                "step_id": pending.get("step_id"),
                                "delay_seconds": SMOKE_BOMB_ANIMATION_DELAY_SECONDS,
                                "poll_seconds": SMOKE_BOMB_STATE_POLL_SECONDS,
                                "timeout_seconds": SMOKE_BOMB_TRANSITION_TIMEOUT_SECONDS,
                                "at": now_iso(),
                            }
                        )
                    pending = None

                current_message = message
                smoke_bomb_timeout_error: str | None = None
                if smoke_bomb_transition is not None:
                    remaining = float(smoke_bomb_transition["resume_at"]) - time.monotonic()
                    if remaining > 0:
                        print(f"waiting {remaining:.1f}s for Smoke Bomb animation", file=sys.stderr, flush=True)
                        time.sleep(remaining)
                        # Do not ask the model about the acknowledgement
                        # state: it can still be the pre-escape combat screen.
                        current_message = None
                        print("STATE", flush=True)
                        continue
                    if smoke_bomb_escape_in_progress(message):
                        remaining = float(smoke_bomb_transition["deadline"]) - time.monotonic()
                        if remaining > 0:
                            delay = min(SMOKE_BOMB_STATE_POLL_SECONDS, remaining)
                            print(
                                f"Smoke Bomb transition still in combat; polling again in {delay:.1f}s",
                                file=sys.stderr,
                                flush=True,
                            )
                            time.sleep(delay)
                            current_message = None
                            print("STATE", flush=True)
                            continue
                        smoke_bomb_timeout_error = (
                            "烟雾弹逃跑动画在 15 秒后仍停留在原战斗；已暂停自动执行，请确认游戏界面后手动操作。"
                        )
                        write_event(
                            {
                                "event": "smoke_bomb_transition_timeout",
                                "run_id": smoke_bomb_transition.get("run_id"),
                                "step_id": smoke_bomb_transition.get("step_id"),
                                "at": now_iso(),
                            }
                        )
                        smoke_bomb_transition = None
                    else:
                        # The first non-combat observation reopens the normal
                        # action pipeline. environment_auto_action() runs
                        # before the teacher, so visible reward automation
                        # finishes before another model decision is requested.
                        write_event(
                            {
                                "event": "smoke_bomb_transition_complete",
                                "run_id": smoke_bomb_transition.get("run_id"),
                                "step_id": smoke_bomb_transition.get("step_id"),
                                "at": now_iso(),
                            }
                        )
                        smoke_bomb_transition = None
                if not message.get("in_game"):
                    # A main-menu observation starts a fresh run context; do
                    # not leak the previous run's action history into it.
                    action_history.clear()
                    declined_card_reward_keys.clear()
                if online_restart_config is not None:
                    if message.get("in_game"):
                        exit_actions = semantic_actions(message, visited_shop_rooms, declined_card_reward_keys)
                        exit_action = next(
                            (item for item in exit_actions if item.get("kind") == "confirm"),
                            None,
                        )
                        if exit_action is not None:
                            pending = make_pending(
                                message,
                                run_id,
                                run_config,
                                step_id,
                                exit_action,
                                None,
                                visited_shop_rooms,
                                action_history,
                                declined_card_reward_keys,
                            )
                            step_id += 1
                            print(str(exit_action["command"]), flush=True)
                            continue
                        print("online RL waiting for terminal transition", file=sys.stderr, flush=True)
                        print("STATE", flush=True)
                        continue
                    restart_action = {
                        "id": "START_RUN",
                        "kind": "start_run",
                        "character": online_restart_config["character"],
                        "ascension": online_restart_config["ascension"],
                        "seed": online_restart_config["seed"],
                    }
                    restart_action["command"] = (
                        f"START {restart_action['character']} {restart_action['ascension']}"
                        + (f" {restart_action['seed']}" if restart_action["seed"] else "")
                    )
                    online_restart_config = None
                    command = begin_run(restart_action, "remote_online_rl")
                    write_event({"event": "online_rl_auto_restart", "run_id": run_id, "at": now_iso()})
                    print(command, flush=True)
                    continue
                if prompt_evolution_restart_config is not None:
                    if message.get("in_game"):
                        exit_actions = semantic_actions(message, visited_shop_rooms, declined_card_reward_keys)
                        exit_action = next(
                            (item for item in exit_actions if item.get("kind") == "confirm"),
                            None,
                        )
                        if exit_action is not None:
                            pending = make_pending(
                                message,
                                run_id,
                                run_config,
                                step_id,
                                exit_action,
                                None,
                                visited_shop_rooms,
                                action_history,
                                declined_card_reward_keys,
                            )
                            step_id += 1
                            print(str(exit_action["command"]), flush=True)
                            continue
                        print("prompt evolution waiting for terminal transition", file=sys.stderr, flush=True)
                        print("STATE", flush=True)
                        continue
                    restart_action = {
                        "id": "START_RUN",
                        "kind": "start_run",
                        "character": prompt_evolution_restart_config["character"],
                        "ascension": prompt_evolution_restart_config["ascension"],
                        "seed": prompt_evolution_restart_config["seed"],
                    }
                    restart_action["command"] = (
                        f"START {restart_action['character']} {restart_action['ascension']}"
                        + (f" {restart_action['seed']}" if restart_action["seed"] else "")
                    )
                    prompt_evolution_restart_config = None
                    command = begin_run(restart_action, "prompt_evolution")
                    write_event({"event": "prompt_evolution_auto_restart", "run_id": run_id, "at": now_iso()})
                    print(command, flush=True)
                    continue
                if policy.runtime == "remote_api" and run_id is None and message.get("in_game"):
                    # Attaching mid-run is allowed, but is explicitly partial
                    # and must not count as a full autonomous baseline game.
                    attached_state = game_state(message)
                    begin_run({"character": attached_state.get("class", "UNKNOWN"),
                               "ascension": attached_state.get("ascension_level", 0),
                               "command": None}, "baseline_inference")
                    assert run_config is not None
                    run_config["partial_run"] = True
                    write_event({"event": "run_attached", "run_id": run_id, "partial_run": True, "at": now_iso()})
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
                    error=smoke_bomb_timeout_error,
                    policy=policy,
                    teacher=teacher,
                    run_config=run_config,
                    visited_shop_rooms=visited_shop_rooms,
                    action_history=action_history,
                    declined_card_reward_keys=declined_card_reward_keys,
                    config=config,
                )
                if result is None:
                    return 0
                command, action, decision = result
                if action.get("kind") == "start_run":
                    collection_mode = "codex" if config["framework"]["control_mode"] == "codex" else (
                        "prompt_evolution" if prompt_evolution.enabled else ("teacher_sft" if teacher.enabled else (
                            "remote_online_rl" if policy.runtime == "remote_online_rl" else (
                                "baseline_inference" if policy.runtime == "remote_api" else "interactive"
                            )
                        ))
                    )
                    print(begin_run(action, collection_mode), flush=True)
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
                        declined_card_reward_keys,
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
    declined_card_reward_keys: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    normalized = normalize_state(message)
    attach_runtime_context(normalized, visited_shop_rooms, action_history, declined_card_reward_keys)
    return {
        "run_id": run_id,
        "run_config": run_config.copy() if run_config is not None else None,
        "step_id": step_id,
        "screen_type": normalized.get("screen_type"),
        "pre_state": normalized,
        "legal_actions": semantic_actions(message, visited_shop_rooms, declined_card_reward_keys),
        "decision": decision,
        "action": action,
        "command": action.get("command"),
    }


def advance_semantic_execution(pending: dict[str, Any], message: dict[str, Any]) -> str | None:
    """Expand one room-level action into CommunicationMod commands.

    ``None`` means that the semantic action has reached its terminal state and
    the normal pending-step completion path should run.  A command string
    means an internal continuation was sent and the caller must skip policy
    inference for this message.  This keeps multi-click UI details out of the
    model while retaining one trace item for the user's actual decision.
    """
    action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
    execution = action.get("execution") if isinstance(action.get("execution"), dict) else None
    if execution is None:
        return None
    phase = str(execution.get("phase") or "await_grid")
    kind = str(execution.get("type") or "")
    state = game_state(message)
    screen = str(state.get("screen_type") or "").upper()
    screen_state = state.get("screen_state") if isinstance(state.get("screen_state"), dict) else {}

    def send(command: str, next_phase: str) -> str:
        execution["phase"] = next_phase
        execution["last_command"] = command
        action["execution"] = execution
        return command

    if kind not in {"campfire_smith", "shop_purge"}:
        return None
    card_uuid = str(execution.get("card_uuid") or "")
    if phase == "await_grid":
        if screen != "GRID":
            return None
        cards = screen_state.get("cards") if isinstance(screen_state.get("cards"), list) else []
        card_index = next(
            (index for index, card in enumerate(cards) if isinstance(card, dict) and str(card.get("uuid") or "") == card_uuid),
            None,
        )
        if card_index is None:
            return None
        return send(f"CHOOSE {card_index}", "await_confirm")
    if phase == "await_confirm":
        if screen != "GRID":
            return None
        confirm_up = bool(screen_state.get("confirm_up") or screen_state.get("confirm_screen_up"))
        commands = command_set(message)
        if confirm_up and "confirm" in commands:
            return send("CONFIRM", "await_done")
        # Some CommunicationMod versions expose the confirm overlay one
        # update after the card click.  Keep the executor alive rather than
        # asking the model to operate the raw GRID.
        return "STATE"
    if phase == "await_done":
        # The action is complete when the grid closes.  If a lagging packet
        # still reports GRID, leave the pending execution in place and poll.
        if screen == "GRID":
            return "STATE"
        return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
