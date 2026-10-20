"""Public model presentation: runtime keyword explanations and readable references.

Only operates on a model-owned observation, never the recorder/control identity.
New Recorder rewards supply stable references. Older snapshots receive aliases
consistent within this observation/history, not a guessed persistent identity.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from sts_agent.policy.reward_text import project_reward_text


_REWARD_KEYS = ("reward_id", "linked_reward_id", "source_reward_id")
_UUID = re.compile(r"(?:[a-z]+-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def prepare_model_context(observation: dict[str, Any]) -> dict[str, Any]:
    """Mutate this model-owned projection, preserving actual type IDs/effects."""
    rewards: dict[str, str] = {}
    used: set[str] = set()

    def register(value: Any) -> None:
        if isinstance(value, dict):
            for key in _REWARD_KEYS:
                raw, ref = value.get(key), value.get(key.removesuffix("_id") + "_ref")
                if raw is not None and isinstance(ref, str) and re.fullmatch(r"R[1-9]\d*", ref):
                    rewards[str(raw)] = ref
                    used.add(ref)
                # Makes the projection idempotent on cached SFT/model contexts.
                if isinstance(raw, str) and re.fullmatch(r"R[1-9]\d*", raw):
                    rewards.setdefault(raw, raw)
                    used.add(raw)
            for child in value.values():
                register(child)
        elif isinstance(value, list):
            for child in value:
                register(child)

    register(observation)
    keywords: dict[str, dict[str, str]] = {}
    next_ref = 1
    instance_refs: dict[str, str] = {}

    def project(value: Any, in_history: bool = False) -> None:
        nonlocal next_ref
        if isinstance(value, dict):
            for tip in value.pop("keyword_tooltips", []) or []:
                if not in_history and isinstance(tip, dict) and tip.get("header") and tip.get("body"):
                    name, body = str(tip["header"]).strip(), str(tip["body"]).strip()
                    if name and body:
                        keywords.setdefault(name.casefold() + "\n" + body, {"name": name, "description": body})
            for key in list(value):
                if key in {"uuid", "card_uuid", "state_id", "transaction_id"} or key in {
                    "reward_ref", "linked_reward_ref", "source_reward_ref",
                }:
                    value.pop(key)
                    continue
                raw = value[key]
                if key in _REWARD_KEYS and raw is not None:
                    raw = str(raw)
                    if raw not in rewards:
                        while f"R{next_ref}" in used:
                            next_ref += 1
                        rewards[raw] = f"R{next_ref}"
                        used.add(rewards[raw])
                        next_ref += 1
                    value[key] = rewards[raw]
                    rewards.setdefault(value[key], value[key])
                elif isinstance(raw, str) and _UUID.search(raw):
                    # No opaque UUIDs in action labels/legacy macro IDs either.
                    # Human-readable text and stable card type IDs remain intact.
                    value[key] = _UUID.sub(lambda match: instance_refs.setdefault(
                        match.group(), f"I{len(instance_refs) + 1}"), raw)
                else:
                    project(raw, in_history or key == "room_history")
        elif isinstance(value, list):
            for child in value:
                project(child, in_history)

    # Establish fallback IDs from history first, then the current reward list.
    # Actual stable Recorder refs always take precedence over these legacy aliases.
    project(observation.get("room_history"), True)
    project(observation)
    existing = observation.get("keyword_explanations", [])
    for item in existing if isinstance(existing, list) else []:
        if isinstance(item, dict) and item.get("name") and item.get("description"):
            keywords.setdefault(str(item["name"]).casefold() + "\n" + str(item["description"]), item)
    if keywords:
        observation["keyword_explanations"] = sorted(keywords.values(), key=lambda x: x["name"])
    return observation


_OPTIONAL_CARD_FLAGS = {
    "exhaust", "ethereal", "innate", "is_innate", "retain", "self_retain", "purge_on_use",
    "free_to_play_once", "exhaust_on_use_once", "is_cost_modified", "is_cost_modified_for_turn",
    "in_bottle_flame", "in_bottle_lightning", "in_bottle_tornado",
}
_CARD_EFFECT_VALUES = {
    "damage", "base_damage", "final_single_hit_damage", "final_damage",
    "block", "base_block", "final_block", "magic_number", "base_magic_number", "hit_count",
}
_CARD_CONTAINERS = {
    "deck", "deck_summary", "cards", "hand", "selected", "selected_cards",
    "draw_pile", "discard_pile", "exhaust_pile", "card", "confirmation_card", "upgrade_preview",
}


def text_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Compact a display copy; keep original model/audit facts untouched.

    Complete live rules text represents card effects; incomplete descriptions
    keep their numeric fallbacks. Unknown flags and control facts remain.
    """
    view = copy.deepcopy(observation)
    project_reward_text(view)
    combat = view.get("combat")
    live_hand = combat.get("hand") if isinstance(combat, dict) else None
    live_hand_cards = {id(card) for card in live_hand if isinstance(card, dict)} if isinstance(live_hand, list) else set()
    monsters = combat.get("monsters") if isinstance(combat, dict) else None

    def target_damage(values: list[Any]) -> list[dict[str, Any]] | None:
        if not isinstance(monsters, list) or len(monsters) != len(values):
            return None
        indexed: dict[int, dict[str, Any]] = {}
        for monster in monsters:
            if not isinstance(monster, dict):
                return None
            index = monster.get("monster_index")
            if type(index) is not int or index < 0 or index in indexed:
                return None
            indexed[index] = monster
        if set(indexed) != set(range(len(values))):
            return None
        return [{"monster_index": index, "name": indexed[index].get("name"), "damage": damage}
                for index, damage in enumerate(values)]

    def comparable_text(text: str) -> str:
        text = re.sub(r"\bNL\b", "\n", text)
        text = re.sub(r"#[rgbypow]|\[#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\]|\[\]|[~@]", "", text)
        for icon, label in (("[E]", "能量"), ("[R]", "红色能量"), ("[G]", "绿色能量"),
                            ("[B]", "蓝色能量"), ("[W]", "紫色能量")):
            text = text.replace(icon, label)
        return " ".join(text.split())

    def clean(value: Any, card_context: bool = False) -> None:
        if isinstance(value, list):
            for item in value:
                clean(item, card_context)
            return
        if not isinstance(value, dict):
            return
        description = value.get("description")
        raw_description = value.get("raw_description")
        same_effect = (isinstance(raw_description, str) and isinstance(description, str)
                       and bool(comparable_text(raw_description))
                       and comparable_text(description).startswith(comparable_text(raw_description)))
        # A readable legacy/preview description can stand on its own when no
        # contrary template is supplied. If raw text has an unrepresented
        # effect, preserve it and the numeric fallback rather than guessing.
        raw_covered = raw_description is None or isinstance(raw_description, str) and (
            not raw_description.strip() or same_effect)
        complete = (isinstance(description, str) and bool(description.strip())
                    and (value.get("description_complete") is True
                         or ("description_complete" not in value and raw_covered))
                    and value.get("unresolved_description_variables", []) in ([], "")
                    and not re.search(r"![^!\s]+!", description))
        value.pop("description_source", None)
        if complete:
            value.pop("raw_description", None)
            if value.get("description_complete") is True:
                value.pop("description_complete")
        if isinstance(value.get("tooltips"), list):
            normalized = " ".join(description.split()) if isinstance(description, str) else ""
            tips = []
            for tip in value["tooltips"]:
                if isinstance(tip, dict):
                    body = tip.get("body", tip.get("description"))
                    if normalized and isinstance(body, str) and " ".join(body.split()) == normalized:
                        continue
                    tips.append({"name": tip.get("header", tip.get("name")), "description": body})
                else:
                    tips.append(tip)
            value["tooltips"] = tips
        is_card = card_context or any(key in value for key in ("cost", "cost_for_turn", "is_playable", "base_damage", "base_block"))
        if is_card:
            # Master-deck and selection copies aren't directly playable. Keep
            # this contextual flag only for actual cards in the current hand.
            if id(value) not in live_hand_cards:
                value.pop("is_playable", None)
            for flag in _OPTIONAL_CARD_FLAGS:
                if value.get(flag) is False:
                    value.pop(flag)
            if complete:
                for field in _CARD_EFFECT_VALUES:
                    value.pop(field, None)
                multiple = value.get("multi_damage")
                known_multiple = (isinstance(multiple, list) and bool(multiple)
                                  and all(type(number) in (int, float) and math.isfinite(number) and number >= 0
                                          for number in multiple))
                if multiple is None or multiple == []:
                    value.pop("multi_damage", None)
                elif known_multiple and len(set(multiple)) == 1:
                    value.pop("multi_damage")
                elif known_multiple and id(value) in live_hand_cards:
                    targets = target_damage(multiple)
                    if targets is not None:
                        value["target_damage"] = targets
                        value.pop("multi_damage")
            for original, duplicate in (("base_cost", "cost"), ("cost_for_turn", "cost")):
                # Do not equate a missing value, bool or unknown with zero.
                left, right = value.get(original), value.get(duplicate)
                if (type(left) in (int, float) and type(right) in (int, float) and left == right):
                    value.pop(original)
            if value.get("upgrade_preview_status") == "available" and isinstance(value.get("upgrade_preview"), dict):
                value.pop("upgrade_preview_status", None)
                value.pop("upgrade_preview_source", None)
        if (type(value.get("intent_damage")) in (int, float)
                and type(value.get("intent_base_damage")) in (int, float)
                and value.get("intent_base_damage") == value["intent_damage"]):
            value.pop("intent_base_damage", None)
        for field in ("draw_pile_missing_count", "discard_pile_missing_count", "exhaust_pile_missing_count"):
            if field in value and (value[field] is None or type(value[field]) is int and value[field] == 0):
                value.pop(field)
        for field, item in value.items():
            clean(item, field in _CARD_CONTAINERS)

    clean(view)
    combat = view.get("combat")
    player = combat.get("player") if isinstance(combat, dict) else None
    if isinstance(player, dict):
        for key in ("current_hp", "max_hp"):
            if type(view.get(key)) in (int, float) and player.get(key) == view[key]:
                player.pop(key)
    memory = view.get("run_memory")
    if isinstance(memory, dict):
        visited = memory.pop("visited_shop_rooms", None)
        # These coordinate lists are control bookkeeping, not a second history.
        memory.pop("declined_card_rewards", None)
        if str(view.get("screen_type") or "").upper() == "SHOP_ROOM" and isinstance(visited, list):
            if any(isinstance(room, dict) and room.get("act") == view.get("act")
                   and room.get("floor") == view.get("floor") for room in visited):
                memory["shop_previously_opened"] = True
    return view
