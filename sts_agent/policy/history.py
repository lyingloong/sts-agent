"""Public room-local decision history, shared by live play and recording.

Never store full observations here: they contain the history itself, the map,
and many repeated entities. Internal audit items can contain observed changes;
model history projects only action/selection context, not before/after changes.
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


def _identity(value: Any) -> dict[str, Any]:
    """Historical identity, not an entity's attributes or effect description."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ("id", "name", "upgrades", "reward_id", "reward_type")
              if value.get(key) is not None}
    for key in ("card", "relic", "potion"):
        if child := _identity(value.get(key)):
            result[key] = child
    return result


def compact_history_step(item: dict[str, Any]) -> dict[str, Any]:
    """Shared live/SFT/UI projection; raw audit history remains untouched."""
    result = {key: copy.deepcopy(item[key]) for key in (
        "step_id", "screen_before", "screen_after", "turn_before", "turn_after",
        "action_kind", "action_id", "action", "role", "execution_status",
    ) if key in item}
    raw = item.get("details") or {}
    details = {key: copy.deepcopy(raw[key]) for key in (
        "choice_text", "selection_mode", "source_reward_id", "board_position",
    ) if raw.get(key) is not None}
    for key in ("object", "confirmation_card", "known_card_at_position"):
        if entity := _identity(raw.get(key)):
            details[key] = entity
    if isinstance(raw.get("target"), dict):
        details["target"] = _identity(raw["target"])
        if raw["target"].get("index") is not None:
            details["target"]["index"] = raw["target"]["index"]
    if isinstance(raw.get("selected_cards"), list):
        details["selected_cards"] = [_identity(card) for card in raw["selected_cards"]]
    context = raw.get("selection_context")
    if isinstance(context, dict):
        lean = {key: context[key] for key in ("kind", "selection_type", "selection_mode")
                if context.get(key) is not None}
        if origin := _identity(context.get("source")):
            lean["source"] = origin
        if lean:
            details["selection_context"] = lean
    mode = details.get("selection_mode") or (details.get("selection_context") or {}).get("selection_mode")
    # HAND_SELECT/modded selections sometimes have only an explicit reason.
    # Preserve that purpose instead of guessing from a card's name/effect.
    if not mode or mode == "SELECT":
        reason = raw.get("selection_reason") or raw.get("selection_instruction")
        if reason:
            details["selection_reason"] = reason
    if details:
        result["details"] = details
    return result


def _name(entity: Any) -> str:
    if not isinstance(entity, dict):
        return ""
    for child in ("card", "relic", "potion"):
        if entity.get(child):
            return _name(entity[child])
    name = str(entity.get("name") or entity.get("id") or entity.get("reward_type") or "")
    upgrades = entity.get("upgrades")
    if isinstance(upgrades, int) and upgrades > 0 and not re.search(r"\+\d*$", name):
        name += "+" if upgrades == 1 else f"+{upgrades}"
    return name


def history_step_text(item: dict[str, Any]) -> str:
    """One semantic line, rendered identically for the model and Overlay."""
    step = compact_history_step(item)
    details = step.get("details") or {}
    kind, screen = step.get("action_kind"), step.get("screen_before")
    label = str(step.get("action") or step.get("action_id") or "未知动作").removeprefix("自动执行 ")
    obj = _name(details.get("object"))
    choice = str(details.get("choice_text") or label)
    context = details.get("selection_context") or {}
    mode = details.get("selection_mode") or context.get("selection_mode")
    purposes = {"UPGRADE": "升级", "PURGE": "删除", "TRANSFORM": "变化", "TRANSFORM_UPGRADE": "变化并升级",
                "COPY": "复制", "BOTTLE": "指定开场手牌", "RELIC_CARD_REWARD": "遗物卡牌奖励",
                "DISCARD": "弃牌", "EXHAUST": "消耗"}
    purpose = purposes.get(mode, mode if mode and mode != "SELECT" else details.get("selection_reason"))
    if purpose:
        purpose = purposes.get(str(purpose).upper(), purpose)
    notes = []
    if kind == "play" and obj:
        action = f"使用{obj}"
    elif kind in {"potion_use", "potion_discard"} and obj:
        action = f"{'使用' if kind == 'potion_use' else '丢弃'}药水：{obj}"
    elif kind == "choose" and details.get("board_position") is not None:
        action = f"翻开位置{details['board_position']}"
        if face := _name(details.get("known_card_at_position")):
            notes.append(f"当前位置已公开：{face}")
    elif kind == "choose":
        if purpose:
            action = f"选择{obj or choice}用于{purpose}"
        elif screen == "SHOP_SCREEN" and obj:
            action = f"购买{obj}"
        elif screen == "SHOP_SCREEN" and choice.casefold() in {"purge", "删牌", "移除卡牌"}:
            action = "打开删牌界面"
        elif screen == "REST":
            rest = {"smith": "锻造", "rest": "休息", "recall": "获取红钥匙", "lift": "举重", "dig": "挖掘"}
            action = f"选择营火动作：{rest.get(choice.casefold(), choice)}"
        elif screen == "BOSS_REWARD":
            action = f"选择Boss遗物：{obj or choice}"
        elif screen == "COMBAT_REWARD":
            reward_type = str((details.get("object") or {}).get("reward_type") or "").upper()
            action = "打开卡牌奖励" if reward_type == "CARD" else f"领取奖励：{obj or choice}"
        elif screen == "CARD_REWARD":
            action = f"选择卡牌：{obj or choice}"
        elif screen == "EVENT":
            action = f"选择事件选项：{choice}"
        else:
            action = f"选择：{obj or choice}"
    elif kind in {"confirm", "return", "cancel", "skip"}:
        action = {"confirm": "确认", "return": "返回", "cancel": "取消", "skip": "跳过"}[kind]
        selected = [_name(card) for card in details.get("selected_cards", [])]
        if not any(selected) and (card := _name(details.get("confirmation_card"))):
            selected = [card]
        if any(selected):
            notes.append("已选牌：" + "、".join(name for name in selected if name))
        if purpose:
            notes.append(f"用途：{purpose}")
    else:
        action = "结束回合" if kind == "end_turn" else label
    target = details.get("target") or {}
    if target:
        index, name = target.get("index"), _name(target)
        action += f" → 敌人{index if index is not None else '?'}" + (f"（{name}）" if name else "")
    if origin := _name(context.get("source")):
        notes.append(f"由{origin}触发")
    reward = details.get("source_reward_id") or (details.get("object") or {}).get("reward_id")
    if reward is not None:
        notes.append(f"奖励ID={reward}")
    if step.get("role"):
        notes.append(step["role"])
    before, after = step.get("screen_before"), step.get("screen_after")
    if before is not None and after is not None and before != after:
        screens = {"MAP": "地图", "EVENT": "事件", "GRID": "网格选牌", "HAND_SELECT": "手牌选择",
                   "CARD_REWARD": "卡牌选择", "COMBAT_REWARD": "战斗奖励", "REST": "营火",
                   "SHOP_SCREEN": "商人", "SHOP_ROOM": "商店房间", "NONE": "战斗/房间"}
        if step.get("turn_before") is not None or step.get("turn_after") is not None:
            screens["NONE"] = "战斗"
        notes.append(f"{screens.get(before, before)}→{screens.get(after, after)}")
    status = step.get("execution_status")
    if status in {"cancelled", "skipped", "rejected"}:
        action = "尝试（未执行）：" + action
    elif status and status not in {"executed", "not_tracked"}:
        notes.append(f"执行状态={status}")
    heading = f"步骤{step.get('step_id', '?')}"
    if step.get("turn_before") is not None:
        heading += f"｜回合{step['turn_before']}"
    return heading + "：" + action + ("（" + "；".join(map(str, notes)) + "）" if notes else "")


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
    if not details.get("selection_mode"):
        for flag, mode in (("for_purge", "PURGE"), ("for_transform", "TRANSFORM"), ("for_upgrade", "UPGRADE")):
            if content.get(flag):
                details["selection_mode"] = mode
                break
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
        entry = compact_history_step(item)
        if isinstance(item.get("room_before"), dict) and _room_key(item["room_before"]) != key:
            entry["role"] = "进入当前房间"
        position = (entry.get("details") or {}).get("board_position")
        if position in known_cards:
            # Recover a slow flip's known face from durable public board memory
            # without pretending that its original post-snapshot was settled.
            entry.setdefault("details", {})["known_card_at_position"] = _identity(known_cards[position])
        steps.append(entry)
    result = {"room": room, "total_steps": len(entries), "shown_steps": len(steps),
              "omitted_steps": len(entries) - len(steps), "limit": limit, "steps": steps}
    if opening:
        result["event_opening"] = opening
    return result
