"""Local Qwen action ranker for Candidate Assist.

The server reads one JSON request per line from stdin and writes one JSON
response per line to stdout.  It never talks to CommunicationMod directly;
the controller remains responsible for legal-action validation and command
compilation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from sts_agent.config import ConfigError, load_agent_config
from sts_agent.core.contracts import validate_action_list, validate_observation
from sts_agent.core.policy import PolicyBackend
from sts_agent.policy.formats import DEFAULT_PROMPT_FORMAT, require_prompt_format
from sts_agent.policy.gpu import select_configured_gpu
from sts_agent.policy.presentation import text_observation
from sts_agent.policy.scenes import scene_key, screen_key


def configure_jsonl_stdio() -> None:
    """Keep JSONL protocol Unicode-safe on Windows pipes."""
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")


def parse_gpu_device(value: str) -> str | int:
    if value in {"auto", "visible"}:
        return value
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gpu device must be 'visible', 'auto', or a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("gpu device must be 'visible', 'auto', or a non-negative integer")
    return result


def configure_inference_gpu(device: str | int, minimum_free_memory_mib: int) -> dict[str, int] | None:
    """Restrict model loading to one selected GPU before importing Torch."""

    if device == "visible":
        return None
    if minimum_free_memory_mib < 1:
        raise ValueError("minimum_free_memory_mib must be positive")
    selected = select_configured_gpu(device, minimum_free_memory_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    return selected


def action_id(action: dict[str, Any]) -> str:
    return str(action.get("id") or action.get("action_id") or "")


def action_label(action: dict[str, Any]) -> str:
    label = action.get("label")
    if label:
        return str(label)
    kind = action.get("kind")
    command = action.get("command")
    return str(kind or command or "")


_FIELD_LABELS = {
    "screen_type": "界面",
    "screen_name": "界面名称",
    "screen_up": "界面已打开",
    "room_phase": "房间阶段",
    "action_phase": "行动阶段",
    "room_type": "房间类型",
    "floor": "楼层",
    "act": "幕",
    "class": "角色",
    "ascension_level": "进阶",
    "current_hp": "当前生命",
    "max_hp": "最大生命",
    "gold": "金币",
    "act_boss": "本幕首领",
    "relics": "遗物",
    "potions": "药水",
    "deck": "当前牌组",
    "choices": "当前选择",
    "historical_choices": "当时可选项",
    "screen_state": "当前界面内容",
    "rest_options": "营火选项标识",
    "rest_option_details": "营火选项说明",
    "has_rested": "营火流程已完成",
    "choice_index": "选项索引",
    "card_index": "手牌位置",
    "monster_index": "敌人序号",
    "slot": "药水栏位",
    "choice": "选项标识",
    "usable": "可用",
    "activity_rule": "营火活动规则",
    "combat": "战斗状态",
    "map": "地图全貌",
    "recent_actions": "最近动作",
    "event_history": "事件历史",
    "room_history": "当前房间历史",
    "choice_text": "选择文本",
    "object": "操作对象",
    "index": "序号",
    "selection_context": "选牌触发来源",
    "selection_reason": "选牌说明",
    "source_reward_id": "所属奖励编号",
    "reward_id": "奖励编号",
    "linked_reward_id": "互斥奖励编号",
    "reward_index": "当前奖励索引",
    "reward_type": "奖励类型",
    "reward_relationships": "奖励互斥关系",
    "target_damage": "按目标区分的单段伤害",
    "multi_damage": "多目标伤害（对应关系未确认）",
    "observed_after": "随后观察值",
    "before": "动作前值",
    "entity": "实体",
    "count_delta": "数量变化",
    "revealed_cards": "本次公开的牌面",
    "board_position": "翻牌位置",
    "known_card_at_position": "该位置当前已公开的牌面",
    "run_memory": "本局记忆",
    "decision_constraints": "必须遵守的限制",
    "state_warning": "状态提示",
    "id": "ID",
    "name": "名称",
    "description": "描述",
    "type": "类型",
    "color": "颜色",
    "rarity": "稀有度",
    "cost": "费用",
    "cost_for_turn": "本回合费用",
    "base_cost": "原始费用",
    "base_damage": "基础伤害",
    "base_block": "基础格挡",
    "magic_number": "效果参数",
    "base_magic_number": "基础效果参数",
    "amount": "数值/层数",
    "raw_description": "原始说明",
    "description_complete": "说明完整",
    "unresolved_description_variables": "未解析的效果变量",
    "upgrade_preview": "升级后预览",
    "upgrade_preview_status": "升级预览状态",
    "upgrade_preview_source": "升级预览来源",
    "tooltips": "补充说明",
    "header": "名称",
    "body": "说明",
    "free_to_play_once": "本次免费打出",
    "exhaust_on_use_once": "本次打出后消耗",
    "is_innate": "固有",
    "innate": "固有",
    "exhaust": "消耗",
    "ethereal": "虚无",
    "retain": "保留",
    "self_retain": "自带保留",
    "purge_on_use": "打出后移出本场战斗",
    "is_cost_modified": "基础费用已修改",
    "is_cost_modified_for_turn": "本回合费用已修改",
    "in_bottle_flame": "瓶装火焰指定牌",
    "in_bottle_lightning": "瓶装闪电指定牌",
    "in_bottle_tornado": "瓶装旋风指定牌",
    "base_energy_per_turn": "基础每回合能量基数",
    "energy_per_turn": "当前每回合能量基数",
    "master_hand_size": "基础每回合抽牌基数",
    "game_hand_size": "当前每回合抽牌基数",
    "cards_played_this_turn": "本回合已出牌数",
    "shop_previously_opened": "本房间曾打开商人",
    "turn": "回合",
    "hit_count": "段数",
    "draw_count": "抽牌堆数量",
    "discard_count": "弃牌堆数量",
    "exhaust_count": "消耗牌堆数量",
    "upgrades": "升级次数",
    "count": "数量",
    "price": "价格",
    "target": "目标",
    "playable": "可打出",
    "is_playable": "可打出",
    "just_for_confirming": "仅展示待确认卡牌",
    "reference_choice_index": "原选择位置（仅供参考）",
    "can_use": "可使用",
    "final_single_hit_damage": "当前牌面单段伤害",
    "final_damage": "伤害预览",
    "final_block": "当前牌面格挡",
    "damage": "伤害",
    "block": "格挡",
    "intent": "意图",
    "intent_base_damage": "意图基础单段伤害",
    "intent_damage": "意图单段伤害",
    "intent_hits": "意图段数",
    "intent_total_damage": "意图总伤害",
    "is_gone": "已离场",
    "card": "卡牌",
    "cards": "卡牌",
    "text": "文本",
    "num_cards": "选牌数量",
    "for_purge": "用于移除",
    "for_upgrade": "用于升级",
    "for_transform": "用于变化",
    "phase": "阶段",
    "powers": "状态效果",
    "nodes": "节点",
    "current_node": "当前位置",
    "next_nodes": "可到达节点",
    "boss_available": "首领节点",
    "hand": "手牌",
    "draw_pile": "抽牌堆组成（不表示抽牌顺序）",
    "discard_pile": "弃牌堆组成",
    "exhaust_pile": "消耗牌堆组成",
    "draw_pile_missing_count": "抽牌堆中详情未知的卡牌数",
    "discard_pile_missing_count": "弃牌堆中详情未知的卡牌数",
    "exhaust_pile_missing_count": "消耗牌堆中详情未知的卡牌数",
    "energy": "能量",
    "stance": "姿态",
    "player": "我方",
    "monsters": "敌人",
    "selected": "已选",
    "selected_cards": "已选牌",
    "selection_mode": "选牌模式",
    "selection_instruction": "选牌要求",
    "required_cards": "需要选择数量",
    "max_cards": "最多选择数量",
    "options": "选项",
    "body_text": "事件说明",
    "event_name": "事件名称",
    "event_id": "事件 ID",
    "match_game": "翻牌配对状态",
    "board": "公开棋盘",
    "position": "位置",
    "row": "行",
    "column": "列",
    "face_up": "当前翻开",
    "revealed": "曾经揭示",
    "matched": "已配对",
    "remaining_attempts": "剩余尝试",
    "matched_pairs": "已配对数量",
    "selected_positions": "已翻位置",
    "awaiting_resolution": "等待结算",
    "game_done": "小游戏结束",
    "confirmation_card": "待确认卡牌",
}

_SCREEN_LABELS = {
    "MAP": "地图",
    "COMBAT": "战斗",
    "EVENT": "事件",
    "CARD_REWARD": "卡牌奖励",
    "COMBAT_REWARD": "战斗奖励",
    "SHOP": "商店",
    "SHOP_ROOM": "商店房间",
    "SHOP_SCREEN": "商人界面",
    "REST": "营火",
    "BOSS_REWARD": "Boss奖励",
    "GRID": "选牌",
    "HAND_SELECT": "从手牌选择",
    "TRANSITION": "房间切换",
    "NONE": "游戏界面",
    "CHEST": "宝箱",
    "GAME_OVER": "终局",
    "VICTORY": "胜利",
}

_SCENE_GUIDANCE = {
    "MAP": "这是地图界面。结合完整公开地图、当前可达节点、生命、牌组、资源和本幕首领比较路线；问号房的隐藏内容未知。",
    "COMBAT": "这是战斗界面。结合敌人意图与状态安排攻防，合理使用药水；仍有有效出牌或药水收益时，不要无故结束回合。",
    "CARD_REWARD": "这是永久卡牌奖励。比较全部候选与牌组缺口，再决定领取或跳过；前期需要能解决当前战斗的过渡牌。",
    "COMBAT_CARD_SELECTION": "这是战斗内生成选牌。按触发来源确认进入位置、临时费用和持续时间，比较对当前战斗的即时收益；不能因担心永久加牌而跳过临时手牌。",
    "RELIC_PICKUP_CARD_SELECTION": "这是遗物触发的选牌。按触发来源、用途、数量和已选项目判断本次选择。",
    "COMBAT_REWARD": "这是战斗奖励界面。通常应领取尚未领取的金币和可用资源；遗物、钥匙等互斥奖励需权衡，药水满栏时可先处理现有药水。",
    "EVENT": "这是事件界面。结合当前正文、选项和房间历史判断事件阶段，权衡生命、金币、牌组与长期收益。",
    "SHOP_ROOM": "这是商店房间。根据金币、牌组缺口和补强需求，决定是否查看商人。",
    "SHOP_SCREEN": "这是商人界面。比较商品效果、价格、牌组缺口、删牌与保留金币的收益。",
    "REST": "这是营火界面。根据生命、牌组和后续路线，比较休息、升级、钥匙等收益。",
    "BOSS_REWARD": "这是Boss奖励界面。比较各候选遗物的收益、限制与当前牌组和后续资源需求。",
    "GRID_SELECT": "这是网格选牌。按本次用途、来源、数量限制和已选项目选择卡牌。",
    "GRID_PURGE": "这是移除卡牌的选择。结合本次用途、牌组缺口和协同，比较删除候选的价值。",
    "GRID_UPGRADE": "这是升级卡牌的选择。结合升级后预览与牌组需求比较收益；升级是否永久生效，以触发来源为准。未提供预览时保持未知。",
    "GRID_TRANSFORM": "这是变化卡牌的选择。根据牌组缺口、协同和本次数量选择替换对象；尚未公开的变化结果未知。",
    "GRID_CONFIRM": "这是选牌后的确认界面。核对待确认对象、本次用途和效果。",
    "HAND_SELECT": "这是从手牌选择的界面。按触发来源、用途、数量和已选项目完成当前效果。",
    "EVENT_MATCH": "这是翻牌配对。利用已公开牌面和位置记忆寻找有价值的配对，或探索未知位置；重新盖回的已知牌仍可用于配对。",
    "CHEST": "这是宝箱界面。权衡遗物与蓝钥匙等互斥奖励。",
    "TRANSITION": "当前处于房间过渡。",
    "GAME_OVER": "本局已结束，处理当前终局确认或返回。",
    "VICTORY": "本局已胜利结束，处理当前终局确认或返回。",
    "GENERIC": "根据当前公开状态、房间历史和选择要求处理本步。",
}

PROMPT_TEMPLATES = ("canonical", "state_first", "decision_first")


def _label(key: Any) -> str:
    key_text = str(key)
    return _FIELD_LABELS.get(key_text, key_text.replace("_", " "))


def _scalar_text(value: Any) -> str:
    if value is None:
        return "未知"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _field_text(key: str, value: Any) -> str:
    if key in {"cost", "cost_for_turn"} and type(value) in (int, float):
        if value == -1:
            return "X"
        if value == -2:
            return "无常规费用"
    if key in {"base_damage", "base_block"} and value == -1:
        return "未初始化/不适用"
    if key == "upgrade_preview_status":
        return {
            "available": "可用", "unsupported_card": "未取得安全升级预览",
            "unavailable": "未取得升级预览", "not_upgradable": "不能继续升级",
        }.get(str(value), _scalar_text(value))
    return _scalar_text(value)


def _field_label(key: str, value: Any) -> str:
    return "需要敌人目标" if key == "target" and isinstance(value, bool) else _label(key)


def _is_empty(value: Any) -> bool:
    return value == "" or value == [] or value == {}


def _entity_line(value: dict[str, Any], indent: str) -> list[str]:
    """Render a card/relic/potion/monster as a compact natural-language line."""
    scalar_parts: list[str] = []
    nested: list[tuple[str, Any]] = []
    preferred = (
        "card_index", "monster_index", "slot", "choice_index", "name", "id", "type", "color", "rarity", "cost", "cost_for_turn", "upgrades", "count", "price",
        "target", "playable", "is_playable", "can_use", "description", "damage", "block",
        "final_single_hit_damage", "final_damage", "final_block", "intent",
    )
    keys = list(dict.fromkeys([key for key in preferred if key in value] + list(value)))
    for key in keys:
        item = value.get(key)
        if _is_empty(item):
            continue
        if isinstance(item, (dict, list)):
            nested.append((key, item))
        else:
            scalar_parts.append(f"{_field_label(key, item)}={_field_text(key, item)}")
    if not scalar_parts and not nested:
        return []
    lines = [indent + "- " + "；".join(scalar_parts)] if scalar_parts else []
    for key, item in nested:
        lines.extend(_natural_lines(item, indent + "  ", _label(key)))
    return lines


def _natural_lines(value: Any, indent: str = "", title: str | None = None) -> list[str]:
    """Render nested public state without exposing JSON punctuation everywhere."""
    lines: list[str] = []
    if isinstance(value, dict):
        child_indent = indent + "  " if title else indent
        scalar_parts: list[str] = []
        complex_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if _is_empty(item):
                continue
            if isinstance(item, (dict, list)):
                complex_items.append((key, item))
            else:
                scalar_parts.append(f"{_field_label(key, item)}={_field_text(key, item)}")
        if scalar_parts:
            lines.append(child_indent + "；".join(scalar_parts))
        for key, item in complex_items:
            lines.extend(_natural_lines(item, child_indent, _label(key)))
        return [f"{indent}{title}：", *lines] if title and lines else lines
    if isinstance(value, list):
        child_indent = indent + "  " if title else indent
        for item in value:
            if _is_empty(item):
                continue
            if isinstance(item, dict):
                lines.extend(_entity_line(item, child_indent))
            elif isinstance(item, list):
                lines.extend(_natural_lines(item, child_indent + "  "))
            else:
                lines.append(child_indent + "- " + _scalar_text(item))
        return [f"{indent}{title}：", *lines] if title and lines else lines
    if title and not _is_empty(value):
        lines.append(f"{indent}{title}：{_scalar_text(value)}")
    return lines


def _prompt_screen(observation: dict[str, Any]) -> str:
    return screen_key(observation)


def render_room_history(history: dict[str, Any], current_event: dict[str, Any] | None = None) -> list[str]:
    from sts_agent.policy.history import history_step_text
    lines: list[str] = []
    if history.get("omitted_steps"):
        lines.append(f"已省略更早的{history['omitted_steps']}步。")
    opening = history.get("event_opening")
    if isinstance(opening, dict):
        # Only explicit option wording can cover a historical choice. In
        # particular, a short label must not erase a choice with extra costs
        # or effects merely because it is a substring of that choice.
        covered_choices: set[str] = set()
        options = opening.get("options")
        for option in options if isinstance(options, list) else []:
            if not isinstance(option, dict):
                continue
            for field in ("label", "name", "text"):
                text = option.get(field)
                if not isinstance(text, str) or not text.strip():
                    continue
                text = text.strip()
                covered_choices.add(text)
                prefix = re.match(r"^\[\s*([^\[\]]+?)\s*\]", text)
                if prefix:
                    covered_choices.add(prefix.group(1).strip())
                    covered_choices.add(text[:prefix.end()])
        # Keep only facts not already displayed in the current event. The
        # opening remains intact in the structured room history and raw trace.
        opening = {key: value for key, value in opening.items()
                   if not current_event or key not in current_event or current_event[key] != value}
        choices = opening.pop("choices", None)
        if isinstance(choices, list):
            remaining = [choice for choice in choices
                         if not isinstance(choice, str) or choice.strip() not in covered_choices]
            if remaining:
                opening["historical_choices"] = remaining
        elif choices is not None:
            opening["historical_choices"] = choices
        lines.extend(_natural_lines(opening, "", "进入事件时的说明与选项"))
    for item in history.get("steps", []):
        lines.append(history_step_text(item))
    return ["【当前房间历史】", *lines] if lines else []


def render_observation(observation: dict[str, Any]) -> str:
    """Render routed state as concise Chinese prose instead of minified JSON.

    ``route_policy_observation`` is the security boundary for map visibility;
    this renderer also skips ``map`` on non-MAP states so callers cannot
    accidentally reintroduce the full route graph into recurring decisions.
    """
    observation = text_observation(observation)
    screen_type = _prompt_screen(observation)
    lines: list[str] = [f"【当前场景】{_SCREEN_LABELS.get(screen_type, screen_type)}"]
    global_keys = (
        "room_phase", "room_type", "floor", "act", "class",
        "ascension_level", "current_hp", "max_hp", "gold", "act_boss",
    )
    global_parts = [
        f"{_label(key)}={_SCREEN_LABELS.get(str(observation.get(key)), _scalar_text(observation.get(key)))}"
        if key == "screen_type" else f"{_label(key)}={_scalar_text(observation.get(key))}"
        for key in global_keys
        if key in observation and not _is_empty(observation.get(key))
    ]
    if global_parts:
        lines.append("【全局状态】" + "；".join(global_parts))

    # Scheduler readiness is not a decision fact. Suppress obsolete goal
    # metadata as well when rendering older observations.
    consumed = set(global_keys) | {"screen_type", "screen_name", "screen_up", "action_phase", "task_goal", "choices"}
    keys = observation.get("keys")
    if isinstance(keys, dict):
        key_labels = ("ruby", "emerald", "sapphire")
        labels = {"ruby": "红宝石钥匙", "emerald": "翡翠钥匙", "sapphire": "蓝宝石钥匙"}
        lines.append(
            "【钥匙】" + "；".join(
                f"{labels[key]}={('已持有' if keys.get(key) is True else '未持有' if keys.get(key) is False else '未知')}"
                for key in key_labels
            )
        )
        lines.append("集齐红、绿、蓝三把钥匙，才能在击败第三幕Boss后进入第四幕。")
    consumed.add("keys")
    if observation.get("keyword_explanations"):
        lines.extend(_natural_lines(observation["keyword_explanations"], "", "【词条解释】"))
    consumed.add("keyword_explanations")
    for key, title in (("deck", "【当前牌组】"), ("relics", "【遗物】")):
        if observation.get(key):
            lines.extend(_natural_lines(observation[key], "", title))
        consumed.add(key)
    potions = observation.get("potions")
    if isinstance(potions, list):
        # CommunicationMod includes empty Potion Slot entries. An actual []
        # means zero slots; a missing inventory is represented separately.
        slots = [item for item in potions if isinstance(item, dict) and (item.get("id") or item.get("name"))]
        empty = [item for item in slots if str(item.get("id") or "").casefold() == "potion slot"]
        held = [item for item in slots if str(item.get("id") or "").casefold() != "potion slot"]
        capacity = observation.get("potion_slot_count", len(potions))
        known_capacity = type(capacity) is int and capacity >= len(slots)
        lines.extend(["【药水】", f"栏位数={capacity if known_capacity else '未知'}；空槽数={len(empty)}"])
        if known_capacity and capacity > len(slots):
            lines.append(f"另有{capacity - len(slots)}个栏位的内容未知。")
        for index, item in enumerate(slots):
            if item in held:
                lines.extend(_entity_line({"slot": index, **item}, "  "))
    elif "potions" in observation:
        lines.append("【药水】未知")
    consumed.add("potions")
    consumed.add("potion_slot_count")

    if screen_type == "MAP" and observation.get("map"):
        lines.extend(_natural_lines(observation["map"], "", "【地图全貌】"))
    consumed.add("map")

    screen_title = {"EVENT": "【事件内容】", "SHOP_SCREEN": "【商人商品】", "SHOP_ROOM": "【商店房间】",
                    "REST": "【营火选项】", "GRID": "【当前选牌】", "HAND_SELECT": "【当前选牌】",
                    "CARD_REWARD": "【卡牌奖励】", "COMBAT_REWARD": "【战斗奖励】",
                    "BOSS_REWARD": "【Boss奖励】"}.get(screen_type, "【当前界面内容】")
    for key, title in (("combat", "【战斗状态】"), ("screen_state", screen_title),
                       ("decision_constraints", "【必须遵守的限制】"), ("run_memory", "【本局记忆】"),
                       ("event_history", "【事件历史】"), ("recent_actions", "【最近动作】")):
        if observation.get(key):
            lines.extend(_natural_lines(observation[key], "", title))
        consumed.add(key)

    consumed.add("room_history")

    # Keep newly-added public fields visible without having to update this
    # renderer for every CommunicationMod release.
    for key, value in observation.items():
        if key not in consumed and not _is_empty(value):
            lines.extend(_natural_lines(value, "", f"【{_label(key)}】"))

    if isinstance(observation.get("room_history"), dict):
        event = observation.get("screen_state") if screen_type == "EVENT" else None
        if isinstance(event, dict):
            event = {**event, "choices": observation.get("choices")}
        lines.extend(render_room_history(observation["room_history"], event))

    return "\n".join(lines)


def action_prompt_lines(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[str]:
    # Never slice characters: token budgets must reject/report a full sample.
    lines = ["【可见游戏状态】", render_observation(observation),
             "【合法动作】"]
    for action in legal_actions:
        label = action_label(action)
        references = []
        for key, title, pattern in (
            ("card_index", "手牌位置", r"手牌(?:位置)?\s*=?\s*{n}(?!\d)"),
            ("target", "敌人序号", r"敌人\s*{n}(?!\d)"),
            ("slot", "药水栏位", r"(?:栏位|槽位)\s*{n}(?!\d)"),
        ):
            value = action.get(key)
            if type(value) is int and not re.search(pattern.format(n=value), label):
                references.append(f"{title}={value}")
        index = action.get("choice_index", action.get("index"))
        if type(index) is int and not re.search(rf"(?:选择位置|选项索引)\s*=?\s*{index}(?!\d)", label):
            references.append(f"选择位置={index}")
        # Labels are human-readable context, never the identity used by the
        # controller. Obsolete macro labels may still contain an instance UUID.
        label = re.sub(r"(?:[a-z]+-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", "[实例]", label, flags=re.I)
        if references:
            label += "（" + "；".join(references) + "）"
        lines.append(f"- {action_id(action)}：{label}")
    return lines


def _policy_prompt_prefix(observation: dict[str, Any]) -> list[str]:
    """Shared natural-language instructions for ranking, scoring and SFT."""
    lines = [
        "你正在游玩《杀戮尖塔》。",
        "结合当前状态与房间历史，权衡生存、牌组成长和资源收益。",
        "默认每回合获得3能量、抽5张牌；普通战斗后不自动回血，击败幕末Boss后通常大量回血。",
    ]
    if observation.get("combat") or _prompt_screen(observation) == "COMBAT":
        lines.extend([
            "战斗默认：回合结束失去未保留的能量并弃掉未保留手牌；格挡在自己下回合开始时清除。X费通常消耗全部剩余能量。",
            "手牌中的伤害、格挡是当前牌面数值；已反映在牌面上的状态修正不要重复计算。实际扣血还需结合目标状态、格挡和攻击段数判断。",
        ])
    return lines


def build_prompt(observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> str:
    scene = scene_key(observation)
    lines = _policy_prompt_prefix(observation) + [
        f"按优先级返回最多{top_k}个当前合法动作ID，只输出JSON：{{\"ranked_action_ids\":[\"A1\",\"A2\"]}}。",
    ]
    lines.append(f"当前场景提示：{_SCENE_GUIDANCE[scene]}")
    lines.extend(action_prompt_lines(observation, legal_actions))
    lines.append("JSON:")
    return "\n".join(lines)


def build_scoring_prompt(
    observation: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    template: str = "canonical",
    *,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
) -> str:
    """Build one single-action prompt from equivalent deterministic layouts."""
    require_prompt_format(prompt_format)
    scene_type = scene_key(observation)
    instructions = _policy_prompt_prefix(observation) + [
        "只输出当前合法动作中的一个动作ID（如A2），不要输出JSON或解释。",
    ]
    scene = [f"当前场景提示：{_SCENE_GUIDANCE[scene_type]}"]
    state = action_prompt_lines(observation, legal_actions)
    # Keep the continuation on the next line so aliases such as A1 and A10
    # are scored with an explicit newline boundary rather than prefix matching.
    suffix = ["请输出唯一的动作 ID："]
    layouts = {
        "canonical": [*instructions, *scene, *state, *suffix],
        "state_first": [*state, *scene, *instructions, *suffix],
        "decision_first": [*instructions, *state, *scene, *suffix],
    }
    if template not in layouts:
        raise ValueError(f"unknown prompt template: {template}")
    lines = layouts[template]
    return "\n".join(lines) + "\n"


def ranked_ids_from_output(text: str, legal_actions: list[dict[str, Any]], top_k: int) -> list[str]:
    """Parse generated JSON and defensively keep only current legal IDs."""
    legal_ids = [action_id(action) for action in legal_actions]
    legal_set = set(legal_ids)
    generated: list[str] = []
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            items = value.get("ranked_action_ids", []) if isinstance(value, dict) else []
            if isinstance(items, list):
                generated = [str(item) for item in items]
        except json.JSONDecodeError:
            pass
    # A small fallback makes imperfect JSON harmless while still requiring an
    # exact legal ID; sort by occurrence to preserve model preference order.
    if not generated:
        positions = []
        for candidate in legal_ids:
            match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])", text)
            if match:
                positions.append((match.start(), candidate))
        generated = [candidate for _, candidate in sorted(positions)]
    result: list[str] = []
    for candidate in generated:
        canonical_id = candidate
        if canonical_id not in legal_set:
            # Small models often omit the target suffix for a single-target
            # card (for example, "play:0" instead of "play:0:0"). Accept
            # that shorthand only when it resolves to exactly one legal ID.
            matching_ids = [legal_id for legal_id in legal_ids if legal_id.startswith(f"{candidate}:")]
            if len(matching_ids) == 1:
                canonical_id = matching_ids[0]
        if canonical_id in legal_set and canonical_id not in result:
            result.append(canonical_id)
        if len(result) >= top_k:
            break
    return result


class MockRanker:
    """Deterministic ranker used to verify controller/terminal integration."""

    model_id = "mock-candidate-ranker"

    def prepare(self) -> None:
        return None

    def close(self) -> None:
        return None

    def rank(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        del observation
        # Keep END_TURN as a useful fallback, but prefer concrete actions in
        # the mock so Candidate Assist can be tested without model downloads.
        ordered = sorted(
            legal_actions,
            key=lambda action: (action_id(action) == "END_TURN", action_id(action)),
        )
        result = []
        for rank, action in enumerate(ordered[:top_k], start=1):
            result.append(
                {
                    "rank": rank,
                    "action_id": action_id(action),
                    "label": action_label(action),
                    "score": None,
                }
            )
        return result

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return deterministic but non-uniform values for protocol tests."""
        ordered = self.rank(observation, legal_actions, len(legal_actions))
        # These are unnormalized log weights. The controller applies the only
        # normalization that matters: over the current legal action set.
        return [
            {
                "action_id": item["action_id"],
                "label": item["label"],
                "logprob": -2.0 * rank,
            }
            for rank, item in enumerate(ordered)
        ]


class QwenRanker:
    def __init__(self, model_name: str, load_in_4bit: bool, local_files_only: bool, adapter: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.model_id = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_source = model_name
        if local_files_only and not Path(model_name).is_dir():
            # Passing a Hub model ID to recent Transformers can still trigger
            # metadata requests during tokenizer setup, despite
            # local_files_only=True. Resolve it to the cached snapshot first
            # so game-time inference is strictly offline.
            from huggingface_hub import snapshot_download

            model_source = snapshot_download(repo_id=model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device.type == "cuda":
            model_kwargs["dtype"] = torch.float16
            if load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["device_map"] = "auto"
                except (ImportError, Exception) as exc:
                    raise RuntimeError(
                        "4-bit loading is required for the local 4GB GPU; install a CUDA-compatible bitsandbytes build "
                        "or start with --no-load-in-4bit only on a machine with enough VRAM."
                    ) from exc
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=local_files_only,
            **model_kwargs,
        )
        if not model_kwargs.get("device_map"):
            self.model.to(self.device)
        if adapter:
            adapter_path = Path(adapter)
            if not adapter_path.is_dir():
                raise RuntimeError(f"LoRA adapter directory does not exist: {adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("install peft to load a LoRA adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), is_trainable=False)
        self.model.eval()
        # Qwen ships sampling defaults. They are irrelevant for deterministic
        # candidate ranking and otherwise cause a warning on every turn.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model_device = next(self.model.parameters()).device

    def prepare(self) -> None:
        # Model loading in __init__ is the warm-up boundary for the local
        # backend. Keeping this method makes it interchangeable with remote
        # backends that need an explicit readiness handshake.
        return None

    def close(self) -> None:
        return None

    def rank(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not legal_actions:
            return []
        user_prompt = build_prompt(observation, legal_actions, top_k)
        tokenizer = self.tokenizer
        try:
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older Transformers versions do not expose Qwen3's
            # enable_thinking option, but still provide the chat template.
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model_device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                do_sample=False,
                # The response is a compact JSON list of A1/A2-style IDs.
                # A short cap prevents a malformed generation from holding up
                # the terminal on the 4GB local GPU.
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_length = encoded["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True).strip()
        ranked_ids = ranked_ids_from_output(generated, legal_actions, top_k)
        actions_by_id = {action_id(action): action for action in legal_actions}
        result = []
        for rank, candidate_id in enumerate(ranked_ids, start=1):
            action = actions_by_id[candidate_id]
            result.append(
                {
                    "rank": rank,
                    "action_id": action_id(action),
                    "label": action_label(action),
                    "score": None,
                }
            )
        return result

    def _chat_prefix(self, user_prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score every exact legal alias with one shared prompt forward pass.

        Scores are the log probability of ``A<n>`` followed by a newline. The
        newline is important: it prevents A1 from being treated as a prefix of
        A10. The controller later normalizes these scores only over legal
        actions, which is the desired masked policy distribution.
        """
        if not legal_actions:
            return []
        tokenizer = self.tokenizer
        prefix = self._chat_prefix(build_scoring_prompt(observation, legal_actions))
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prefix_ids = prefix_ids.to(self.model_device)
        prefix_length = prefix_ids.shape[1]

        continuation_ids: dict[str, list[int]] = {}
        for action in legal_actions:
            alias = action_id(action)
            combined_ids = tokenizer(prefix + alias + "\n", add_special_tokens=False)["input_ids"]
            prefix_as_list = prefix_ids[0].tolist()
            if combined_ids[:prefix_length] != prefix_as_list:
                raise RuntimeError("tokenizer changed the scoring prompt boundary")
            suffix = combined_ids[prefix_length:]
            if not suffix:
                raise RuntimeError(f"empty token continuation for action {alias}")
            continuation_ids[alias] = suffix

        with self.torch.inference_mode():
            prefix_output = self.model(input_ids=prefix_ids, use_cache=True)
            first_logprobs = self.torch.log_softmax(prefix_output.logits[0, -1].float(), dim=-1)
            result: list[dict[str, Any]] = []
            for action in legal_actions:
                alias = action_id(action)
                suffix = continuation_ids[alias]
                logprob = float(first_logprobs[suffix[0]].item())
                if len(suffix) > 1:
                    continuation = self.torch.tensor([suffix[:-1]], device=self.model_device)
                    continuation_output = self.model(
                        input_ids=continuation,
                        # DynamicCache.update mutates the supplied cache even
                        # with use_cache=False. Each alternative must start
                        # from the same untouched prompt cache.
                        past_key_values=copy.deepcopy(prefix_output.past_key_values),
                        use_cache=False,
                    )
                    next_logprobs = self.torch.log_softmax(continuation_output.logits[0].float(), dim=-1)
                    for position, token_id in enumerate(suffix[1:]):
                        logprob += float(next_logprobs[position, token_id].item())
                result.append(
                    {
                        "action_id": alias,
                        "label": action_label(action),
                        "logprob": logprob,
                    }
                )
        return result


def serve(args: argparse.Namespace) -> int:
    configure_jsonl_stdio()
    # Hugging Face libraries read this process-local setting. The user-facing
    # source of truth remains config/agent.json or an explicit CLI override.
    os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        selected_gpu = configure_inference_gpu(args.gpu_device, args.minimum_free_memory_mib)
    except Exception as exc:
        print(f"policy server GPU selection failed: {exc}", file=sys.stderr, flush=True)
        return 2
    ranker: PolicyBackend
    if args.mock:
        ranker = MockRanker()
        model_id = "mock-candidate-ranker"
    else:
        try:
            ranker = QwenRanker(args.model, args.load_in_4bit, args.local_files_only, args.adapter or None)
            model_id = args.model if not args.adapter else f"{args.model}+{Path(args.adapter).name}"
        except Exception as exc:
            print(f"policy server failed to load {args.model}: {exc}", file=sys.stderr, flush=True)
            if args.debug:
                traceback.print_exc(file=sys.stderr)
            return 2
    ranker.prepare()

    print(
        json.dumps(
            {"type": "ready", "model_id": model_id, "endpoint": args.endpoint, "gpu": selected_gpu},
            ensure_ascii=False,
        ),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Any = None
        try:
            request = json.loads(line)
            if request.get("type") == "shutdown":
                break
            request_type = request.get("type")
            if request_type not in {"rank", "score"}:
                raise ValueError("expected request type rank or score")
            observation = validate_observation(request.get("observation"))
            actions = validate_action_list(request.get("legal_actions"))
            legal_actions = [item for item in actions if action_id(item)]
            top_k = max(1, min(int(request.get("top_k", args.top_k)), len(legal_actions) or 1))
            if request_type == "rank":
                candidates = ranker.rank(observation, legal_actions, top_k)
                response = {
                    "type": "ranking",
                    "request_id": request.get("request_id"),
                    "model_id": model_id,
                    "candidates": candidates,
                }
            else:
                response = {
                    "type": "scoring",
                    "request_id": request.get("request_id"),
                    "model_id": model_id,
                    "action_scores": ranker.score(observation, legal_actions),
                }
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"type": "error", "request_id": request.get("request_id") if isinstance(request, dict) else None, "error": str(exc)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.debug:
                traceback.print_exc(file=sys.stderr)
    ranker.close()
    return 0


def parse_args() -> argparse.Namespace:
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config", default="config/agent.json")
    preliminary, _ = preliminary_parser.parse_known_args()
    try:
        policy = load_agent_config(config_file=preliminary.config)["policy"]
    except ConfigError as exc:
        raise SystemExit(f"invalid agent configuration: {exc}") from exc
    parser = argparse.ArgumentParser(description="Qwen ranker for StS Candidate Assist")
    parser.add_argument("--config", default="config/agent.json", help="项目根目录下的配置文件路径")
    parser.add_argument("--model", default=policy["model"]["base"])
    parser.add_argument(
        "--adapter",
        default=policy["model"]["initial_adapter"],
        help="local LoRA adapter directory to attach to the base model",
    )
    parser.add_argument("--endpoint", default=policy["model"]["hub_endpoint"])
    parser.add_argument("--top-k", type=int, default=policy["request"]["top_k"])
    parser.add_argument(
        "--gpu-device",
        type=parse_gpu_device,
        default="visible",
        help="auto or physical GPU index; visible leaves CUDA_VISIBLE_DEVICES unchanged",
    )
    parser.add_argument("--minimum-free-memory-mib", type=int, default=1)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认只从本地 Hugging Face 缓存加载；请先运行 download_qwen.py",
    )
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=policy["development"]["mock"],
        help="不加载模型，用于测试协议和终端交互",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=policy["development"]["debug"],
        help="将模型服务异常完整写入 stderr",
    )
    return parser.parse_args()


def main() -> int:
    return serve(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
