"""Local Qwen action ranker for Candidate Assist.

The server reads one JSON request per line from stdin and writes one JSON
response per line to stdout.  It never talks to CommunicationMod directly;
the controller remains responsible for legal-action validation and command
compilation.
"""

from __future__ import annotations

import argparse
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
from sts_agent.policy.gpu import select_configured_gpu
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
    "screen_state": "当前界面内容",
    "combat": "战斗状态",
    "map": "地图全貌",
    "recent_actions": "最近动作",
    "event_history": "事件历史",
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
    "upgrades": "升级次数",
    "count": "数量",
    "price": "价格",
    "target": "目标",
    "playable": "可打出",
    "is_playable": "可打出",
    "can_use": "可使用",
    "final_single_hit_damage": "最终单段伤害",
    "final_damage": "最终伤害",
    "final_block": "最终格挡",
    "damage": "伤害",
    "block": "格挡",
    "intent": "意图",
    "powers": "状态效果",
    "nodes": "节点",
    "current_node": "当前位置",
    "next_nodes": "可到达节点",
    "boss_available": "首领节点",
    "hand": "手牌",
    "draw_pile": "抽牌堆数量",
    "discard_pile": "弃牌堆数量",
    "exhaust_pile": "消耗牌堆数量",
    "energy": "能量",
    "player": "我方",
    "monsters": "敌人",
    "powers": "状态效果",
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
    "MAP": "这是地图界面。根据完整公开地图、当前可达节点、生命、牌组、金币、药水和本幕首领比较后续路线；兼顾补强、营火恢复或升级、商店和精英风险，只选择当前合法的可达节点，不臆测问号房内容。",
    "COMBAT": "这是战斗界面。先看我方生命、能量、手牌和敌人意图，优先计算本回合能造成的伤害和格挡；有可用能量和有效动作时不要无故结束回合，必要时合理使用药水，目标是减少掉血而不是机械保留资源。",
    "CARD_REWARD": "这是卡牌奖励界面。先读取并比较全部可见卡牌，再决定获得或跳过；前期通常需要能解决当前战斗的过渡牌，不要因为稀有度或单张牌面看起来强就忽略牌组协同。",
    "COMBAT_CARD_SELECTION": "这是战斗内生成选牌，不是普通战后永久卡牌奖励。读取触发来源、描述和最近动作，确认卡牌进入手牌或其他位置、临时费用和持续时间；结合当前能量、敌人意图和手牌比较即时收益，不要因担心永久牌组变大而跳过。不知道的来源效果保持未知。",
    "RELIC_PICKUP_CARD_SELECTION": "这是遗物触发的选牌。读取 selection_context 的来源、选择目的、数量和已选项目，按本次遗物效果比较候选；不要套用普通卡牌奖励或其他选牌流程。",
    "COMBAT_REWARD": "这是战斗奖励界面。先确认可见金币、药水和遗物等奖励是否仍未领取，再处理卡牌奖励；没有明确理由不要跳过尚未领取的资源。",
    "EVENT": "这是事件界面。先读事件说明、选项和事件历史，确认当前选项是否已经改变事件阶段；比较生命、金币、牌组和后续路线的长期代价，避免重复上一阶段的选项。",
    "SHOP_ROOM": "这是商店房间，还未打开商品界面。区分进入商店房间、打开商人和离开房间；按合法动作查看商人，不猜测未展示商品。若本局记忆表明已经访问并离开该商人，不要重复进入。",
    "SHOP_SCREEN": "这是商人界面。比较全部可见商品的效果、价格、当前金币和牌组缺口，以及移除牌的价值；考虑后续资源需求，不按名称或稀有度盲目购买。离开商人界面和离开商店房间是不同动作。",
    "REST": "这是营火界面。根据当前生命、牌组缺口和下一场精英或首领的风险，在休息、升级及其他可见选项之间权衡。",
    "BOSS_REWARD": "这是Boss奖励界面。比较所有Boss遗物和钥匙等可见选项对当前牌组、资源和后续路线的影响，不要把未查看的选项当成不存在。",
    "GRID_SELECT": "这是选牌界面。读取选择目的、触发来源、数量和已选项目，按本次效果选择；多选时不要重复选择已选项目。若合法动作允许取消、少选或跳过，可按效果权衡，不把它当作普通卡牌奖励或翻牌配对。",
    "GRID_PURGE": "这是移除卡牌的选择。删除会永久改变牌组，按当前牌组缺口、重复数量和协同比较候选，不要误删关键伤害、防御或成长牌；以本次选择要求为准。",
    "GRID_UPGRADE": "这是升级卡牌的选择。比较本次升级对费用、效果和牌组需求的改善；是否永久升级或仅本场战斗生效，以触发来源和选择要求为准。",
    "GRID_TRANSFORM": "这是变化卡牌的选择。按牌组缺口和协同选择值得替换的牌；结果未公开时不要假定会得到理想卡牌。遵守本次变化数量和要求。",
    "GRID_CONFIRM": "这是选牌后的确认界面。查看待确认卡牌、已选项目和本次升级、移除或变化目的，决定确认或按合法动作返回重选；不是重新选择另一张牌的阶段。",
    "HAND_SELECT": "这是从手牌选择的界面。先读清楚触发该选择的牌、遗物或事件及其要求，只选择能完成当前效果的牌，不要把它当作普通卡牌奖励。",
    "EVENT_MATCH": "这是事件内的翻牌配对。根据公开棋盘的位置、曾经揭示的牌面、已配对项目、当前翻开位置和剩余尝试选择；未揭示牌面未知，不得猜作已知。有已知配对时优先比较配对价值，否则探索未揭示的合法位置，不重复已证伪的组合。",
    "CHEST": "这是宝箱界面。区分打开宝箱与领取奖励，按可见内容和合法动作继续；遗物与蓝钥匙可能互斥，需结合钥匙持有情况和本局目标权衡，不能假定两者都能领取。",
    "TRANSITION": "这是房间切换或未完成流程的过渡状态。不要把上一场景的状态当成当前可执行决策；只使用当前合法动作，未提供合法动作时等待控制器更新。",
    "GAME_OVER": "本局已结束。只处理当前合法的终局确认或返回动作，不继续提交战斗或地图动作。",
    "VICTORY": "本局已胜利结束。只处理当前合法的终局确认或返回动作，不自行开始新局。",
    "GENERIC": "根据当前界面内容、最近动作、明确的选择要求和合法动作处理本步；未知效果保持未知，不猜测隐藏内容或重复已经完成的动作。",
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


def _is_empty(value: Any) -> bool:
    return value == "" or value == [] or value == {}


def _entity_line(value: dict[str, Any], indent: str) -> list[str]:
    """Render a card/relic/potion/monster as a compact natural-language line."""
    scalar_parts: list[str] = []
    nested: list[tuple[str, Any]] = []
    preferred = (
        "name", "id", "type", "color", "rarity", "cost", "upgrades", "count", "price",
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
            scalar_parts.append(f"{_label(key)}={_scalar_text(item)}")
    lines = [indent + "- " + "；".join(scalar_parts)] if scalar_parts else [indent + "-"]
    for key, item in nested:
        lines.extend(_natural_lines(item, indent + "  ", _label(key)))
    return lines


def _natural_lines(value: Any, indent: str = "", title: str | None = None) -> list[str]:
    """Render nested public state without exposing JSON punctuation everywhere."""
    lines: list[str] = []
    if isinstance(value, dict):
        if title:
            lines.append(f"{indent}{title}：")
            indent += "  "
        scalar_parts: list[str] = []
        complex_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if _is_empty(item):
                continue
            if isinstance(item, (dict, list)):
                complex_items.append((key, item))
            else:
                scalar_parts.append(f"{_label(key)}={_scalar_text(item)}")
        if scalar_parts:
            lines.append(indent + "；".join(scalar_parts))
        for key, item in complex_items:
            lines.extend(_natural_lines(item, indent, _label(key)))
        return lines
    if isinstance(value, list):
        if title:
            lines.append(f"{indent}{title}：")
            indent += "  "
        for item in value:
            if _is_empty(item):
                continue
            if isinstance(item, dict):
                lines.extend(_entity_line(item, indent))
            elif isinstance(item, list):
                lines.extend(_natural_lines(item, indent + "  "))
            else:
                lines.append(indent + "- " + _scalar_text(item))
        return lines
    if title and not _is_empty(value):
        lines.append(f"{indent}{title}：{_scalar_text(value)}")
    return lines


def _prompt_screen(observation: dict[str, Any]) -> str:
    return screen_key(observation)


def render_observation(observation: dict[str, Any]) -> str:
    """Render routed state as concise Chinese prose instead of minified JSON.

    ``route_policy_observation`` is the security boundary for map visibility;
    this renderer also skips ``map`` on non-MAP states so callers cannot
    accidentally reintroduce the full route graph into recurring decisions.
    """
    screen_type = _prompt_screen(observation)
    lines: list[str] = [f"【当前场景】{_SCREEN_LABELS.get(screen_type, screen_type)}"]
    global_keys = (
        "room_phase", "action_phase", "room_type", "floor", "act", "class",
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

    consumed = set(global_keys) | {"screen_type", "screen_name", "screen_up"}
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
    consumed.add("keys")
    for key, title in (("deck", "【当前牌组】"), ("relics", "【遗物】"), ("potions", "【药水】")):
        if observation.get(key):
            lines.extend(_natural_lines(observation[key], "", title))
        consumed.add(key)

    if screen_type == "MAP" and observation.get("map"):
        lines.extend(_natural_lines(observation["map"], "", "【地图全貌】"))
    consumed.add("map")

    for key, title in (("combat", "【战斗状态】"), ("screen_state", "【当前界面内容】"),
                       ("decision_constraints", "【必须遵守的限制】"), ("run_memory", "【本局记忆】"),
                       ("event_history", "【事件历史】"), ("recent_actions", "【最近动作】")):
        if observation.get(key):
            lines.extend(_natural_lines(observation[key], "", title))
        consumed.add(key)

    # Keep newly-added public fields visible without having to update this
    # renderer for every CommunicationMod release.
    for key, value in observation.items():
        if key not in consumed and not _is_empty(value):
            lines.extend(_natural_lines(value, "", f"【{_label(key)}】"))

    return "\n".join(lines)


def action_prompt_lines(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[str]:
    # Never slice characters: token budgets must reject/report a full sample.
    lines = ["【可见游戏状态】", render_observation(observation),
             "【合法动作】"]
    for action in legal_actions:
        lines.append(f"- {action_id(action)}：{action_label(action)}")
    return lines


def _policy_prompt_prefix() -> list[str]:
    """Shared natural-language instructions for ranking, scoring and SFT."""
    return [
        "你正在游玩《杀戮尖塔》，负责完成整局游戏。当前要选择下一步动作。",
        "优先保证存活和减少不必要的掉血，再权衡牌组成长、资源、路线和本幕首领。必须遵守状态中的限制、选牌要求和最近动作。",
        "默认规则：回合开始通常获得3能量并抽5张牌；回合结束会失去能量并弃手牌；普通战斗结束不回血，击败Boss后通常大量回血。当前状态中的真实数值和效果优先于默认规则。",
    ]


def build_prompt(observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> str:
    scene = scene_key(observation)
    lines = _policy_prompt_prefix() + [
        "只能使用合法动作中的 ID。下面的 ID 是程序令牌，必须原样输出，不得输出游戏命令。",
        f"请按优先级返回最多 {top_k} 个动作 ID，第一项是首选。只能输出一个 JSON 对象：{{\"ranked_action_ids\":[\"A1\",\"A2\"]}}。不要输出解释、Markdown 或列表外的 ID。",
    ]
    lines.append(f"当前场景提示：{_SCENE_GUIDANCE[scene]}")
    lines.extend(action_prompt_lines(observation, legal_actions))
    lines.append("JSON:")
    return "\n".join(lines)


def build_scoring_prompt(
    observation: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    template: str = "canonical",
) -> str:
    """Build one single-action prompt from equivalent deterministic layouts."""
    scene_type = scene_key(observation)
    instructions = _policy_prompt_prefix() + [
        "只能从合法动作中选择一个。下面的 ID 是程序令牌，必须原样输出。不要输出 JSON、解释、Markdown、命令或其他文字。",
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
                        past_key_values=prefix_output.past_key_values,
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
