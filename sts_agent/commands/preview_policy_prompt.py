"""Print developer-readable prompts from deterministic public-state examples.

Run ``python -m sts_agent.commands.preview_policy_prompt``. This does not
connect to the game/model; every example uses the real normalization, room
history, routing and prompt builder. Examples are illustrative, not game logs.
"""

from __future__ import annotations

import copy
import sys

from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.runtime.controller import accepted_action_history_item, normalize_state, policy_observation, semantic_actions


def _card(identifier, name, description, card_type="ATTACK", **extra):
    return {"id": identifier, "name": name, "type": card_type, "cost": 1,
            "upgrades": 0, "description": description, **extra}


def _message(screen, floor, room, **extra):
    return {"in_game": True, "game_state": {
        "screen_type": screen, "room_type": room, "room_phase": "COMPLETE",
        "action_phase": "WAITING_ON_USER", "act": 1, "floor": floor,
        "class": "IRONCLAD", "ascension_level": 0, "current_hp": 60, "max_hp": 80,
        "gold": 99, "act_boss": "The Guardian",
        "keys": {"ruby": False, "emerald": False, "sapphire": False},
        "deck": [_card("Strike_R", "打击", "造成6点伤害") for _ in range(5)]
                + [_card("Defend_R", "防御", "获得5点格挡", "SKILL") for _ in range(4)]
                + [_card("Bash", "痛击", "造成8点伤害，施加2层易伤", cost=2)],
        "relics": [{"id": "Burning Blood", "name": "燃烧之血", "description": "每场战斗结束时，回复6点生命。"}],
        "potions": [{"id": "Potion Slot", "name": "药水栏位"} for _ in range(3)],
        "screen_state": {}, "choice_list": [], **extra,
    }}


def _history(before, after, step, action, source="human_game"):
    return accepted_action_history_item({
        "step_id": step, "screen_type": before["game_state"]["screen_type"],
        "pre_state": normalize_state(before), "action": action,
        "decision": {"source": source},
    }, after)


def example_prompts() -> dict[str, str]:
    combat = _message("NONE", 5, "MonsterRoom", room_phase="COMBAT", combat_state={
        "turn": 2, "player": {"current_hp": 60, "max_hp": 80, "energy": 3, "block": 0, "powers": []},
        "hand": [_card("Defend_R", "防御", "获得5点格挡", "SKILL", block=5, is_playable=True),
                 _card("Strike_R", "打击", "造成6点伤害", damage=6, is_playable=True, has_target=True),
                 _card("Strike_R", "打击", "造成6点伤害", damage=6, is_playable=True, has_target=True,
                       cost_for_turn=0, is_cost_modified_for_turn=True)],
        "monsters": [{"id": "JawWorm", "name": "大颚虫", "current_hp": 20, "max_hp": 42,
                      "block": 0, "is_gone": False, "intent": "ATTACK", "move_adjusted_damage": 7,
                      "move_hits": 1, "powers": []}],
        "draw_pile": [_card("Defend_R", "防御", "获得5点格挡", "SKILL") for _ in range(3)]
                     + [_card("Strike_R", "打击", "造成6点伤害") for _ in range(3)]
                     + [_card("Bash", "痛击", "造成8点伤害，施加2层易伤", cost=2)],
        "discard_pile": [], "exhaust_pile": [],
    })
    after_defend = copy.deepcopy(combat)
    for card in after_defend["game_state"]["deck"]:
        if card["id"] == "Defend_R":
            card["keyword_tooltips"] = [{"header": "格挡", "body": "在下个回合前，阻挡受到的伤害。"}]
    after_defend["game_state"]["combat_state"]["hand"].pop(0)
    after_defend["game_state"]["combat_state"]["player"].update(energy=2, block=5)
    after_defend["game_state"]["combat_state"]["discard_pile"] = [combat["game_state"]["combat_state"]["hand"][0]]
    history = [_history(combat, after_defend, 21,
                        {"kind": "play", "id": "PLAY:card=1", "label": "出牌 防御"})]
    prompts = {"战斗": build_scoring_prompt(policy_observation(after_defend, action_history=history),
                 [{"id": "A1", "label": "出牌 打击 -> 敌人 0", "card_index": 1, "target": 0},
                  {"id": "A2", "label": "出牌 打击 -> 敌人 0", "card_index": 2, "target": 0},
                  {"id": "A3", "label": "结束回合"}])}

    event = _message("EVENT", 8, "EventRoom", choice_list=[f"card{i}" for i in range(12)], screen_state={
        "event_id": "Match and Keep!", "event_name": "对对碰！", "body_text": "翻开两张牌，相同就加入牌组。",
        "options": [], "match_game": {"phase": "PLAY", "remaining_attempts": 5, "matched_pairs": 0,
            "board": [{"position": i, "row": i // 4, "column": i % 4, "revealed": False,
                       "face_up": False, "matched": False, "card": None} for i in range(12)]},
    })
    flip_one = copy.deepcopy(event)
    flip_one["game_state"]["screen_state"]["match_game"]["board"][0].update(
        revealed=True, face_up=True, card=_card("Bash", "痛击", "造成8点伤害，施加2层易伤", cost=2))
    flip_two = copy.deepcopy(flip_one)
    flip_two["game_state"]["screen_state"]["match_game"]["board"][0]["face_up"] = False
    flip_two["game_state"]["screen_state"]["match_game"]["board"][3].update(
        revealed=True, face_up=False, card=_card("Defend_R", "防御", "获得5点格挡", "SKILL"))
    flip_two["game_state"]["screen_state"]["match_game"]["remaining_attempts"] = 4
    history = [_history(event, flip_one, 31, {"kind": "choose", "id": "CHOOSE:index=0", "label": "card0"}),
               _history(flip_one, flip_two, 32, {"kind": "choose", "id": "CHOOSE:index=3", "label": "card3"})]
    prompts["翻牌事件"] = build_scoring_prompt(policy_observation(flip_two, action_history=history),
        [{"id": f"A{i+1}", "label": f"card{i}"} for i in range(12)])

    event = _message("EVENT", 10, "EventRoom", choice_list=["净化"], screen_state={
        "event_id": "The Purifier", "event_name": "牧师", "body_text": "可以免费从牌组中移除一张牌。",
        "options": [{"text": "净化：移除一张牌。"}],
    })
    grid = _message("GRID", 10, "EventRoom", choice_list=["打击", "防御", "痛击"], screen_state={
        "cards": [event["game_state"]["deck"][i] for i in (0, 5, 9)], "for_purge": True, "num_cards": 1,
    })
    history = [_history(event, grid, 40, {"kind": "choose", "id": "CHOOSE:index=0", "label": "净化"})]
    prompts["事件中的删牌二级界面"] = build_scoring_prompt(policy_observation(grid, action_history=history),
        [{"id": "A1", "label": "打击"}, {"id": "A2", "label": "防御"},
         {"id": "A3", "label": "痛击"}, {"id": "A4", "label": "RETURN"}])

    campfire = _message("REST", 12, "RestRoom", current_hp=40, room_phase="INCOMPLETE",
        choice_list=["rest", "smith", "recall"], screen_state={
            "has_rested": False, "rest_options": ["rest", "smith", "recall"],
            "rest_option_details": [
                {"choice_index": 0, "choice": "rest", "name": "休息", "description": "回复 24 点生命。", "usable": True},
                {"choice_index": 1, "choice": "smith", "name": "锻造", "description": "升级一张牌。", "usable": True},
                {"choice_index": 2, "choice": "recall", "name": "回忆", "description": "获得红宝石钥匙。", "usable": True},
            ],
        })
    campfire["available_commands"] = ["choose", "state"]
    actions = semantic_actions(campfire)
    prompts["营火"] = build_scoring_prompt(policy_observation(campfire),
        [{"id": f"A{i}", "label": action["label"]} for i, action in enumerate(actions, 1)])

    upgrading = copy.deepcopy(campfire)
    upgrading["game_state"]["screen_type"] = "GRID"
    cards = copy.deepcopy(upgrading["game_state"]["deck"])
    for card in cards:
        description = {"Strike_R": "造成9点伤害", "Defend_R": "获得8点格挡",
                       "Bash": "造成10点伤害，施加3层易伤"}[card["id"]]
        card.update(upgrade_preview={**card, "name": card["name"] + "+", "upgrades": 1,
                                     "description": description},
                    upgrade_preview_status="available", upgrade_preview_source="game_copy")
    upgrading["game_state"].update(choice_list=[card["name"] for card in cards],
        screen_state={"cards": cards, "for_upgrade": True, "num_cards": 1, "selected_cards": []})
    history = [_history(campfire, upgrading, 50, {"kind": "choose", "id": "CHOOSE:index=1", "label": "锻造"})]
    actions = semantic_actions(upgrading)
    prompts["营火升级候选与升级后预览"] = build_scoring_prompt(policy_observation(upgrading, action_history=history),
        [{**action, "id": f"A{i}"} for i, action in enumerate(actions, 1)])

    linked_rewards = _message("COMBAT_REWARD", 9, "TreasureRoom", choice_list=["relic", "sapphire_key"],
        screen_state={"rewards": [
            {"reward_type": "RELIC", "reward_id": "preview-relic", "reward_index": 0,
             "linked_reward_id": "preview-key", "relic": {"id": "Anchor", "name": "锚",
             "description": "在每场战斗开始时，获得10点格挡。"}},
            {"reward_type": "SAPPHIRE_KEY", "reward_id": "preview-key", "reward_index": 1,
             "linked_reward_id": "preview-relic"},
        ]})
    linked_rewards["available_commands"] = ["choose", "proceed", "state"]
    prompts["宝箱奖励互斥关系"] = build_scoring_prompt(policy_observation(linked_rewards),
        [{**action, "id": f"A{i}"} for i, action in enumerate(semantic_actions(linked_rewards), 1)])
    return prompts


def preview_markdown() -> str:
    lines = ["# 模型 Prompt Preview（原格式：plain_text）", "",
        "本文件由 `python -m sts_agent.commands.preview_policy_prompt` 的输出生成。",
        "使用真实的状态规范化、房间历史和 `build_scoring_prompt` 拼接器；示例数据是人工构造的，不是采集轨迹。",
        "不要直接修改生成内容；通用指令和场景提示在 `sts_agent/policy/local_qwen.py` 中维护。", "",
        "这是默认启用的 `plain_text` 格式。并列的 [Markdown 方案](POLICY_PROMPT_PREVIEW_MARKDOWN.md) 标为待开发，暂不接入运行。", "",
        "完整字段、动作空间、历史保留和未知信息规则见 [模型输入与动作空间](MODEL_OBSERVATION_AND_ACTIONS.md)。", "",
        "默认 `canonical` 顺序：通用规则 → 场景提示 → 当前状态 → 当前房间历史 → 合法动作 → 单动作输出要求。",
        "`state_first`、`decision_first` 使用相同内容，仅改变区块顺序。完整地图只在 MAP 提供。", "",
        "下面包含战斗、翻牌、事件二级选牌、营火、升级预览和奖励互斥六个完整输入，均输出本次合法动作的单个别名（如 `A1`）。",
        "战斗示例包含两张同名、不同本回合费用的手牌，以展示实例位置与动作的对应。", ""]
    for title, prompt in example_prompts().items():
        lines.extend([f"## {title}", ""])
        if title == "营火":
            lines.extend([
                "示例说明文字是人工构造的按钮运行时文本，用于展示字段和动作映射；实局使用游戏当前语言的",
                "`label/description`。恢复数字是 UI 说明，不保证实际净回血。",
                "`A1/A2/A3` 分别映射到原来的 `CHOOSE 0/1/2`；进入锻造后仍需选牌、确认或取消返回。",
                "“营火活动规则”仅在 `REST` 输入中提供，使用/丢弃药水不计入一次营火活动。", "",
            ])
        lines.extend(["```text", prompt.rstrip(), "```", ""])
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(preview_markdown(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
