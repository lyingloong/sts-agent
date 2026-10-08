"""Print developer-readable prompts from deterministic public-state examples.

Run ``python -m sts_agent.commands.preview_policy_prompt``. This does not
connect to the game/model; every example uses the real normalization, room
history, routing and prompt builder. Examples are illustrative, not game logs.
"""

from __future__ import annotations

import copy
import sys

from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.runtime.controller import accepted_action_history_item, normalize_state, policy_observation


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
        "potions": [], "screen_state": {}, "choice_list": [], **extra,
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
                 _card("Strike_R", "打击", "造成6点伤害", damage=6, is_playable=True)],
        "monsters": [{"id": "JawWorm", "name": "大颚虫", "current_hp": 20, "max_hp": 42,
                      "block": 0, "is_gone": False, "intent": "ATTACK", "move_adjusted_damage": 7,
                      "move_hits": 1, "powers": []}],
        "draw_pile": [{}, {}, {}], "discard_pile": [], "exhaust_pile": [],
    })
    after_defend = copy.deepcopy(combat)
    after_defend["game_state"]["combat_state"]["hand"].pop(0)
    after_defend["game_state"]["combat_state"]["player"].update(energy=2, block=5)
    after_defend["game_state"]["combat_state"]["discard_pile"] = [combat["game_state"]["combat_state"]["hand"][0]]
    history = [_history(combat, after_defend, 21,
                        {"kind": "play", "id": "PLAY:card=1", "label": "出牌 防御"})]
    prompts = {"战斗": build_scoring_prompt(policy_observation(after_defend, action_history=history),
                 [{"id": "A1", "label": "出牌 打击 -> 敌人 0"}, {"id": "A2", "label": "结束回合"}])}

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
    return prompts


def preview_markdown() -> str:
    lines = ["# 模型 Prompt Preview", "",
        "本文件由 `python -m sts_agent.commands.preview_policy_prompt` 的输出生成。",
        "使用真实的状态规范化、房间历史和 `build_scoring_prompt` 拼接器；示例数据是人工构造的，不是采集轨迹。",
        "不要直接修改生成内容；通用指令和场景提示在 `sts_agent/policy/local_qwen.py` 中维护。", "",
        "完整字段、动作空间、历史保留和未知信息规则见 [模型输入与动作空间](MODEL_OBSERVATION_AND_ACTIONS.md)。", "",
        "默认 `canonical` 顺序：通用规则 → 场景提示 → 当前状态 → 当前房间历史 → 合法动作 → 单动作输出要求。",
        "`state_first`、`decision_first` 使用相同内容，仅改变区块顺序。完整地图只在 MAP 提供。", "",
        "下面包含战斗、翻牌和事件二级选牌三个完整输入，均输出本次合法动作的单个别名（如 `A1`）。", ""]
    for title, prompt in example_prompts().items():
        lines.extend([f"## {title}", "", "```text", prompt.rstrip(), "```", ""])
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(preview_markdown(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
