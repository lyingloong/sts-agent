from __future__ import annotations

import unittest
from pathlib import Path

from sts_agent.commands.preview_policy_prompt import preview_markdown
from sts_agent.policy import local_qwen as policy_server
from sts_agent.runtime.controller import normalize_combat, policy_observation
from tests.test_trajectory_assembler import combat_state


class PolicyPromptTests(unittest.TestCase):
    def test_checked_in_preview_is_the_real_assembled_prompt(self) -> None:
        path = Path(__file__).resolve().parents[1] / "doc" / "POLICY_PROMPT_PREVIEW.md"
        self.assertEqual(path.read_text(encoding="utf-8").rstrip(), preview_markdown().rstrip())

    def test_runtime_stance_reaches_all_prompt_layouts_and_combat_selections(self) -> None:
        for stance_id, name in (("Neutral", "正常"), ("Calm", "平静"), ("Wrath", "愤怒"), ("Divinity", "神格")):
            for screen in ("NONE", "GRID", "HAND_SELECT", "CARD_REWARD"):
                with self.subTest(stance=stance_id, screen=screen):
                    state = combat_state() | {"class": "WATCHER", "screen_type": screen}
                    stance = {"id": stance_id, "name": name, "description": f"运行时姿态说明：{name}"}
                    state["combat_state"]["player"]["stance"] = stance
                    observation = policy_observation({"in_game": True, "game_state": state})
                    self.assertEqual(observation["combat"]["player"]["stance"], stance)
                    for template in policy_server.PROMPT_TEMPLATES:
                        prompt = policy_server.build_scoring_prompt(observation, [{"id": "A1", "label": "确认"}], template)
                        self.assertIn("姿态：", prompt)
                        self.assertIn(f"ID={stance_id}", prompt)
                        self.assertIn(stance["description"], prompt)

    def test_missing_stance_is_unknown_not_assumed_neutral(self) -> None:
        for player in ({}, {"stance": None}, {"stance": "Wrath"}):
            with self.subTest(player=player):
                normalized = normalize_combat({"player": player})
                self.assertIsNone(normalized["player"]["stance"])
                rendered = policy_server.render_observation({"screen_type": "COMBAT", "combat": normalized})
                self.assertIn("姿态=未知", rendered)

    def test_noncombat_screens_do_not_leak_stale_stance(self) -> None:
        for screen in ("MAP", "COMBAT_REWARD", "SHOP_SCREEN"):
            with self.subTest(screen=screen):
                state = combat_state() | {"screen_type": screen, "room_phase": "COMPLETE"}
                state["combat_state"]["player"]["stance"] = {"id": "Wrath", "name": "愤怒", "description": "旧战斗状态"}
                observation = policy_observation({"in_game": True, "game_state": state})
                self.assertNotIn("combat", observation)
                self.assertNotIn("旧战斗状态", policy_server.render_observation(observation))

    def test_prompt_is_chinese_and_keeps_action_protocol(self) -> None:
        prompt = policy_server.build_scoring_prompt(
            {
                "screen_type": "COMBAT",
                "current_hp": 40,
                "max_hp": 80,
                "combat": {"energy": 3},
            },
            [{"id": "A1", "label": "结束回合"}],
        )
        self.assertIn("你正在游玩《杀戮尖塔》", prompt)
        self.assertIn("战斗界面", prompt)
        self.assertIn("A1：结束回合", prompt)
        self.assertIn("请输出唯一的动作 ID", prompt)
        self.assertNotIn("You are", prompt)

    def test_map_is_only_rendered_on_map_screen(self) -> None:
        observation = {
            "screen_type": "COMBAT",
            "map": {"nodes": [{"x": 0, "y": 0, "type": "ELITE"}]},
            "combat": {"energy": 3},
        }
        rendered = policy_server.render_observation(observation)
        self.assertNotIn("地图全貌", rendered)
        self.assertNotIn("ELITE", rendered)

        observation["screen_type"] = "MAP"
        rendered = policy_server.render_observation(observation)
        self.assertIn("地图全貌", rendered)
        self.assertIn("ELITE", rendered)

    def test_cards_and_history_are_rendered_without_json_punctuation(self) -> None:
        rendered = policy_server.render_observation(
            {
                "screen_type": "COMBAT",
                "deck": [{"id": "Strike", "count": 5}],
                "combat": {"hand": [{"name": "防御", "cost": 1, "final_block": 5}]},
                "recent_actions": [{"action": "出牌 防御"}],
            }
        )
        self.assertIn("当前牌组", rendered)
        self.assertIn("数量=5", rendered)
        self.assertIn("最终格挡=5", rendered)
        self.assertIn("最近动作", rendered)
        self.assertNotIn('"deck"', rendered)

    def test_keys_are_rendered_as_public_run_state(self) -> None:
        rendered = policy_server.render_observation(
            {
                "screen_type": "MAP",
                "keys": {"ruby": True, "emerald": False, "sapphire": True},
            }
        )
        self.assertIn("红宝石钥匙=已持有", rendered)
        self.assertIn("翡翠钥匙=未持有", rendered)
        self.assertIn("蓝宝石钥匙=已持有", rendered)


if __name__ == "__main__":
    unittest.main()
