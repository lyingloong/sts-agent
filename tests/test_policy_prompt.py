from __future__ import annotations

import unittest

from sts_agent.policy import local_qwen as policy_server


class PolicyPromptTests(unittest.TestCase):
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
