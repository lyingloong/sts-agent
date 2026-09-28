from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.learning import posthoc_feedback


def episode() -> dict:
    return {
        "terminal_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 7, "current_hp": 0, "max_hp": 80},
        "steps": [
            {
                "step_id": 3,
                "observation": {
                    "screen_type": "COMBAT",
                    "floor": 5,
                    "combat": {
                        "player": {"energy": 2, "powers": [{"id": "Strength", "amount": 1, "description": "static"}]},
                        "hand": [{"id": "Strike_R", "name": "打击", "cost": 1, "is_playable": True, "description": "Deal 6 damage."}],
                        "monsters": [{"id": "JawWorm", "current_hp": 40, "intent": "ATTACK", "intent_damage": 11}],
                    },
                    "recent_actions": [{"step_id": 2, "action": "出牌 防御", "source": "online_policy_sample"}],
                },
                "legal_actions": [{"id": "A1", "label": "出牌 打击"}, {"id": "A2", "label": "结束回合"}],
                "selected_alias": "A1",
            },
            {
                "step_id": 4,
                "observation": {"screen_type": "COMBAT", "floor": 5},
                "legal_actions": [{"id": "A1", "label": "结束回合"}],
                "selected_alias": "A1",
            },
        ],
    }


class PosthocFeedbackTests(unittest.TestCase):
    def test_action_ledger_contains_selected_and_legal_actions_with_dynamic_state(self) -> None:
        ledger = posthoc_feedback.action_ledger(episode())
        self.assertEqual([item["step_id"] for item in ledger], [3, 4])
        self.assertEqual(ledger[0]["selected_action"]["label"], "出牌 打击")
        self.assertEqual(ledger[0]["legal_actions"], [{"alias": "A1", "label": "出牌 打击"}, {"alias": "A2", "label": "结束回合"}])
        combat = ledger[0]["state"]["combat"]
        self.assertEqual(combat["player"]["energy"], 2)
        self.assertEqual(combat["hand"][0]["name"], "打击")
        self.assertEqual(combat["monsters"][0]["intent_damage"], 11)
        self.assertEqual(ledger[0]["state"]["recent_actions"][0]["action"], "出牌 防御")
        self.assertNotIn("description", str(ledger))

    def test_action_ledger_deduplicates_repeated_map_nodes(self) -> None:
        map_nodes = [{"x": 0, "y": 0, "symbol": "M", "children": [{"x": 0, "y": 1}]}]
        value = episode()
        for step in value["steps"]:
            step["observation"] = {
                "screen_type": "MAP",
                "floor": step["step_id"],
                "map": {"nodes": map_nodes, "current_node": {"x": 0, "y": 0}, "next_nodes": [{"x": 0, "y": 1}]},
            }
        ledger = posthoc_feedback.action_ledger(value)
        self.assertEqual(ledger[0]["state"]["map"]["map_id"], "map-1")
        self.assertIn("nodes", ledger[0]["state"]["map"])
        self.assertEqual(ledger[1]["state"]["map"]["map_id"], "map-1")
        self.assertNotIn("nodes", ledger[1]["state"]["map"])

    def test_trace_action_review_preserves_model_and_automatic_actions(self) -> None:
        records = [
            {
                "step_id": 1,
                "accepted": True,
                "screen_type": "COMBAT_REWARD",
                "pre_state": {"act": 1, "floor": 3},
                "decision": {"source": "environment_auto"},
                "action": {"label": "自动拾取金币 25"},
            },
            {
                "step_id": 2,
                "accepted": True,
                "screen_type": "NONE",
                "pre_state": {"act": 1, "floor": 4},
                "decision": {"source": "online_policy_sample"},
                "action": {"label": "出牌 打击 -> 敌人 0"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
            review = posthoc_feedback.action_review_from_trace(path)
        self.assertEqual(len(review), 2)
        self.assertEqual(review[0]["source"], "自动")
        self.assertEqual(review[0]["action"], "自动拾取金币 25")
        self.assertEqual(review[1]["source"], "模型")
        self.assertEqual(review[1]["floor"], 4)

    def test_validation_filters_low_confidence_and_conflicts(self) -> None:
        result = posthoc_feedback.validate_attributions(
            {
                "reinforce": [
                    {"step_id": 3, "confidence": 0.91, "severity": 2, "evidence": "保留能量"},
                    {"step_id": 4, "confidence": 0.74, "severity": 3},
                ],
                "weaken": [{"step_id": 3, "confidence": 0.95, "severity": 3, "evidence": "不该打"}],
            },
            episode(),
            min_confidence=0.75,
            max_applied_per_direction=8,
        )
        self.assertEqual(result["applied"], [])
        reasons = {item["reason"] for item in result["rejected"]}
        self.assertIn("conflicting directions", reasons)
        self.assertIn("confidence below configured threshold 0.75", reasons)

    def test_validation_caps_each_direction_and_keeps_highest_confidence(self) -> None:
        value = {
            "reinforce": [
                {"step_id": 3, "confidence": 0.80, "severity": 1},
                {"step_id": 4, "confidence": 0.95, "severity": 1},
            ],
            "weaken": [],
        }
        result = posthoc_feedback.validate_attributions(value, episode(), min_confidence=0.75, max_applied_per_direction=1)
        self.assertEqual(result["applied"][0]["step_id"], 4)
        self.assertEqual(result["applied"][0]["direction"], "reinforce")
        self.assertTrue(any(item["reason"] == "per-direction cap" for item in result["rejected"]))

    def test_validation_respects_human_feedback_direction(self) -> None:
        result = posthoc_feedback.validate_attributions(
            {"reinforce": [{"step_id": 3, "confidence": 0.9, "severity": 2}], "weaken": []},
            episode(),
            min_confidence=0.75,
            max_applied_per_direction=8,
            allowed_directions={"weaken"},
        )
        self.assertEqual(result["applied"], [])
        self.assertEqual(result["rejected"][0]["reason"], "human reflection did not authorize this direction")

    def test_prompt_is_bounded_and_requests_both_directions(self) -> None:
        messages = posthoc_feedback.build_prompt(
            {"reinforce": "保留格挡", "weaken": "不该空过"},
            episode(),
            max_candidates_per_direction=10,
            max_input_chars=20000,
        )
        self.assertIn("reinforce", messages[0]["content"])
        self.assertIn("sampled_action_ledger", messages[1]["content"])
        self.assertIn("保留格挡", messages[1]["content"])

    def test_auto_prompt_allows_teacher_to_choose_both_directions(self) -> None:
        messages = posthoc_feedback.build_prompt(
            {"mode": "auto", "reinforce": "", "weaken": ""},
            episode(),
            max_candidates_per_direction=10,
            max_input_chars=20000,
        )
        self.assertIn("自动归因模式", messages[0]["content"])
        self.assertIn("终局死亡、低楼层、胜利或奖励数值本身不是任何单步的正负证据", messages[0]["content"])
        self.assertIn("当时合法的替代动作", messages[0]["content"])
        self.assertIn("通常每个方向 0-3 条", messages[0]["content"])
        self.assertIn('"feedback_mode":"auto"', messages[1]["content"])
        client = object.__new__(posthoc_feedback.PosthocFeedbackClient)
        self.assertEqual(client.automatic_reflection()["reason"], "teacher_auto")

    def test_human_prompt_keeps_human_direction_authoritative(self) -> None:
        messages = posthoc_feedback.build_prompt(
            {"mode": "human", "reinforce": "合理防御", "weaken": ""},
            episode(),
            max_candidates_per_direction=10,
            max_input_chars=20000,
        )
        self.assertIn("人类复盘是唯一的价值方向", messages[0]["content"])
        self.assertNotIn("通常每个方向 0-3 条", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
