from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import posthoc_feedback  # noqa: E402


def episode() -> dict:
    return {
        "terminal_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 7, "current_hp": 0, "max_hp": 80},
        "steps": [
            {
                "step_id": 3,
                "observation": {"screen_type": "COMBAT", "floor": 5, "combat": {"player": {"energy": 2}}},
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
    def test_action_ledger_only_contains_selected_actions(self) -> None:
        ledger = posthoc_feedback.action_ledger(episode())
        self.assertEqual([item["step_id"] for item in ledger], [3, 4])
        self.assertEqual(ledger[0]["selected_action"]["label"], "出牌 打击")
        self.assertNotIn("A2", str(ledger))

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


if __name__ == "__main__":
    unittest.main()
