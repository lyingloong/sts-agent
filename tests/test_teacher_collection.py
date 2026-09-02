from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import export_teacher_sft  # noqa: E402
import teacher_client  # noqa: E402
import teacher_prompts  # noqa: E402


class TeacherClientTests(unittest.TestCase):
    def test_reads_key_field_without_exposing_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secret.txt"
            path.write_text("base_url: https://example.invalid\nkey: test-key\n", encoding="utf-8")
            self.assertEqual(teacher_client.read_api_key(path), "test-key")

    def test_parses_only_current_legal_action_alias(self) -> None:
        self.assertEqual(teacher_client.parse_action_id('{"action_id":"A2"}', {"A1", "A2"}), "A2")
        with self.assertRaises(teacher_client.TeacherError):
            teacher_client.parse_action_id('{"action_id":"A3"}', {"A1", "A2"})

    def test_prompt_lists_actions_and_requires_compact_json(self) -> None:
        prompt = teacher_client.build_prompt(
            {"screen_type": "EVENT", "choices": ["Take", "Leave"]},
            [{"id": "A1", "label": "Take"}, {"id": "A2", "label": "Leave"}],
        )
        self.assertIn('{"action_id":"A1"}', prompt)
        self.assertIn("A2 | Leave", prompt)

    def test_prompt_uses_configurable_input_limit(self) -> None:
        observation = {"screen_type": "GRID", "cards": [{"description": "x" * 200}]}
        truncated = teacher_client.build_prompt(observation, [], max_input_chars=80)
        unbounded = teacher_client.build_prompt(observation, [], max_input_chars=0)
        self.assertIn("...<truncated>", truncated)
        self.assertNotIn("...<truncated>", unbounded)

    def test_teacher_prompt_selects_chinese_scenario_guidance(self) -> None:
        prompt = teacher_prompts.build_system_prompt(
            {
                "screen_type": "GRID",
                "screen_state": {"selection_mode": "PURGE"},
            }
        )
        self.assertIn("删牌决策", prompt)
        self.assertIn("不可逆", prompt)
        self.assertNotIn("【战斗决策】", prompt)

    def test_teacher_prompt_distinguishes_generated_combat_card_choice(self) -> None:
        prompt = teacher_prompts.build_system_prompt(
            {
                "screen_type": "CARD_REWARD",
                "selection_context": {"kind": "COMBAT_GENERATED_CARD_CHOICE"},
            }
        )
        self.assertIn("战斗内生成选牌", prompt)
        self.assertIn("攻击药水", prompt)
        self.assertNotIn("【卡牌奖励决策】", prompt)

    def test_teacher_prompt_uses_relic_pickup_selection_guidance(self) -> None:
        prompt = teacher_prompts.build_system_prompt(
            {
                "screen_type": "GRID",
                "screen_state": {"selection_mode": "BOTTLE"},
                "selection_context": {"kind": "RELIC_PICKUP_CARD_SELECTION"},
            }
        )
        self.assertIn("遗物触发选牌", prompt)
        self.assertIn("瓶装遗物", prompt)

    def test_combat_prompt_prioritizes_avoiding_preventable_damage(self) -> None:
        prompt = teacher_prompts.build_system_prompt(
            {"screen_type": "NONE", "room_phase": "COMBAT"}
        )
        self.assertIn("减少本回合及后续几回合可避免掉血", prompt)
        self.assertIn("不是机械地追求每回合零伤害", prompt)
        self.assertIn("把 `END_TURN` 当作最后备选", prompt)

    def test_teacher_payload_uses_scenario_system_prompt(self) -> None:
        client = teacher_client.RemoteTeacherClient.__new__(teacher_client.RemoteTeacherClient)
        client.model = "deepseek-v4-pro"
        client.temperature = 0.0
        client.max_tokens = 512
        client.max_input_chars = 32000
        client.thinking = False
        client.json_mode = True
        client.streaming = True
        payload = client._payload(
            {"screen_type": "SHOP_SCREEN"},
            [{"id": "A1", "label": "购买"}],
        )
        self.assertIn("商店决策", payload["messages"][0]["content"])
        self.assertIn("A1 | 购买", payload["messages"][1]["content"])

    def test_select_extracts_reasoning_content_without_using_it_as_label(self) -> None:
        client = teacher_client.RemoteTeacherClient.__new__(teacher_client.RemoteTeacherClient)
        client.enabled = True
        client.base_url = "https://example.invalid/v1"
        client.model = "deepseek-v4-pro"
        client.timeout_seconds = 1.0
        client.max_retries = 0
        client.temperature = 0.0
        client.max_tokens = 512
        client.max_input_chars = 32000
        client.thinking = True
        client.json_mode = True
        client.streaming = False
        client.request_id = 0
        client.last_error = None
        client._api_key = "test"
        client._request = lambda payload, callback=None: {
            "model": "deepseek-v4-pro",
            "choices": [{
                "message": {
                    "reasoning_content": "先比较收益，再选择离开。",
                    "content": '{"action_id":"A2"}',
                },
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 12},
        }
        result = client.select({"screen_type": "EVENT"}, [{"id": "A1"}, {"id": "A2"}])
        self.assertEqual(result["action_id"], "A2")
        self.assertEqual(result["thinking"], "先比较收益，再选择离开。")
        self.assertEqual(result["output"], '{"action_id":"A2"}')


class TeacherSftExportTests(unittest.TestCase):
    def test_export_rebuilds_dynamic_alias_and_ignores_non_teacher_steps(self) -> None:
        record = {
            "accepted": True,
            "stable_after_action": True,
            "run_id": "run-1",
            "step_id": 7,
            "screen_type": "EVENT",
            "pre_state": {"screen_type": "EVENT"},
            "legal_actions": [
                {"id": "CHOOSE:index=0", "label": "Take"},
                {"id": "CHOOSE:index=1", "label": "Leave"},
            ],
            "action": {"id": "CHOOSE:index=1"},
            "decision": {
                "source": "teacher_auto",
                "teacher": {"model_id": "deepseek-v4-pro", "latency_ms": 10},
            },
        }
        sample = export_teacher_sft.export_record(record)
        assert sample is not None
        self.assertEqual(sample["target"], {"alias": "A2", "action_id": "CHOOSE:index=1"})
        self.assertEqual(sample["legal_actions"][0]["action_id"], "CHOOSE:index=0")
        record["decision"]["source"] = "human_only"
        self.assertIsNone(export_teacher_sft.export_record(record))

    def test_terminal_outcome_requires_a_stable_terminal_state(self) -> None:
        records = [
            {"accepted": True, "stable_after_action": True, "terminal": True, "post_state": {"screen_type": "VICTORY"}},
        ]
        self.assertEqual(export_teacher_sft.terminal_outcome(records), "victory")
        self.assertIsNone(export_teacher_sft.terminal_outcome([{"terminal": True, "post_state": {"screen_type": "NONE"}}]))

    def test_export_rejects_action_absent_from_recorded_legal_actions(self) -> None:
        record = {
            "accepted": True,
            "stable_after_action": True,
            "pre_state": {"screen_type": "EVENT"},
            "legal_actions": [{"id": "CHOOSE:index=0", "label": "Take"}],
            "action": {"id": "CHOOSE:index=1"},
            "decision": {"source": "teacher_auto"},
        }
        self.assertIsNone(export_teacher_sft.export_record(record))


if __name__ == "__main__":
    unittest.main()
