from __future__ import annotations

import copy
import unittest

from sts_agent.policy.local_qwen import PROMPT_TEMPLATES, build_scoring_prompt, render_room_history


class EventHistoryRenderingTests(unittest.TestCase):
    def test_grid_opening_deduplicates_choices_and_keeps_complete_options(self):
        opening = {
            "event_name": "升级神龛", "body_text": "被遗忘神灵的神龛。",
            "choices": ["祈祷", "离开"], "options": [
                {"label": "祈祷", "text": "[祈祷] 支付10金币，升级一张牌。", "disabled": False},
                {"label": "离开", "text": "[离开]", "disabled": True},
            ],
        }
        observation = {"screen_type": "GRID", "screen_state": {"for_upgrade": True},
                       "room_history": {"event_opening": opening, "steps": []}}
        original = copy.deepcopy(observation)
        for template in PROMPT_TEMPLATES:
            with self.subTest(template=template):
                prompt = build_scoring_prompt(observation, [{"id": "A1", "label": "选牌"}], template)
                self.assertNotIn("当前选择", prompt)
                self.assertNotIn("当时可选项", prompt)
                for text in ("升级神龛", "被遗忘神灵的神龛。", "支付10金币，升级一张牌。", "disabled=是", "disabled=否"):
                    self.assertIn(text, prompt)
        self.assertEqual(observation, original)

    def test_choice_coverage_requires_exact_wording_or_a_clear_bracket_prefix(self):
        opening = {"choices": ["完全相同", "祈祷", "离开", "[离开] 失去20金币。", "获得遗物，失去10生命。"],
                   "options": [{"text": "完全相同"}, {"text": "[ 祈祷 ] 升级一张牌。"},
                               {"name": "离开", "text": "[离开] 失去5金币。"},
                               {"label": "获得遗物", "text": "[获得遗物]"}]}
        text = "\n".join(render_room_history({"event_opening": opening}))
        self.assertIn("当时可选项", text)
        self.assertNotIn("  - 完全相同", text)
        self.assertNotIn("  - 祈祷", text)
        self.assertNotIn("  - 离开\n", text)
        self.assertIn("[离开] 失去20金币。", text)
        self.assertIn("获得遗物，失去10生命。", text)
        self.assertIn("[离开] 失去5金币。", text)

    def test_partial_options_keep_uncovered_historical_choices(self):
        history = {"event_opening": {"body_text": "事件开场", "choices": ["祈祷", "离开"],
                   "options": [{"label": "祈祷", "text": "[祈祷] 升级一张牌。", "disabled": False}]}}
        original = copy.deepcopy(history)
        text = "\n".join(render_room_history(history))
        self.assertIn("事件开场", text)
        self.assertIn("当时可选项", text)
        self.assertIn("- 离开", text)
        self.assertNotIn("- 祈祷", text)
        self.assertIn("升级一张牌。", text)
        self.assertEqual(history, original)

    def test_empty_missing_and_incomplete_options_keep_choice_only_fallback(self):
        for options in ({}, {"options": []}, {"options": None}, {"options": [{"disabled": True}]},
                        {"options": [{"text": "只解释代价但未指明选项"}]}):
            with self.subTest(options=options):
                history = {"event_opening": {"choices": ["选择甲", "选择乙"], **options}}
                original = copy.deepcopy(history)
                text = "\n".join(render_room_history(history))
                self.assertIn("当时可选项", text)
                self.assertIn("- 选择甲", text)
                self.assertIn("- 选择乙", text)
                self.assertNotIn("当前选择", text)
                self.assertEqual(history, original)

    def test_equal_current_event_fields_remain_deduplicated(self):
        event = {"event_name": "神龛", "body_text": "相同开场", "choices": ["祈祷"],
                 "options": [{"label": "祈祷", "text": "[祈祷] 升级一张牌。"}]}
        history = {"event_opening": copy.deepcopy(event), "steps": []}
        original = copy.deepcopy(history)
        self.assertEqual(render_room_history(history, event), [])
        # Even if choices were not copied into current_event, its identical
        # detailed options already represent the same historical label.
        without_choices = {key: value for key, value in event.items() if key != "choices"}
        self.assertEqual(render_room_history(history, without_choices), [])
        self.assertEqual(history, original)

    def test_old_opening_options_remain_when_current_event_has_advanced(self):
        history = {"event_opening": {"event_name": "涅奥", "choices": ["对话"],
                   "options": [{"label": "对话", "text": "[对话]", "disabled": False}]}}
        current = {"event_name": "涅奥", "options": [{"label": "获得遗物", "text": "[获得遗物]"}]}
        text = "\n".join(render_room_history(history, current))
        self.assertIn("[对话]", text)
        self.assertIn("disabled=否", text)
        self.assertNotIn("当前选择", text)
        self.assertNotIn("当时可选项", text)


if __name__ == "__main__":
    unittest.main()
