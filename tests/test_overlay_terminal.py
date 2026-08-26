from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import overlay_terminal  # noqa: E402


class OverlayFormattingTests(unittest.TestCase):
    def test_state_panel_excludes_model_candidate_block(self) -> None:
        rendered = overlay_terminal.state_text_only(
            "STS AGENT\n"
            "模型置信度：p1=90%\n"
            "模型候选（输入 a 接受第一项）\n"
            "  [1] CHOOSE:index=0\n"
            "合法动作\n"
            "  CHOOSE:index=0"
        )
        self.assertNotIn("模型", rendered)
        self.assertIn("合法动作", rendered)
        self.assertIn("CHOOSE:index=0", rendered)

    def test_state_panel_compacts_controller_dividers(self) -> None:
        rendered = overlay_terminal.state_text_only("=" * 78 + "\nstate\n" + "-" * 78, divider_columns=36)
        self.assertEqual(rendered.splitlines(), ["=" * 36, "state", "-" * 36])

    def test_state_panel_renders_compact_summary_and_action_columns(self) -> None:
        rendered = overlay_terminal.state_panel_text(
            {
                "summary": "RUN test | IRONCLAD A0 | Act 1 Floor 2 | NONE\nHP 75/88 | Gold 117 | Room COMBAT",
                "deck": [
                    {"name": "打击", "upgrades": 0, "count": 5},
                    {"name": "痛击", "upgrades": 1, "count": 1},
                ],
                "recent_actions": [
                    {"step_id": 12, "action": "净化"},
                    {"step_id": 13, "action": "选择 打击"},
                ],
                "legal_actions": [
                    {"type": "出牌", "selection": "双重打击", "target": "敌人 0", "command": "p 1 0"},
                    {"type": "结束回合", "selection": "", "target": "", "command": "e"},
                ],
            }
        )
        self.assertIn("RUN test", rendered)
        self.assertIn("当前牌组", rendered)
        self.assertIn("打击*5  痛击+1*1", rendered)
        self.assertIn("最近动作", rendered)
        self.assertIn("#12 净化\n#13 选择 打击", rendered)
        self.assertIn("动作类型\t选择\t目标\t动作指令", rendered)
        self.assertIn("出牌\t双重打击\t敌人 0\tp 1 0", rendered)
        self.assertIn("结束回合\t\t\te", rendered)

    def test_candidate_button_contains_label_and_probability_once(self) -> None:
        rendered = overlay_terminal.candidate_button_text(
            {"label": "结束回合", "score": 0.381},
            1,
        )
        self.assertEqual(rendered, "接受一: 结束回合  p=38.1%")

    def test_teacher_panel_uses_thinking_and_final_output(self) -> None:
        rendered = overlay_terminal.teacher_panel_text(
            {
                "mode": "teacher_collection",
                "source": "teacher_auto",
                "model_action": "CHOOSE:index=1",
                "teacher": {
                    "model_id": "deepseek-v4-pro",
                    "latency_ms": 321,
                    "thinking": "保留资源，离开事件。",
                    "output": '{"action_id":"A2"}',
                },
            }
        )
        self.assertIn("远程教师模式 · 思考已开启", rendered)
        self.assertIn("自动执行", rendered)
        self.assertIn("保留资源，离开事件。", rendered)
        self.assertIn('{"action_id":"A2"}', rendered)

    def test_history_line_falls_back_to_action_id(self) -> None:
        self.assertEqual(overlay_terminal.history_line({"step_id": 2, "action_id": "END_TURN"}), "#2 END_TURN")

    def test_status_table_columns_are_evenly_spaced(self) -> None:
        self.assertEqual(overlay_terminal.status_tab_stops(560), (135, 270, 405))
        self.assertEqual(overlay_terminal.status_tab_stops(200), (60, 120, 180))


if __name__ == "__main__":
    unittest.main()
