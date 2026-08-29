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

    def test_online_rl_action_distinguishes_sampled_action_from_top_candidates(self) -> None:
        action = overlay_terminal.online_rl_action_text(
            {
                "model_action_label": "出牌 双重打击 -> 敌人 0",
                "model_action_probability": 0.193,
            }
        )
        distribution = overlay_terminal.online_rl_distribution_text(
            {"rank": 1, "label": "结束回合", "score": 0.381}
        )
        self.assertEqual(action, "本步已执行：出牌 双重打击 -> 敌人 0  p=19.3%")
        self.assertEqual(distribution, "策略 Top 1: 结束回合  p=38.1%")

    def test_online_rl_panel_reports_rollout_and_update_progress(self) -> None:
        rendered = overlay_terminal.online_rl_panel_text(
            {
                "policy_version": "v000003",
                "iteration": 3,
                "group_id": "g-000004",
                "group_size": 4,
                "rollout_index": 2,
                "episodes_collected": 1,
                "stage": "正在执行第 2/4 次 rollout",
                "gpu": {"index": 5, "free_memory_mib": 18432, "utilization_percent": 12},
            },
            "/data/wanghao/sts-agent/models/base/Qwen3-0.6B",
        )
        self.assertIn("模型：Qwen3-0.6B", rendered)
        self.assertIn("策略版本：v000003  |  参数更新：第 3 次", rendered)
        self.assertIn("Rollout：当前 2/4  |  已完成 1/4", rendered)
        self.assertIn("GPU 5 · 剩余 18432 MiB · 利用率 12%", rendered)

    def test_model_name_uses_only_the_final_path_component(self) -> None:
        self.assertEqual(
            overlay_terminal.display_model_name("/data/wanghao/sts-agent/models/base/Qwen3-0.6B+sft-v0-r001"),
            "Qwen3-0.6B+sft-v0-r001",
        )

    def test_posthoc_reflection_summary_uses_terminal_facts(self) -> None:
        rendered = overlay_terminal.posthoc_reflection_text(
            {"terminal_state": {"screen_type": "GAME_OVER", "act": 2, "floor": 24, "current_hp": 0, "max_hp": 88}}
        )
        self.assertEqual(rendered, "GAME_OVER  |  Act 2 Floor 24  |  HP 0/88")

    def test_posthoc_action_review_lists_all_actions_with_context(self) -> None:
        rendered = overlay_terminal.action_review_text(
            [
                {
                    "step_id": 1,
                    "act": 1,
                    "floor": 3,
                    "screen_type": "COMBAT_REWARD",
                    "source": "自动",
                    "action": "自动拾取金币 25",
                },
                {
                    "step_id": 2,
                    "act": 1,
                    "floor": 4,
                    "screen_type": "NONE",
                    "source": "模型",
                    "action": "出牌 打击 -> 敌人 0",
                },
            ]
        )
        self.assertIn("#1  A1 F3  COMBAT_REWARD  [自动] 自动拾取金币 25", rendered)
        self.assertIn("#2  A1 F4  NONE  [模型] 出牌 打击 -> 敌人 0", rendered)

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
