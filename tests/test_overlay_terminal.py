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

    def test_model_panel_keeps_confidence_and_candidates(self) -> None:
        rendered = overlay_terminal.model_feedback(
            {
                "confidence": {
                    "top_probability": 0.9,
                    "margin": 0.8,
                    "entropy": 0.3,
                    "route": "human_confirm",
                },
                "model_candidates": [{"rank": 1, "action_id": "CHOOSE:index=0", "score": 0.9}],
            }
        )
        self.assertIn("90.0%", rendered)
        self.assertIn("CHOOSE:index=0", rendered)
        self.assertIn("等待人工确认", rendered)


if __name__ == "__main__":
    unittest.main()
