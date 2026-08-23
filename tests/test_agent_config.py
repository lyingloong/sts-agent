from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import agent_config  # noqa: E402


class AgentConfigTests(unittest.TestCase):
    def test_default_config_has_one_confidence_gate(self) -> None:
        config = agent_config.validate_config(agent_config.DEFAULT_CONFIG)
        confidence = config["policy"]["confidence_auto"]
        self.assertTrue(confidence["enabled"])
        self.assertEqual(confidence["min_probability"], 0.85)
        self.assertNotIn("min_margin", confidence)

    def test_unknown_field_is_rejected(self) -> None:
        with self.assertRaises(agent_config.ConfigError):
            agent_config._merge(agent_config.DEFAULT_CONFIG, {"policy": {"min_margin": 0.35}})

    def test_invalid_overlay_geometry_is_rejected(self) -> None:
        config = agent_config._merge(agent_config.DEFAULT_CONFIG, {"overlay": {"status": {"geometry": "wide"}}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_current_project_file_is_valid(self) -> None:
        config = agent_config.load_agent_config(TOOLS_DIR.parent)
        self.assertEqual(config["server"]["port"], 8765)
        self.assertEqual(config["policy"]["screens"], ["ALL"])
        self.assertTrue(config["overlay"]["always_on_top"])
        self.assertTrue(config["overlay"]["frameless"])
        self.assertEqual(config["overlay"]["status"]["divider_columns"], 36)


if __name__ == "__main__":
    unittest.main()
