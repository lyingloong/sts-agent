from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent import config as agent_config
from sts_agent.paths import PROJECT_ROOT


class AgentConfigTests(unittest.TestCase):
    def test_default_framework_is_direct(self) -> None:
        config = agent_config.validate_config(agent_config.DEFAULT_CONFIG)
        self.assertEqual(config["framework"]["game"], "sts1")
        self.assertEqual(config["framework"]["control_mode"], "manual")
        self.assertEqual(config["framework"]["policy_backend"], "none")

    def test_current_project_config_is_direct_prompt_evolution(self) -> None:
        config = agent_config.load_agent_config(PROJECT_ROOT)
        self.assertEqual(config["framework"]["policy_backend"], "remote_teacher")
        self.assertEqual(config["framework"]["learning_method"], "prompt_evolution")
        self.assertTrue(config["teacher"]["enabled"])
        self.assertFalse(config["policy"]["enabled"])
        self.assertTrue(config["prompt_evolution"]["enabled"])

    def test_framework_backend_projects_to_runtime_components(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "framework": {
                    "control_mode": "autonomous",
                    "policy_backend": "remote_qwen",
                    "learning_method": "online_rl",
                },
                "policy": {"model": {"initial_adapter": "outputs/sft/example"}},
            },
        )
        validated = agent_config.validate_config(config)
        self.assertTrue(validated["policy"]["enabled"])
        self.assertEqual(validated["policy"]["runtime"], "remote_online_rl")

    def test_collect_human_uses_unified_recording_config(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "framework": {
                    "control_mode": "manual",
                    "policy_backend": "none",
                    "learning_method": "collect_human",
                },
            },
        )
        validated = agent_config.validate_config(config)
        self.assertEqual(validated["recording"]["listen"]["port"], 8766)
        self.assertEqual(
            validated["recording"]["output_dir"],
            "data/trajectories/human",
        )

    def test_collect_human_rejects_policy_backend(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "framework": {
                    "control_mode": "manual",
                    "policy_backend": "local_qwen",
                    "learning_method": "collect_human",
                },
            },
        )
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_unknown_framework_field_is_rejected(self) -> None:
        with self.assertRaises(agent_config.ConfigError):
            agent_config._merge(agent_config.DEFAULT_CONFIG, {"framework": {"mode": "manual"}})

    def test_invalid_framework_value_is_rejected(self) -> None:
        config = agent_config._merge(agent_config.DEFAULT_CONFIG, {"framework": {"policy_backend": "teacher"}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_teacher_and_policy_cannot_be_enabled_together(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "framework": {"control_mode": "autonomous", "policy_backend": "remote_teacher", "learning_method": "none"},
                "teacher": {"enabled": True},
            },
        )
        # The framework is the source of truth and disables policy here.
        self.assertFalse(agent_config.validate_config(config)["policy"]["enabled"])

    def test_json_file_is_strict_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            path.write_text(json.dumps(agent_config.DEFAULT_CONFIG), encoding="utf-8")
            loaded = agent_config.load_agent_config(Path(directory), path)
            self.assertEqual(loaded["framework"]["game"], "sts1")


if __name__ == "__main__":
    unittest.main()
