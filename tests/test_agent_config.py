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
        self.assertEqual(config["teacher"]["max_input_chars"], 32000)
        self.assertEqual(config["posthoc_feedback"]["max_input_chars"], 96000)
        self.assertEqual(config["posthoc_feedback"]["mode"], "human")

    def test_teacher_input_limit_must_be_a_non_negative_integer(self) -> None:
        config = agent_config._merge(agent_config.DEFAULT_CONFIG, {"teacher": {"max_input_chars": -1}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_unknown_field_is_rejected(self) -> None:
        with self.assertRaises(agent_config.ConfigError):
            agent_config._merge(agent_config.DEFAULT_CONFIG, {"policy": {"min_margin": 0.35}})

    def test_teacher_requires_local_policy_to_be_disabled(self) -> None:
        config = agent_config._merge(agent_config.DEFAULT_CONFIG, {"teacher": {"enabled": True}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_teacher_config_is_valid_with_local_policy_disabled(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {"policy": {"enabled": False}, "teacher": {"enabled": True}},
        )
        validated = agent_config.validate_config(config)
        self.assertTrue(validated["teacher"]["enabled"])
        self.assertEqual(validated["teacher"]["model"], "deepseek-v4-pro")

    def test_posthoc_feedback_requires_remote_online_rl(self) -> None:
        config = agent_config._merge(agent_config.DEFAULT_CONFIG, {"posthoc_feedback": {"enabled": True}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_posthoc_feedback_is_valid_for_remote_online_rl(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "policy": {
                    "runtime": "remote_online_rl",
                    "model": {"initial_adapter": "outputs/sft/example"},
                },
                "posthoc_feedback": {"enabled": True},
            },
        )
        validated = agent_config.validate_config(config)
        self.assertTrue(validated["posthoc_feedback"]["enabled"])
        self.assertEqual(validated["posthoc_feedback"]["max_candidates_per_direction"], 10)

    def test_posthoc_auto_mode_is_validated(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "policy": {
                    "runtime": "remote_online_rl",
                    "model": {"initial_adapter": "outputs/sft/example"},
                },
                "posthoc_feedback": {"enabled": True, "mode": "auto"},
            },
        )
        self.assertEqual(agent_config.validate_config(config)["posthoc_feedback"]["mode"], "auto")
        invalid = agent_config._merge(agent_config.DEFAULT_CONFIG, {"posthoc_feedback": {"mode": "automatic"}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(invalid)
        malformed = agent_config._merge(agent_config.DEFAULT_CONFIG, {"posthoc_feedback": {"mode": ["auto"]}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(malformed)

    def test_invalid_overlay_geometry_is_rejected(self) -> None:
        config = agent_config._merge(agent_config.DEFAULT_CONFIG, {"overlay": {"status": {"geometry": "wide"}}})
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_current_project_file_is_valid(self) -> None:
        config = agent_config.load_agent_config(TOOLS_DIR.parent)
        self.assertEqual(config["server"]["port"], 8765)
        self.assertEqual(config["policy"]["request"]["screens"], ["ALL"])
        self.assertEqual(config["policy"]["model"]["initial_policy_version"], "rl-001")
        self.assertTrue(config["overlay"]["always_on_top"])
        self.assertFalse(config["overlay"]["frameless"])
        self.assertEqual(config["overlay"]["status"]["divider_columns"], 36)
        self.assertEqual(config["policy"]["remote"]["inference_gpu"]["device"], "auto")
        self.assertEqual(config["posthoc_feedback"]["mode"], "auto")

    def test_mode_profiles_are_valid_sparse_overrides(self) -> None:
        mode_root = TOOLS_DIR.parent / "config" / "modes"
        expected = {"manual", "candidate_assist", "teacher_sft", "prompt_evolution", "remote_inference", "remote_online_rl"}
        self.assertTrue(expected.issubset({path.stem for path in mode_root.glob("*.json")}))
        for name in expected:
            config = agent_config.load_agent_config(TOOLS_DIR.parent, mode_root / f"{name}.json")
            self.assertEqual(config["schema_version"], 1)

    def test_remote_inference_gpu_selector_is_validated(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {"policy": {"remote": {"inference_gpu": {"device": -1}}}},
        )
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(config)

    def test_online_rl_requires_one_initial_adapter_unless_mocked(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {"policy": {"runtime": "remote_online_rl"}},
        )
        with self.assertRaisesRegex(agent_config.ConfigError, "initial_adapter"):
            agent_config.validate_config(config)

        mocked = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {"policy": {"runtime": "remote_online_rl", "development": {"mock": True}}},
        )
        self.assertTrue(agent_config.validate_config(mocked)["policy"]["development"]["mock"])

    def test_prompt_evolution_requires_teacher_and_disables_other_runtime_loops(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "policy": {"enabled": False},
                "teacher": {"enabled": True},
                "prompt_evolution": {"enabled": True},
            },
        )
        validated = agent_config.validate_config(config)
        self.assertTrue(validated["prompt_evolution"]["enabled"])

        invalid = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {"prompt_evolution": {"enabled": True}},
        )
        with self.assertRaises(agent_config.ConfigError):
            agent_config.validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
