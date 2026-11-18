from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from sts_agent import config as agent_config
from sts_agent.paths import PROJECT_ROOT


class AgentConfigTests(unittest.TestCase):
    def test_default_config_uses_public_example_until_local_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            shutil.copytree(PROJECT_ROOT / "config" / "modes", config_dir / "modes")
            shutil.copy(PROJECT_ROOT / "config" / "agent.example.json", config_dir)
            local = config_dir / "agent.json"
            for selection in (None, "config/agent.json", local):
                config = agent_config.load_agent_config(root, selection)
                self.assertEqual(config["framework"]["learning_method"], "collect_human")
            with self.assertRaisesRegex(agent_config.ConfigError, "missing configuration file"):
                agent_config.load_agent_config(root, "config/custom.json")

            local.write_text('{"schema_version": 2, "active_mode": "baseline_api"}', encoding="utf-8")
            config = agent_config.load_agent_config(root, "config/agent.json")
            self.assertEqual(config["policy"]["api"]["provider"], "openai_compatible")
            local.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(agent_config.ConfigError, "invalid JSON"):
                agent_config.load_agent_config(root)

    def test_public_example_is_offline_human_collection(self) -> None:
        config = agent_config.load_agent_config(PROJECT_ROOT, "config/agent.example.json")
        self.assertEqual(config["framework"]["learning_method"], "collect_human")
        self.assertFalse(config["policy"]["enabled"])
        self.assertFalse(config["teacher"]["enabled"])
        self.assertEqual(config["policy"]["remote"]["ssh_host"], "gpu-server")

    def test_default_framework_is_direct(self) -> None:
        config = agent_config.validate_config(agent_config.DEFAULT_CONFIG)
        self.assertEqual(config["framework"]["game"], "sts1")
        self.assertEqual(config["framework"]["control_mode"], "manual")
        self.assertEqual(config["framework"]["policy_backend"], "none")

    def test_human_profile_selects_human_collection(self) -> None:
        config = agent_config.load_agent_config(PROJECT_ROOT, mode="collect_human")
        self.assertEqual(config["framework"]["control_mode"], "manual")
        self.assertEqual(config["framework"]["policy_backend"], "none")
        self.assertEqual(config["framework"]["learning_method"], "collect_human")
        self.assertFalse(config["teacher"]["enabled"])
        self.assertFalse(config["policy"]["enabled"])
        self.assertFalse(config["prompt_evolution"]["enabled"])

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

    def test_collect_codex_mode_routes_to_its_own_run_directory(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {
                "framework": {
                    "control_mode": "codex",
                    "policy_backend": "none",
                    "learning_method": "collect_codex",
                },
            },
        )
        validated = agent_config.validate_config(config)
        self.assertFalse(validated["policy"]["enabled"])
        self.assertFalse(validated["teacher"]["enabled"])
        self.assertEqual(validated["codex"]["character"], "IRONCLAD")
        self.assertEqual(validated["codex"]["ascension"], 0)
        self.assertEqual(validated["codex"]["output_dir"], "data/trajectories/codex")

    def test_collect_codex_requires_external_codex_control(self) -> None:
        config = agent_config._merge(
            agent_config.DEFAULT_CONFIG,
            {"framework": {"learning_method": "collect_codex"}},
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
            root = Path(directory)
            (root / "modes").mkdir()
            path = root / "agent.json"
            path.write_text(json.dumps({"schema_version": 2, "active_mode": "collect_human"}), encoding="utf-8")
            (root / "modes" / "collect_human.json").write_text("{}", encoding="utf-8")
            loaded = agent_config.load_agent_config(root, path)
            self.assertEqual(loaded["framework"]["game"], "sts1")

    def test_mode_profile_applies_mode_specific_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "modes").mkdir()
            config_file = root / "agent.json"
            config_file.write_text(json.dumps({"schema_version": 2, "active_mode": "online_rl"}), encoding="utf-8")
            (root / "modes" / "online_rl.json").write_text(
                json.dumps({"posthoc_feedback": {"enabled": True, "mode": "auto"}, "policy": {"model": {"initial_adapter": "adapter"}}}),
                encoding="utf-8",
            )
            loaded = agent_config.load_agent_config(root, config_file)
            self.assertEqual(loaded["framework"]["learning_method"], "online_rl")
            self.assertTrue(loaded["policy"]["enabled"])
            self.assertEqual(loaded["policy"]["runtime"], "remote_online_rl")
            self.assertEqual(loaded["posthoc_feedback"]["mode"], "auto")

    def test_mode_selector_rejects_framework_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "agent.json"
            config_file.write_text(
                json.dumps({"schema_version": 2, "active_mode": "collect_human", "framework": {"control_mode": "manual"}}),
                encoding="utf-8",
            )
            with self.assertRaises(agent_config.ConfigError):
                agent_config.load_agent_config(root, config_file)

    def test_every_named_mode_profile_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modes_dir = root / "modes"
            shutil.copytree(PROJECT_ROOT / "config" / "modes", modes_dir)
            config_file = root / "agent.json"
            descriptor = {
                "schema_version": 2,
                "shared": {"policy": {"model": {"initial_adapter": "outputs/sft/test"}}},
            }
            for mode_name in agent_config.MODE_FRAMEWORKS:
                descriptor["active_mode"] = mode_name
                config_file.write_text(json.dumps(descriptor), encoding="utf-8")
                resolved = agent_config.load_agent_config(root, config_file)
                self.assertEqual(
                    resolved["framework"],
                    agent_config.MODE_FRAMEWORKS[mode_name],
                    mode_name,
                )


if __name__ == "__main__":
    unittest.main()
