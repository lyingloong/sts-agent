from __future__ import annotations

import copy
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from sts_agent import config as agent_config
from sts_agent.runtime import controller as communication_controller
from sts_agent.core.contracts import (
    DecisionRecord,
    validate_action_list,
    validate_observation,
    validate_policy_response,
)
from sts_agent.core.modes import resolve_framework_mode
from sts_agent.core.learning import learning_entrypoint


class AgentFrameworkTests(unittest.TestCase):
    def test_direct_framework_resolves_without_mode_names(self) -> None:
        self.assertEqual(
            resolve_framework_mode({
                "game": "sts1",
                "control_mode": "autonomous",
                "policy_backend": "remote_qwen",
                "learning_method": "online_rl",
            }),
            {
                "game": "sts1",
                "control_mode": "autonomous",
                "policy_backend": "remote_qwen",
                "learning_method": "online_rl",
            },
        )

    def test_contracts_reject_duplicate_or_illegal_actions(self) -> None:
        actions = validate_action_list([{"id": "A1"}, {"id": "A2"}])
        validate_observation({"screen_type": "COMBAT"})
        validate_policy_response({"candidates": [{"action_id": "A2"}]}, actions)
        with self.assertRaises(ValueError):
            validate_action_list([{"id": "A1"}, {"id": "A1"}])
        with self.assertRaises(ValueError):
            validate_policy_response({"selected_action_id": "A3"}, actions)

    def test_decision_record_is_serializable(self) -> None:
        record = DecisionRecord(
            step_id=1,
            observation={"screen_type": "MAP"},
            legal_actions=[{"id": "A1", "label": "前往"}],
            selected_action_id="A1",
            source="human",
        )
        self.assertEqual(record.to_dict()["selected_action_id"], "A1")

    def test_controller_exposes_sts1_adapter_boundary(self) -> None:
        adapter = communication_controller.make_sts1_game_adapter()
        self.assertEqual(adapter.__class__.__name__, "Sts1Adapter")

    def test_learning_methods_have_canonical_entrypoints(self) -> None:
        self.assertEqual(learning_entrypoint("teacher_sft").entrypoint, "sts_agent.learning.sft")
        self.assertEqual(learning_entrypoint("online_rl").artifact, "outputs/online_rl")
        self.assertIsNone(learning_entrypoint("none"))

    def test_local_policy_process_uses_package_module_entrypoint(self) -> None:
        config = copy.deepcopy(agent_config.DEFAULT_CONFIG)
        config["policy"]["enabled"] = True
        config["policy"]["runtime"] = "local"
        config["policy"]["development"]["mock"] = True
        process = MagicMock()
        process.poll.return_value = None
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps({"type": "ready", "model_id": "mock"}) + "\n")
        with patch.object(communication_controller.subprocess, "Popen", return_value=process) as popen:
            client = communication_controller.PolicyClient(agent_config.project_root(), config)
            self.assertTrue(client._start())
        command = popen.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "sts_agent.policy.local_qwen"])
        self.assertFalse(any("tools" in part for part in command))


if __name__ == "__main__":
    unittest.main()
