from __future__ import annotations

import copy
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sts_agent.runtime import controller
from sts_agent.runtime.batch_client import BatchClientGuard, client_status


class BatchStatusTests(unittest.TestCase):
    def test_menu_capability_is_explicit_not_inferred_from_null_scene(self):
        api = SimpleNamespace(run_summaries={}, output_dir=Path("data/test"), model="test-model")
        guard = BatchClientGuard()
        menu = {"in_game": False, "ready_for_command": True}
        status = client_status(api, menu, {"start", "state"}, None, guard)
        self.assertTrue(status["can_start_run"])
        self.assertEqual(status["mode"], "baseline_inference")
        self.assertTrue(Path(status["experiment_dir"]).is_absolute())
        for message, commands in (({**menu, "in_game": True}, {"start"}),
                                  ({**menu, "ready_for_command": False}, {"start"}),
                                  (menu, {"state"}), ({"ready_for_command": True}, {"start"})):
            self.assertFalse(client_status(api, message, commands, None, guard)["can_start_run"])

    def test_summary_is_detached_and_large_terminal_observation_omitted(self):
        summary = {"terminal": True, "terminal_state": {"deck": ["large"]},
                   "run_config": {"seed": "ABC", "internal_seed": 123},
                   "evaluation": {"progress_score": 15}}
        api = SimpleNamespace(run_summaries={"run-1": summary}, output_dir=None, model="test")
        status = client_status(api, {}, set(), "run-1", None)
        self.assertNotIn("terminal_state", status["run_summary"])
        status["run_summary"]["run_config"]["seed"] = "CHANGED"
        self.assertEqual(summary["run_config"]["seed"], "ABC")
        self.assertIsNone(client_status(api, {}, set(), "unknown", None)["run_summary"])


class BatchGuardTests(unittest.TestCase):
    def pair(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        return local, peer

    def test_ordinary_ui_disconnect_does_not_pause(self):
        local, peer = self.pair()
        peer.close()
        guard = BatchClientGuard()
        self.assertIsNone(guard.error(local))
        guard.disconnected()
        self.assertIsNone(guard.failure)

    def test_batch_eof_pauses_until_explicit_recovery(self):
        local, peer = self.pair()
        guard = BatchClientGuard()
        guard.attach()
        peer.close()
        self.assertIn("已断开", guard.error(local))
        self.assertFalse(guard.attached)
        self.assertEqual(guard.error(None), guard.failure)
        guard.detach()
        self.assertIsNone(guard.error(None))

    def test_peek_does_not_consume_next_client_command(self):
        local, peer = self.pair()
        guard = BatchClientGuard()
        guard.attach()
        peer.sendall(b"command\n")
        self.assertIsNone(guard.error(local))
        self.assertEqual(local.recv(100), b"command\n")

    def test_clean_detach_and_missing_socket(self):
        guard = BatchClientGuard()
        guard.attach()
        self.assertIsNotNone(guard.error(None))
        guard.attach()
        guard.detach()
        guard.disconnected()
        self.assertIsNone(guard.failure)


class BatchControllerTests(unittest.TestCase):
    MENU = {"in_game": False, "ready_for_command": True, "available_commands": ["start", "state"]}
    EVENT = {"in_game": True, "ready_for_command": True, "available_commands": ["choose", "state"],
             "game_state": {"screen_type": "EVENT", "choice_list": ["Take", "Leave"], "screen_state": {}}}

    def setUp(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        self.peer = peer
        self.bridge = Mock(connection=local, batch_guard=BatchClientGuard())
        self.bridge.receive.return_value = {"type": "input", "text": "s"}
        self.bridge.send.side_effect = lambda packet: self.packets.append(copy.deepcopy(packet))
        self.packets = []
        api = SimpleNamespace(run_summaries={}, output_dir=Path("data/test"), model="test", pending="cached")
        self.policy = Mock(runtime="remote_api", screens=("ALL",), api=api)
        self.policy.status_decision.return_value = {"mode": "baseline_inference"}
        self.policy.requesting_decision.return_value = {"mode": "baseline_inference", "assist_status": "请求中"}
        self.environment = patch.object(controller, "environment_auto_action", return_value=None)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.sleep = patch.object(controller.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_attach_ack_then_clean_detach_at_ready_menu(self):
        self.bridge.receive.side_effect = [
            {"type": "baseline_batch_attach", "protocol_version": 1},
            {"type": "baseline_batch_detach"},
            {"type": "input", "text": "s IRONCLAD 0 ABC"},
        ]
        result = controller.ask_action(self.bridge, self.MENU, None, policy=self.policy)
        self.assertEqual(result[0], "START IRONCLAD 0 ABC")
        self.assertEqual([packet["baseline"]["batch_attached"] for packet in self.packets], [False, True, False])
        self.policy.rank.assert_not_called()
        self.peer.close()
        self.assertIsNone(self.bridge.batch_guard.error(self.bridge.connection))

    def test_attach_wrong_protocol_is_not_acknowledged(self):
        self.bridge.receive.side_effect = [
            {"type": "baseline_batch_attach", "protocol_version": "1"},
            {"type": "input", "text": "s IRONCLAD 0 ABC"},
        ]
        controller.ask_action(self.bridge, self.MENU, None, policy=self.policy)
        self.assertFalse(self.bridge.batch_guard.attached)
        self.assertIn("协议版本不匹配", self.packets[-1]["baseline"]["controller_error"])

    def test_cannot_attach_in_game_or_while_controller_paused(self):
        for message, error in ((self.EVENT, "existing error"), (self.MENU, "existing error")):
            self.bridge.receive.side_effect = [
                {"type": "baseline_batch_attach", "protocol_version": 1},
                {"type": "input", "text": "state"},
            ]
            result = controller.ask_action(self.bridge, message, None, error=error, policy=self.policy)
            self.assertEqual(result[0], "STATE")
            self.assertFalse(self.bridge.batch_guard.attached)
        self.policy.rank.assert_not_called()

    def test_disconnect_before_auto_prevents_environment_and_model_actions(self):
        self.bridge.batch_guard.attach()
        self.peer.close()
        environment_action = {"id": "CHOOSE:index=0", "command": "CHOOSE 0"}
        with patch.object(controller, "environment_auto_action", return_value=(environment_action, {})) as auto:
            result = controller.ask_action(self.bridge, self.EVENT, "run-1", policy=self.policy)
        self.assertEqual(result[0], "STATE")  # Explicit mocked human refresh, never a model action.
        auto.assert_not_called()
        self.policy.rank.assert_not_called()
        self.assertIsNone(self.policy.api.pending)
        self.assertIn("已断开", self.packets[-1]["decision"]["assist_reason"])

    def test_disconnect_during_inference_discards_pending_instead_of_auto_revalidation(self):
        self.bridge.batch_guard.attach()

        def rank(*args, **kwargs):
            self.peer.close()
            self.policy.api.pending = "new answer"
            return {"mode": "baseline_inference", "model_action": "CHOOSE:index=0",
                    "model_candidates": [{"action_id": "CHOOSE:index=0"}], "api": {"revalidated": False}}

        self.policy.rank.side_effect = rank
        result = controller.ask_action(self.bridge, self.EVENT, "run-1", policy=self.policy)
        self.assertEqual(result[0], "STATE")
        self.bridge.receive.assert_called_once()  # Must wait for explicit input, not return autonomous STATE.
        self.assertIsNone(self.policy.api.pending)
        view = next(packet for packet in self.packets if packet["type"] == "view")
        self.assertIsNone(view["decision"]["model_action"])
        self.assertIn("已断开", view["decision"]["assist_reason"])

    def test_disconnect_during_automatic_delay_prevents_committing_action(self):
        self.bridge.batch_guard.attach()
        self.policy.rank.return_value = {"mode": "baseline_inference", "model_action": "CHOOSE:index=0",
                                         "api": {"revalidated": True}}
        with patch.object(controller.time, "sleep", side_effect=lambda delay: self.peer.close()):
            result = controller.ask_action(self.bridge, self.EVENT, "run-1", policy=self.policy)
        self.assertEqual(result[0], "STATE")
        self.bridge.receive.assert_called_once()
        self.assertIsNone(self.policy.api.pending)


if __name__ == "__main__":
    unittest.main()
