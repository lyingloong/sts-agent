from __future__ import annotations

import json
import io
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sts_agent.codex_mcp import ControllerClient, observation_payload
from sts_agent.config import load_agent_config
from sts_agent.runtime import controller


class CodexMcpTests(unittest.TestCase):
    def test_first_start_creates_configured_output_directory(self) -> None:
        config = load_agent_config(mode="collect_codex")
        config["framework"]["control_mode"] = "codex"
        config["framework"]["learning_method"] = "collect_codex"
        config["codex"]["output_dir"] = "data/trajectories/new-codex-directory"
        menu = {"available_commands": ["start", "state"], "ready_for_command": True, "in_game": False}
        bridge = Mock()
        bridge.receive.return_value = {"type": "input", "text": "s IRONCLAD 0"}
        evolution = Mock(enabled=False)
        policy = Mock(enabled=False, runtime="disabled", screens=[])
        teacher = Mock(enabled=False, prompt_evolution=evolution)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = io.StringIO()
            with (
                patch.object(controller, "PROJECT_ROOT", root),
                patch.object(controller, "load_agent_config", return_value=config),
                patch.object(controller, "TerminalBridge", return_value=bridge),
                patch.object(controller, "PromptEvolutionManager", return_value=evolution),
                patch.object(controller, "PolicyClient", return_value=policy),
                patch.object(controller, "TeacherClient", return_value=teacher),
                patch.object(controller, "PosthocFeedbackClient", return_value=Mock(enabled=False)),
                patch("sys.argv", ["controller"]),
                patch("sys.stdin", io.StringIO(json.dumps(menu) + "\n")),
                patch("sys.stdout", stdout),
            ):
                self.assertEqual(controller.main(), 0)
            self.assertIn("START IRONCLAD 0", stdout.getvalue())
            output = root / config["codex"]["output_dir"]
            files = list(output.glob("*.jsonl"))
            self.assertEqual(len(files), 2)
            records = [json.loads(line) for file in files for line in file.read_text(encoding="utf-8").splitlines()]
            start = next(item for item in records if item.get("event") == "run_start")
            self.assertEqual(start["run_config"]["collection_mode"], "codex")

    def test_reset_during_read_disconnects_client_not_controller(self) -> None:
        bridge = controller.TerminalBridge("127.0.0.1", 0)
        try:
            bridge.connection = Mock()
            bridge.connection.recv.side_effect = ConnectionResetError("client exited")
            with patch.object(controller.select, "select", return_value=([bridge.connection], [], [])):
                self.assertIsNone(bridge.receive())
            self.assertIsNone(bridge.connection)
            self.assertEqual(bridge.input_buffer, bytearray())
            self.assertGreaterEqual(bridge.server.fileno(), 0)
        finally:
            bridge.close()

    def test_reset_during_view_republishes_decision(self) -> None:
        bridge = Mock()
        bridge.send.side_effect = [ConnectionResetError("client exited"), None]
        bridge.receive.return_value = {"type": "input", "text": "s IRONCLAD 0"}
        command, _, _ = controller.ask_action(
            bridge, {"available_commands": ["start"], "ready_for_command": True, "in_game": False},
            None, config={"framework": {"control_mode": "codex"}},
        )
        self.assertEqual(command, "START IRONCLAD 0")
        self.assertEqual(bridge.send.call_count, 2)
        bridge.close_client.assert_called_once()

    def test_closed_client_cannot_serve_cached_observation(self) -> None:
        client = self.client_with_view({"state_id": "stale"})
        client.closed = True
        client.error = None
        with self.assertRaises(ConnectionError):
            client.wait_for_view(1)

    def test_controller_menu_without_game_state_can_start_via_mcp(self) -> None:
        # The real CommunicationMod menu shape has no screen or game_state.
        message = {"available_commands": ["start", "state"], "ready_for_command": True, "in_game": False}
        bridge = Mock()
        bridge.receive.return_value = {"type": "input", "text": "s IRONCLAD 0"}
        command, _, _ = controller.ask_action(
            bridge, message, None, config={"framework": {"control_mode": "codex"}},
        )
        self.assertEqual(command, "START IRONCLAD 0")
        packet = bridge.send.call_args.args[0]
        payload = observation_payload(packet)
        self.assertEqual(payload["game_status"], "ready_to_start")
        self.assertTrue(payload["can_start_run"])
        self.assertEqual(payload["legal_actions"], [])
        self.assertIsNone(payload["observation"]["screen_type"])
        self.assertEqual(payload["available_commands"], ["start", "state"])

        client = self.client_with_view(packet)
        submitted = client.start_run("IRONCLAD", 0, "", 1)
        client._send_text.assert_called_once_with("s IRONCLAD 0")
        self.assertEqual(submitted["state_id"], payload["state_id"])
        self.assertIsNone(client.current_view)

    @staticmethod
    def client_with_view(packet: dict) -> ControllerClient:
        client = ControllerClient.__new__(ControllerClient)
        client.condition = threading.Condition()
        client.current_view = packet
        client.closed = False
        client._send_text = Mock()
        return client

    def test_empty_actions_alone_do_not_authorize_start(self) -> None:
        menu = {
            "state_id": "menu-1", "in_game": False, "observation": {}, "legal_actions": [],
            "available_commands": ["start", "state"], "ready_for_command": True,
        }
        for overrides in (
            {"available_commands": []}, {"available_commands": ["state"]},
            {"ready_for_command": False}, {"ready_for_command": None},
            {"in_game": True}, {"in_game": None}, {"error": "game error"},
        ):
            with self.subTest(overrides=overrides):
                packet = {**menu, **overrides}
                self.assertFalse(observation_payload(packet)["can_start_run"])
                client = self.client_with_view(packet)
                with self.assertRaises(ValueError):
                    client.start_run("IRONCLAD", 0, "", 1)
                client._send_text.assert_not_called()
                self.assertIs(client.current_view, packet)

    def test_observation_payload_requires_structured_codex_packet(self) -> None:
        packet = {
            "type": "view",
            "state_id": "state-1",
            "in_game": True,
            "observation": {"screen_type": "MAP"},
            "legal_actions": [{"id": "A1", "label": "前往"}],
        }
        result = observation_payload(packet)
        self.assertEqual(result["state_id"], "state-1")
        self.assertEqual(result["observation"]["screen_type"], "MAP")
        self.assertEqual(result["legal_actions"][0]["id"], "A1")
        self.assertIn("already been executed", result["automatic_actions_note"])

    def test_controller_client_rejects_stale_or_illegal_actions(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()
        received: list[dict[str, object]] = []

        def serve() -> None:
            connection, _ = listener.accept()
            with connection:
                reader = connection.makefile("r", encoding="utf-8")
                writer = connection.makefile("w", encoding="utf-8")
                writer.write(json.dumps({"type": "hello"}) + "\n")
                writer.write(json.dumps({
                    "type": "view",
                    "state_id": "state-1",
                    "in_game": True,
                    "observation": {"screen_type": "MAP"},
                    "legal_actions": [{"id": "A1", "command": "CHOOSE 0"}],
                }) + "\n")
                writer.flush()
                raw = reader.readline()
                received.append(json.loads(raw))
                writer.write(json.dumps({
                    "type": "view",
                    "state_id": "state-2",
                    "in_game": True,
                    "observation": {"screen_type": "COMBAT"},
                    "legal_actions": [],
                }) + "\n")
                writer.flush()
                threading.Event().wait(0.2)
                reader.close()
                writer.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        client = ControllerClient(host, port, connect_timeout=2)
        try:
            with self.assertRaises(ValueError):
                client.execute_action("A2", "state-1", 2)
            with self.assertRaises(ValueError):
                client.execute_action("A1", "stale", 2)
            client.execute_action("A1", "state-1", 2)
            next_view = client.wait_for_view(2, after_state_id="state-1")
            self.assertEqual(next_view["state_id"], "state-2")
        finally:
            client.close()
            listener.close()
        thread.join(timeout=2)
        self.assertEqual(received[0]["text"], "A1")
        self.assertEqual(received[0]["actor"], "codex")


if __name__ == "__main__":
    unittest.main()
