from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import single_seed_solver as solver  # noqa: E402


class SingleSeedSolverTests(unittest.TestCase):
    def test_build_start_command(self) -> None:
        self.assertEqual(
            solver.build_start_command("THE_SILENT", 5, "4QTKQ7VBMFX47"),
            "s THE_SILENT 5 4QTKQ7VBMFX47",
        )

    def test_extract_terminal_view_status(self) -> None:
        text = "\n".join(
            (
                "STS AGENT",
                "RUN 20260823-run-001 | IRONCLAD A0 | Act 3 Floor 51 | VICTORY",
            )
        )
        self.assertEqual(
            solver.view_status(text),
            {
                "run_id": "20260823-run-001",
                "screen": "VICTORY",
                "act": "3",
                "floor": "51",
            },
        )

    def test_top_candidate_requires_structured_candidate(self) -> None:
        self.assertEqual(
            solver.top_candidate({"decision": {"model_candidates": [{"action_id": "END_TURN"}]}}),
            {"action_id": "END_TURN"},
        )
        self.assertIsNone(solver.top_candidate({"decision": {"model_candidates": []}}))

    def test_autoplay_protocol_cycle(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        received: list[dict[str, object]] = []
        server_error: list[BaseException] = []

        def send_packet(writer, packet: dict[str, object]) -> None:
            writer.write(json.dumps(packet) + "\n")
            writer.flush()

        def receive_packet(reader) -> None:
            line = reader.readline()
            self.assertTrue(line)
            received.append(json.loads(line))

        def serve() -> None:
            try:
                connection, _ = server.accept()
                with connection:
                    reader = connection.makefile("r", encoding="utf-8", newline="\n")
                    writer = connection.makefile("w", encoding="utf-8", newline="\n")
                    with reader, writer:
                        send_packet(writer, {"type": "hello"})
                        send_packet(writer, {"type": "view", "in_game": False, "text": "STS AGENT\n状态：主菜单"})
                        receive_packet(reader)
                        send_packet(
                            writer,
                            {
                                "type": "view",
                                "in_game": True,
                                "text": "RUN fake-run | IRONCLAD A0 | Act 1 Floor 1 | NONE",
                                "decision": {"model_candidates": [{"action_id": "PLAY:card=1"}]},
                            },
                        )
                        receive_packet(reader)
                        send_packet(
                            writer,
                            {
                                "type": "view",
                                "in_game": True,
                                "text": "RUN fake-run | IRONCLAD A0 | Act 1 Floor 5 | GAME_OVER",
                                "decision": {"model_candidates": []},
                            },
                        )
                        receive_packet(reader)
                        send_packet(writer, {"type": "view", "in_game": False, "text": "STS AGENT\n状态：主菜单"})
            except BaseException as exc:  # Surface server thread failures in the test.
                server_error.append(exc)
            finally:
                server.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                seed="4QTKQ7VBMFX47",
                character="IRONCLAD",
                ascension=0,
                episodes=1,
                max_actions=10,
                max_repeated_views=3,
                action_delay=0.0,
                host="127.0.0.1",
                port=port,
                connect_timeout=2.0,
                report=str(Path(temp_dir) / "report.jsonl"),
                no_report=False,
                verbose=False,
            )
            self.assertEqual(solver.run_solver(args), 0)
            report = (Path(temp_dir) / "report.jsonl").read_text(encoding="utf-8")
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(server_error)
        self.assertEqual([item["text"] for item in received], ["s IRONCLAD 0 4QTKQ7VBMFX47", "a", "x"])
        self.assertTrue(all(item.get("autoplay") is True for item in received))
        self.assertIn('"event":"episode_terminal_seen"', report)
        self.assertIn('"event":"solver_complete"', report)


if __name__ == "__main__":
    unittest.main()
