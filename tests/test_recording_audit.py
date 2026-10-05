from __future__ import annotations

import json
import queue
import socket
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from sts_agent.recording.communication_bridge import CommunicationBridgeRecorder, MarkerServer, _state_message
from sts_agent.recording.trajectory_assembler import ActionRecorderEnricher, resolve_recorded_action
from sts_agent.runtime.controller import main as controller_main, normalize_screen_state, semantic_actions
from tests.test_communication_bridge import state


class RecordingAuditTests(unittest.TestCase):
    def test_controller_dispatches_human_collection_to_passive_bridge(self):
        with patch("sys.argv", ["controller", "--config", "config/agent.json"]), \
                patch("sts_agent.runtime.controller.load_agent_config", return_value={
                    "framework": {"learning_method": "collect_human"}}), \
                patch("sts_agent.recording.communication_bridge.main", return_value=0) as bridge_main:
            self.assertEqual(controller_main(), 0)
            bridge_main.assert_called_once_with()

    def test_selection_reason_and_confirmation_card_survive_normalization(self):
        hand = normalize_screen_state("HAND_SELECT", {"hand": [], "selected": [],
                                      "selection_reason": "选择一张手牌消耗", "up_to": True})
        self.assertEqual(hand["selection_instruction"], "选择一张手牌消耗")
        grid = normalize_screen_state("GRID", {"cards": [], "for_transform": True, "num_cards": 3,
                                      "selection_reason": "选择3张牌变化", "confirm_screen_up": True,
                                      "confirmation_card": {"id": "Strike_R", "name": "打击", "uuid": "u"}})
        self.assertEqual(grid["selection_instruction"], "选择3张牌变化")
        self.assertTrue(grid["confirm_screen_up"])
        self.assertEqual(grid["confirmation_card"]["uuid"], "u")

    def test_delayed_pre_state_is_matched_by_id_not_receipt_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = CommunicationBridgeRecorder(Path(directory))
            bridge.start()
            bridge.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx",
                                  "expected_screen": "SHOP_ROOM", "before_state_id": "java:state-1",
                                  "timestamp_ms": 1})
            bridge.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": "tx",
                                  "action": {"id": "OPEN:SHOP", "kind": "shop_opened"}})
            bridge.ingest_state(state("SHOP_SCREEN", seq=2))
            bridge.ingest_state(state("SHOP_ROOM", seq=1))
            bridge.close()
            record = json.loads(next(Path(directory).glob("human-*.jsonl")).read_text().splitlines()[0])
            self.assertEqual(record["state_match"]["match_status"], "exact_state_id")
            self.assertEqual(record["pre_state"]["screen_type"], "SHOP_ROOM")
            self.assertEqual(record["post_state"]["screen_type"], "SHOP_SCREEN")

    def test_two_channels_deduplicate_same_state(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = CommunicationBridgeRecorder(Path(directory))
            original = bridge.ingest_state(state("MAP"))
            bridge.ingest_marker({"type": "state_published", "message": state("MAP")})
            self.assertEqual(len(bridge.states), 1)
            self.assertIs(original, bridge.ingest_state(state("MAP")))
            bridge.close()

    def test_exact_id_cannot_override_wrong_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = CommunicationBridgeRecorder(Path(directory))
            bridge.ingest_state(state("MAP", floor=1))
            bridge.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx",
                                  "before_state_id": "java:state-1", "context": {"floor": 2}})
            self.assertIsNone(bridge.transactions["tx"].before)
            self.assertEqual(bridge.transactions["tx"].before_status, "unmatched_context")
            bridge.close()

    def test_menu_has_no_fabricated_game_state(self):
        message = _state_message({"in_game": False, "available_commands": ["start"]})
        self.assertFalse(message["in_game"])
        self.assertNotIn("in_game", message["game_state"])

    def test_raw_state_strips_seed_and_non_map_graph_only(self):
        raw = state("EVENT")
        raw["game_state"].update(seed=123, map=[{"x": 1}], custom_mod_field={"value": 5})
        message = _state_message(raw)
        self.assertIsNone(message["game_state"]["map"])
        self.assertNotIn("seed", message["game_state"])
        self.assertEqual(message["game_state"]["custom_mod_field"], {"value": 5})
        self.assertEqual(raw["game_state"]["seed"], 123)

    def test_recording_does_not_expand_smith_or_hide_reentry(self):
        message = state("REST")
        message["game_state"]["choice_list"] = ["rest", "smith", "toke"]
        actions = semantic_actions(message, fine_grained=True)
        self.assertEqual([a["id"] for a in actions], ["CHOOSE:index=0", "CHOOSE:index=1", "CHOOSE:index=2"])
        message = state("SHOP_ROOM")
        self.assertEqual(semantic_actions(message, fine_grained=True)[0]["id"], "CHOOSE:index=0")

    def test_confirm_maps_to_real_command_and_wrong_entity_is_not_guessed(self):
        for identifier in ("SELECT_CARDS:CONFIRM", "SELECT_CARDS:HAND_CONFIRM"):
            action, valid, _ = resolve_recorded_action({"id": identifier}, [{"id": "CONFIRM"}])
            self.assertTrue(valid)
            self.assertEqual(action["id"], "CONFIRM")
        _, valid, _ = resolve_recorded_action(
            {"id": "REWARD:TAKE:POTION", "kind": "reward_potion_claimed"},
            [{"id": "CHOOSE:index=0", "kind": "choose", "label": "Gold: 25"}],
        )
        self.assertFalse(valid)

    def test_preview_card_uuid_resolves_without_selected_cards(self):
        action, valid, _ = resolve_recorded_action(
            {"id": "SELECT_CARDS:grid:b;", "kind": "card_selection_changed", "screen": "grid", "card_uuid": "b"},
            [{"id": "CHOOSE:index=0"}, {"id": "CHOOSE:index=1"}],
            {"screen_state": {"cards": [{"uuid": "a"}, {"uuid": "b"}], "selected_cards": []}},
        )
        self.assertTrue(valid)
        self.assertEqual(action["id"], "CHOOSE:index=1")

    def test_positional_play_requires_same_card_uuid(self):
        _, valid, reason = resolve_recorded_action(
            {"id": "PLAY:card=1", "card_uuid": "wrong"},
            [{"id": "PLAY:card=1", "card_index": 1}],
            {"combat_state": {"hand": [{"uuid": "correct"}]}},
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "card_identity_mismatch")

    def test_idle_and_reconnected_marker_client_does_not_kill_server(self):
        markers = queue.Queue()
        server = MarkerServer("127.0.0.1", 0, markers)
        server.start()
        try:
            deadline = time.monotonic() + 3
            while (server.server is None or server.server.getsockname()[1] == 0) and time.monotonic() < deadline:
                time.sleep(0.01)
            port = server.server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                time.sleep(1.1)  # Previously exceeded the text-stream timeout.
                client.sendall(b'{"type":"hello"}\n')
                self.assertEqual(markers.get(timeout=2)["type"], "hello")
            with socket.create_connection(("127.0.0.1", port)) as client:
                client.sendall(b'{"type":"run_started"}\n')
                self.assertEqual(markers.get(timeout=2)["type"], "run_started")
            self.assertTrue(server.is_alive())
        finally:
            server.close()
            server.join(timeout=2)
        self.assertFalse(server.is_alive())

    def test_terminal_result_is_written_even_without_leaving_end_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ActionRecorderEnricher(Path(directory))
            writer.consume({"type": "run_started", "run_id": "r", "character": "IRONCLAD", "seed": "0", "ascension": 0})
            writer.consume({"type": "run_finished", "run_id": "r", "reason": "victory", "terminal": True})
            writer.close()
            record = json.loads(next(Path(directory).glob("human-*.jsonl")).read_text().splitlines()[0])
            self.assertEqual(record["reason"], "victory")
            self.assertTrue(record["terminal"])

    def test_save_resume_restores_recent_history_and_appends_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            started = {"type": "run_started", "run_id": "r", "character": "IRONCLAD", "seed": "123", "ascension": 0}
            writer = ActionRecorderEnricher(Path(directory))
            writer.consume(started)
            before = state("SHOP_ROOM")
            writer.consume({"type": "action_observed", "run_id": "r", "step_id": "tx",
                            "observation_before": before["game_state"], "raw_message": before,
                            "chosen_action": {"id": "OPEN:SHOP", "kind": "shop_opened"},
                            "state_match": {"match_status": "exact_state_id"}})
            after = state("SHOP_SCREEN", seq=2)
            writer.consume({"type": "step_resolved", "run_id": "r", "step_id": "tx",
                            "observation_after": after["game_state"], "raw_message": after,
                            "resolution": "stable_state"})
            path = writer.runs["r"].output_path
            history = writer.runs["r"].action_history.copy()
            writer.close()
            self.assertEqual(len(history), 1)
            resumed = ActionRecorderEnricher(Path(directory))
            resumed.consume(started)
            self.assertEqual(resumed.runs["r"].output_path, path)
            self.assertEqual(resumed.runs["r"].action_history, history)
            resumed.consume({"type": "run_finished", "run_id": "r", "reason": "death", "terminal": True})
            resumed.close()
            self.assertEqual(len(list(Path(directory).glob("human-*.jsonl"))), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_occupied_marker_port_is_reported_at_startup(self):
        with socket.create_server(("127.0.0.1", 0)) as occupied:
            server = MarkerServer("127.0.0.1", occupied.getsockname()[1], queue.Queue())
            server.start()
            self.assertTrue(server.ready_event.wait(timeout=2))
            server.join(timeout=2)
            self.assertIsNotNone(server.startup_error)
            self.assertFalse(server.is_alive())


if __name__ == "__main__":
    unittest.main()
