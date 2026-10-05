from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.recording.communication_bridge import CommunicationBridgeRecorder


def state(screen: str, *, floor: int = 1, room_phase: str = "COMPLETE", seq: int = 1) -> dict:
    return {
        "recorder_state_id": f"java:state-{seq}",
        "recorder_state_seq": seq,
        "recorder_session": "java",
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "wait"],
        "game_state": {
            "screen_type": screen,
            "screen_name": screen,
            "room_phase": room_phase,
            "floor": floor,
            "act": 1,
            "class": "IRONCLAD",
            "ascension_level": 0,
            "choice_list": ["shop"],
            "screen_state": {"next_nodes": [{"x": 1, "y": 0}]},
        },
    }


class CommunicationBridgeTests(unittest.TestCase):
    def test_transaction_uses_state_before_marker_and_waits_for_after(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            recorder = CommunicationBridgeRecorder(tmp_path)
            recorder.start()
            try:
                recorder.ingest_state(state("SHOP_ROOM"))
                recorder.ingest_marker(
                    {"type": "run_started", "run_id": "run-1", "character": "IRONCLAD", "ascension": 0, "seed": "ABC123"}
                )
                recorder.ingest_marker(
                    {"type": "action_begin", "run_id": "run-1", "transaction_id": "tx-1", "expected_screen": "SHOP_ROOM", "before_state_id": "java:state-1"}
                )
                recorder.ingest_marker(
                    {"type": "action_accepted", "run_id": "run-1", "transaction_id": "tx-1", "action": {"id": "OPEN:SHOP", "kind": "shop_opened"}}
                )
                self.assertIn("tx-1", recorder.transactions)
                recorder.ingest_state(state("SHOP_SCREEN", room_phase="INCOMPLETE", seq=2))
                self.assertNotIn("tx-1", recorder.transactions)
            finally:
                recorder.close()

            traces = list(tmp_path.glob("human-*.jsonl"))
            self.assertTrue(traces)
            record = json.loads(traces[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["step_id"], "tx-1")
            self.assertEqual(record["screen_type"], "SHOP_ROOM")
            self.assertEqual(record["post_state"]["screen_type"], "SHOP_SCREEN")

    def test_unmatched_transaction_is_not_silently_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            recorder = CommunicationBridgeRecorder(tmp_path)
            recorder.start()
            try:
                recorder.ingest_state(state("MAP"))
                recorder.ingest_marker(
                    {"type": "action_begin", "run_id": "run-2", "transaction_id": "tx-2", "expected_screen": "SHOP_SCREEN"}
                )
                recorder.ingest_marker(
                    {"type": "action_accepted", "run_id": "run-2", "transaction_id": "tx-2", "action": {"id": "SHOP:PURGE", "kind": "shop_purge_opened"}}
                )
                self.assertIn("tx-2", recorder.transactions)
                log = next(tmp_path.glob("communication-bridge-*.jsonl"))
            finally:
                recorder.close()
            log = next(tmp_path.glob("communication-bridge-*.jsonl"))
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(entry.get("type") == "transaction_unmatched" for entry in entries))
            self.assertFalse(list(tmp_path.glob("human-*.jsonl")))
