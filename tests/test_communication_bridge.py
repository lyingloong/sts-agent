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
    def test_save_checkpoint_is_metadata_not_a_rejected_or_completed_action(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CommunicationBridgeRecorder(Path(directory))
            recorder.start()
            marker = {"type": "run_checkpoint", "run_id": "r", "segment_id": "a",
                      "checkpoint_id": "cp", "checkpoint_state_id": "java:state-1", "timestamp_ms": 123}
            try:
                recorder.ingest_marker({"type": "run_started", "run_id": "r", "segment_id": "a",
                                        "character": "IRONCLAD", "ascension": 0, "seed": "123"})
                recorder.ingest_state(state("SHOP_ROOM") | {"recorder_segment_id": "a"})
                recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx",
                                        "before_state_id": "java:state-1", "segment_id": "a"})
                recorder.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": "tx",
                                        "action": {"id": "OPEN:SHOP", "kind": "shop_opened"}})
                history = [{"action": "此前动作"}]
                recorder.enricher.runs["r"].action_history[:] = history
                recorder.ingest_marker(marker)
                self.assertIn("tx", recorder.transactions)
                self.assertEqual(recorder.enricher.runs["r"].action_history, history)
            finally:
                recorder.close()
            events = [json.loads(line) for line in next(Path(directory).glob("communication-bridge-*.jsonl")).read_text(encoding="utf-8").splitlines()]
            checkpoints = [e for e in events if e["type"] == "run_checkpoint_marker"]
            self.assertEqual(checkpoints, [marker | {"type": "run_checkpoint_marker"}])
            self.assertFalse(any(e["type"] == "marker_rejected" for e in events))
            rows = [json.loads(line) for line in next(Path(directory).glob("human-*.jsonl")).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step_id"], "tx")

    def test_duplicate_card_rewards_with_source_id_are_usable_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CommunicationBridgeRecorder(Path(directory))
            recorder.start()
            try:
                recorder.ingest_marker({"type": "run_started", "run_id": "r", "character": "IRONCLAD",
                                        "ascension": 0, "seed": "123", "segment_id": "a"})
                before = state("COMBAT_REWARD") | {"recorder_segment_id": "a"}
                before["game_state"].update(choice_list=["card", "card"], screen_state={"rewards": [
                    {"reward_type": "CARD", "reward_id": "a", "reward_index": 0},
                    {"reward_type": "CARD", "reward_id": "b", "reward_index": 1}]})
                recorder.ingest_state(before)
                recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx",
                                        "before_state_id": "java:state-1", "segment_id": "a"})
                recorder.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": "tx",
                                        "segment_id": "a", "action": {"id": "REWARD:TAKE:CARD", "kind": "reward_card_claimed",
                                        "reward_type": "CARD", "reward_id": "b", "reward_index": 1}})
                after = state("CARD_REWARD", seq=2) | {"recorder_segment_id": "a"}
                after["game_state"].update(screen_state={"cards": [], "source_reward_id": "b"})
                recorder.ingest_state(after)
            finally:
                recorder.close()
            row = json.loads(next(Path(directory).glob("human-*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["usable_for_sft"])
            self.assertEqual(row["selected_action_id"], "CHOOSE:index=1")
            self.assertEqual(row["raw_message_before"]["recorder_segment_id"], "a")
            self.assertEqual(row["training_policy_observation"]["screen_state"]["rewards"][1]["reward_id"], "b")

    def test_explicit_state_id_from_other_segment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CommunicationBridgeRecorder(Path(directory))
            recorder.start()
            try:
                recorder.ingest_state(state("SHOP_ROOM") | {"recorder_segment_id": "a"})
                recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx",
                                        "before_state_id": "java:state-1", "segment_id": "b"})
                self.assertEqual(recorder.transactions["tx"].before_status, "unmatched_segment")
            finally:
                recorder.close()
    def test_resume_boundary_closes_pending_without_new_state_and_keeps_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CommunicationBridgeRecorder(Path(directory))
            recorder.start()
            metadata = {"run_id": "r", "character": "IRONCLAD", "ascension": 0, "seed": "123", "act": 1}
            try:
                recorder.ingest_marker(metadata | {"type": "run_started", "segment_id": "a"})
                recorder.ingest_state(state("SHOP_ROOM") | {"recorder_segment_id": "a"})
                recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx-1",
                                        "before_state_id": "java:state-1", "segment_id": "a"})
                recorder.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": "tx-1",
                                        "segment_id": "a", "action": {"id": "OPEN:SHOP", "kind": "shop_opened"}})
                self.assertIn("tx-1", recorder.transactions)
                recorder.ingest_marker(metadata | {"type": "run_resumed", "segment_id": "b", "checkpoint_id": "cp"})
                self.assertNotIn("tx-1", recorder.transactions)
                self.assertEqual(recorder.enricher.runs["r"].action_history, [])
            finally:
                recorder.close()
            files = list(Path(directory).glob("human-*.jsonl"))
            self.assertEqual(len(files), 1)
            rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["segment_id"], "a")
            self.assertIsNone(rows[0]["raw_message_after"])
            self.assertEqual(rows[1]["type"], "run_resumed")
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
