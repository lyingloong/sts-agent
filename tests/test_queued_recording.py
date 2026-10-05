from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.recording.communication_bridge import CommunicationBridgeRecorder
from tests.test_communication_bridge import state


def combat(seq, cards, *, energy=3, executing=False, turn=1):
    message = state("NONE", room_phase="COMBAT", seq=seq)
    message["ready_for_command"] = not executing
    message["available_commands"] = ["play", "end", "potion"]
    message["game_state"].update(action_phase="EXECUTING_ACTIONS" if executing else "WAITING_ON_USER",
                                 deck=[], relics=[], potions=[], gold=0, current_hp=80, max_hp=80,
                                 combat_state={"turn": turn, "player": {"energy": energy, "current_hp": 80},
                                               "hand": cards, "monsters": [{"id": "JawWorm", "current_hp": 40}]})
    return message


def card(uuid, name="Strike_R"):
    return {"id": name, "uuid": uuid, "name": name, "cost": 1,
            "is_playable": True, "has_target": True, "type": "ATTACK"}


def play(uuid, index):
    return {"kind": "play_card", "id": f"PLAY:card={index}:target=0",
            "card_uuid": uuid, "card_id": "Strike_R", "target_index": 0, "target_id": "JawWorm"}


class QueuedRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recorder = CommunicationBridgeRecorder(Path(self.tmp.name))
        self.recorder.start()
        self.recorder.ingest_marker({"type": "run_started", "run_id": "r", "character": "IRONCLAD",
                                     "ascension": 0, "seed": "123"})

    def tearDown(self):
        self.recorder.close()
        self.tmp.cleanup()

    def submit(self, tx, before, action, *, tracked=True):
        self.recorder.ingest_state(before)
        self.recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": tx,
                                    "before_state_id": before["recorder_state_id"], "expected_screen": "NONE"})
        self.recorder.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": tx,
                                    "action": action, "execution_tracking": tracked})

    def execute(self, tx, before, action, *, publish=True):
        if publish:
            self.recorder.ingest_state(before)
        self.recorder.ingest_marker({"type": "action_execution_begin", "run_id": "r", "transaction_id": tx,
                                    "execution_before_state_id": before["recorder_state_id"],
                                    "logical_boundary": True, "action": action, "context": {"floor": 1, "act": 1}})
        self.recorder.ingest_marker({"type": "action_execution_result", "run_id": "r", "transaction_id": tx,
                                    "status": "executed"})

    def settle(self, tx, after):
        self.recorder.ingest_state(after)
        self.recorder.ingest_marker({"type": "action_effects_settled", "run_id": "r", "transaction_id": tx,
                                    "after_state_id": after["recorder_state_id"], "logical_boundary": True})

    def rows(self):
        path = next(Path(self.tmp.name).glob("human-*.jsonl"))
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if '"step_id"' in line]

    def test_rapid_clicks_keep_submission_but_train_execution_index_and_energy(self):
        a, b = card("a"), card("b")
        self.submit("a", combat(1, [a, b]), play("a", 1))
        self.submit("b", combat(2, [a, b], executing=True), play("b", 2))
        self.assertEqual(self.rows(), [])  # Do not settle A merely because B was clicked.
        self.execute("a", combat(3, [a, b], executing=True), play("a", 1))
        self.settle("a", combat(4, [b], energy=2, executing=True))
        self.execute("b", combat(5, [b], energy=2, executing=True), play("b", 1))
        self.settle("b", combat(6, [], energy=1))
        rows = self.rows()
        self.assertEqual([x["step_id"] for x in rows], ["a", "b"])
        row = rows[1]
        self.assertEqual(row["action"]["card_index"], 2)
        self.assertEqual(row["training_action"]["card_index"], 1)
        self.assertEqual(row["pre_state"]["combat"]["player"]["energy"], 3)
        self.assertEqual(row["training_pre_state"]["combat"]["player"]["energy"], 2)
        self.assertTrue(row["usable_for_sft"])
        self.assertTrue(rows[0]["stable_after_action"])  # Exact logical boundary, not animation timer.

    def test_validation_failure_keeps_raw_action_but_excludes_training(self):
        self.submit("a", combat(1, [card("a")], executing=True), play("a", 1))
        self.recorder.ingest_marker({"type": "action_execution_result", "run_id": "r", "transaction_id": "a",
                                    "status": "skipped", "reason": "queue_validation_failed"})
        row = self.rows()[0]
        self.assertFalse(row["usable_for_sft"])
        self.assertEqual(row["training_exclusion_reason"], "execution_not_performed")
        self.assertIsNone(row["training_action"])
        self.assertFalse(row["stable_after_action"])

    def test_cancelled_later_action_does_not_reorder_earlier_action(self):
        self.submit("a", combat(1, [card("a"), card("b")]), play("a", 1))
        self.submit("b", combat(2, [card("a"), card("b")], executing=True), play("b", 2))
        self.recorder.ingest_marker({"type": "action_execution_result", "run_id": "r", "transaction_id": "b", "status": "cancelled"})
        self.assertEqual(self.rows(), [])
        self.execute("a", combat(3, [card("a"), card("b")], executing=True), play("a", 1))
        self.settle("a", combat(4, [card("b")], energy=2))
        self.assertEqual([r["step_id"] for r in self.rows()], ["a", "b"])
        self.assertEqual(self.rows()[1]["execution_status"], "cancelled")

    def test_delayed_execution_state_is_resolved_by_id(self):
        self.submit("a", combat(1, [card("a")]), play("a", 1))
        self.execute("a", combat(2, [card("a")], executing=True), play("a", 1), publish=False)
        self.recorder.ingest_state(combat(3, [], energy=2))
        self.assertIn("a", self.recorder.transactions)
        self.recorder.ingest_state(combat(2, [card("a")], executing=True))
        self.assertEqual(self.rows()[0]["execution_state_match"]["before_state_id"], "java:state-2")
        self.assertTrue(self.rows()[0]["usable_for_sft"])

    def test_execution_state_cannot_be_from_another_floor(self):
        self.submit("a", combat(1, [card("a")]), play("a", 1))
        wrong = combat(2, [card("a")], executing=True)
        wrong["game_state"]["floor"] = 2
        self.execute("a", wrong, play("a", 1))
        self.settle("a", combat(3, [], energy=2))
        self.recorder.close()
        self.assertFalse(self.rows()[0]["usable_for_sft"])

    def test_executed_status_without_execution_snapshot_is_not_trainable(self):
        self.submit("a", combat(1, [card("a")]), play("a", 1))
        self.recorder.ingest_marker({"type": "action_execution_result", "run_id": "r", "transaction_id": "a", "status": "executed"})
        self.recorder.ingest_state(combat(2, [], energy=2))
        self.recorder.close()
        self.assertFalse(self.rows()[0]["usable_for_sft"])

    def test_old_animation_boundary_is_neither_stable_nor_training_pre(self):
        self.submit("a", combat(1, [card("a")]), play("a", 1), tracked=False)
        self.submit("b", combat(2, [card("b")], executing=True), play("b", 1), tracked=False)
        self.recorder.ingest_state(combat(3, [], energy=1))
        a, b = self.rows()
        self.assertFalse(a["stable_after_action"])
        self.assertFalse(b["usable_for_sft"])
        self.assertEqual(b["training_exclusion_reason"], "combat_submission_during_execution_without_execution_state")

    def test_end_turn_uses_execution_energy_and_waits_for_next_turn(self):
        end = {"id": "END_TURN", "kind": "end_turn", "turn": 1}
        self.submit("end", combat(1, [card("a")], energy=2, executing=True), end)
        self.execute("end", combat(2, [], energy=0, executing=True), end)
        after = combat(3, [], energy=0)
        self.recorder.ingest_state(after)
        self.assertIn("end", self.recorder.transactions)
        self.settle("end", combat(4, [card("b")], turn=2))
        row = self.rows()[0]
        self.assertTrue(row["usable_for_sft"])
        self.assertEqual(row["training_action"]["id"], "END_TURN")

    def test_interactive_selection_during_executing_actions_is_valid(self):
        before = combat(1, [card("a")], executing=True)
        before["game_state"].update(screen_type="HAND_SELECT", screen_name="HAND_SELECT",
                                    choice_list=["Strike_R"], screen_state={"hand": [card("a")], "selected": []})
        before["available_commands"] = ["choose", "confirm"]
        before["ready_for_command"] = True
        self.recorder.ingest_state(before)
        self.recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "choose",
                                    "before_state_id": before["recorder_state_id"], "expected_screen": "HAND_SELECT"})
        self.recorder.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": "choose",
                                    "action": {"id": "SELECT_CARDS:hand:a;", "kind": "card_selection_changed",
                                               "screen": "hand", "card_uuid": "a", "selected_cards": [{"card_uuid": "a"}]}})
        self.recorder.ingest_state(combat(2, []))
        self.assertTrue(self.rows()[0]["usable_for_sft"])

    def test_wrong_execution_card_uuid_is_not_trainable(self):
        self.submit("a", combat(1, [card("a"), card("b")]), play("a", 1))
        self.execute("a", combat(2, [card("a"), card("b")], executing=True), play("b", 2))
        self.settle("a", combat(3, [card("b")], energy=2))
        self.assertFalse(self.rows()[0]["usable_for_sft"])


if __name__ == "__main__":
    unittest.main()
