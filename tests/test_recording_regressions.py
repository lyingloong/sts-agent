from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts_agent.recording.communication_bridge import CommunicationBridgeRecorder, main as bridge_main
from sts_agent.recording.trajectory_assembler import ActionRecorderEnricher, resolve_recorded_action
from tests.test_communication_bridge import state


class RecordingRegressionTests(unittest.TestCase):
    def test_errors_do_not_enter_state_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CommunicationBridgeRecorder(Path(directory))
            self.assertIsNone(recorder.ingest_state({"error": "Missing argument", "ready_for_command": True}))
            self.assertEqual(len(recorder.states), 0)
            recorder.close()

    def test_passive_process_sends_only_handshake_not_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            incoming = [{"in_game": False, "ready_for_command": True, "available_commands": ["start"]},
                        {"error": "Missing argument", "ready_for_command": True}]
            with patch("sys.argv", ["bridge", "--port", "0", "--output-dir", directory]), \
                    patch("sys.stdin", io.StringIO("\n".join(map(json.dumps, incoming)))), \
                    patch("sys.stdout", output):
                self.assertEqual(bridge_main(), 0)
            self.assertEqual(output.getvalue(), "ready\n")

    def test_potion_reward_is_not_confused_with_discarding_held_potion(self):
        before = {"screen_type": "COMBAT_REWARD", "screen_state": {"rewards": [
            {"reward_type": "POTION", "potion": {"id": "Strength Potion"}}, {"reward_type": "CARD"}]}}
        action, valid, reason = resolve_recorded_action(
            {"id": "REWARD:TAKE:POTION", "kind": "reward_potion_claimed", "reward_type": "POTION", "potion_id": "Strength Potion"},
            [{"id": "POTION_DISCARD:slot=0", "kind": "potion_discard", "label": "丢弃药水 格挡药水"},
             {"id": "CHOOSE:index=0", "kind": "choose", "label": "potion"},
             {"id": "CHOOSE:index=1", "kind": "choose", "label": "card"}], before)
        self.assertTrue(valid)
        self.assertEqual(action["id"], "CHOOSE:index=0")
        self.assertEqual(reason, "reward_entry")

    def test_chest_requires_unopened_chest_state(self):
        action = {"id": "CHEST:OPEN:BossChest", "kind": "chest_opened"}
        legal = [{"id": "CHOOSE:index=0", "kind": "choose", "label": "open"},
                 {"id": "PROCEED", "kind": "confirm"}]
        self.assertTrue(resolve_recorded_action(action, legal, {
            "screen_type": "CHEST", "screen_state": {"chest_open": False}})[1])
        self.assertFalse(resolve_recorded_action(action, legal[1:], {
            "screen_type": "CHEST", "screen_state": {"chest_open": True}})[1])

    def test_key_reward_waits_until_key_is_actually_present(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CommunicationBridgeRecorder(Path(directory))
            before = state("COMBAT_REWARD")
            before["game_state"].update(keys={"sapphire": False}, choice_list=["sapphire_key"],
                                        screen_state={"rewards": [{"reward_type": "SAPPHIRE_KEY"}]})
            recorder.ingest_state(before)
            recorder.ingest_marker({"type": "run_started", "run_id": "r", "character": "IRONCLAD", "ascension": 0, "seed": "0"})
            recorder.ingest_marker({"type": "action_begin", "run_id": "r", "transaction_id": "tx", "before_state_id": "java:state-1"})
            recorder.ingest_marker({"type": "action_accepted", "run_id": "r", "transaction_id": "tx", "action": {
                "id": "REWARD:TAKE:SAPPHIRE_KEY", "kind": "reward_sapphire_key_claimed", "reward_type": "SAPPHIRE_KEY"}})
            after = state("COMBAT_REWARD", seq=2)
            after["game_state"].update(keys={"sapphire": False}, choice_list=[], screen_state={"rewards": []})
            recorder.ingest_state(after)
            self.assertIn("tx", recorder.transactions)
            after["recorder_state_id"] = "java:state-3"
            after["recorder_state_seq"] = 3
            after["game_state"]["keys"]["sapphire"] = True
            recorder.ingest_state(after)
            self.assertNotIn("tx", recorder.transactions)
            recorder.close()
            row = json.loads(next(Path(directory).glob("human-*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["observation_after"]["keys"]["sapphire"])

    def test_cancelled_preview_excluded_but_final_reselection_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ActionRecorderEnricher(Path(directory))
            writer.consume({"type": "run_started", "run_id": "r", "character": "IRONCLAD", "ascension": 0, "seed": "0"})
            grid = state("GRID")
            grid["game_state"].update(choice_list=["Strike", "Defend"], screen_state={
                "cards": [{"id": "Strike_R", "uuid": "a"}, {"id": "Defend_R", "uuid": "b"}], "selected_cards": []})

            def step(identifier, action, before, after):
                writer.consume({"type": "action_observed", "run_id": "r", "step_id": identifier,
                                "observation_before": before["game_state"], "raw_message": before,
                                "chosen_action": action, "state_match": {"match_status": "exact_state_id"}})
                writer.consume({"type": "step_resolved", "run_id": "r", "step_id": identifier,
                                "observation_after": after["game_state"], "raw_message": after, "resolution": "next_decision"})

            preview = state("GRID", seq=2)
            preview["available_commands"] = ["confirm", "cancel"]
            preview["game_state"]["screen_state"] = {"confirm_screen_up": True}
            step("wrong", {"id": "SELECT_CARDS:grid:a;", "kind": "card_selection_changed", "card_uuid": "a", "selected_cards": [{"card_uuid": "a"}]}, grid, preview)
            step("cancel", {"id": "RETURN", "kind": "return_button"}, preview, grid)
            step("final", {"id": "SELECT_CARDS:grid:b;", "kind": "card_selection_changed", "card_uuid": "b", "selected_cards": [{"card_uuid": "b"}]}, grid, preview)
            step("confirm", {"id": "SELECT_CARDS:CONFIRM", "kind": "card_selection_confirmed"}, preview, state("EVENT", seq=3))
            writer.close()
            rows = [json.loads(line) for line in next(Path(directory).glob("human-*.jsonl")).read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["step_id"] for row in rows], ["wrong", "cancel", "final", "confirm"])
            self.assertEqual([row["usable_for_sft"] for row in rows], [False, False, True, True])


if __name__ == "__main__":
    unittest.main()
