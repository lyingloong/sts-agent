from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.recording.action_recorder import ActionRecorderEnricher, resolve_recorded_action
from sts_agent.runtime.controller import normalize_state, route_policy_observation


def combat_state() -> dict:
    return {
        "screen_type": "NONE",
        "screen_name": "NONE",
        "is_screen_up": False,
        "room_phase": "COMBAT",
        "action_phase": "WAITING_ON_USER",
        "room_type": "MonsterRoom",
        "floor": 1,
        "act": 1,
        "class": "IRONCLAD",
        "ascension_level": 0,
        "current_hp": 80,
        "max_hp": 80,
        "gold": 99,
        "act_boss": "TheGuardian",
        "map": [{"x": 0, "y": 0, "symbol": "M", "children": []}],
        "deck": [{"id": "Strike_R", "name": "打击", "upgrades": 0}],
        "relics": [],
        "potions": [],
        "choice_list": [],
        "combat_state": {
            "turn": 1,
            "hand": [{"id": "Strike_R", "name": "打击", "is_playable": True, "has_target": True}],
            "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "monsters": [{"id": "JawWorm", "name": "大颚虫", "current_hp": 42, "max_hp": 42,
                          "block": 0, "is_gone": False, "powers": [], "intent": "ATTACK",
                          "move_base_damage": 11, "move_adjusted_damage": 11, "move_hits": 1}],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
        },
    }


class ActionRecorderRecordingTests(unittest.TestCase):
    def test_enricher_uses_live_agent_observation_and_action_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            enricher = ActionRecorderEnricher(Path(directory))
            enricher.consume({
                "type": "run_started", "run_id": "run-1", "character": "IRONCLAD",
                "ascension": 0, "seed": "ABC", "act": 1,
            })
            enricher.consume({
                "type": "action_observed", "run_id": "run-1", "step_id": "run-1:1",
                "timestamp_ms": 1, "observation_before": combat_state(),
                "available_actions": {"commands": ["play", "end"], "choices": []},
                "chosen_action": {"id": "END_TURN", "kind": "end_turn"},
            })
            record = enricher.consume({
                "type": "step_resolved", "run_id": "run-1", "step_id": "run-1:1",
                "timestamp_ms": 2, "observation_after": combat_state(), "resolution": "stable_state",
            })
            enricher.close()

            assert record is not None
            self.assertTrue(record["chosen_action_legal"])
            self.assertTrue(record["usable_for_sft"])
            self.assertEqual(record["selected_action_id"], "END_TURN")
            self.assertIn("PLAY:card=1:target=0", {item["id"] for item in record["legal_actions"]})
            self.assertIn("knowledge", record["pre_state"]["combat"]["monsters"][0])
            self.assertNotIn("map", record["policy_observation"])
            output = next(Path(directory).glob("*.jsonl"))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8").splitlines()[0])["step_id"], "run-1:1")

    def test_resolves_game_side_reward_by_agent_choice_label(self) -> None:
        action, legal, reason = resolve_recorded_action(
            {"id": "REWARD:TAKE:GOLD", "kind": "reward_gold_claimed", "gold": 25},
            [
                {"id": "CHOOSE:index=0", "kind": "choose", "label": "Gold: 25"},
                {"id": "PROCEED", "kind": "confirm"},
            ],
        )
        self.assertTrue(legal)
        self.assertEqual(action["id"], "CHOOSE:index=0")
        self.assertEqual(reason, "choice_label")

    def test_resolves_map_click_by_public_node_coordinates(self) -> None:
        action, legal, reason = resolve_recorded_action(
            {"id": "MAP:x=2:y=4", "kind": "map_node_selected", "x": 2, "y": 4},
            [
                {"id": "CHOOSE:index=0", "kind": "choose", "label": "x=1"},
                {"id": "CHOOSE:index=1", "kind": "choose", "label": "x=2"},
            ],
            {"screen_state": {"next_nodes": [{"x": 1, "y": 4}, {"x": 2, "y": 4}]}},
        )
        self.assertTrue(legal)
        self.assertEqual(action["id"], "CHOOSE:index=1")
        self.assertEqual(reason, "map_coordinates")

    def test_normalizes_heart_keys_and_keeps_them_in_combat_policy_state(self) -> None:
        message = {"game_state": combat_state() | {
            "keys": {"ruby": True, "emerald": False, "sapphire": True},
        }}
        normalized = normalize_state(message)
        self.assertEqual(normalized["keys"], {"ruby": True, "emerald": False, "sapphire": True})
        routed = route_policy_observation(normalized)
        self.assertEqual(routed["keys"]["ruby"], True)

    def test_missing_heart_keys_remain_unknown(self) -> None:
        normalized = normalize_state({"game_state": combat_state()})
        self.assertNotIn("keys", normalized)


if __name__ == "__main__":
    unittest.main()
