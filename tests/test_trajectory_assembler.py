from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.recording.trajectory_assembler import ActionRecorderEnricher, resolve_recorded_action
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
    def test_identical_rewards_resolve_by_identity_and_current_index(self):
        state = {"screen_type": "COMBAT_REWARD", "screen_state": {"rewards": [
            {"reward_type": "CARD", "reward_id": "a", "reward_index": 0},
            {"reward_type": "CARD", "reward_id": "b", "reward_index": 1}]}}
        legal = [{"id": f"CHOOSE:index={i}", "kind": "choose"} for i in range(2)]
        raw = {"id": "REWARD:TAKE:CARD", "kind": "reward_card_claimed", "reward_type": "CARD",
               "reward_id": "b", "reward_index": 1}
        action, valid, reason = resolve_recorded_action(raw, legal, state)
        self.assertTrue(valid)
        self.assertEqual((action["id"], reason), ("CHOOSE:index=1", "reward_identity"))
        self.assertFalse(resolve_recorded_action(raw | {"reward_index": 0}, legal, state)[1])
        self.assertFalse(resolve_recorded_action(raw | {"reward_id": "missing"}, legal, state)[1])
        self.assertFalse(resolve_recorded_action(raw | {"reward_type": "GOLD"}, legal, state)[1])
        state["screen_state"]["rewards"] = [{"reward_type": "CARD", "reward_id": "b", "reward_index": 0}]
        self.assertTrue(resolve_recorded_action(raw | {"reward_index": 0}, legal, state)[1])
        self.assertFalse(resolve_recorded_action({k: v for k, v in raw.items() if k not in {"reward_id", "reward_index"}},
                                                legal, {"screen_type": "COMBAT_REWARD", "screen_state": {"rewards": [
                                                    {"reward_type": "CARD"}, {"reward_type": "CARD"}]}})[1])

    def test_child_card_choice_validates_source_reward(self):
        raw = {"id": "CHOOSE:index=0", "kind": "card_reward_selected", "source_reward_id": "a"}
        legal = [{"id": "CHOOSE:index=0", "kind": "choose"}]
        self.assertTrue(resolve_recorded_action(raw, legal, {"screen_state": {"source_reward_id": "a"}})[1])
        self.assertFalse(resolve_recorded_action(raw, legal, {"screen_state": {"source_reward_id": "b"}})[1])

    def test_boss_rewards_and_normalized_sources_keep_identity(self):
        relics = [{"id": "Empty Cage", "name": "空鸟笼", "reward_id": "cage", "reward_index": 0}]
        game = combat_state() | {"screen_type": "BOSS_REWARD", "room_phase": "COMPLETE",
                                 "screen_state": {"relics": relics}}
        obs = normalize_state({"game_state": game})
        self.assertEqual(obs["screen_state"]["relics"][0]["reward_id"], "cage")
        raw = {"id": "CHOOSE:index=0", "kind": "boss_relic_selected", "reward_id": "wrong", "reward_index": 0}
        legal = [{"id": "CHOOSE:index=0", "kind": "choose"}]
        self.assertFalse(resolve_recorded_action(raw, legal, game)[1])
        self.assertTrue(resolve_recorded_action(raw | {"reward_id": "cage"}, legal, game)[1])
        game.update(screen_type="CARD_REWARD", screen_state={"cards": [], "source_reward_id": "cards-a"})
        self.assertEqual(normalize_state({"game_state": game})["screen_state"]["source_reward_id"], "cards-a")

    def test_resume_after_receiver_restart_does_not_reuse_old_history(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"run_id": "r", "character": "IRONCLAD", "ascension": 0, "seed": "123", "act": 1}
            path = Path(directory) / "human-ironclad-A0-seed-123-r.jsonl"
            path.write_text(json.dumps({"accepted": True, "action": {"id": "END_TURN"}, "step_id": "old"}) + "\n",
                            encoding="utf-8")
            receiver = ActionRecorderEnricher(Path(directory))
            receiver.consume(metadata | {"type": "run_resumed", "segment_id": "b"})
            self.assertEqual(receiver.runs["r"].action_history, [])
            receiver.close()
            self.assertEqual(len(list(Path(directory).glob("human-*.jsonl"))), 1)

    def test_resume_appends_one_file_and_resets_roll_back_history(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver = ActionRecorderEnricher(Path(directory))
            metadata = {"run_id": "r", "character": "IRONCLAD", "ascension": 0, "seed": "123", "act": 1}
            receiver.consume(metadata | {"type": "run_started", "segment_id": "a"})
            context = receiver.runs["r"]
            context.action_history.append({"action_id": "old"})
            context.visited_shop_rooms.add((1, 14))
            context.declined_card_reward_keys.add((1, 2))
            receiver.consume(metadata | {"type": "run_resumed", "segment_id": "b", "checkpoint_id": "cp"})
            self.assertEqual(context.action_history, [])
            self.assertEqual(context.visited_shop_rooms, set())
            self.assertEqual(context.declined_card_reward_keys, set())
            receiver.consume({"type": "action_observed", "run_id": "r", "step_id": "s:t", "segment_id": "b",
                              "state_match": {"match_status": "exact_state_id"}, "observation_before": combat_state(),
                              "available_actions": {"commands": ["end"], "choices": []},
                              "chosen_action": {"id": "END_TURN", "kind": "end_turn"}})
            record = receiver.consume({"type": "step_resolved", "run_id": "r", "step_id": "s:t",
                                       "observation_after": combat_state()})
            self.assertEqual(record["segment_id"], "b")
            receiver.close()
            self.assertEqual(len(list(Path(directory).glob("human-*.jsonl"))), 1)
            rows = [json.loads(line) for line in next(Path(directory).glob("human-*.jsonl")).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["type"], "run_resumed")
            self.assertEqual(rows[0]["checkpoint_id"], "cp")

    def test_smith_and_purge_entry_are_fine_grained_choices(self) -> None:
        for screen, prefix, kind, choices in (
            ("REST", "CAMPFIRE:SMITH", "campfire_action", ["rest", "smith"]),
            ("SHOP_SCREEN", "SHOP:PURGE", "shop_purge_opened", ["purge"]),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                receiver = ActionRecorderEnricher(Path(directory))
                before = combat_state() | {
                    "screen_type": screen, "room_phase": "INCOMPLETE",
                    "choice_list": choices, "screen_state": {},
                    "deck": [{"id": "Strike_R", "name": "打击", "uuid": "card-1", "upgrades": 0}],
                }
                selected_id = f"CHOOSE:index={len(choices)-1}"
                # The mod can emit auxiliary lifecycle events, but the receiver
                # must not need any of them to reconstruct the actual decision.
                receiver.consume({"type": "decision_started", "run_id": "r", "decision": prefix})
                receiver.consume({
                    "type": "action_observed", "run_id": "r", "step_id": "r:1",
                    "state_match": {"match_status": "exact_state_id"},
                    "observation_before": before,
                    "available_actions": {"commands": ["choose"], "choices": choices},
                    "chosen_action": {"id": prefix, "kind": kind},
                })
                record = receiver.consume({
                    "type": "step_resolved", "run_id": "r", "step_id": "r:1",
                    "observation_after": before, "resolution": "stable_state",
                })
                receiver.close()
                self.assertIsNotNone(record)
                self.assertTrue(record["usable_for_sft"])
                self.assertEqual(record["action_resolution"], "choice_label")
                self.assertEqual(record["selected_action_id"], selected_id)
                self.assertEqual(record["pre_state"]["screen_type"], screen)
                self.assertEqual(len(receiver.runs["r"].action_history), 1)
                lines = next(Path(directory).glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)

    def test_cancelled_room_decision_creates_no_success_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receiver = ActionRecorderEnricher(Path(directory))
            receiver.consume({"type": "decision_started", "run_id": "r", "decision": "CAMPFIRE:SMITH"})
            receiver.consume({"type": "decision_cancelled", "run_id": "r", "reason": "cancelled_by_player"})
            receiver.close()
            self.assertEqual(receiver.runs["r"].action_history, [])
            self.assertEqual(list(Path(directory).glob("*.jsonl")), [])

    def test_enricher_uses_live_agent_observation_and_action_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            enricher = ActionRecorderEnricher(Path(directory))
            enricher.consume({
                "type": "run_started", "run_id": "run-1", "character": "IRONCLAD",
                "ascension": 0, "seed": "ABC", "act": 1,
            })
            enricher.consume({
                "type": "action_observed", "run_id": "run-1", "step_id": "run-1:1",
                "state_match": {"match_status": "exact_state_id"},
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

    def test_card_reward_claim_is_one_opening_choice(self) -> None:
        action, legal, reason = resolve_recorded_action(
            {"id": "REWARD:TAKE:CARD", "kind": "reward_card_claimed"},
            [
                {"id": "CHOOSE:index=0", "kind": "choose", "label": "Gold: 25"},
                {"id": "CHOOSE:index=1", "kind": "choose", "label": "Card Reward"},
                {"id": "CHOOSE:index=2", "kind": "choose", "label": "Potion"},
            ],
        )
        self.assertTrue(legal)
        self.assertEqual(action["id"], "CHOOSE:index=1")
        self.assertEqual(reason, "choice_label")

    def test_discovery_and_choose_one_use_card_reward_choice_index(self) -> None:
        choices = [
            {"id": "CHOOSE:index=0", "kind": "choose", "label": "卡牌甲"},
            {"id": "CHOOSE:index=1", "kind": "choose", "label": "卡牌乙"},
        ]
        for mode in ("discovery", "choose_one"):
            with self.subTest(mode=mode):
                action, legal, reason = resolve_recorded_action(
                    {"id": "CHOOSE:index=1", "kind": "card_option_selected",
                     "selection_mode": mode, "card_id": "SomeCard"},
                    choices,
                )
                self.assertTrue(legal)
                self.assertEqual(action["id"], "CHOOSE:index=1")
                self.assertEqual(reason, "exact_id")

    def test_discovery_choice_becomes_a_usable_sft_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receiver = ActionRecorderEnricher(Path(directory))
            before = combat_state() | {
                "screen_type": "CARD_REWARD", "room_phase": "COMBAT",
                "choice_list": ["卡牌甲", "卡牌乙"],
                "screen_state": {"cards": [
                    {"id": "A", "name": "卡牌甲"},
                    {"id": "B", "name": "卡牌乙"},
                ]},
            }
            receiver.consume({
                "type": "action_observed", "run_id": "r", "step_id": "r:1",
                "observation_before": before,
                "available_actions": {"commands": ["choose"], "choices": before["choice_list"]},
                "chosen_action": {"id": "CHOOSE:index=1", "kind": "card_option_selected",
                                  "selection_mode": "discovery", "card_id": "B"},
                "state_match": {"match_status": "exact_state_id"},
            })
            record = receiver.consume({
                "type": "step_resolved", "run_id": "r", "step_id": "r:1",
                "observation_after": combat_state(), "resolution": "next_decision",
            })
            receiver.close()
            self.assertIsNotNone(record)
            self.assertTrue(record["usable_for_sft"])
            self.assertEqual(record["selected_action_id"], "CHOOSE:index=1")
            self.assertEqual(record["pre_state"]["screen_type"], "CARD_REWARD")

    def test_single_card_grid_choice_becomes_a_usable_sft_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receiver = ActionRecorderEnricher(Path(directory))
            cards = [
                {"id": "A", "name": "卡牌甲", "uuid": "a"},
                {"id": "B", "name": "卡牌乙", "uuid": "b"},
            ]
            before = combat_state() | {
                "screen_type": "GRID", "room_phase": "COMBAT",
                "choice_list": ["卡牌甲", "卡牌乙"],
                "screen_state": {"cards": cards, "selected_cards": [], "num_cards": 1},
            }
            receiver.consume({
                "type": "action_observed", "run_id": "r", "step_id": "r:1",
                "observation_before": before,
                "available_actions": {"commands": ["choose"], "choices": before["choice_list"]},
                "chosen_action": {"id": "SELECT_CARDS:grid:b;", "kind": "card_selection_changed",
                                  "screen": "grid", "selected_cards": [{"card_id": "B", "card_uuid": "b"}]},
                "state_match": {"match_status": "exact_state_id"},
            })
            record = receiver.consume({
                "type": "step_resolved", "run_id": "r", "step_id": "r:1",
                "observation_after": combat_state(), "resolution": "next_decision",
            })
            receiver.close()
            self.assertIsNotNone(record)
            self.assertTrue(record["usable_for_sft"])
            self.assertEqual(record["selected_action_id"], "CHOOSE:index=1")
            self.assertEqual(record["pre_state"]["screen_type"], "GRID")

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
