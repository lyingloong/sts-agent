from __future__ import annotations

import copy
import unittest

from sts_agent.runtime import controller
from tests.test_trajectory_assembler import combat_state


class PolicyStateProjectionTests(unittest.TestCase):
    def test_hand_and_enemy_positions_keep_protocol_indices_after_filtering(self):
        state = combat_state()
        card = {"id": "Strike_R", "name": "打击", "has_target": True}
        state["combat_state"]["hand"] = [
            card | {"is_playable": False}, None,
            card | {"is_playable": True, "cost_for_turn": 0, "free_to_play_once": True},
            card | {"is_playable": True, "cost_for_turn": 1},
        ]
        state["combat_state"]["monsters"] = [None, {"name": "敌人", "current_hp": 20, "is_gone": False}]
        message = {"in_game": True, "available_commands": ["play", "end"], "game_state": state}
        original = copy.deepcopy(message)
        observation = controller.normalize_state(message)
        actions = controller.semantic_actions(message, fine_grained=True)
        self.assertEqual([card["card_index"] for card in observation["combat"]["hand"]], [1, 3, 4])
        self.assertEqual(observation["combat"]["monsters"][0]["monster_index"], 1)
        plays = [action for action in actions if action["kind"] == "play"]
        self.assertEqual([action["command"] for action in plays], ["PLAY 3 1", "PLAY 4 1"])
        self.assertIn("手牌 3", plays[0]["label"])
        self.assertIn("手牌 4", plays[1]["label"])
        self.assertEqual(message, original)

    def test_potion_slots_use_inventory_positions(self):
        state = combat_state()
        potion = {"id": "FirePotion", "name": "火焰药水", "can_use": True,
                  "can_discard": True, "requires_target": True}
        state["potions"] = [{"id": "Potion Slot", "name": "Potion Slot"}, None, potion, potion]
        message = {"in_game": True, "available_commands": ["potion"], "game_state": state}
        observation = controller.normalize_state(message)
        self.assertEqual([potion["slot"] for potion in observation["potions"]], [0, 2, 3])
        actions = controller.semantic_actions(message, fine_grained=True)
        self.assertEqual([action["command"] for action in actions], [
            "POTION USE 2 0", "POTION DISCARD 2", "POTION USE 3 0", "POTION DISCARD 3",
        ])
        for action in actions:
            self.assertIn(f"栏位 {action['slot']}", action["label"])

    def test_potion_unknown_and_known_zero_slots_remain_distinct(self):
        for has_field, value in ((False, None), (True, None), (True, [])):
            with self.subTest(has_field=has_field, value=value):
                state = combat_state()
                state.pop("potions")
                if has_field:
                    state["potions"] = value
                normalized = controller.normalize_state({"game_state": state})
                observation = controller.route_policy_observation(copy.deepcopy(normalized))
                if has_field and value == []:
                    self.assertEqual(normalized["potions"], [])
                    self.assertEqual(normalized["potion_slot_count"], 0)
                    self.assertEqual(observation["potion_slot_count"], 0)
                else:
                    self.assertIsNone(normalized["potions"])
                    self.assertIsNone(normalized["potion_slot_count"])
                    self.assertIsNone(observation["potions"])
                    self.assertIsNone(observation["potion_slot_count"])

    def test_sparse_potion_inventory_keeps_capacity_and_original_slots(self):
        state = combat_state()
        potion = {"id": "FirePotion", "name": "火焰药水", "can_use": True}
        state["potions"] = [None, {"id": "Potion Slot", "name": "Potion Slot"}, potion, None, potion]
        original = copy.deepcopy(state)
        normalized = controller.normalize_state({"game_state": state})
        self.assertEqual(normalized["potion_slot_count"], 5)
        self.assertEqual([item["slot"] for item in normalized["potions"]], [1, 2, 4])
        observation = controller.route_policy_observation(normalized)
        self.assertEqual(observation["potion_slot_count"], 5)
        self.assertEqual(state, original)

    def test_runtime_card_flags_and_unknowns_survive_without_static_facts(self):
        source = {"id": "Card", "name": "实例", "innate": True, "free_to_play_once": True,
                  "exhaust_on_use_once": False, "in_bottle_flame": True,
                  "in_bottle_lightning": None, "is_cost_modified_for_turn": True,
                  "base_magic_number": 2, "magic_number": 4}
        projected = controller.visible_card(source)
        for field in ("free_to_play_once", "exhaust_on_use_once", "in_bottle_flame",
                      "in_bottle_lightning", "is_cost_modified_for_turn", "base_magic_number", "magic_number"):
            self.assertEqual(projected[field], source[field])
        self.assertIs(projected["is_innate"], True)
        self.assertNotIn("innate", projected)
        self.assertIs(controller.visible_card(source | {"is_innate": False})["is_innate"], False)
        self.assertNotIn("free_to_play_once", controller.visible_card({"id": "unknown"}))

    def test_player_turn_fields_preserve_zero_and_unknown(self):
        combat = combat_state()["combat_state"]
        fields = {"base_energy_per_turn": 3, "energy_per_turn": 4, "master_hand_size": 5,
                  "game_hand_size": None, "cards_played_this_turn": 0}
        combat["player"].update(fields)
        projected = controller.normalize_combat(combat)["player"]
        for key, value in fields.items():
            self.assertEqual(projected[key], value)

    def test_deck_groups_only_equal_public_instances(self):
        card = {"id": "Strike_R", "name": "打击", "type": "ATTACK", "cost": 1,
                "description": "造成伤害", "base_damage": 6, "in_bottle_flame": False}
        deck = controller.deck_summary([card, card, card | {"in_bottle_flame": True},
                                        card | {"base_damage": 9}])
        self.assertEqual(len(deck), 3)
        self.assertEqual(sum(item["count"] for item in deck), 4)
        self.assertEqual(next(item["count"] for item in deck if item["in_bottle_flame"]), 1)
        self.assertEqual(next(item["count"] for item in deck if item["base_damage"] == 9), 1)

    def test_candidate_positions_survive_gaps_and_selected_order(self):
        cards = [{"name": "同名", "uuid": "first"}, None, {"name": "同名", "uuid": "last"}]
        raw = {"cards": cards, "selected_cards": [cards[2]], "confirmation_card": cards[2]}
        projected = controller.normalize_screen_state("GRID", raw)
        self.assertEqual([card["choice_index"] for card in projected["cards"]], [0, 2])
        self.assertEqual(projected["selected_cards"][0]["choice_index"], 2)
        self.assertEqual(projected["confirmation_card"]["choice_index"], 2)
        # Equal names alone do not establish which candidate was selected.
        projected = controller.normalize_screen_state("GRID", raw | {"selected_cards": [{"name": "同名"}]})
        self.assertNotIn("choice_index", projected["selected_cards"][0])
        hand = controller.normalize_screen_state("HAND_SELECT", {"hand": cards, "selected": [cards[2]]})
        self.assertEqual([card["choice_index"] for card in hand["hand"]], [0, 2])
        self.assertEqual(hand["selected"][0]["choice_index"], 2)
        explicit = cards[2] | {"choice_index": 7}
        projected = controller.normalize_screen_state("GRID", {"cards": [explicit], "selected_cards": [cards[2]]})
        self.assertEqual(projected["cards"][0]["choice_index"], 7)
        self.assertEqual(projected["selected_cards"][0]["choice_index"], 7)

    def test_upgrade_preview_and_complete_confirmation_only_in_upgrade_screens(self):
        preview = {"id": "Strike_R", "name": "打击+", "cost": 0, "damage": 9,
                   "base_damage": 9, "magic_number": 2, "description": "造成9点伤害"}
        card = {"id": "Strike_R", "name": "打击", "uuid": "card-1", "cost": 1, "damage": 6,
                "base_damage": 6, "base_magic_number": 1, "magic_number": 1,
                "is_innate": False, "upgrade_preview": preview,
                "upgrade_preview_source": "game_ui", "upgrade_preview_status": "available"}
        raw = {"cards": [card], "selected_cards": [card], "confirmation_card": card, "for_upgrade": True}
        original = copy.deepcopy(raw)
        projected = controller.normalize_screen_state("GRID", raw)
        for entity in (projected["cards"][0], projected["selected_cards"][0], projected["confirmation_card"]):
            self.assertEqual(entity["upgrade_preview"]["damage"], 9)
            self.assertEqual(entity["upgrade_preview_status"], "available")
            self.assertEqual(entity["base_magic_number"], 1)
            self.assertIs(entity["is_innate"], False)
        self.assertEqual(projected["confirmation_card"]["uuid"], "card-1")
        for screen in ("GRID", "CARD_REWARD"):
            normal = controller.normalize_screen_state(screen, raw | {"for_upgrade": False})
            for entity in (normal["cards"][0], normal["confirmation_card"]):
                self.assertNotIn("upgrade_preview", entity)
                self.assertNotIn("upgrade_preview_status", entity)
        self.assertEqual(raw, original)

    def test_unavailable_upgrade_preview_stays_explicitly_unknown(self):
        card = {"name": "模组牌", "upgrade_preview": None,
                "upgrade_preview_source": None, "upgrade_preview_status": "unsupported_card"}
        projected = controller.normalize_screen_state("GRID", {"cards": [card], "for_upgrade": True})["cards"][0]
        self.assertIsNone(projected["upgrade_preview"])
        self.assertIsNone(projected["upgrade_preview_source"])
        self.assertEqual(projected["upgrade_preview_status"], "unsupported_card")

    def test_grid_confirmation_keeps_effects_without_fabricating_choices(self):
        card = {"id": "Strike_R", "uuid": "card-1", "name": "打击", "damage": 6,
                "description": "造成6点伤害", "upgrade_preview": {"name": "打击+", "damage": 9},
                "upgrade_preview_status": "available", "upgrade_preview_source": "game_ui"}
        for flag in ("confirm_up", "confirm_screen_up", "confirmScreenUp", "just_for_confirming",
                     "is_just_for_confirming", "for_just_confirming", "isJustForConfirming",
                     "forJustConfirming", "justForConfirming", "for_confirm", None):
            with self.subTest(flag=flag):
                raw = {"cards": [card], "selected_cards": [card], "confirmation_card": card,
                       "for_upgrade": True}
                if flag:
                    raw[flag] = True
                projected = controller.normalize_screen_state("GRID", raw, choice_list=[] if flag is None else None)
                for entity in (projected["cards"][0], projected["selected_cards"][0], projected["confirmation_card"]):
                    self.assertNotIn("choice_index", entity)
                    self.assertEqual(entity["damage"], 6)
                    self.assertEqual(entity["upgrade_preview"]["damage"], 9)
                    self.assertEqual(entity["upgrade_preview_status"], "available")
                self.assertEqual(projected["confirmation_card"]["uuid"], "card-1")
                if flag in {"confirm_up", "confirm_screen_up"}:
                    self.assertIs(projected[flag], True)
                elif flag == "confirmScreenUp":
                    self.assertIs(projected["confirm_screen_up"], True)
                elif flag:
                    self.assertIs(projected["just_for_confirming"], True)

    def test_grid_confirmation_explicit_source_position_is_only_a_reference(self):
        card = {"name": "牌", "uuid": "card-1", "choice_index": 4}
        projected = controller.normalize_screen_state("GRID", {
            "cards": [card], "selected_cards": [card], "confirmation_card": card,
        }, choice_list=[])
        for entity in (projected["cards"][0], projected["selected_cards"][0], projected["confirmation_card"]):
            self.assertNotIn("choice_index", entity)
            self.assertEqual(entity["reference_choice_index"], 4)

    def test_fine_grained_choices_reference_original_legal_positions(self):
        for screen in ("GRID", "HAND_SELECT", "CARD_REWARD", "COMBAT_REWARD", "BOSS_REWARD", "SHOP_SCREEN"):
            message = {"in_game": True, "available_commands": ["choose"], "game_state": {
                "screen_type": screen, "room_phase": "COMPLETE", "choice_list": ["重复项", "重复项"],
                "screen_state": {}, "potions": [],
            }}
            actions = controller.semantic_actions(message, fine_grained=True)
            self.assertEqual([action["command"] for action in actions], ["CHOOSE 0", "CHOOSE 1"])
            self.assertEqual([action["choice_index"] for action in actions], [0, 1])
            for index, action in enumerate(actions):
                self.assertIn(f"选择位置 {index}", action["label"])

    def test_shop_uses_only_explicit_command_positions(self):
        state = {"cards": [{"name": "same"}, {"name": "same", "choice_index": 5}],
                 "potions": [{"name": "药水", "choice_index": 2}], "relics": []}
        projected = controller.normalize_screen_state("SHOP_SCREEN", state, choice_list=["same", "药水"])
        self.assertNotIn("choice_index", projected["cards"][0])
        self.assertEqual(projected["cards"][1]["choice_index"], 5)
        self.assertEqual(projected["potions"][0]["choice_index"], 2)
        self.assertNotIn("slot", projected["potions"][0])

    def test_public_piles_are_grouped_without_source_order_or_uuid(self):
        strike = {"id": "Strike_R", "name": "打击", "type": "ATTACK", "cost": 1,
                  "base_damage": 6, "description": "造成伤害", "in_bottle_flame": False}
        bash = {"id": "Bash", "name": "痛击", "type": "ATTACK", "cost": 2,
                "base_damage": 8, "description": "造成伤害并施加易伤"}
        source = [strike | {"uuid": "second", "card_index": 2}, bash | {"uuid": "first"},
                  strike | {"uuid": "third", "card_index": 1}]
        original = copy.deepcopy(source)
        first, missing = controller.card_pile_summary(source)
        second, _ = controller.card_pile_summary(list(reversed(source)))
        self.assertEqual(first, second)
        self.assertEqual(missing, 0)
        self.assertEqual(len(first), 2)
        self.assertEqual(next(card["count"] for card in first if card["id"] == "Strike_R"), 2)
        for card in first:
            for field in ("uuid", "card_index", "choice_index", "index"):
                self.assertNotIn(field, card)
        self.assertEqual(source, original)

    def test_pile_cards_keep_base_and_instance_facts_without_target_values(self):
        card = {"id": "Strike_R", "name": "打击", "type": "ATTACK", "base_damage": 6,
                "base_block": 0, "base_magic_number": 2, "description": "源当前说明",
                "in_bottle_flame": True, "free_to_play_once": None, "is_playable": True,
                "damage": 99, "block": 30, "multi_damage": [99, 1],
                "final_single_hit_damage": 99, "final_damage": 99, "final_block": 30}
        details, missing = controller.card_pile_summary([card, card | {"damage": 8, "block": 1}])
        self.assertEqual(missing, 0)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["count"], 2)
        for field in ("base_damage", "base_block", "base_magic_number", "description", "in_bottle_flame", "free_to_play_once"):
            self.assertEqual(details[0][field], card[field])
        for field in ("is_playable", "damage", "block", "multi_damage", "final_single_hit_damage", "final_damage", "final_block"):
            self.assertNotIn(field, details[0])

    def test_pile_unknown_empty_and_partial_details_remain_distinct(self):
        for value in (None, "unavailable"):
            self.assertEqual(controller.card_pile_summary(value), (None, None))
        self.assertEqual(controller.card_pile_summary([]), ([], 0))
        self.assertEqual(controller.card_pile_summary([{}, None, {"id": None, "name": None}]), ([], 3))
        combat = combat_state()["combat_state"]
        combat["draw_pile"] = [{}, {"id": "Strike_R", "name": "打击"}, None]
        combat["discard_pile"] = []
        combat["exhaust_pile"] = None
        projected = controller.normalize_combat(combat)
        self.assertEqual(projected["draw_count"], 3)
        self.assertEqual(projected["draw_pile_missing_count"], 2)
        self.assertEqual(sum(card["count"] for card in projected["draw_pile"]), 1)
        self.assertEqual(projected["discard_count"], 0)
        self.assertEqual(projected["discard_pile"], [])
        self.assertEqual(projected["discard_pile_missing_count"], 0)
        self.assertIsNone(projected["exhaust_count"])
        self.assertIsNone(projected["exhaust_pile"])
        self.assertIsNone(projected["exhaust_pile_missing_count"])

    def test_pile_details_survive_combat_choices_and_drop_with_stale_battle(self):
        state = combat_state()
        for field in ("draw_pile", "discard_pile", "exhaust_pile"):
            state["combat_state"][field] = [{"id": field, "name": field}]
        original = copy.deepcopy(state)
        for screen in ("NONE", "COMBAT", "GRID", "HAND_SELECT", "CARD_REWARD"):
            with self.subTest(screen=screen, phase="COMBAT"):
                current = copy.deepcopy(state) | {"screen_type": screen, "screen_state": {}}
                observation = controller.route_policy_observation(controller.normalize_state({"game_state": current}))
                for field in ("draw_pile", "discard_pile", "exhaust_pile"):
                    self.assertEqual(observation["combat"][field][0]["name"], field)
        for screen in ("MAP", "EVENT", "GRID", "HAND_SELECT", "CARD_REWARD", "COMBAT_REWARD"):
            with self.subTest(screen=screen, phase="COMPLETE"):
                current = copy.deepcopy(state) | {"screen_type": screen, "room_phase": "COMPLETE", "screen_state": {}}
                observation = controller.route_policy_observation(controller.normalize_state({"game_state": current}))
                self.assertNotIn("combat", observation)
        self.assertEqual(state, original)


if __name__ == "__main__":
    unittest.main()
