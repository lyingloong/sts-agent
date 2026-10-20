from __future__ import annotations

import copy
import unittest

from sts_agent.policy.presentation import text_observation


EFFECT_VALUES = {
    "damage": 9, "base_damage": 6, "final_single_hit_damage": 9, "final_damage": 9,
    "block": 0, "base_block": -1, "final_block": None,
    "magic_number": 2, "base_magic_number": 1, "hit_count": 3,
}


class CardTextProjectionTests(unittest.TestCase):
    def display(self, card, **observation):
        source = {"screen_type": "COMBAT", "combat": {"hand": [card]}, **observation}
        original = copy.deepcopy(source)
        view = text_observation(source)
        self.assertEqual(source, original)
        self.assertEqual(text_observation(view), view)
        return view["combat"]["hand"][0]

    def test_complete_runtime_description_replaces_redundant_effect_numbers(self):
        card = {"name": "连击", "cost": 1, "cost_for_turn": 0, "is_playable": True,
                "description": "造成9点伤害3次，并施加2层虚弱。", "description_complete": True,
                "unresolved_description_variables": [], "raw_description": "造成!D!点伤害3次，并施加!M!层虚弱。",
                "free_to_play_once": True, "exhaust_on_use_once": True, "retain": None, **EFFECT_VALUES}
        view = self.display(card)
        self.assertEqual(view["description"], card["description"])
        for field in EFFECT_VALUES:
            self.assertNotIn(field, view)
        self.assertNotIn("raw_description", view)
        self.assertNotIn("description_complete", view)
        self.assertEqual(view["cost"], 1)
        self.assertEqual(view["cost_for_turn"], 0)
        self.assertIs(view["is_playable"], True)
        self.assertIs(view["free_to_play_once"], True)
        self.assertIs(view["exhaust_on_use_once"], True)
        self.assertIsNone(view["retain"])

    def test_incomplete_or_unknown_descriptions_keep_numeric_fallbacks(self):
        cases = [
            {"description_complete": False}, {"description_complete": None},
            {"description_complete": True, "unresolved_description_variables": ["CUSTOM"]},
            {"description_complete": True, "unresolved_description_variables": None},
            {"description_complete": True, "description": "造成!D!点伤害。"},
            {"description_complete": True, "description": ""},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                card = {"cost": 1, "description": "造成9点伤害。", **EFFECT_VALUES, **overrides}
                view = self.display(card)
                for field, value in EFFECT_VALUES.items():
                    self.assertIn(field, view)
                    self.assertEqual(view[field], value)

    def test_readable_legacy_description_without_a_conflicting_template_is_sufficient(self):
        for metadata in ({}, {"raw_description": None}, {"raw_description": ""},
                         {"raw_description": "造成#b9点伤害。"}):
            with self.subTest(metadata=metadata):
                view = self.display({"cost": 1, "description": "造成9点伤害。", "damage": 9, **metadata})
                self.assertNotIn("damage", view)
                self.assertNotIn("raw_description", view)

    def test_legacy_description_does_not_erase_unique_raw_effects_or_unknown_variables(self):
        for raw in ("造成9点伤害并抽1张牌。", "造成!CUSTOM!点伤害。", "造成!D!点伤害。"):
            with self.subTest(raw=raw):
                view = self.display({"cost": 1, "description": "造成9点伤害。", "damage": 9,
                                     "raw_description": raw})
                self.assertEqual(view["damage"], 9)
                self.assertEqual(view["raw_description"], raw)

    def test_costs_keep_the_base_card_cost_and_only_remove_equal_duplicates(self):
        for cost in (0, 1, -1):
            with self.subTest(cost=cost):
                view = self.display({"cost": cost, "base_cost": cost, "cost_for_turn": cost,
                                     "description": "完整效果", "description_complete": True})
                self.assertEqual(view["cost"], cost)
                self.assertNotIn("base_cost", view)
                self.assertNotIn("cost_for_turn", view)
        view = self.display({"cost": 1, "base_cost": 2, "cost_for_turn": 0,
                             "description": "完整效果", "description_complete": True})
        self.assertEqual((view["cost"], view["base_cost"], view["cost_for_turn"]), (1, 2, 0))
        view = self.display({"cost": None, "base_cost": 0, "cost_for_turn": False,
                             "description": "完整效果", "description_complete": True})
        self.assertIsNone(view["cost"])
        self.assertEqual(view["base_cost"], 0)
        self.assertIs(view["cost_for_turn"], False)

    def test_upgrade_previews_and_pile_cards_are_cleaned_without_playability_claims(self):
        source = {"screen_type": "GRID", "screen_state": {"cards": [{
            "name": "打击", "damage": 6, "description": "造成6点伤害。",
            "upgrade_preview": {"name": "打击+", "damage": 9, "description": "造成9点伤害。"}}]},
            "combat": {"draw_pile": [{"name": "防御", "block": 5, "description": "获得5点格挡。",
                                         "is_playable": False, "count": 2}]}}
        original = copy.deepcopy(source)
        view = text_observation(source)
        card = view["screen_state"]["cards"][0]
        self.assertNotIn("damage", card)
        self.assertNotIn("damage", card["upgrade_preview"])
        pile_card = view["combat"]["draw_pile"][0]
        self.assertNotIn("block", pile_card)
        self.assertNotIn("is_playable", pile_card)
        self.assertEqual(pile_card["count"], 2)
        self.assertEqual(source, original)
        self.assertEqual(text_observation(view), view)

    def test_player_block_and_enemy_numbers_are_never_card_effect_duplicates(self):
        source = {"combat": {"player": {"block": 20, "energy": 0, "description": "玩家状态"},
            "monsters": [{"monster_index": 0, "name": "敌人", "block": 10,
                          "intent_damage": 12, "intent_hits": 2, "intent_total_damage": 24,
                          "description": "敌人说明", "description_complete": True}]}}
        view = text_observation(source)
        self.assertEqual(view["combat"]["player"]["block"], 20)
        monster = view["combat"]["monsters"][0]
        self.assertEqual((monster["block"], monster["intent_damage"], monster["intent_hits"], monster["intent_total_damage"]),
                         (10, 12, 2, 24))

    def test_distinct_multi_target_damage_uses_explicit_monster_indices(self):
        card = {"cost": 1, "description": "对所有敌人造成伤害。", "description_complete": True,
                "multi_damage": [9, 6], "damage": 9}
        source = {"combat": {"hand": [card], "monsters": [
            {"monster_index": 1, "name": "敌人乙"}, {"monster_index": 0, "name": "敌人甲"}]}}
        original = copy.deepcopy(source)
        view = text_observation(source)
        shown = view["combat"]["hand"][0]
        self.assertEqual(shown["target_damage"], [{"monster_index": 0, "name": "敌人甲", "damage": 9},
                                                 {"monster_index": 1, "name": "敌人乙", "damage": 6}])
        self.assertNotIn("multi_damage", shown)
        self.assertNotIn("damage", shown)
        self.assertEqual(source, original)
        self.assertEqual(text_observation(view), view)

    def test_equal_multi_target_damage_is_redundant_with_complete_text(self):
        view = self.display({"cost": 1, "description": "对所有敌人造成6点伤害。",
                             "description_complete": True, "multi_damage": [6, 6]})
        self.assertNotIn("multi_damage", view)

    def test_unknown_or_unmatched_multi_target_arrays_remain_unmapped_fallbacks(self):
        cases = [
            ([9, None], [{"monster_index": 0}, {"monster_index": 1}]),
            ([9, -1], [{"monster_index": 0}, {"monster_index": 1}]),
            ([9, False], [{"monster_index": 0}, {"monster_index": 1}]),
            ([9, 6], [{"name": "无索引甲"}, {"name": "无索引乙"}]),
            ([9, 6], [{"monster_index": 0}, {"monster_index": 0}]),
            ([9, 6], [{"monster_index": 0}, {"monster_index": 2}]),
            ([9, 6], [{"monster_index": 0}]),
        ]
        for multiple, monsters in cases:
            with self.subTest(multiple=multiple, monsters=monsters):
                card = {"cost": 1, "description": "完整效果", "description_complete": True, "multi_damage": multiple}
                source = {"combat": {"hand": [card], "monsters": monsters}}
                view = text_observation(source)["combat"]["hand"][0]
                self.assertEqual(view["multi_damage"], multiple)
                self.assertNotIn("target_damage", view)

    def test_incomplete_multi_target_effect_and_nonhand_arrays_keep_fallbacks(self):
        card = {"cost": 1, "description": "效果尚未完整", "description_complete": False,
                "damage": 9, "multi_damage": [9, 6]}
        source = {"combat": {"hand": [card], "monsters": [{"monster_index": 0}, {"monster_index": 1}]}}
        shown = text_observation(source)["combat"]["hand"][0]
        self.assertEqual(shown["damage"], 9)
        self.assertEqual(shown["multi_damage"], [9, 6])
        self.assertNotIn("target_damage", shown)
        source["deck"] = [dict(card, description_complete=True)]
        shown = text_observation(source)["deck"][0]
        self.assertEqual(shown["multi_damage"], [9, 6])
        self.assertNotIn("target_damage", shown)


if __name__ == "__main__":
    unittest.main()
