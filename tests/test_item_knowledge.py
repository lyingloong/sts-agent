from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import communication_controller as controller  # noqa: E402
import card_knowledge  # noqa: E402
import fetch_card_knowledge  # noqa: E402
import item_knowledge  # noqa: E402
import power_knowledge  # noqa: E402


class ItemKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_cache = item_knowledge._CACHE
        self.old_card_cache = card_knowledge._CACHE
        self.old_power_cache = power_knowledge._CACHE
        item_knowledge._CACHE = {
            "relic": {
                "BURNINGBLOOD": {
                    "id": "BURNING_BLOOD",
                    "name": "Burning Blood",
                    "description": "At the end of combat, heal 6 HP.",
                    "tier": "Starter",
                    "color": "red",
                }
            },
            "potion": {
                "ATTACKPOTION": {
                    "id": "ATTACKPOTION",
                    "name": "Attack Potion",
                    "description": "Choose an Attack card.",
                    "rarity": "Common",
                    "target": "None",
                    "is_thrown": False,
                }
            },
        }
        card_knowledge._CACHE = {
            "TEST_CARD": {
                "id": "TEST_CARD",
                "name": "Test Card",
                "description": "Deal 8 damage.",
                "cost": 1,
                "type": "Attack",
                "rarity": "Common",
                "target": "Enemy",
                "color": "red",
                "damage": 8,
                "block": None,
                "magic_number": None,
                "hit_count": None,
                "exhaust": False,
                "ethereal": False,
                "innate": False,
                "retain": False,
                "self_retain": False,
                "purge_on_use": False,
                "tags": [],
                "keywords": [],
                "upgrade": {},
            }
        }
        power_knowledge._CACHE = {
            "powers": {
                "STRENGTH": {"id": "STRENGTH", "name": "Strength", "description": "Gain X Strength."},
            },
            "card_powers": {
                "LEGACYPOWER": {"id": "LegacyPower", "name": "Legacy Power", "description": "Perform the legacy effect."},
            },
        }

    def tearDown(self) -> None:
        item_knowledge._CACHE = self.old_cache
        card_knowledge._CACHE = self.old_card_cache
        power_knowledge._CACHE = self.old_power_cache

    def test_shop_items_receive_static_effects_and_prices(self) -> None:
        message = {
            "game_state": {
                "screen_type": "SHOP_SCREEN",
                "screen_state": {
                    "relics": [{"id": "Burning Blood", "name": "燃烧之血", "price": 150}],
                    "potions": [{"id": "AttackPotion", "name": "攻击药水", "price": 50}],
                },
            }
        }
        shop = controller.normalize_state(message)["screen_state"]
        relic = shop["relics"][0]
        potion = shop["potions"][0]
        self.assertEqual(relic["price"], 150)
        self.assertEqual(relic["description"], "At the end of combat, heal 6 HP.")
        self.assertNotIn("tier", relic)
        self.assertNotIn("effect", relic)
        self.assertEqual(potion["price"], 50)
        self.assertEqual(potion["description"], "Choose an Attack card.")
        self.assertEqual(potion["target"], "None")
        self.assertNotIn("effect", potion)

    def test_event_and_reward_only_enrich_explicit_entities(self) -> None:
        event = controller.normalize_event_option(
            {"text": "拿走药水", "potion": {"id": "AttackPotion", "name": "攻击药水"}}
        )
        self.assertEqual(event["potion"]["target"], "None")
        bare_reward = controller.normalize_reward({"reward_type": "RELIC"})
        self.assertEqual(bare_reward, {"reward_type": "RELIC"})
        revealed_reward = controller.normalize_reward({"reward_type": "RELIC", "id": "Burning Blood", "name": "燃烧之血"})
        self.assertEqual(revealed_reward["relic"]["description"], "At the end of combat, heal 6 HP.")

    def test_neow_event_is_kept_as_local_static_entity(self) -> None:
        payload = fetch_card_knowledge.normalized_payload("events.json", [])
        self.assertEqual(payload[0]["id"], "NEOW_EVENT")
        self.assertIsNotNone(controller.entity_profile("event", "Neow Event"))
        normalized = controller.normalize_state(
            {
                "game_state": {
                    "screen_type": "EVENT",
                    "screen_state": {
                        "event_id": "Neow Event",
                        "event_name": "Neow Event",
                        "options": [],
                    },
                }
            }
        )
        self.assertEqual(normalized["screen_state"]["knowledge"]["id"], "NEOW_EVENT")
        references = controller.visible_entity_references(normalized)
        self.assertEqual(references[0]["entity_id"], "NEOW_EVENT")

    def test_drops_internal_reward_uuid(self) -> None:
        reward = controller.normalize_reward(
            {"reward_type": "POTION", "id": "AttackPotion", "name": "攻击药水", "uuid": "hidden-object-id"}
        )
        self.assertNotIn("uuid", reward)
        self.assertEqual(reward["potion"]["name"], "攻击药水")

    def test_card_merges_runtime_values_and_static_fallbacks_at_top_level(self) -> None:
        card = controller.visible_card(
            {
                "id": "TEST_CARD",
                "name": "运行时卡牌",
                "upgrades": 1,
                "is_playable": True,
                "damage": 17,
                "description": "Deal 17 damage.",
                "exhausts": True,
            }
        )
        self.assertEqual(card["name"], "运行时卡牌")
        self.assertEqual(card["type"], "Attack")
        self.assertEqual(card["cost"], 1)
        self.assertEqual(card["upgrades"], 1)
        self.assertTrue(card["is_playable"])
        self.assertEqual(card["description"], "Deal 17 damage.")
        self.assertNotIn("damage", card)
        self.assertNotIn("target", card)
        self.assertNotIn("effect", card)

    def test_potion_use_and_relic_counter_remain_runtime_fields(self) -> None:
        potion = controller.visible_potion({"id": "AttackPotion", "can_use": True})
        relic = controller.visible_relic({"id": "Burning Blood", "counter": -1})
        self.assertTrue(potion["can_use"])
        self.assertNotIn("counter", relic)

    def test_active_powers_receive_static_description_with_card_power_fallback(self) -> None:
        powers = controller.powers_summary(
            [
                {"id": "Strength", "name": "力量", "amount": 3, "type": "BUFF"},
                {"id": "Legacy Power", "name": "旧效果", "amount": 1, "type": "BUFF"},
            ]
        )
        self.assertEqual(powers[0], {"id": "Strength", "name": "力量", "amount": 3, "type": "BUFF", "description": "Gain X Strength."})
        self.assertEqual(powers[1]["description"], "Perform the legacy effect.")

    def test_knowledge_generator_repairs_known_potion_description_defects(self) -> None:
        samples = {
            "ATTACKPOTION": "Choose 1 of 3 random Attack cards to add to your hand, it costs 0 this turn.1Choose 1 of 3 random Attack cards and add 2 copies to your hand, they cost 0 this turn.",
            "COLORLESSPOTION": "Choose 1 of 3 random Colorless cards to add to your hand, it costs 0 this turn.1Choose 1 of 3 random Colorless cards and add 2 copies to your hand, they cost 0 this turn.",
            "DUPLICATIONPOTION": "This turn, your next card is played twice.1This turn, your next 2 cards are played twice.",
            "LIQUIDMEMORIES": "Choose a card in your discard pile and return it to your hand. It costs 0 this turn.1Choose cards in your discard pile and return them to your hand. They cost 0 this turn.",
            "POWERPOTION": "Choose 1 of 3 random Power cards to add to your hand, it costs 0 this turn.1Choose 1 of 3 random Power cards and add 2 copies to your hand, they cost 0 this turn.",
            "SKILLPOTION": "Choose 1 of 3 random Skill cards to add to your hand, it costs 0 this turn.1Choose 1 of 3 random Skill cards and add 2 copies to your hand, they cost 0 this turn.",
            "SMOKEBOMB": "Escape from a non-boss combat. Receive no rewards.0I'm Surrounded!",
            "SWIFT_POTION": "3Draw cards.",
        }
        expected = {
            "ATTACKPOTION": "Choose 1 of 3 random Attack cards to add to your hand, it costs 0 this turn.",
            "COLORLESSPOTION": "Choose 1 of 3 random Colorless cards to add to your hand, it costs 0 this turn.",
            "DUPLICATIONPOTION": "This turn, your next card is played twice.",
            "LIQUIDMEMORIES": "Choose a card in your discard pile and return it to your hand. It costs 0 this turn.",
            "POWERPOTION": "Choose 1 of 3 random Power cards to add to your hand, it costs 0 this turn.",
            "SKILLPOTION": "Choose 1 of 3 random Skill cards to add to your hand, it costs 0 this turn.",
            "SMOKEBOMB": "Escape from a non-boss combat. Receive no rewards.",
            "SWIFT_POTION": "Draw 3 cards.",
        }

        for item_id, description in samples.items():
            entity = fetch_card_knowledge.normalize_entity("potions", {"id": item_id, "description": description})
            self.assertEqual(entity["description"], expected[item_id])

    def test_knowledge_generator_repairs_empty_impulse_description(self) -> None:
        entity = fetch_card_knowledge.normalize_entity(
            "cards",
            {"id": "IMPULSE", "description": "", "upgrade": {"description": None}},
        )
        self.assertEqual(entity["description"], "Use the Passive effect of all your Channeled Orbs. Exhaust.")
        self.assertEqual(entity["upgrade"]["description"], "Use the Passive effect of all your Channeled Orbs.")

    def test_knowledge_generator_repairs_known_power_description_defects(self) -> None:
        samples = {
            "DARK_EMBRACE": "Whenever a card is Exhausted, draw X1 card.Whenever a card is Exhausted, draw X",
            "DUPLICATIONPOWER": "Your next card is played twice this turn.Your next X",
            "THEBOMB": "At the end of X%1$s turns, deal X%2$s damage to ALL enemies.At the end of this turn, deal X%1$s damage to ALL enemies.",
        }
        expected = {
            "DARK_EMBRACE": "Whenever a card is Exhausted, draw 1 card.",
            "DUPLICATIONPOWER": "Your next card is played twice this turn.",
            "THEBOMB": "At the end of 3 turns, deal massive damage to ALL enemies.",
        }

        for power_id, description in samples.items():
            entity = fetch_card_knowledge.normalize_entity("powers", {"id": power_id, "description": description})
            self.assertEqual(entity["description"], expected[power_id])


if __name__ == "__main__":
    unittest.main()
