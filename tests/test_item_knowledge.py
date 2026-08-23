from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import communication_controller as controller  # noqa: E402
import item_knowledge  # noqa: E402


class ItemKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_cache = item_knowledge._CACHE
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

    def tearDown(self) -> None:
        item_knowledge._CACHE = self.old_cache

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
        self.assertEqual(relic["effect"]["description"], "At the end of combat, heal 6 HP.")
        self.assertEqual(potion["price"], 50)
        self.assertEqual(potion["effect"]["description"], "Choose an Attack card.")

    def test_event_and_reward_only_enrich_explicit_entities(self) -> None:
        event = controller.normalize_event_option(
            {"text": "拿走药水", "potion": {"id": "AttackPotion", "name": "攻击药水"}}
        )
        self.assertEqual(event["potion"]["effect"]["rarity"], "Common")
        bare_reward = controller.normalize_reward({"reward_type": "RELIC"})
        self.assertEqual(bare_reward, {"reward_type": "RELIC"})
        revealed_reward = controller.normalize_reward({"reward_type": "RELIC", "id": "Burning Blood", "name": "燃烧之血"})
        self.assertEqual(revealed_reward["relic"]["effect"]["tier"], "Starter")

    def test_drops_internal_reward_uuid(self) -> None:
        reward = controller.normalize_reward(
            {"reward_type": "POTION", "id": "AttackPotion", "name": "攻击药水", "uuid": "hidden-object-id"}
        )
        self.assertNotIn("uuid", reward)
        self.assertEqual(reward["potion"]["effect"]["name"], "Attack Potion")


if __name__ == "__main__":
    unittest.main()
