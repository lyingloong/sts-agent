from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import communication_controller as controller  # noqa: E402


class NoBridge:
    def send(self, packet):
        if packet.get("auto_execute"):
            return
        raise AssertionError(f"automatic decision unexpectedly requested terminal input: {packet}")


class FakePolicy:
    def __init__(self, decision):
        self.decision = decision
        self.screens = ("ALL",)
        self.calls = 0

    def rank(self, observation, legal_actions, on_update=None):
        self.calls += 1
        del observation, legal_actions, on_update
        return self.decision

    def status_decision(self):
        return self.decision


class FakeTeacher:
    enabled = True
    screens = ("ALL",)
    model = "teacher-test"

    def rank(self, observation, legal_actions, on_update=None):
        del observation, legal_actions, on_update
        return {
            "mode": "teacher_collection",
            "model_id": "teacher-test",
            "candidate_source": "remote_teacher",
            "model_action": "CHOOSE:index=1",
            "model_candidates": [{"action_id": "CHOOSE:index=1", "rank": 1}],
            "teacher": {"model_id": "teacher-test", "prompt_version": "test-v1", "latency_ms": 12},
        }


def event_message() -> dict:
    return {
        "in_game": True,
        "available_commands": ["choose"],
        "game_state": {
            "screen_type": "EVENT",
            "choice_list": ["Take reward", "Leave"],
            "screen_state": {"event_name": "Test", "options": []},
        },
    }


def combat_message() -> dict:
    return {
        "in_game": True,
        "available_commands": ["play", "end"],
        "game_state": {
            "screen_type": "NONE",
            "room_phase": "COMBAT",
            "class": "IRONCLAD",
            "ascension_level": 0,
            "act": 1,
            "floor": 2,
            "current_hp": 75,
            "max_hp": 88,
            "gold": 117,
            "deck": [
                {"name": "Strike", "id": "Strike_R", "upgrades": 0},
                {"name": "Strike", "id": "Strike_R", "upgrades": 0},
                {"name": "Bash", "id": "Bash", "upgrades": 1},
            ],
            "combat_state": {
                "hand": [{"name": "Strike", "is_playable": True, "has_target": True}],
                "monsters": [{"name": "Louse", "is_gone": False}],
            },
        },
    }


def combat_reward_message() -> dict:
    return {
        "in_game": True,
        "available_commands": ["choose", "proceed"],
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "room_phase": "COMPLETE",
            "act": 1,
            "floor": 3,
            "choice_list": ["gold", "card"],
            "screen_state": {
                "rewards": [
                    {"reward_type": "GOLD", "gold": 11},
                    {"reward_type": "CARD"},
                ]
            },
        },
    }


def full_combat_reward_message() -> dict:
    return {
        "in_game": True,
        "available_commands": ["choose", "proceed"],
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "room_phase": "COMPLETE",
            "act": 1,
            "floor": 3,
            "choice_list": ["gold", "card", "potion", "relic"],
            "screen_state": {
                "rewards": [
                    {"reward_type": "GOLD", "gold": 19},
                    {"reward_type": "CARD"},
                    {"reward_type": "POTION", "potion": {"id": "DexterityPotion", "name": "敏捷药水"}},
                    {"reward_type": "RELIC", "relic": {"id": "Anchor", "name": "锚"}},
                ]
            },
        },
    }


def grid_selection_message() -> dict:
    cards = [
        {"id": "Strike_R", "name": "Strike", "upgrades": 0, "uuid": "strike-a"},
        {"id": "Strike_R", "name": "Strike", "upgrades": 0, "uuid": "strike-b"},
        {"id": "Bash", "name": "Bash", "upgrades": 0, "uuid": "bash-a"},
    ]
    return {
        "in_game": True,
        "available_commands": ["choose"],
        "game_state": {
            "screen_type": "GRID",
            "choice_list": ["Strike", "Strike", "Bash"],
            "screen_state": {
                "cards": cards,
                "selected_cards": [cards[1]],
                "num_cards": 3,
            },
        },
    }


def relic_pickup_grid_message(relic_id: str, relic_name: str) -> dict:
    message = grid_selection_message()
    message["game_state"]["relics"] = [{"id": relic_id, "name": relic_name}]
    message["game_state"]["screen_state"]["selected_cards"] = []
    return message


def relic_pickup_pending() -> dict:
    return {
        "step_id": 4,
        "screen_type": "BOSS_REWARD",
        "pre_state": {
            "screen_type": "BOSS_REWARD",
            "relics": [{"id": "Burning Blood", "name": "Burning Blood"}],
        },
        "action": {"id": "CHOOSE:index=0", "kind": "choose", "label": "领取遗物"},
    }


def shop_room_message() -> dict:
    return {
        "in_game": True,
        "available_commands": ["choose", "proceed"],
        "game_state": {
            "screen_type": "SHOP_ROOM",
            "room_phase": "COMPLETE",
            "choice_list": ["merchant", "leave"],
        },
    }


def shop_continue_message() -> dict:
    return {
        "in_game": True,
        "available_commands": ["proceed"],
        "game_state": {
            "screen_type": "SHOP_SCREEN",
            "room_phase": "COMPLETE",
            "choice_list": [],
        },
    }


def empty_combat_message() -> dict:
    message = combat_message()
    message["available_commands"] = ["play", "end", "potion"]
    message["game_state"]["combat_state"] = {
        "hand": [],
        "monsters": [{"name": "Louse", "is_gone": False}],
    }
    message["game_state"]["potions"] = []
    return message


class ConfidenceAutoTests(unittest.TestCase):
    def test_environment_automation_collects_combat_rewards_in_order(self) -> None:
        message = full_combat_reward_message()
        result = controller.environment_auto_action(message, controller.semantic_actions(message))
        self.assertIsNotNone(result)
        assert result is not None
        action, detail = result
        self.assertEqual(action["command"], "CHOOSE 0")
        self.assertEqual(detail["reward_type"], "GOLD")
        self.assertEqual(action["label"], "自动拾取金币 19")

    def test_combat_observation_calculates_per_hit_card_values_and_intent(self) -> None:
        message = combat_message()
        combat = message["game_state"]["combat_state"]
        combat["hand"] = [
            {
                "id": "STRIKE_R",
                "name": "Strike",
                "type": "Attack",
                "damage": 6,
                "upgrades": 0,
                "is_playable": True,
                "has_target": True,
            },
            {
                "id": "DEFEND_R",
                "name": "Defend",
                "type": "Skill",
                "block": 5,
                "upgrades": 0,
                "is_playable": True,
                "has_target": False,
            },
        ]
        combat["player"] = {
            "current_hp": 75,
            "max_hp": 88,
            "block": 0,
            "energy": 3,
            "powers": [
                {"id": "Strength", "name": "力量", "amount": 1},
                {"id": "Weak", "name": "虚弱", "amount": 1},
                {"id": "Dexterity", "name": "敏捷", "amount": 2},
                {"id": "Frail", "name": "脆弱", "amount": 1},
                {"id": "Vulnerable", "name": "易伤", "amount": 1},
            ],
        }
        combat["monsters"] = [
            {
                "id": "Louse",
                "name": "Louse",
                "current_hp": 15,
                "max_hp": 15,
                "is_gone": False,
                "intent": "ATTACK",
                "move_base_damage": 6,
                "move_adjusted_damage": 6,
                "move_hits": 2,
                "powers": [],
            }
        ]

        observation = controller.normalize_state(message)
        strike, defend = observation["combat"]["hand"]
        monster = observation["combat"]["monsters"][0]
        # floor((6 + Strength 1) * 0.75), per hit; it is not multiplied by
        # the card's or enemy's hit count here.
        self.assertEqual(strike["final_single_hit_damage"], 5)
        self.assertEqual(defend["final_block"], 5)  # floor((5 + 2) * .75)
        self.assertEqual(monster["intent_final_damage"], 9)  # floor(6 * 1.5)
        self.assertEqual(monster["intent_final_total_damage"], 18)

    def test_combat_observation_falls_back_to_base_intent_and_enemy_powers(self) -> None:
        message = combat_message()
        combat = message["game_state"]["combat_state"]
        combat["player"] = {"powers": []}
        combat["monsters"] = [
            {
                "id": "Louse",
                "name": "Louse",
                "is_gone": False,
                "move_base_damage": 6,
                "move_adjusted_damage": -1,
                "move_hits": 2,
                "powers": [{"id": "Strength", "amount": 1}],
            }
        ]
        observation = controller.normalize_state(message)
        monster = observation["combat"]["monsters"][0]
        self.assertEqual(monster["intent_final_damage"], 7)
        self.assertEqual(monster["intent_final_total_damage"], 14)

    def test_dynamic_card_without_live_formula_does_not_claim_a_fixed_value(self) -> None:
        message = combat_message()
        combat = message["game_state"]["combat_state"]
        combat["player"] = {"block": 12, "powers": []}
        combat["hand"] = [
            {
                "id": "Body Slam",
                "name": "Body Slam",
                "type": "ATTACK",
                "is_playable": True,
            }
        ]
        observation = controller.normalize_state(message)
        self.assertNotIn("final_single_hit_damage", observation["combat"]["hand"][0])

    def test_environment_automation_leaves_relic_reward_for_policy(self) -> None:
        message = full_combat_reward_message()
        message["game_state"]["screen_state"]["rewards"] = [
            {"reward_type": "RELIC", "relic": {"id": "Key", "name": "钥匙"}},
        ]
        message["game_state"]["choice_list"] = ["relic"]
        self.assertIsNone(controller.environment_auto_action(message, controller.semantic_actions(message)))

    def test_environment_automation_skips_full_potion_and_opens_card_reward(self) -> None:
        message = full_combat_reward_message()
        message["game_state"]["potions"] = [
            {"id": "StrengthPotion"},
            {"id": "DexterityPotion"},
            {"id": "FirePotion"},
        ]
        message["game_state"]["screen_state"]["rewards"] = [
            {"reward_type": "POTION", "potion": {"id": "FruitJuice"}},
            {"reward_type": "CARD"},
        ]
        message["game_state"]["choice_list"] = ["potion", "card"]
        result = controller.environment_auto_action(message, controller.semantic_actions(message))
        self.assertIsNotNone(result)
        assert result is not None
        action, detail = result
        self.assertEqual(action["command"], "CHOOSE 1")
        self.assertEqual(detail["reward_type"], "CARD")

    def test_sozu_masks_potion_rewards_and_shop_potions(self) -> None:
        message = full_combat_reward_message()
        message["game_state"]["relics"] = [{"id": "Sozu", "name": "Sozu"}]
        message["game_state"]["screen_state"]["rewards"] = [
            {"reward_type": "POTION", "potion": {"id": "DexterityPotion", "name": "敏捷药水"}},
            {"reward_type": "CARD"},
        ]
        message["game_state"]["choice_list"] = ["potion", "card"]
        self.assertEqual(
            [action["id"] for action in controller.semantic_actions(message)],
            ["CHOOSE:index=1", "PROCEED"],
        )
        _, _, error = controller.parse_human_input("0", message)
        self.assertIn("苏祖", error or "")

        shop = {
            "in_game": True,
            "available_commands": ["choose", "proceed"],
            "game_state": {
                "screen_type": "SHOP_SCREEN",
                "choice_list": ["敏捷药水 50", "打击"],
                "relics": [{"id": "Sozu", "name": "Sozu"}],
                "screen_state": {"potions": [{"id": "DexterityPotion", "name": "敏捷药水"}]},
            },
        }
        self.assertEqual(
            [action["id"] for action in controller.semantic_actions(shop)],
            ["CHOOSE:index=1", "PROCEED"],
        )

    def test_map_actions_filter_stale_potion_operations(self) -> None:
        message = {
            "in_game": True,
            "available_commands": ["choose", "potion", "return"],
            "game_state": {
                "screen_type": "MAP",
                "room_phase": "COMPLETE",
                "choice_list": ["x=2", "使用药水", "x=3"],
            },
        }
        actions = controller.semantic_actions(message)
        leaked = [
            {"id": "POTION_USE:slot=0", "command": "POTION USE 0", "kind": "potion_use"},
            {"id": "A3", "command": "USE 0", "kind": "choose", "label": "使用药水"},
        ]
        self.assertEqual([action["id"] for action in actions], ["CHOOSE:index=0", "CHOOSE:index=2", "RETURN"])
        self.assertTrue(all(controller.is_map_potion_action(action) for action in leaked))

    def test_non_map_choice_screen_exposes_potion_operations(self) -> None:
        message = {
            "in_game": True,
            "available_commands": ["choose", "potion"],
            "game_state": {
                "screen_type": "COMBAT_REWARD",
                "room_phase": "COMPLETE",
                "choice_list": ["potion", "card"],
                "potions": [
                    {
                        "id": "AttackPotion",
                        "name": "攻击药水",
                        "requires_target": True,
                        "can_use": True,
                        "can_discard": True,
                    },
                    {"id": "BloodPotion", "name": "鲜血药水", "can_use": True, "can_discard": True},
                    {"id": "SwiftPotion", "name": "迅捷药水", "can_use": True, "can_discard": True},
                ],
            },
        }
        actions = controller.semantic_actions(message)
        self.assertEqual(
            [action["id"] for action in actions],
            [
                "POTION_DISCARD:slot=0",
                "POTION_USE:slot=1",
                "POTION_DISCARD:slot=1",
                "POTION_DISCARD:slot=2",
                "CHOOSE:index=1",
            ],
        )
        self.assertEqual(controller.parse_human_input("u 1", message)[0], "POTION USE 1")
        self.assertEqual(controller.parse_human_input("d 0", message)[0], "POTION DISCARD 0")

    def test_noncombat_potion_that_cannot_be_used_is_discard_only(self) -> None:
        message = {
            "in_game": True,
            "available_commands": ["potion"],
            "game_state": {
                "screen_type": "EVENT",
                "room_phase": "EVENT",
                "potions": [{"id": "SwiftPotion", "name": "迅捷药水", "can_use": True, "can_discard": True}],
            },
        }
        actions = controller.semantic_actions(message)
        self.assertEqual([action["id"] for action in actions], ["POTION_DISCARD:slot=0"])
        self.assertIsNone(controller.environment_auto_action(message, actions))

    def test_ectoplasm_masks_gold_reward_and_automation_skips_it(self) -> None:
        message = combat_reward_message()
        message["game_state"]["relics"] = [{"id": "Ectoplasm", "name": "灵体外质"}]
        actions = controller.semantic_actions(message)
        self.assertEqual([action["id"] for action in actions], ["CHOOSE:index=1", "PROCEED"])
        result = controller.environment_auto_action(message, actions)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0]["command"], "CHOOSE 1")
        _, _, error = controller.parse_human_input("0", message)
        self.assertIn("灵体外质", error or "")

    def test_boss_profile_and_relic_constraints_are_model_visible(self) -> None:
        message = combat_message()
        message["game_state"]["act_boss"] = "Hexaghost"
        message["game_state"]["relics"] = [{"id": "Velvet Choker", "name": "添水"}]
        history = [
            {"action_kind": "play"},
            {"action_kind": "play"},
        ]
        observation = controller.policy_observation(message, action_history=history)
        self.assertEqual(observation["act_boss_profile"]["id"], "HEXAGHOST")
        self.assertIn("strategy", observation["act_boss_profile"])
        strategy = observation["act_boss_profile"]["strategy"]
        self.assertIn("长线输出", strategy)
        self.assertIn("尽快结束战斗", strategy)
        self.assertGreater(len(strategy), 100)
        self.assertNotIn("moves", observation["act_boss_profile"])
        strike = next(item for item in observation["deck"] if item["id"] == "Strike_R")
        self.assertEqual(strike["count"], 2)
        choker = next(item for item in observation["decision_constraints"] if item["kind"] == "CARD_PLAY_LIMIT")
        self.assertEqual(choker["plays_remaining_this_turn"], 4)

    def test_combat_observation_adds_static_enemy_knowledge(self) -> None:
        message = combat_message()
        monster = message["game_state"]["combat_state"]["monsters"][0]
        monster.update({"id": "JAWWORM", "name": "Jaw Worm", "current_hp": 41, "max_hp": 46})
        observation = controller.policy_observation(message)
        knowledge = observation["combat"]["monsters"][0]["knowledge"]
        self.assertEqual(knowledge["id"], "JAWWORM")
        self.assertEqual(knowledge["type"], "Normal")
        self.assertEqual(knowledge["min_hp"], 40)
        self.assertEqual(knowledge["max_hp_ascension"], 46)
        self.assertTrue(any(move["id"] == "CHOMP" for move in knowledge["moves"]))
        self.assertEqual(knowledge["source"], "static_knowledge")

    def test_noncombat_observation_drops_enemy_knowledge(self) -> None:
        message = combat_message()
        message["game_state"].update({"screen_type": "EVENT", "room_phase": "EVENT"})
        observation = controller.policy_observation(message)
        self.assertNotIn("combat", observation)

    def test_runic_dome_hides_live_enemy_intents_from_model_and_terminal(self) -> None:
        message = combat_message()
        message["game_state"]["relics"] = [{"id": "Runic Dome", "name": "符文圆顶"}]
        monster = message["game_state"]["combat_state"]["monsters"][0]
        monster.update({"intent": "ATTACK", "move_base_damage": 8, "move_adjusted_damage": 9, "move_hits": 1})
        observation = controller.policy_observation(message)
        model_monster = observation["combat"]["monsters"][0]
        self.assertNotIn("intent", model_monster)
        self.assertNotIn("intent_damage", model_monster)
        self.assertIn("Intent=HIDDEN", controller.format_view(message, "run-1"))

    def test_environment_automation_opens_shop_before_policy(self) -> None:
        command, action, decision = controller.ask_action(NoBridge(), shop_room_message(), "run-1", policy=FakePolicy({}))
        self.assertEqual(command, "CHOOSE 0")
        self.assertEqual(action["label"], "自动打开商人界面")
        assert decision is not None
        self.assertEqual(decision["source"], "environment_auto")
        self.assertEqual(decision["environment"]["rule"], "shop_open_merchant")

    def test_automatic_action_waits_briefly_before_command(self) -> None:
        with patch.object(controller.time, "sleep") as sleep:
            command, action, decision = controller.ask_action(
                NoBridge(), shop_room_message(), "run-1", policy=FakePolicy({})
            )
        self.assertEqual(command, "CHOOSE 0")
        self.assertEqual(action["kind"], "choose")
        self.assertIsNotNone(decision)
        sleep.assert_called_once_with(controller.AUTOMATIC_ACTION_DELAY_SECONDS)

    def test_single_shop_continue_is_auto_executed_without_policy(self) -> None:
        policy = FakePolicy({})
        command, action, decision = controller.ask_action(
            NoBridge(), shop_continue_message(), "run-1", policy=policy
        )
        self.assertEqual(command, "PROCEED")
        self.assertEqual(action["id"], "PROCEED")
        self.assertEqual(action["label"], "自动执行 PROCEED")
        self.assertEqual(policy.calls, 0)
        assert decision is not None
        self.assertEqual(decision["source"], "environment_auto")
        self.assertEqual(decision["environment"]["rule"], "single_legal_action")

    def test_empty_combat_auto_ends_turn_without_policy(self) -> None:
        policy = FakePolicy({})
        command, action, decision = controller.ask_action(
            NoBridge(), empty_combat_message(), "run-1", policy=policy
        )
        self.assertEqual(command, "END")
        self.assertEqual(action["id"], "END_TURN")
        self.assertEqual(action["label"], "自动执行 结束回合")
        self.assertEqual(policy.calls, 0)
        assert decision is not None
        self.assertEqual(decision["source"], "environment_auto")
        self.assertEqual(decision["environment"]["rule"], "single_legal_action")

    def test_environment_automation_is_in_recent_action_history(self) -> None:
        message = full_combat_reward_message()
        result = controller.environment_auto_action(message, controller.semantic_actions(message))
        assert result is not None
        action, detail = result
        pending = controller.make_pending(
            message,
            "run-1",
            None,
            2,
            action,
            controller.environment_auto_decision(action, detail),
        )
        history = controller.accepted_action_history_item(pending, {"game_state": {"screen_type": "COMBAT_REWARD"}})
        self.assertEqual(history["source"], "environment_auto")
        self.assertEqual(controller.recent_action_summary([history])[0]["source"], "environment_auto")

    def test_detects_smoke_bomb_from_pending_pre_action_inventory(self) -> None:
        message = combat_message()
        message["available_commands"] = ["potion"]
        message["game_state"]["potions"] = [{"id": "SmokeBomb", "name": "烟雾弹", "can_use": True}]
        action = {
            "id": "POTION_USE:slot=0:target=0",
            "kind": "potion_use",
            "command": "POTION 0 0",
            "slot": 0,
        }
        pending = controller.make_pending(message, "run-1", None, 0, action)

        self.assertTrue(controller.is_smoke_bomb_potion_use(pending))

    def test_does_not_mistake_other_potions_for_smoke_bomb(self) -> None:
        message = combat_message()
        message["available_commands"] = ["potion"]
        message["game_state"]["potions"] = [{"id": "AttackPotion", "name": "攻击药水", "can_use": True}]
        action = {"id": "POTION_USE:slot=0:target=0", "kind": "potion_use", "command": "POTION 0 0", "slot": 0}
        pending = controller.make_pending(message, "run-1", None, 0, action)

        self.assertFalse(controller.is_smoke_bomb_potion_use(pending))

    def test_smoke_bomb_transition_blocks_only_the_old_combat_state(self) -> None:
        self.assertTrue(controller.smoke_bomb_escape_in_progress(combat_message()))
        self.assertFalse(controller.smoke_bomb_escape_in_progress(combat_reward_message()))
        self.assertFalse(controller.smoke_bomb_escape_in_progress({"in_game": False, "game_state": {}}))

    def test_normalizes_only_current_legal_actions(self) -> None:
        scores = controller.normalize_action_logprobs(
            [
                {"action_id": "A1", "logprob": -0.1},
                {"action_id": "A2", "logprob": -2.1},
            ]
        )
        assert scores is not None
        self.assertEqual([item["action_id"] for item in scores], ["A1", "A2"])
        self.assertAlmostEqual(sum(item["probability"] for item in scores), 1.0)
        summary = controller.confidence_summary(scores, 0.8)
        self.assertEqual(summary["route"], "auto_execute")

    def test_low_top_probability_requires_human_confirmation(self) -> None:
        scores = controller.normalize_action_logprobs(
            [
                {"action_id": "A1", "logprob": -0.69},
                {"action_id": "A2", "logprob": -0.70},
            ]
        )
        assert scores is not None
        summary = controller.confidence_summary(scores, 0.85)
        self.assertEqual(summary["route"], "human_confirm")

    def test_auto_action_stays_inside_legal_actions(self) -> None:
        actions = controller.semantic_actions(event_message())
        decision = {
            "model_action": "CHOOSE:index=0",
            "confidence": {"route": "auto_execute"},
        }
        action = controller.auto_action_for_decision(decision, actions)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["command"], "CHOOSE 0")
        decision["model_action"] = "PLAY:card=1"
        self.assertIsNone(controller.auto_action_for_decision(decision, actions))

    def test_high_confidence_skips_terminal_and_marks_trace_source(self) -> None:
        decision = {
            "mode": "confidence_auto",
            "model_action": "CHOOSE:index=0",
            "model_candidates": [{"action_id": "CHOOSE:index=0"}],
            "confidence": {"route": "auto_execute", "top_probability": 0.91, "margin": 0.67},
        }
        command, action, final = controller.ask_action(NoBridge(), event_message(), "run-1", policy=FakePolicy(decision))
        self.assertEqual(command, "CHOOSE 0")
        self.assertEqual(action["id"], "CHOOSE:index=0")
        assert final is not None
        self.assertEqual(final["source"], "model_auto")
        self.assertFalse(final["overridden"])

    def test_teacher_action_skips_terminal_and_marks_teacher_source(self) -> None:
        command, action, final = controller.ask_action(NoBridge(), event_message(), "run-1", teacher=FakeTeacher())
        self.assertEqual(command, "CHOOSE 1")
        self.assertEqual(action["id"], "CHOOSE:index=1")
        assert final is not None
        self.assertEqual(final["source"], "teacher_auto")
        self.assertFalse(final["overridden"])
        self.assertEqual(final["teacher"]["prompt_version"], "test-v1")

    def test_disabled_teacher_does_not_shadow_policy(self) -> None:
        decision = {
            "mode": "confidence_auto",
            "model_id": "policy-test",
            "model_action": "CHOOSE:index=0",
            "model_candidates": [{"action_id": "CHOOSE:index=0"}],
            "confidence": {"route": "auto_execute", "top_probability": 0.91, "margin": 0.67},
        }
        teacher = FakeTeacher()
        teacher.enabled = False
        command, action, final = controller.ask_action(
            NoBridge(),
            event_message(),
            "run-1",
            policy=FakePolicy(decision),
            teacher=teacher,
        )
        self.assertEqual(command, "CHOOSE 0")
        self.assertEqual(action["id"], "CHOOSE:index=0")
        assert final is not None
        self.assertEqual(final["source"], "model_auto")

    def test_teacher_action_must_remain_legal(self) -> None:
        action = controller.teacher_action_for_decision(
            {"mode": "teacher_collection", "model_action": "CHOOSE:index=99", "model_candidates": [{}]},
            controller.semantic_actions(event_message()),
        )
        self.assertIsNone(action)

    def test_teacher_thinking_is_stripped_from_durable_trace_decision(self) -> None:
        decision = {
            "mode": "teacher_collection",
            "teacher": {"thinking": "private reasoning", "output": '{"action_id":"A1"}', "latency_ms": 2},
        }
        durable = controller.trace_decision(decision)
        assert durable is not None
        self.assertNotIn("thinking", durable["teacher"])
        self.assertNotIn("output", durable["teacher"])
        self.assertEqual(durable["teacher"]["latency_ms"], 2)

    def test_skipped_combat_card_reward_cannot_reopen_the_same_offer(self) -> None:
        message = combat_reward_message()
        declined: set[tuple[int, int]] = set()
        remembered = controller.remember_declined_combat_card_reward(
            {"screen_type": "CARD_REWARD", "action": {"id": "SKIP", "kind": "skip"}},
            message,
            declined,
        )
        self.assertTrue(remembered)
        self.assertEqual(declined, {(1, 3)})
        action_ids = [item["id"] for item in controller.semantic_actions(message, declined_card_reward_keys=declined)]
        self.assertEqual(action_ids, ["CHOOSE:index=0", "PROCEED"])
        self.assertEqual(
            controller.run_memory_summary(set(), declined),
            {"visited_shop_rooms": [], "declined_card_rewards": [{"act": 1, "floor": 3}]},
        )

    def test_scoring_decision_records_all_aliases_and_confidence(self) -> None:
        policy = controller.PolicyClient.__new__(controller.PolicyClient)
        policy.top_k = 2
        policy.auto_min_probability = 0.85
        policy.model = "test-model"
        aliases = {
            "A1": {"id": "CHOOSE:index=0", "command": "CHOOSE 0", "label": "Take"},
            "A2": {"id": "CHOOSE:index=1", "command": "CHOOSE 1", "label": "Leave"},
        }
        decision = policy._scoring_decision(
            {
                "model_id": "test-model",
                "action_scores": [
                    {"action_id": "A2", "logprob": -2.0},
                    {"action_id": "A1", "logprob": 0.0},
                ],
            },
            aliases,
        )
        self.assertEqual(decision["model_action"], "CHOOSE:index=0")
        self.assertEqual(decision["confidence"]["route"], "auto_execute")
        self.assertEqual(len(decision["action_scores"]), 2)
        self.assertAlmostEqual(sum(item["probability"] for item in decision["action_scores"]), 1.0)

    def test_overlay_view_contains_only_summary_and_action_table_fields(self) -> None:
        message = combat_message()
        payload = controller.overlay_view_data(message, "run-1", controller.semantic_actions(message))
        self.assertEqual(
            payload["summary"],
            "RUN run-1 | IRONCLAD A0 | Act 1 Floor 2 | NONE\nHP 75/88 | Gold 117 | Room COMBAT",
        )
        self.assertEqual(
            payload["deck"],
            [
                {"name": "Bash", "upgrades": 1, "count": 1},
                {"name": "Strike", "upgrades": 0, "count": 2},
            ],
        )
        self.assertEqual(payload["recent_actions"], [])
        self.assertEqual(
            payload["legal_actions"],
            [
                {"type": "出牌", "selection": "Strike", "target": "敌人 0", "command": "p 1 0"},
                {"type": "结束回合", "selection": "", "target": "", "command": "e"},
            ],
        )

    def test_overlay_input_hints_are_directly_accepted_by_terminal(self) -> None:
        message = event_message()
        choice = controller.semantic_actions(message)[1]
        self.assertEqual(controller.overlay_input_hint(choice), "1")
        self.assertEqual(controller.overlay_input_hint({"kind": "confirm", "command": "PROCEED"}), "c")
        self.assertEqual(controller.overlay_input_hint({"kind": "return", "command": "RETURN"}), "x")

    def test_grid_normalization_exposes_selection_purpose(self) -> None:
        message = {
            "in_game": True,
            "game_state": {
                "screen_type": "GRID",
                "screen_state": {
                    "cards": [{"id": "Strike_R", "name": "Strike"}],
                    "for_purge": True,
                    "num_cards": 1,
                },
            },
        }
        screen_state = controller.normalize_state(message)["screen_state"]
        self.assertEqual(screen_state["selection_mode"], "PURGE")
        self.assertEqual(screen_state["selection_instruction"], "选择 1 张牌移除")

    def test_grid_selection_hides_exact_selected_card_instance(self) -> None:
        message = grid_selection_message()
        actions = controller.semantic_actions(message)
        self.assertEqual(
            [action["id"] for action in actions],
            ["CHOOSE:index=0", "CHOOSE:index=2"],
        )
        command, action, error = controller.parse_human_input("1", message)
        self.assertIsNone(command)
        self.assertIsNone(action)
        self.assertEqual(error, "该选项当前不可选择。")

    def test_hand_selection_hides_exact_selected_card_instance(self) -> None:
        message = grid_selection_message()
        cards = message["game_state"]["screen_state"].pop("cards")
        message["game_state"]["screen_type"] = "HAND_SELECT"
        message["game_state"]["screen_state"] = {"hand": cards, "selected": [cards[1]], "max_cards": 3}
        self.assertEqual(
            [action["id"] for action in controller.semantic_actions(message)],
            ["CHOOSE:index=0", "CHOOSE:index=2"],
        )

    def test_astrolabe_pickup_marks_first_continuous_grid_selection(self) -> None:
        message = relic_pickup_grid_message("Astrolabe", "Astrolabe")
        history = [controller.accepted_action_history_item(relic_pickup_pending(), message)]
        observation = controller.policy_observation(message, action_history=history)
        context = observation["selection_context"]
        self.assertEqual(context["kind"], "RELIC_PICKUP_CARD_SELECTION")
        self.assertEqual(context["selection_type"], "ASTROLABE_TRANSFORM_UPGRADE")
        self.assertEqual(context["required_cards"], 3)
        self.assertEqual(observation["screen_state"]["selection_mode"], "TRANSFORM_UPGRADE")
        self.assertIn("变化为随机牌并升级", observation["screen_state"]["selection_instruction"])

        history.append({"screen_before": "GRID", "screen_after": "GRID"})
        continued = controller.policy_observation(message, action_history=history)
        self.assertEqual(continued["selection_context"]["selection_type"], "ASTROLABE_TRANSFORM_UPGRADE")

        history.append({"screen_before": "EVENT", "screen_after": "GRID"})
        self.assertNotIn("selection_context", controller.policy_observation(message, action_history=history))

    def test_empty_cage_pickup_marks_two_card_purge(self) -> None:
        message = relic_pickup_grid_message("Empty Cage", "Empty Cage")
        history = [controller.accepted_action_history_item(relic_pickup_pending(), message)]
        observation = controller.policy_observation(message, action_history=history)
        self.assertEqual(observation["selection_context"]["selection_type"], "EMPTY_CAGE_PURGE")
        self.assertEqual(observation["screen_state"]["selection_mode"], "PURGE")
        self.assertEqual(observation["selection_context"]["required_cards"], 2)

    def test_recent_action_summary_is_bounded_and_keeps_semantics(self) -> None:
        history = [
            {
                "step_id": index,
                "screen_before": "EVENT",
                "action_kind": "choose",
                "action_id": f"CHOOSE:index={index}",
                "action": "净化",
                "screen_after": "GRID",
            }
            for index in range(8)
        ]
        summary = controller.recent_action_summary(history)
        self.assertEqual(len(summary), 6)
        self.assertEqual(summary[0]["step_id"], 2)
        self.assertEqual(summary[-1]["action"], "净化")

    def test_policy_observation_carries_recent_actions(self) -> None:
        observation = controller.policy_observation(
            event_message(),
            action_history=[
                {
                    "step_id": 4,
                    "screen_before": "EVENT",
                    "action_kind": "choose",
                    "action_id": "CHOOSE:index=1",
                    "action": "净化",
                    "screen_after": "GRID",
                }
            ],
        )
        self.assertEqual(observation["recent_actions"][0]["screen_after"], "GRID")
        self.assertEqual(observation["recent_actions"][0]["action"], "净化")

    def test_event_observation_keeps_event_specific_transition_history(self) -> None:
        message = event_message()
        message["game_state"]["choice_list"] = ["Bash", "Dropkick", "card2"]
        message["game_state"]["screen_state"] = {
            "event_id": "Match and Keep!",
            "event_name": "对对碰！",
            "body_text": "",
            "options": [],
        }
        history = [
            {
                "step_id": 12,
                "screen_before": "EVENT",
                "action_kind": "choose",
                "action_id": "CHOOSE:index=0",
                "action": "card0",
                "screen_after": "EVENT",
                "event_before": {
                    "event_id": "Match and Keep!",
                    "event_name": "对对碰！",
                    "body_text": "十二张牌，搭对就拿走。",
                    "choices": ["card0", "card1", "card2"],
                    "options": [],
                },
                "event_after": {
                    "event_id": "Match and Keep!",
                    "event_name": "对对碰！",
                    "body_text": "",
                    "choices": ["Bash", "card1", "card2"],
                    "options": [],
                },
            }
        ]
        observation = controller.policy_observation(message, action_history=history)
        self.assertEqual(observation["event_history"][0]["action"], "card0")
        self.assertEqual(observation["event_history"][0]["before"]["body_text"], "十二张牌，搭对就拿走。")
        self.assertEqual(observation["event_history"][0]["after"]["choices"][0], "Bash")

    def test_combat_generated_card_choice_carries_potion_source(self) -> None:
        pre_message = combat_message()
        pre_message["game_state"]["potions"] = [
            {
                "id": "AttackPotion",
                "name": "攻击药水",
                "description": "Choose 1 of 3 random Attack cards to add to your hand, it costs 0 this turn.",
            }
        ]
        action = {
            "id": "POTION_USE:slot=0",
            "command": "POTION USE 0",
            "kind": "potion_use",
            "label": "使用药水 攻击药水",
        }
        pending = controller.make_pending(pre_message, None, None, 1, action)
        post_message = {
            "in_game": True,
            "available_commands": ["choose", "skip"],
            "game_state": {
                "screen_type": "CARD_REWARD",
                "room_phase": "COMBAT",
                "choice_list": ["Anger", "Skip"],
                "screen_state": {"cards": [{"id": "Anger", "name": "Anger"}], "skip_available": True},
            },
        }
        history = [controller.accepted_action_history_item(pending, post_message)]
        observation = controller.policy_observation(post_message, action_history=history)
        context = observation["selection_context"]
        self.assertEqual(context["kind"], "COMBAT_GENERATED_CARD_CHOICE")
        self.assertEqual(context["source"]["source_type"], "POTION")
        self.assertEqual(context["source"]["id"], "AttackPotion")

    def test_combat_generated_card_choice_carries_played_card_source(self) -> None:
        pre_message = combat_message()
        pre_message["game_state"]["combat_state"]["hand"] = [
            {
                "id": "Discovery",
                "name": "发现",
                "type": "SKILL",
                "cost": 1,
                "is_playable": True,
                "description": "Choose 1 of 3 random cards to add into your hand. It costs 0 this turn.",
            }
        ]
        action = {
            "id": "PLAY:card=1",
            "command": "PLAY 1",
            "kind": "play",
            "label": "出牌 发现",
        }
        pending = controller.make_pending(pre_message, None, None, 1, action)
        post_message = {
            "in_game": True,
            "available_commands": ["choose", "skip"],
            "game_state": {
                "screen_type": "CARD_REWARD",
                "room_phase": "COMBAT",
                "choice_list": ["Apotheosis", "Skip"],
                "screen_state": {"cards": [{"id": "Apotheosis", "name": "Apotheosis"}], "skip_available": True},
            },
        }
        history = [controller.accepted_action_history_item(pending, post_message)]
        observation = controller.policy_observation(post_message, action_history=history)
        self.assertEqual(observation["selection_context"]["source"]["source_type"], "CARD")
        self.assertEqual(observation["selection_context"]["source"]["name"], "发现")


if __name__ == "__main__":
    unittest.main()
