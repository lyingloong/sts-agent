from __future__ import annotations

import copy
import unittest

from sts_agent.commands.preview_policy_prompt import example_prompts
from sts_agent.game.sts1.campfire import CAMPFIRE_ACTIVITY_RULE
from sts_agent.learning.export_human_sft import export_record
from sts_agent.learning.export_teacher_sft import alias_actions
from sts_agent.policy.local_qwen import PROMPT_TEMPLATES, build_prompt, build_scoring_prompt
from sts_agent.policy.teacher import build_prompt as build_teacher_prompt
from sts_agent.runtime.controller import normalize_state, policy_observation, semantic_actions


def campfire_message():
    choices = ["rest", "smith", "recall"]
    labels = [("休息", "回复 24 点生命。"), ("锻造", "升级一张牌。"), ("回忆", "获得红宝石钥匙。")]
    return {
        "in_game": True, "ready_for_command": True, "available_commands": ["choose", "state"],
        "game_state": {"screen_type": "REST", "room_type": "RestRoom", "room_phase": "INCOMPLETE",
            "act": 1, "floor": 9, "current_hp": 40, "max_hp": 80, "choice_list": choices,
            "deck": [], "relics": [], "potions": [], "screen_state": {
                "rest_options": choices[:], "has_rested": False,
                "rest_option_details": [
                    {"choice_index": i, "choice": choice, "name": name, "description": description,
                     "usable": True, "raw_description": "RAW " + description,
                     "description_source": "game_runtime", "option_class": "synthetic." + choice}
                    for i, (choice, (name, description)) in enumerate(zip(choices, labels))
                ],
            },
        },
    }


class CampfireContextTests(unittest.TestCase):
    def test_runtime_descriptions_and_activity_rule_reach_models_without_raw_duplicate_fields(self):
        message = campfire_message()
        original = copy.deepcopy(message)
        for observation in (normalize_state(message), policy_observation(message)):
            screen = observation["screen_state"]
            self.assertEqual(screen["activity_rule"], CAMPFIRE_ACTIVITY_RULE)
            self.assertEqual(screen["rest_option_details"][0], {
                "choice_index": 0, "choice": "rest", "name": "休息", "description": "回复 24 点生命。", "usable": True})
            for detail in screen["rest_option_details"]:
                self.assertNotIn("raw_description", detail)
                self.assertNotIn("option_class", detail)
        observation = policy_observation(message)
        actions, _ = alias_actions(semantic_actions(message))
        prompts = [build_scoring_prompt(observation, actions, template) for template in PROMPT_TEMPLATES]
        prompts.extend([build_prompt(observation, actions, 3), build_teacher_prompt(observation, actions)])
        for prompt in prompts:
            for text in (CAMPFIRE_ACTIVITY_RULE, "回复 24 点生命。", "升级一张牌。", "获得红宝石钥匙。"):
                self.assertIn(text, prompt)
            for text in ("RAW ", "synthetic.rest"):
                self.assertNotIn(text, prompt)
        self.assertEqual(message, original)

    def test_only_labels_change_and_fine_grained_option_ids_and_commands_are_preserved(self):
        actions = semantic_actions(campfire_message())
        self.assertEqual([a["id"] for a in actions], ["CHOOSE:index=0", "CHOOSE:index=1", "CHOOSE:index=2"])
        self.assertEqual([a["command"] for a in actions], ["CHOOSE 0", "CHOOSE 1", "CHOOSE 2"])
        self.assertEqual([a["label"] for a in actions], ["休息——回复 24 点生命。", "锻造——升级一张牌。", "回忆——获得红宝石钥匙。"])
        self.assertTrue(all(a["kind"] == "choose" for a in actions))
        self.assertFalse(any(a.get("execution") for a in actions))

    def test_special_and_unknown_mod_options_keep_their_runtime_descriptions(self):
        message = campfire_message()
        choices = ["lift", "dig", "toke", "newmod"]
        message["game_state"]["choice_list"] = choices
        message["game_state"]["screen_state"]["rest_options"] = choices[:]
        message["game_state"]["screen_state"]["rest_option_details"] = [
            {"choice_index": i, "choice": choice, "name": f"活动{i}", "description": f"实际效果{i}", "usable": True}
            for i, choice in enumerate(choices)
        ]
        self.assertEqual([a["label"] for a in semantic_actions(message)], [f"活动{i}——实际效果{i}" for i in range(4)])

    def test_missing_list_is_unknown_and_missing_individual_fields_are_not_invented(self):
        for value in (None, {}, "unavailable"):
            message = campfire_message()
            message["game_state"]["screen_state"]["rest_option_details"] = value
            self.assertIsNone(normalize_state(message)["screen_state"]["rest_option_details"])
            self.assertEqual([a["label"] for a in semantic_actions(message)], ["rest", "smith", "recall"])
        message = campfire_message()
        details = message["game_state"]["screen_state"]["rest_option_details"]
        details[0].update(name=None, usable=None)
        details[1]["description"] = None
        self.assertEqual([a["label"] for a in semantic_actions(message)], ["rest——回复 24 点生命。", "锻造", "回忆——获得红宝石钥匙。"])

    def test_misaligned_or_malformed_details_never_explain_a_different_choice(self):
        for mutation in ("count", "order", "identity", "duplicate_index", "boolean_index", "bad_row"):
            with self.subTest(mutation=mutation):
                message = campfire_message()
                details = message["game_state"]["screen_state"]["rest_option_details"]
                if mutation == "count":
                    details.pop()
                elif mutation == "order":
                    details.reverse()
                elif mutation == "identity":
                    details[0]["choice"] = "smith"
                elif mutation == "duplicate_index":
                    details[1]["choice_index"] = 0
                elif mutation == "boolean_index":
                    details[0]["choice_index"] = False
                else:
                    details[0] = "bad"
                self.assertIsNone(normalize_state(message)["screen_state"]["rest_option_details"])
                self.assertEqual([a["label"] for a in semantic_actions(message)], ["rest", "smith", "recall"])

    def test_room_options_must_match_the_entire_actual_choose_space(self):
        for choices in (["smith", "rest", "recall"], ["another_choice", "smith", "recall"],
                        ["rest", "smith"], [], None):
            with self.subTest(choices=choices):
                message = campfire_message()
                if choices is None:
                    message["game_state"].pop("choice_list")
                else:
                    message["game_state"]["choice_list"] = choices
                for observation in (normalize_state(message), policy_observation(message)):
                    self.assertIsNone(observation["screen_state"]["rest_option_details"])
                    self.assertEqual(observation["screen_state"]["activity_rule"], CAMPFIRE_ACTIVITY_RULE)
                actions = semantic_actions(message)
                self.assertEqual([a["label"] for a in actions], choices or [])
                self.assertEqual([a["command"] for a in actions],
                                 [f"CHOOSE {i}" for i in range(len(choices or []))])

    def test_empty_used_campfire_only_proceeds_and_potions_do_not_change_choice_indices(self):
        message = campfire_message()
        message["available_commands"].append("potion")
        message["game_state"]["potions"] = [{"id": "BloodPotion", "name": "鲜血药水", "can_use": True, "can_discard": True}]
        actions = semantic_actions(message)
        self.assertEqual([a["command"] for a in actions], ["POTION USE 0", "POTION DISCARD 0", "CHOOSE 0", "CHOOSE 1", "CHOOSE 2"])
        self.assertEqual(actions[2]["label"], "休息——回复 24 点生命。")
        message["available_commands"] = ["proceed", "state"]
        message["game_state"]["choice_list"] = []
        message["game_state"]["screen_state"].update(rest_options=[], rest_option_details=[], has_rested=True)
        self.assertEqual([a["id"] for a in semantic_actions(message)], ["PROCEED"])
        self.assertEqual(normalize_state(message)["screen_state"]["rest_option_details"], [])

    def test_other_screens_do_not_inherit_campfire_rule_or_choice_descriptions(self):
        for screen in ("MAP", "EVENT", "GRID", "SHOP_ROOM", "SHOP_SCREEN"):
            message = campfire_message()
            message["game_state"]["screen_type"] = screen
            self.assertNotIn("activity_rule", normalize_state(message)["screen_state"])
            self.assertFalse(any("回复 24" in a.get("label", "") for a in semantic_actions(message)))

    def test_sft_uses_the_same_prompt_and_option_explanations_as_inference(self):
        message = campfire_message()
        observation, legal = policy_observation(message), semantic_actions(message)
        aliases, _ = alias_actions(legal)
        row = {"run_id": "campfire-test", "step_id": "option-1", "usable_for_sft": True,
               "training_policy_observation": observation, "training_legal_actions": legal,
               "training_action": legal[1]}
        sample, reason = export_record(row)
        self.assertEqual(reason, "kept")
        self.assertEqual(sample["target"], {"alias": "A2", "action_id": "CHOOSE:index=1"})
        for template in PROMPT_TEMPLATES:
            self.assertEqual(sample["prompt_variants"][template], build_scoring_prompt(observation, aliases, template))
            self.assertIn("A2：锻造——升级一张牌。", sample["prompt_variants"][template])
        self.assertIn(CAMPFIRE_ACTIVITY_RULE, sample["prompt"])

    def test_preview_has_an_actual_fully_assembled_campfire_example(self):
        prompt = example_prompts()["营火"]
        for text in ("营火活动规则", CAMPFIRE_ACTIVITY_RULE, "A1：休息——回复 24 点生命。", "A2：锻造——升级一张牌。"):
            self.assertIn(text, prompt)


if __name__ == "__main__":
    unittest.main()
