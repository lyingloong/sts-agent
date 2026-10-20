from __future__ import annotations

import copy
import json
import unittest

from sts_agent.policy.presentation import prepare_model_context
from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.runtime.controller import normalize_state, policy_observation, route_policy_observation
from tests.test_room_history import message, record


class ModelPresentationTests(unittest.TestCase):
    def test_same_keyword_name_with_different_actual_meaning_is_not_erased(self):
        observation = {"deck": [{"keyword_tooltips": [
            {"header": "愤怒", "body": "姿态：伤害翻倍。"},
            {"header": "愤怒", "body": "每打出攻击牌获得格挡。"}]}]}
        result = prepare_model_context(observation)
        self.assertEqual(len(result["keyword_explanations"]), 2)
        self.assertEqual(result, prepare_model_context(copy.deepcopy(result)))

    def test_stable_reward_references_survive_consumption_and_linked_keys(self):
        a, b = "reward-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "reward-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        state = {"room_history": {"steps": [{"details": {"object": {"reward_id": a, "reward_ref": "R8"}}}]},
                 "screen_state": {"rewards": [{"reward_id": b, "reward_ref": "R9", "linked_reward_id": a,
                                               "linked_reward_ref": "R8", "reward_index": 0}]}}
        raw = copy.deepcopy(state)
        projected = prepare_model_context(state)
        self.assertEqual(projected["screen_state"]["rewards"][0]["reward_id"], "R9")
        self.assertEqual(projected["screen_state"]["rewards"][0]["linked_reward_id"], "R8")
        self.assertEqual(projected["room_history"]["steps"][0]["details"]["object"]["reward_id"], "R8")
        self.assertEqual(projected, prepare_model_context(copy.deepcopy(projected)))
        self.assertEqual(raw["screen_state"]["rewards"][0]["reward_id"], b)
        self.assertNotIn("aaaaaaaa", json.dumps(projected))
        self.assertNotIn("reward_ref", json.dumps(projected))

    def test_legacy_identity_aliases_are_unique_and_match_history_and_source(self):
        state = {"room_history": {"steps": [{"details": {"source_reward_id": "a"}}]},
                 "screen_state": {"source_reward_id": "a", "rewards": [{"reward_id": "a"}, {"reward_id": "b"}]}}
        result = prepare_model_context(state)
        self.assertEqual([x["reward_id"] for x in result["screen_state"]["rewards"]], ["R1", "R2"])
        self.assertEqual(result["screen_state"]["source_reward_id"], "R1")
        self.assertEqual(result["room_history"]["steps"][0]["details"]["source_reward_id"], "R1")
        self.assertEqual(result, prepare_model_context(copy.deepcopy(result)))

    def test_card_uuid_hidden_but_type_identity_and_selection_purpose_retained(self):
        card = {"id": "Strike_R", "name": "打击", "uuid": "44ca978d-f908-40ce-98bf-448f012a72af"}
        msg = message("GRID", screen_state={"cards": [card], "confirmation_card": card, "for_purge": True})
        full = normalize_state(msg)
        self.assertIn("uuid", full["screen_state"]["confirmation_card"])
        observation = route_policy_observation(copy.deepcopy(full))
        self.assertNotIn("uuid", json.dumps(observation))
        self.assertEqual(observation["screen_state"]["confirmation_card"]["id"], "Strike_R")
        self.assertEqual(observation["screen_state"]["selection_mode"], "PURGE")

    def test_current_runtime_keywords_deduplicated_not_taken_from_history(self):
        tip = {"header": "不能被打出", "body": "不能从手牌打出这张牌。"}
        card = {"id": "Burn", "name": "灼伤", "keyword_tooltips": [tip], "description": "不能被打出。"}
        msg = message("CARD_REWARD", screen_state={"cards": [card, card]})
        msg["game_state"]["deck"] = [card]
        observation = policy_observation(msg)
        self.assertEqual(observation["keyword_explanations"], [{"name": tip["header"], "description": tip["body"]}])
        self.assertNotIn("keyword_tooltips", json.dumps(observation))
        for template in ("canonical", "state_first", "decision_first"):
            prompt = build_scoring_prompt(observation, [{"id": "A1", "label": "选择灼伤"}], template)
            self.assertEqual(prompt.count(tip["body"]), 1)
            self.assertIn("【词条解释】", prompt)
        legacy = {"room_history": {"steps": [{"keyword_tooltips": [{"header": "过时", "body": "历史解释"}]}]}}
        self.assertNotIn("keyword_explanations", prepare_model_context(legacy))

    def test_keyword_power_potion_relic_and_combat_selection_are_visible(self):
        tip = {"header": "敏捷", "body": "敏捷增加你从牌中获得的格挡值。"}
        for screen in ("MAP", "NONE", "GRID", "HAND_SELECT", "CARD_REWARD", "EVENT", "SHOP_SCREEN"):
            msg = message(screen)
            msg["game_state"]["potions"] = [{"id": "SpeedPotion", "name": "速度药水", "keyword_tooltips": [tip]}]
            observation = policy_observation(msg)
            self.assertEqual(observation["keyword_explanations"][0]["name"], "敏捷")

    def test_stable_reference_current_reward_and_room_history_agree(self):
        a = "reward-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        pre = message("COMBAT_REWARD", choice_list=["card"], screen_state={"rewards": [
            {"reward_id": a, "reward_ref": "R12", "reward_type": "CARD"}]})
        post = message("CARD_REWARD", screen_state={"cards": [], "source_reward_id": a, "source_reward_ref": "R12"})
        item = record(pre, post)
        observation = policy_observation(post, action_history=[item])
        self.assertEqual(observation["screen_state"]["source_reward_id"], "R12")
        self.assertEqual(observation["room_history"]["steps"][0]["details"]["object"]["reward_id"], "R12")
        self.assertEqual(item["details"]["object"]["reward_id"], a)

    def test_stale_combat_keywords_do_not_leak_into_noncombat_scene(self):
        msg = message("EVENT")
        msg["game_state"]["combat_state"] = {"player": {"powers": [
            {"id": "Dexterity", "keyword_tooltips": [{"header": "敏捷", "body": "STALE_COMBAT"}]}]}}
        self.assertNotIn("STALE_COMBAT", json.dumps(policy_observation(msg)))

    def test_stable_refs_are_unique_with_legacy_fallback_and_opaque_macro_ids(self):
        state = {"screen_state": {"rewards": [{"reward_id": "a", "reward_ref": "R1"}, {"reward_id": "b"}]},
                 "room_history": {"steps": [{"action_id": "SHOP:PURGE:card=aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
                    {"action_id": "SHOP:PURGE:card=bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"}]}}
        result = prepare_model_context(state)
        self.assertEqual([x["reward_id"] for x in result["screen_state"]["rewards"]], ["R1", "R2"])
        actions = [x["action_id"] for x in result["room_history"]["steps"]]
        self.assertNotEqual(actions[0], actions[1])
        self.assertNotIn("aaaa", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
