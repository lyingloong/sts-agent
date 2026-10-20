from __future__ import annotations

import copy
import json
import unittest

from sts_agent.policy.history import compact_history_step, history_step_text
from sts_agent.policy.reward_text import project_reward_text


class RewardTextProjectionTests(unittest.TestCase):
    def test_raw_and_aliased_rewards_keep_the_same_named_exclusive_choice(self):
        raw = {"screen_type": "COMBAT_REWARD", "screen_state": {"rewards": [
            {"reward_id": "reward-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "reward_ref": "R7",
             "reward_type": "RELIC", "choice_index": 0, "reward_index": 0, "relic": {"id": "Anchor", "name": "锚"}},
            {"reward_id": "reward-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "reward_ref": "R8",
             "reward_type": "SAPPHIRE_KEY", "choice_index": 1, "reward_index": 1,
             "linked_reward_id": "reward-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "linked_reward_ref": "R7"},
        ]}}
        original = copy.deepcopy(raw)
        projected = project_reward_text(copy.deepcopy(raw))
        self.assertEqual(raw, original)
        text = projected["screen_state"]["reward_relationships"]
        self.assertEqual(text, ["蓝宝石钥匙（选择位置=1）与遗物：锚（选择位置=0）二选一；领取其中一项会放弃另一项。"])
        aliased = copy.deepcopy(raw)
        for item in aliased["screen_state"]["rewards"]:
            item["reward_id"] = item["reward_ref"]
            if "linked_reward_ref" in item:
                item["linked_reward_id"] = item["linked_reward_ref"]
        self.assertEqual(project_reward_text(aliased), projected)
        self.assertEqual(project_reward_text(copy.deepcopy(projected)), projected)
        self.assertEqual(projected["screen_state"]["rewards"][0]["reward_index"], 0)
        encoded = json.dumps(projected)
        for hidden in ("reward_id", "reward_ref", "linked_reward_id", "linked_reward_ref", "R7", "R8", "aaaaaaaa"):
            self.assertNotIn(hidden, encoded)

    def test_same_named_rewards_are_disambiguated_by_actual_choice_positions(self):
        view = {"screen_state": {"rewards": [
            {"reward_id": "a", "reward_type": "RELIC", "name": "锚", "choice_index": 0},
            {"reward_id": "b", "reward_type": "RELIC", "name": "锚", "choice_index": 2},
            {"reward_id": "k", "reward_type": "SAPPHIRE_KEY", "choice_index": 3, "linked_reward_id": "b"},
        ]}}
        relationship = project_reward_text(view)["screen_state"]["reward_relationships"][0]
        self.assertIn("遗物：锚（选择位置=2）", relationship)
        self.assertNotIn("选择位置=0", relationship)
        self.assertIn("蓝宝石钥匙（选择位置=3）", relationship)

    def test_bidirectional_links_produce_one_relation_and_existing_indices_survive(self):
        view = {"screen_state": {"rewards": [
            {"reward_id": "a", "reward_type": "RELIC", "relic": {"name": "锚"}, "reward_index": 4, "linked_reward_id": "k"},
            {"reward_id": "k", "reward_type": "SAPPHIRE_KEY", "reward_index": 5, "linked_reward_id": "a"},
        ]}}
        projected = project_reward_text(view)
        relationships = projected["screen_state"]["reward_relationships"]
        self.assertEqual(len(relationships), 1)
        self.assertIn("当前奖励索引=4", relationships[0])
        self.assertIn("当前奖励索引=5", relationships[0])
        self.assertEqual(projected["screen_state"]["rewards"][0]["reward_index"], 4)
        self.assertNotIn("choice_index", projected["screen_state"]["rewards"][0])

    def test_missing_linked_reward_keeps_only_known_exclusivity(self):
        view = {"room_history": {"steps": [{"details": {"object": {"reward_id": "old", "name": "锚"}}}]},
                "screen_state": {"rewards": [{"reward_id": "key", "reward_type": "SAPPHIRE_KEY",
                    "linked_reward_id": "old", "choice_index": 0}]}}
        relationship = project_reward_text(view)["screen_state"]["reward_relationships"][0]
        self.assertIn("蓝宝石钥匙（选择位置=0）", relationship)
        self.assertIn("与另一项关联奖励二选一", relationship)
        self.assertIn("当前无法确定", relationship)
        self.assertNotIn("锚", relationship)
        self.assertNotIn("old", json.dumps(view))

    def test_ambiguous_identity_does_not_guess_a_specific_current_entity(self):
        view = {"screen_state": {"rewards": [
            {"reward_id": "duplicate", "reward_type": "RELIC", "name": "锚", "choice_index": 0},
            {"reward_id": "duplicate", "reward_type": "RELIC", "name": "黑星", "choice_index": 1},
            {"reward_id": "key", "reward_type": "SAPPHIRE_KEY", "linked_reward_id": "duplicate", "choice_index": 2},
        ]}}
        relationship = project_reward_text(view)["screen_state"]["reward_relationships"][0]
        self.assertIn("另一项关联奖励", relationship)
        self.assertNotIn("锚", relationship)
        self.assertNotIn("黑星", relationship)

    def test_boss_rewards_and_nested_sources_hide_ids_without_invented_numbering(self):
        view = {"screen_type": "CARD_REWARD", "screen_state": {
            "source_reward_id": "R23", "source_reward_ref": "R23", "reward_index": 0,
            "relics": [{"reward_id": "R24", "reward_ref": "R24", "name": "贤者之石", "choice_index": 0}],
            "cards": [{"name": "打击", "source_reward_id": "R23", "choice_index": 0}]},
            "selection_context": {"kind": "COMBAT_GENERATED_CARD_CHOICE", "source": {"name": "攻击药水", "source_reward_ref": "R23"}},
            "legal_actions": [{"id": "A1", "index": 0, "label": "选择打击"}]}
        original = copy.deepcopy(view)
        projected = project_reward_text(copy.deepcopy(view))
        self.assertEqual(view, original)
        self.assertNotIn("reward_relationships", projected["screen_state"])
        self.assertEqual(projected["screen_state"]["relics"][0], {"name": "贤者之石", "choice_index": 0})
        self.assertEqual(projected["screen_state"]["cards"][0], {"name": "打击", "choice_index": 0})
        self.assertEqual(projected["selection_context"]["source"], {"name": "攻击药水"})
        self.assertEqual(projected["legal_actions"], original["legal_actions"])
        self.assertNotIn("R23", json.dumps(projected))
        self.assertNotIn("R24", json.dumps(projected))
        self.assertNotIn("第", json.dumps(projected, ensure_ascii=False))

    def test_history_hides_reward_numbers_but_keeps_names_origin_and_flow(self):
        step = {"step_id": 12, "screen_before": "CARD_REWARD", "screen_after": "COMBAT_REWARD",
                "action_kind": "choose", "action": "选择打击", "details": {
                    "source_reward_id": "R23", "source_reward_ref": "R23", "object": {"name": "打击", "reward_id": "R23"},
                    "selection_context": {"kind": "COMBAT_GENERATED_CARD_CHOICE", "source": {"name": "攻击药水"}}}}
        original = copy.deepcopy(step)
        rendered = history_step_text(step)
        self.assertIn("选择卡牌：打击", rendered)
        self.assertIn("由攻击药水触发", rendered)
        self.assertIn("卡牌选择→战斗奖励", rendered)
        self.assertNotIn("R23", rendered)
        self.assertNotIn("奖励编号", rendered)
        self.assertEqual(compact_history_step(step)["details"]["source_reward_id"], "R23")
        self.assertEqual(step, original)


if __name__ == "__main__":
    unittest.main()
