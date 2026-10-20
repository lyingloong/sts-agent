from __future__ import annotations

import copy
import unittest

from sts_agent.policy import local_qwen, teacher_prompts
from sts_agent.policy.scenes import SCENE_KEYS, scene_key
from sts_agent.runtime import controller
from tests.test_trajectory_assembler import combat_state


class PolicySceneTests(unittest.TestCase):
    actions = [{"id": "A1", "label": "合法选项"}]

    def test_every_classifier_scene_has_local_and_remote_guidance(self):
        self.assertEqual(SCENE_KEYS, frozenset(local_qwen._SCENE_GUIDANCE))
        self.assertEqual(SCENE_KEYS, frozenset(teacher_prompts.SCENARIO_PROMPTS))

    def test_actual_screens_route_consistently(self):
        expected = {
            "MAP": "MAP", "NONE": "COMBAT", "COMBAT": "COMBAT",
            "SHOP": "SHOP_SCREEN", "SHOP_SCREEN": "SHOP_SCREEN", "SHOP_ROOM": "SHOP_ROOM",
            "EVENT": "EVENT", "CARD_REWARD": "CARD_REWARD", "COMBAT_REWARD": "COMBAT_REWARD",
            "GRID": "GRID_SELECT", "HAND_SELECT": "HAND_SELECT", "REST": "REST",
            "CHEST": "CHEST", "BOSS_REWARD": "BOSS_REWARD", "TRANSITION": "TRANSITION",
            "GAME_OVER": "GAME_OVER", "VICTORY": "VICTORY", "NEW_MOD_SCREEN": "GENERIC",
        }
        for screen, key in expected.items():
            with self.subTest(screen=screen):
                obs = {"screen_type": screen, "room_phase": "COMBAT" if screen in {"NONE", "COMBAT"} else "COMPLETE"}
                self.assertEqual(scene_key(obs), key)
                self.assertEqual(teacher_prompts.scenario_key(obs), key)
                for template in local_qwen.PROMPT_TEMPLATES:
                    prompt = local_qwen.build_scoring_prompt(obs, self.actions, template)
                    self.assertIn(local_qwen._SCENE_GUIDANCE[key], prompt)
                self.assertIn(local_qwen._SCENE_GUIDANCE[key], local_qwen.build_prompt(obs, self.actions, 3))

    def test_map_is_not_combat_and_shops_have_distinct_chinese_labels(self):
        prompt = local_qwen.build_scoring_prompt({"screen_type": "MAP"}, self.actions)
        self.assertIn("这是地图界面", prompt)
        self.assertNotIn("这是战斗界面", prompt)
        for screen, label in (("SHOP_ROOM", "商店房间"), ("SHOP_SCREEN", "商人界面")):
            prompt = local_qwen.build_scoring_prompt({"screen_type": screen}, self.actions)
            self.assertIn("【当前场景】" + label, prompt)
            self.assertIn("当前场景提示", prompt)

    def test_combat_generated_rewards_do_not_use_permanent_reward_guidance(self):
        for context in (None, {"kind": "COMBAT_GENERATED_CARD_CHOICE"}):
            obs = {"screen_type": "CARD_REWARD", "room_phase": "COMBAT", "selection_context": context}
            prompt = local_qwen.build_scoring_prompt(obs, self.actions)
            self.assertIn("战斗内生成选牌", prompt)
            self.assertNotIn(local_qwen._SCENE_GUIDANCE["CARD_REWARD"], prompt)
            self.assertEqual(teacher_prompts.scenario_key(obs), "COMBAT_CARD_SELECTION")
        obs = {"screen_type": "CARD_REWARD", "room_phase": "COMPLETE"}
        self.assertEqual(scene_key(obs), "CARD_REWARD")

    def test_grid_purposes_and_confirmation_are_distinct(self):
        for mode, key in (("PURGE", "GRID_PURGE"), ("UPGRADE", "GRID_UPGRADE"),
                          ("TRANSFORM", "GRID_TRANSFORM"), ("TRANSFORM_UPGRADE", "GRID_TRANSFORM")):
            obs = {"screen_type": "GRID", "screen_state": {"selection_mode": mode}}
            self.assertEqual(scene_key(obs), key)
            obs["screen_state"]["confirm_screen_up"] = True
            self.assertEqual(scene_key(obs), "GRID_CONFIRM")
            prompt = local_qwen.build_scoring_prompt(obs, self.actions)
            self.assertIn("选牌后的确认界面", prompt)

    def test_grid_communication_confirmation_flags_override_selection_purpose(self):
        for flag in ("confirm_up", "confirm_screen_up", "just_for_confirming"):
            observation = {"screen_type": "GRID", "screen_state": {"selection_mode": "UPGRADE", flag: True}}
            with self.subTest(flag=flag):
                self.assertEqual(scene_key(observation), "GRID_CONFIRM")

    def test_relic_selection_is_not_regular_reward_or_purge(self):
        for screen in ("GRID", "CARD_REWARD"):
            obs = {"screen_type": screen, "room_phase": "COMBAT",
                   "screen_state": {"selection_mode": "TRANSFORM"},
                   "selection_context": {"kind": "RELIC_PICKUP_CARD_SELECTION"}}
            self.assertEqual(scene_key(obs), "RELIC_PICKUP_CARD_SELECTION")

    def test_ranking_and_single_action_output_formats_do_not_conflict(self):
        obs = {"screen_type": "NONE", "room_phase": "COMBAT"}
        ranked = local_qwen.build_prompt(obs, self.actions, 3)
        self.assertIn('"ranked_action_ids"', ranked)
        self.assertNotIn("不要输出JSON", ranked)
        self.assertNotIn("只能从合法动作中选择一个", ranked)
        single = local_qwen.build_scoring_prompt(obs, self.actions)
        self.assertIn("不要输出JSON", single)
        self.assertNotIn("ranked_action_ids", single)

    def test_combat_choices_keep_battle_but_not_map(self):
        for screen in ("GRID", "HAND_SELECT", "CARD_REWARD", "COMBAT", "NONE"):
            with self.subTest(screen=screen):
                state = combat_state() | {"screen_type": screen, "screen_state": {"selection_reason": "升级"}}
                obs = controller.route_policy_observation(controller.normalize_state({"game_state": state}))
                self.assertIn("combat", obs)
                self.assertEqual(obs["combat"]["player"]["energy"], 3)
                self.assertEqual(obs["combat"]["monsters"][0]["intent"], "ATTACK")
                self.assertNotIn("map", obs)
                prompt = local_qwen.build_scoring_prompt(obs, self.actions)
                self.assertIn("能量=3", prompt)
                self.assertIn("大颚虫", prompt)

    def test_noncombat_screens_remove_stale_battle(self):
        for screen, phase in (("MAP", "COMPLETE"), ("COMBAT_REWARD", "COMPLETE"),
                              ("SHOP_SCREEN", "COMPLETE"), ("EVENT", "EVENT"),
                              ("GRID", "EVENT"), ("HAND_SELECT", "COMPLETE"),
                              ("CARD_REWARD", "COMPLETE"), ("REST", "COMBAT")):
            obs = controller.route_policy_observation({"screen_type": screen, "room_phase": phase,
                "combat": {"secret_stale": True}, "map": {"nodes": [1]}})
            self.assertNotIn("combat", obs)
            self.assertEqual("map" in obs, screen == "MAP")

    def test_map_renderer_defends_against_direct_nonmap_input(self):
        for screen in ("HAND_SELECT", "CARD_REWARD", "GRID", "EVENT", "SHOP_SCREEN"):
            prompt = local_qwen.build_scoring_prompt({"screen_type": screen,
                "map": {"nodes": ["不应出现的地图节点"]}}, self.actions)
            self.assertNotIn("不应出现的地图节点", prompt)

    def test_public_match_memory_survives_without_hidden_faces(self):
        state = {"event_id": "Match and Keep!", "body_text": "配对相同卡牌",
                 "match_game": {"remaining_attempts": 3, "selected_positions": [0], "board": [
                     {"position": 0, "revealed": True, "face_up": False,
                      "card": {"id": "Strike_R", "name": "已知打击", "description": "造成6点伤害"}},
                     {"position": 1, "revealed": False, "face_up": False,
                      "card": {"id": "Secret", "name": "不可见牌面"}},
                     {"position": 2, "revealed": None, "card": {"name": "未知可见性"}},
                 ]}}
        original = copy.deepcopy(state)
        normalized = controller.normalize_screen_state("EVENT", state)
        self.assertEqual(state, original)
        board = normalized["match_game"]["board"]
        self.assertEqual(board[0]["card"]["name"], "已知打击")
        self.assertIsNone(board[1]["card"])
        self.assertIsNone(board[2]["card"])
        obs = {"screen_type": "EVENT", "screen_state": normalized}
        self.assertEqual(scene_key(obs), "EVENT_MATCH")
        prompt = local_qwen.build_scoring_prompt(obs, self.actions)
        for text in ("翻牌配对", "已知打击", "剩余尝试=3"):
            self.assertIn(text, prompt)
        for text in ("不可见牌面", "未知可见性"):
            self.assertNotIn(text, prompt)

    def test_unknown_match_state_remains_unknown(self):
        self.assertIsNone(controller.normalize_screen_state("EVENT", {"match_game": None})["match_game"])
        self.assertIsNone(controller.normalize_match_game({"board": None})["board"])
        self.assertEqual(scene_key({"screen_type": "EVENT", "screen_state": {"match_game": None}}), "EVENT")


if __name__ == "__main__":
    unittest.main()
