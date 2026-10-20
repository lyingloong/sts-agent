from __future__ import annotations

import copy
import unittest

from sts_agent.policy.local_qwen import PROMPT_TEMPLATES, build_prompt, build_scoring_prompt, render_observation, render_room_history
from sts_agent.policy.presentation import text_observation
from sts_agent.runtime.controller import policy_observation, semantic_actions
from tests.test_trajectory_assembler import combat_state


class PolicyTextTests(unittest.TestCase):
    def test_display_cleanup_preserves_raw_input_and_meaningful_differences(self):
        card = {"id": "Strike_R", "cost": 1, "cost_for_turn": 0, "is_playable": True,
                "description": "造成9点伤害。", "raw_description": "造成!D!点伤害。",
                "description_source": "game_runtime", "description_complete": True,
                "damage": 9, "base_damage": 6, "final_single_hit_damage": 9,
                "ethereal": False, "retain": None, "exhaust_on_use_once": True}
        observation = {"combat": {"hand": [card]}}
        original = copy.deepcopy(observation)
        view = text_observation(observation)["combat"]["hand"][0]
        self.assertEqual(observation, original)
        self.assertNotIn("raw_description", view)
        self.assertNotIn("description_source", view)
        self.assertNotIn("description_complete", view)
        self.assertNotIn("ethereal", view)
        self.assertNotIn("damage", view)
        self.assertNotIn("base_damage", view)
        self.assertNotIn("final_single_hit_damage", view)
        self.assertEqual(view["cost_for_turn"], 0)
        self.assertTrue(view["exhaust_on_use_once"])
        self.assertIsNone(view["retain"])
        self.assertEqual(view, text_observation({"combat": {"hand": [view]}})["combat"]["hand"][0])

    def test_incomplete_effect_and_extra_tooltip_are_not_silently_erased(self):
        card = {"cost": 1, "description": "造成!CUSTOM!点伤害。", "raw_description": "原始效果",
                "description_complete": False, "unresolved_description_variables": ["CUSTOM"],
                "tooltips": [{"header": "附加机制", "body": "需要保留的说明"}]}
        text = render_observation({"screen_type": "GRID", "screen_state": {"cards": [card]}})
        for expected in ("说明完整=否", "CUSTOM", "原始效果", "需要保留的说明"):
            self.assertIn(expected, text)

    def test_missing_completeness_does_not_erase_an_unrepresented_effect(self):
        for marker in ({}, {"description_complete": None}):
            card = {"cost": 1, "description": "造成6点伤害。", "raw_description": "造成6点伤害并抽1张牌。", **marker}
            text = render_observation({"screen_type": "GRID", "screen_state": {"cards": [card]}})
            self.assertIn("造成6点伤害并抽1张牌。", text)

    def test_runtime_formatting_only_raw_effect_is_redundant(self):
        relic = {"name": "燃烧之血", "description": "每场战斗结束时，回复6点生命。",
                 "raw_description": "每场战斗结束时，回复#b6点生命。", "description_source": "game_runtime"}
        text = render_observation({"relics": [relic]})
        self.assertEqual(text.count("回复6点生命"), 1)
        self.assertNotIn("原始说明", text)

    def test_zero_unknown_and_false_remain_distinct(self):
        text = render_observation({"screen_type": "COMBAT", "combat": {"hand": [
            {"cost": 0, "is_playable": False, "damage": 0, "block": None, "retain": None}]}})
        for expected in ("费用=0", "可打出=否", "伤害=0", "格挡=未知", "保留=未知"):
            self.assertIn(expected, text)

    def test_complete_card_descriptions_do_not_show_empty_numeric_diagnostics(self):
        for multiple in (None, []):
            with self.subTest(multiple=multiple):
                observation = {"deck": [{"id": "Strike_R", "name": "打击", "cost": 1,
                    "description": "造成6点伤害。", "description_complete": True,
                    "damage": -1, "block": -1, "base_damage": 6, "magic_number": -1,
                    "multi_damage": multiple}]}
                original = copy.deepcopy(observation)
                text = render_observation(observation)
                self.assertIn("造成6点伤害", text)
                for label in ("伤害=", "格挡=", "效果参数=", "multi damage", "多目标伤害"):
                    self.assertNotIn(label, text)
                self.assertEqual(observation, original)

    def test_playability_is_only_a_live_hand_fact_not_a_deck_or_preview_fact(self):
        observation = {"screen_type": "GRID", "combat": {"hand": [
            {"cost": 1, "is_playable": False, "upgrade_preview": {"cost": 1, "is_playable": False}},
            {"cost": 0, "is_playable": None}]},
            "deck_summary": [{"cost": 1, "is_playable": False}],
            "screen_state": {"cards": [{"cost": 1, "is_playable": None}]}}
        original = copy.deepcopy(observation)
        view = text_observation(observation)
        self.assertEqual(observation, original)
        self.assertIs(view["combat"]["hand"][0]["is_playable"], False)
        self.assertIsNone(view["combat"]["hand"][1]["is_playable"])
        self.assertNotIn("is_playable", view["combat"]["hand"][0]["upgrade_preview"])
        self.assertNotIn("is_playable", view["deck_summary"][0])
        self.assertNotIn("is_playable", view["screen_state"]["cards"][0])
        self.assertEqual(view, text_observation(view))

    def test_only_exact_duplicate_tooltip_is_removed(self):
        observation = {"potions": [{"id": "P", "description": "效果", "tooltips": [
            {"header": "P", "body": "效果"}, {"header": "额外", "body": "额外条件"}]}]}
        view = text_observation(observation)
        self.assertEqual(view["potions"][0]["tooltips"], [{"name": "额外", "description": "额外条件"}])
        self.assertEqual(view, text_observation(view))

    def test_empty_potion_slots_are_compact_but_inventory_positions_survive(self):
        slot = {"id": "Potion Slot", "name": "药水栏位", "description": "空栏位说明"}
        observation = {"screen_type": "COMBAT_REWARD", "potions": [slot,
            {"id": "P", "name": "药水", "description": "获得格挡", "can_use": False, "target": False}, slot]}
        text = render_observation(observation)
        self.assertIn("栏位数=3；空槽数=2", text)
        self.assertIn("药水栏位=1", text)
        self.assertIn("可使用=否", text)
        self.assertIn("需要敌人目标=否", text)
        self.assertNotIn("空栏位说明", text)
        self.assertNotIn("Potion Slot", text)

    def test_potion_zero_capacity_unknown_inventory_and_partial_slots_differ(self):
        text = render_observation({"potions": [], "potion_slot_count": 0})
        self.assertIn("栏位数=0；空槽数=0", text)
        self.assertNotIn("未知", text)
        self.assertIn("【药水】未知", render_observation({"potions": None, "potion_slot_count": None}))
        text = render_observation({"potions": [{"id": "P", "name": "药水", "slot": 2}], "potion_slot_count": 3})
        self.assertIn("栏位数=3；空槽数=0", text)
        self.assertIn("另有2个栏位的内容未知", text)
        self.assertIn("药水栏位=2", text)

    def test_public_piles_are_rendered_with_meaningful_missing_details_only(self):
        observation = {"screen_type": "COMBAT", "combat": {
            "draw_count": 3, "draw_pile": [{"id": "Strike_R", "name": "打击", "count": 2}],
            "draw_pile_missing_count": 1, "discard_count": 0, "discard_pile": [],
            "discard_pile_missing_count": 0, "exhaust_count": None, "exhaust_pile": None,
            "exhaust_pile_missing_count": None}}
        original = copy.deepcopy(observation)
        text = render_observation(observation)
        self.assertIn("抽牌堆组成（不表示抽牌顺序）", text)
        self.assertIn("名称=打击", text)
        self.assertIn("数量=2", text)
        self.assertIn("抽牌堆中详情未知的卡牌数=1", text)
        self.assertIn("弃牌堆数量=0", text)
        self.assertNotIn("弃牌堆中详情未知的卡牌数", text)
        self.assertNotIn("消耗牌堆中详情未知的卡牌数", text)
        self.assertIn("消耗牌堆组成=未知", text)
        self.assertEqual(observation, original)

    def test_empty_sections_and_irrelevant_shop_coordinates_are_not_rendered(self):
        text = render_observation({"screen_type": "MAP", "act": 1, "floor": 3,
            "screen_state": {"rewards": []}, "run_memory": {"visited_shop_rooms": [{"act": 1, "floor": 2}],
                                                           "declined_card_rewards": []}})
        self.assertNotIn("当前界面内容", text)
        self.assertNotIn("本局记忆", text)
        self.assertNotIn("visited shop rooms", text)

    def test_raw_choices_do_not_duplicate_legal_actions_or_erase_option_details(self):
        observation = {"screen_type": "EVENT", "choices": ["查看奖励", "离开"],
                       "screen_state": {"body_text": "选择奖励或离开。", "options": [
                           {"text": "查看奖励", "description": "支付35金币，获得一张牌。"}]}}
        original = copy.deepcopy(observation)
        prompt = build_scoring_prompt(observation, [
            {"id": "A1", "label": "查看奖励"}, {"id": "A2", "label": "离开"}])
        self.assertNotIn("【当前选择】", prompt)
        self.assertIn("- A1：查看奖励", prompt)
        self.assertIn("- A2：离开", prompt)
        self.assertIn("支付35金币，获得一张牌", prompt)
        self.assertEqual(observation, original)

    def test_scene_guidance_avoids_controller_workflow_reminders(self):
        actions = [{"id": "A1", "label": "选择"}]
        for screen, forbidden in (("SHOP_SCREEN", "退出商人和离开房间是两步"),
                                  ("CHEST", "打开宝箱与领取奖励是不同动作"),
                                  ("GRID", "确认、返回或少选是否可用")):
            with self.subTest(screen=screen):
                prompt = build_scoring_prompt({"screen_type": screen}, actions)
                self.assertNotIn(forbidden, prompt)
        prompt = build_scoring_prompt({"screen_type": "CHEST"}, actions)
        self.assertIn("互斥奖励", prompt)

    def test_reward_identifiers_are_hidden_without_losing_exclusive_choice(self):
        observation = {"screen_type": "COMBAT_REWARD", "screen_state": {"rewards": [
            {"reward_type": "RELIC", "reward_id": "R23", "choice_index": 0,
             "relic": {"id": "Anchor", "name": "锚", "description": "战斗开始获得10格挡。"}},
            {"reward_type": "SAPPHIRE_KEY", "reward_id": "R24", "choice_index": 1,
             "linked_reward_id": "R23"}]},
            "room_history": {"steps": [{"step_id": 1, "action_kind": "choose",
                "screen_before": "COMBAT_REWARD", "details": {"object": {
                    "reward_type": "GOLD", "reward_id": "R22"}}}]}}
        original = copy.deepcopy(observation)
        prompt = build_scoring_prompt(observation, [{"id": "A1", "label": "领取锚"},
                                                  {"id": "A2", "label": "领取蓝钥匙"}])
        self.assertIn("蓝宝石钥匙", prompt)
        self.assertIn("遗物：锚", prompt)
        self.assertIn("二选一", prompt)
        self.assertIn("领取其中一项会放弃另一项", prompt)
        for opaque in ("R22", "R23", "R24", "奖励编号", "所属奖励编号", "互斥奖励编号"):
            self.assertNotIn(opaque, prompt)
        self.assertEqual(observation, original)

    def test_current_shop_visit_is_relevant_and_does_not_forbid_reentry(self):
        text = render_observation({"screen_type": "SHOP_ROOM", "act": 1, "floor": 3,
            "run_memory": {"visited_shop_rooms": [{"act": 1, "floor": 3}]}})
        self.assertIn("本房间曾打开商人", text)
        self.assertNotIn("仍可按合法动作重入", text)

    def test_x_cost_and_damage_scope_are_clear(self):
        prompt = build_scoring_prompt({"screen_type": "COMBAT", "combat": {"hand": [
            {"cost": -1, "cost_for_turn": -1, "damage": 8, "final_single_hit_damage": 8}]}},
            [{"id": "A1", "label": "旋风斩"}])
        self.assertIn("费用=X", prompt)
        self.assertNotIn("费用=-1", prompt)
        self.assertIn("X费通常消耗全部剩余能量", prompt)
        self.assertIn("当前牌面单段伤害=8", prompt)
        self.assertIn("已反映在牌面上的状态修正不要重复计算", prompt)
        self.assertIn("实际扣血还需结合目标状态、格挡和攻击段数判断", prompt)
        self.assertNotIn("游戏缓存", prompt)
        self.assertNotIn("最终单段伤害", prompt)

    def test_no_fixed_goal_or_scheduler_phase_in_model_text(self):
        observation = {"screen_type": "COMBAT", "room_phase": "COMBAT",
                       "action_phase": "WAITING_ON_USER", "task_goal": "act3_clear"}
        original = copy.deepcopy(observation)
        actions = [{"id": "A1", "label": "结束回合"}]
        prompts = [build_prompt(observation, actions, 1)] + [
            build_scoring_prompt(observation, actions, template) for template in PROMPT_TEMPLATES]
        for prompt in prompts:
            self.assertIn("结合当前状态与房间历史，权衡生存、牌组成长和资源收益。", prompt)
            self.assertNotIn("必要时可支付生命", prompt)
            self.assertNotIn("当前效果和限制优先", prompt)
            self.assertNotIn("未提供的信息保持未知", prompt)
            self.assertNotIn("本局目标", prompt)
            self.assertNotIn("act3_clear", prompt)
            self.assertNotIn("WAITING_ON_USER", prompt)
            self.assertNotIn("行动阶段", prompt)
            self.assertIn("房间阶段=COMBAT", prompt)
        self.assertEqual(observation, original)
        single = build_scoring_prompt(observation, actions)
        self.assertIn("不要输出JSON或解释", single)
        self.assertNotIn("输出 JSON", single)

    def test_keys_explain_act_four_access_without_requiring_a_particular_goal(self):
        observation = {"keys": {"ruby": True, "emerald": False, "sapphire": None}}
        rendered = render_observation(observation)
        self.assertIn("红宝石钥匙=已持有", rendered)
        self.assertIn("翡翠钥匙=未持有", rendered)
        self.assertIn("蓝宝石钥匙=未知", rendered)
        self.assertIn("集齐红、绿、蓝三把钥匙，才能在击败第三幕Boss后进入第四幕", rendered)

    def test_different_cost_same_name_cards_have_explicit_action_binding(self):
        state = combat_state()
        cards = [{"id": "Strike_R", "name": "打击", "type": "ATTACK", "cost": 1,
                  "cost_for_turn": cost, "is_playable": playable, "has_target": True}
                 for cost, playable in ((1, False), (0, True), (1, True))]
        state["combat_state"]["hand"] = cards
        message = {"in_game": True, "game_state": state, "available_commands": ["play", "end"]}
        actions = semantic_actions(message)
        aliases = [{**action, "id": f"A{i}"} for i, action in enumerate(actions, 1)]
        prompt = build_scoring_prompt(policy_observation(message), aliases)
        self.assertIn("手牌位置=2", prompt)
        self.assertIn("手牌位置=3", prompt)
        self.assertNotEqual(actions[0]["label"], actions[1]["label"])
        self.assertEqual(actions[0]["command"], "PLAY 2 0")
        self.assertEqual(actions[1]["command"], "PLAY 3 0")

    def test_legacy_action_labels_can_use_explicit_safe_metadata(self):
        prompt = build_scoring_prompt({}, [{"id": "A1", "label": "出牌 打击", "card_index": 4, "target": 0}])
        self.assertIn("手牌位置=4", prompt)
        self.assertIn("敌人序号=0", prompt)

    def test_history_is_not_a_blanket_ban_on_repeated_actions(self):
        observation = {"screen_type": "EVENT", "room_history": {"room": {"act": 1, "floor": 2},
            "total_steps": 1, "shown_steps": 1, "omitted_steps": 0, "limit": 30,
            "steps": [{"step_id": 1, "action_text": "翻开位置0"}]}}
        prompt = build_scoring_prompt(observation, [{"id": "A1", "label": "翻开位置0"}])
        self.assertNotIn("历史动作已执行，不要重复提交", prompt)
        self.assertIn("【当前房间历史】", prompt)
        self.assertIn("步骤1", prompt)
        self.assertNotIn("每个房间最多显示", prompt)
        self.assertNotIn("以下为过去操作", prompt)
        self.assertNotIn("当前合法动作允许的再次操作", prompt)
        self.assertNotIn("已省略", prompt)

    def test_history_only_mentions_an_actual_omission_and_keeps_cancelled_markers(self):
        history = {"limit": 30, "omitted_steps": 10,
                   "steps": [{"step_id": 11, "action_kind": "end_turn", "action": "结束回合",
                              "execution_status": "cancelled"}]}
        rendered = "\n".join(render_room_history(history))
        self.assertIn("已省略更早的10步", rendered)
        self.assertIn("尝试（未执行）", rendered)
        self.assertNotIn("最多显示", rendered)
        self.assertNotIn("以下为过去操作", rendered)
        self.assertEqual(render_room_history({"steps": [], "limit": 30}), [])


if __name__ == "__main__":
    unittest.main()
