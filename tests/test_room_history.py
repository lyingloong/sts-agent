from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.policy.history import compact_history_step, history_step_text, room_history_summary
from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.recording.trajectory_assembler import ActionRecorderEnricher, recorded_history_pending
from sts_agent.runtime.controller import accepted_action_history_item, normalize_state, overlay_view_data, policy_observation, route_policy_observation


def message(screen="EVENT", floor=5, room="EventRoom", **extra):
    return {"in_game": True, "game_state": {
        "act": 1, "floor": floor, "room_type": room, "screen_type": screen,
        "room_phase": "COMBAT" if screen == "NONE" else "COMPLETE",
        "current_hp": 60, "max_hp": 80, "gold": 99, "deck": [],
        "choice_list": ["card0", "card1", "card2"],
        "screen_state": {"event_id": "Match and Keep!", "event_name": "对对碰",
                         "body_text": "两张相同牌配对后加入牌组。", "options": []},
        **extra,
    }}


def record(pre, post, step=1, action=None, **extra):
    return accepted_action_history_item({
        "step_id": step, "screen_type": pre["game_state"]["screen_type"],
        "pre_state": normalize_state(pre),
        "action": action or {"kind": "choose", "id": "CHOOSE:index=0", "index": 0, "label": "card0"},
        "decision": {"source": "environment_auto"}, **extra,
    }, post)


class RoomHistoryTests(unittest.TestCase):
    def test_recording_and_restart_retain_more_than_sixteen_steps_without_future_leakage(self):
        current = message()["game_state"]
        with tempfile.TemporaryDirectory() as directory:
            receiver = ActionRecorderEnricher(Path(directory))
            try:
                for step in range(35):
                    receiver.consume({"type": "action_observed", "run_id": "r", "step_id": f"r:{step}",
                        "state_match": {"match_status": "exact_state_id"}, "observation_before": current,
                        "available_actions": {"commands": ["choose"]},
                        "chosen_action": {"id": "CHOOSE:index=0", "kind": "choose", "board_position": 0}})
                    result = receiver.consume({"type": "step_resolved", "run_id": "r", "step_id": f"r:{step}",
                        "observation_after": current, "resolution": "stable_state"})
                    self.assertEqual(result["training_policy_observation"]["room_history"]["shown_steps"], min(step, 30))
                    self.assertTrue(result["usable_for_sft"])
                self.assertEqual(len(receiver.runs["r"].action_history), 35)
            finally:
                receiver.close()
            restarted = ActionRecorderEnricher(Path(directory))
            try:
                restarted.consume({"type": "action_observed", "run_id": "r", "step_id": "r:35",
                    "state_match": {"match_status": "exact_state_id"}, "observation_before": current,
                    "available_actions": {"commands": ["choose"]},
                    "chosen_action": {"id": "CHOOSE:index=0", "kind": "choose"}})
                pending = restarted.runs["r"].pending["r:35"]
                context = pending["training_policy_observation"]["room_history"]
                self.assertEqual(context["shown_steps"], 30)
                self.assertEqual(context["total_steps"], 35)
                self.assertEqual(context["omitted_steps"], 5)
                self.assertEqual(context["steps"][0]["step_id"], "r:5")
                self.assertNotIn("r:35", [step["step_id"] for step in context["steps"]])
            finally:
                restarted.close()

    def test_noncombat_history_shows_last_thirty_across_secondary_screens(self):
        for room, screen in (("EventRoom", "EVENT"), ("RestRoom", "REST"), ("ShopRoom", "SHOP_SCREEN")):
            with self.subTest(room=room):
                pre, post = message(screen, room=room), message("GRID", room=room)
                history = [record(pre, post, step) for step in range(45)]
                observation = policy_observation(post, action_history=history)
                context = observation["room_history"]
                self.assertEqual(context["shown_steps"], 30)
                self.assertEqual(context["total_steps"], 45)
                self.assertEqual(context["limit"], 30)
                self.assertEqual(context["omitted_steps"], 15)
                self.assertEqual(context["steps"][0]["step_id"], 15)
                self.assertNotIn("source", context["steps"][0])
                self.assertEqual(history[0]["source"], "environment_auto")
                self.assertNotIn("recent_actions", observation)
                self.assertNotIn("event_history", observation)

    def test_combat_room_and_its_rewards_show_last_thirty(self):
        pre = message("NONE", room="MonsterRoom")
        history = [record(pre, pre, step) for step in range(37)]
        for current in (pre, message("COMBAT_REWARD", room="MonsterRoom"), message("CARD_REWARD", room="MonsterRoom")):
            context = policy_observation(current, action_history=history)["room_history"]
            self.assertEqual(context["total_steps"], 37)
            self.assertEqual(context["shown_steps"], 30)
            self.assertEqual(context["omitted_steps"], 7)
            self.assertEqual(context["steps"][0]["step_id"], 7)

    def test_room_boundary_does_not_mix_repeated_events(self):
        old, current = message(floor=4), message(floor=5)
        old_history = record(old, old, "old")
        arrival = record(message("MAP", floor=4), current, "arrival")
        new_history = record(current, current, "new")
        context = room_history_summary(normalize_state(current), [old_history, arrival, new_history])
        self.assertEqual([step["step_id"] for step in context["steps"]], ["arrival", "new"])
        self.assertEqual(context["steps"][0]["role"], "进入当前房间")
        # Both kinds of room use the same window; navigation keeps its scope.
        exit_action = record(current, message("NONE", floor=6, room="MonsterRoom"), "exit")
        self.assertEqual(room_history_summary(normalize_state(current), [exit_action])["limit"], 30)

    def test_card_target_and_resource_changes_use_pre_action_objects(self):
        pre = message("NONE", room="MonsterRoom", combat_state={
            "turn": 2, "player": {"energy": 3, "block": 0},
            "hand": [{"id": "Strike_R", "name": "打击", "type": "ATTACK", "damage": 9, "cost_for_turn": 1}],
            "monsters": [{"id": "JawWorm", "name": "大颚虫", "current_hp": 20}],
        })
        post = copy.deepcopy(pre)
        post["game_state"]["combat_state"]["hand"] = [{"id": "Defend_R", "name": "防御"}]
        post["game_state"]["combat_state"]["player"]["energy"] = 2
        post["game_state"]["combat_state"]["monsters"][0]["current_hp"] = 11
        history = record(pre, post, action={"kind": "play", "id": "PLAY:card=1:target=0", "label": "出牌 打击 -> 敌人 0"})
        self.assertEqual(history["details"]["object"]["name"], "打击")
        self.assertEqual(history["details"]["object"]["final_single_hit_damage"], 9)
        self.assertEqual(history["details"]["target"]["name"], "大颚虫")
        self.assertEqual(history["turn_before"], 2)
        self.assertEqual(history["observed_changes"]["energy"], {"before": 3, "observed_after": 2})
        prompt = build_scoring_prompt(policy_observation(post, action_history=[history]), [{"id": "A1", "label": "结束回合"}])
        self.assertIn("步骤1", prompt)
        self.assertIn("使用打击 → 敌人0（大颚虫）", prompt)
        self.assertNotIn("操作对象", prompt)
        self.assertNotIn("随后观察值", prompt)
        self.assertNotIn("observed_changes", policy_observation(post, action_history=[history])["room_history"]["steps"][0])
        self.assertIn("大颚虫", prompt)

    def test_confirmation_names_selected_card_and_purpose(self):
        pre = message("GRID", screen_state={"for_upgrade": True, "num_cards": 1,
                      "selected_cards": [{"id": "Bash", "name": "痛击"}]})
        item = record(pre, message("REST"), action={"kind": "confirm", "id": "CONFIRM", "label": "确认"})
        self.assertEqual(item["details"]["selection_mode"], "UPGRADE")
        self.assertEqual(item["details"]["selected_cards"][0]["name"], "痛击")

    def test_each_reveal_survives_cards_being_turned_face_down(self):
        pre = message()
        board = [{"position": 0, "revealed": False, "face_up": False,
                  "card": {"id": "SECRET", "name": "禁止泄露"}},
                 {"position": 2, "revealed": False, "face_up": False, "card": None}]
        pre["game_state"]["screen_state"]["match_game"] = {"remaining_attempts": 5, "board": board}
        first = copy.deepcopy(pre)
        first["game_state"]["screen_state"]["match_game"]["board"][0].update(
            revealed=True, face_up=True, card={"id": "Bash", "name": "痛击"})
        second = copy.deepcopy(first)
        second["game_state"]["screen_state"]["match_game"]["board"][0]["face_up"] = False
        second["game_state"]["screen_state"]["match_game"]["board"][1].update(
            revealed=True, face_up=False, card={"id": "Defend_R", "name": "防御"})
        second["game_state"]["screen_state"]["match_game"]["remaining_attempts"] = 4
        history = [record(pre, first, 1), record(first, second, 2,
            action={"id": "CHOOSE:index=1", "index": 1, "kind": "choose", "label": "card2"},
            raw_action={"board_position": 2})]
        context = policy_observation(second, action_history=history)["room_history"]
        self.assertNotIn("event_changes", context["steps"][0])
        self.assertNotIn("event_changes", context["steps"][1])
        self.assertEqual(context["steps"][1]["details"]["board_position"], 2)
        self.assertEqual(context["steps"][1]["details"]["known_card_at_position"]["name"], "防御")
        self.assertEqual(policy_observation(second, action_history=history)["screen_state"]["match_game"]["remaining_attempts"], 4)
        self.assertEqual(context["steps"][0]["details"]["known_card_at_position"]["name"], "痛击")
        self.assertNotIn("禁止泄露", json.dumps(context, ensure_ascii=False))
        prompt = build_scoring_prompt(policy_observation(second, action_history=history), [{"id": "A1", "label": "翻牌"}])
        self.assertIn("当前位置已公开：痛击", prompt)
        self.assertIn("痛击", prompt)
        self.assertIn("防御", prompt)

    def test_slow_flip_uses_only_the_current_public_board_to_resolve_its_face(self):
        pre = message()
        pre["game_state"]["screen_state"]["match_game"] = {"board": [
            {"position": 0, "revealed": False, "card": None}]}
        item = record(pre, pre)
        current = copy.deepcopy(pre)
        current["game_state"]["screen_state"]["match_game"]["board"][0].update(
            revealed=True, card={"id": "Bash", "name": "痛击"})
        context = policy_observation(current, action_history=[item])["room_history"]
        self.assertEqual(context["steps"][0]["details"]["known_card_at_position"]["name"], "痛击")
        self.assertNotIn("revealed_cards", context["steps"][0].get("event_changes", {}))

    def test_history_window_keeps_event_opening_and_public_board_memory(self):
        pre = message()
        pre["game_state"]["screen_state"]["match_game"] = {"board": [
            {"position": 0, "revealed": False, "card": None}]}
        current = copy.deepcopy(pre)
        current["game_state"]["screen_state"]["body_text"] = ""
        current["game_state"]["screen_state"]["match_game"]["board"][0].update(
            revealed=True, face_up=False, card={"id": "Bash", "name": "痛击"})
        history = [record(pre, current, 0)] + [record(current, current, i) for i in range(1, 40)]
        observation = policy_observation(current, action_history=history)
        context = observation["room_history"]
        self.assertEqual(context["shown_steps"], 30)
        self.assertEqual(context["steps"][0]["step_id"], 10)
        self.assertEqual(context["event_opening"]["body_text"], "两张相同牌配对后加入牌组。")
        self.assertEqual(observation["screen_state"]["match_game"]["board"][0]["card"]["name"], "痛击")
        prompt = build_scoring_prompt(observation, [{"id": "A1", "label": "翻牌"}])
        self.assertIn("更早的10步未在本次房间历史窗口中显示", prompt)
        self.assertIn("痛击", prompt)

    def test_history_has_no_recursive_observation_or_map_graph(self):
        pre = message("MAP")
        pre["game_state"]["map"] = [{"x": 0, "y": 0, "symbol": "?", "children": []}]
        state = normalize_state(pre)
        state["room_history"] = {"steps": [{"unexpected": "nested"}]}
        item = accepted_action_history_item({"pre_state": state, "action": {"kind": "choose"}}, message())
        serialized = json.dumps(item)
        self.assertNotIn('"map"', serialized)
        self.assertNotIn("nested", serialized)

    def test_missing_post_inventory_is_not_reported_as_losing_all_cards(self):
        pre = message(deck=[{"id": "Strike_R", "name": "打击"}])
        item = record(pre, {"in_game": False, "game_state": {}})
        self.assertNotIn("deck", item["observed_changes"])

    def test_cached_observation_deltas_are_removed_in_routing_and_rendering(self):
        observation = normalize_state(message())
        observation["room_history"] = {
            "room": {"act": 1, "floor": 5}, "limit": 30, "total_steps": 1, "shown_steps": 1,
            "steps": [{"step_id": 9, "action": "card0", "action_kind": "choose",
                       "details": {"board_position": 0},
                       "observed_changes": {"powers": "LONG_OBSERVATION_DELTA"},
                       "event_changes": {"body_text": "LONG_EVENT_DELTA"}}],
        }
        for template in ("canonical", "state_first", "decision_first"):
            # Even callers which render an older cached window directly must
            # not accidentally reintroduce the removed history sections.
            prompt = build_scoring_prompt(observation, [{"id": "A1", "label": "翻牌"}], template)
            self.assertNotIn("LONG_OBSERVATION_DELTA", prompt)
            self.assertNotIn("LONG_EVENT_DELTA", prompt)
            self.assertIn("翻开位置0", prompt)
        routed = route_policy_observation(copy.deepcopy(observation))
        self.assertNotIn("observed_changes", routed["room_history"]["steps"][0])
        self.assertNotIn("event_changes", routed["room_history"]["steps"][0])
        self.assertIn("observed_changes", observation["room_history"]["steps"][0])

    def test_event_history_keeps_selection_and_opening_not_later_state_changes(self):
        pre, post = message(), message()
        post["game_state"]["gold"] = 50
        post["game_state"]["screen_state"]["body_text"] = "当前事件阶段"
        item = record(pre, post)
        self.assertIn("gold", item["observed_changes"])
        self.assertEqual(item["event_after"]["body_text"], "当前事件阶段")
        observation = policy_observation(post, action_history=[item])
        step = observation["room_history"]["steps"][0]
        self.assertNotIn("observed_changes", step)
        self.assertNotIn("event_changes", step)
        self.assertEqual(step["details"]["choice_text"], "card0")
        self.assertEqual(observation["gold"], 50)
        self.assertEqual(observation["screen_state"]["body_text"], "当前事件阶段")
        self.assertEqual(observation["room_history"]["event_opening"]["body_text"], pre["game_state"]["screen_state"]["body_text"])

    def test_queued_execution_boundary_and_cancelled_attempt_are_explicit(self):
        row = {"execution_tracking": True, "execution_status": "executed",
               "pre_state": {"screen_type": "NONE", "gold": 99},
               "training_pre_state": {"screen_type": "NONE", "gold": 77},
               "action": {"id": "PLAY:card=1"}, "training_action": {"id": "PLAY:card=2"}}
        pending = recorded_history_pending(row)
        self.assertEqual(pending["action"]["id"], "PLAY:card=2")
        self.assertEqual(pending["pre_state"]["gold"], 77)
        self.assertEqual(pending["decision"]["source"], "human_game")
        current = message("NONE", room="MonsterRoom")
        item = record(current, current)
        item["execution_status"] = "cancelled"
        prompt = build_scoring_prompt(policy_observation(current, action_history=[item]), [{"id": "A1"}])
        self.assertIn("尝试（未执行）", prompt)

    def test_history_identity_has_no_entity_attributes_or_execution_source(self):
        item = {"step_id": 1, "turn_before": 2, "action_kind": "play", "action": "出牌 痛击",
                "source": "environment_auto", "details": {
                    "object": {"id": "Bash", "name": "痛击", "upgrades": 1, "cost": 2,
                               "description": "LONG_EFFECT", "damage": 12, "final_single_hit_damage": 12},
                    "target": {"index": 0, "id": "SlimeBoss", "name": "史莱姆老大", "block": 5},
                }}
        step = compact_history_step(item)
        self.assertEqual(step["details"]["object"], {"id": "Bash", "name": "痛击", "upgrades": 1})
        self.assertEqual(step["details"]["target"], {"index": 0, "id": "SlimeBoss", "name": "史莱姆老大"})
        self.assertNotIn("source", step)
        self.assertEqual(history_step_text(item), "步骤1｜回合2：使用痛击+ → 敌人0（史莱姆老大）")
        self.assertIn("description", item["details"]["object"])
        self.assertEqual(compact_history_step(step), step)

    def test_event_purge_selection_return_and_confirmation_keep_purpose(self):
        event = message(choice_list=["净化"], screen_state={"event_id": "The Purifier", "body_text": "免费删除一张牌"})
        grid = message("GRID", choice_list=["痛击"], screen_state={"for_purge": True, "num_cards": 1,
                       "cards": [{"id": "Bash", "name": "痛击", "description": "LONG_CARD_EFFECT"}]})
        selected = copy.deepcopy(grid)
        selected["game_state"]["screen_state"]["selected_cards"] = [{"id": "Bash", "name": "痛击"}]
        history = [record(event, grid, 1, {"kind": "choose", "index": 0, "label": "净化"}),
                   record(grid, selected, 2, {"kind": "choose", "index": 0, "label": "痛击"}),
                   record(selected, event, 3, {"kind": "return", "label": "返回"})]
        text = "\n".join(history_step_text(step) for step in policy_observation(event, action_history=history)["room_history"]["steps"])
        self.assertIn("选择事件选项：净化", text)
        self.assertIn("选择痛击用于删除", text)
        self.assertIn("返回（已选牌：痛击；用途：删除", text)
        self.assertNotIn("LONG_CARD_EFFECT", text)
        confirm = record(selected, event, 4, {"kind": "confirm", "label": "确认"})
        self.assertIn("确认（已选牌：痛击；用途：删除", history_step_text(confirm))

    def test_campfire_shop_and_boss_choices_have_explicit_semantics(self):
        cases = [
            (message("REST", room="RestRoom", choice_list=["smith"]), "选择营火动作：锻造"),
            (message("SHOP_SCREEN", room="ShopRoom", choice_list=["purge"]), "打开删牌界面"),
            (message("SHOP_SCREEN", room="ShopRoom", choice_list=["金刚杵"],
                     screen_state={"relics": [{"id": "Vajra", "name": "金刚杵", "price": 150}]}), "购买金刚杵"),
            (message("BOSS_REWARD", choice_list=["星盘"], screen_state={"relics": [{"id": "Astrolabe", "name": "星盘"}]}), "选择Boss遗物：星盘"),
            (message("COMBAT_REWARD", choice_list=["card"], screen_state={"rewards": [{"reward_type": "CARD", "reward_id": "r1"}]}), "打开卡牌奖励（奖励ID=r1）"),
        ]
        for pre, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, history_step_text(record(pre, pre)))

    def test_generated_card_choices_keep_trigger_not_its_description(self):
        for identifier, name in (("AttackPotion", "攻击药水"), ("Discovery", "发现")):
            item = {"step_id": 3, "screen_before": "CARD_REWARD", "action_kind": "choose", "action": "痛击",
                    "details": {"object": {"id": "Bash", "name": "痛击"}, "selection_context": {
                        "kind": "COMBAT_GENERATED_CARD_CHOICE", "guidance": "LONG_GUIDANCE",
                        "source": {"id": identifier, "name": name, "description": "LONG_EFFECT"}}}}
            text = history_step_text(item)
            self.assertIn("选择卡牌：痛击", text)
            self.assertIn(f"由{name}触发", text)
            self.assertNotIn("LONG_", text)
            self.assertNotIn("LONG_", json.dumps(compact_history_step(item)))

    def test_hand_selection_keeps_upgrade_and_exhaust_purpose(self):
        pre = message("HAND_SELECT", screen_state={"hand": [{"id": "Strike_R", "name": "打击"}], "for_upgrade": True})
        self.assertIn("选择打击用于升级", history_step_text(record(pre, pre)))
        pre["game_state"]["screen_state"] = {"hand": [{"id": "Strike_R", "name": "打击"}], "selection_reason": "消耗一张手牌"}
        self.assertIn("选择打击用于消耗一张手牌", history_step_text(record(pre, pre)))

    def test_overlay_uses_model_room_entries_and_identical_lines_with_shorter_window(self):
        from sts_agent.ui.overlay import history_line
        from sts_agent.policy.local_qwen import render_room_history
        current = message("NONE", room="MonsterRoom")
        old = message("EVENT", floor=4)
        history = [record(old, old, 0)] + [record(current, current, i, {"kind": "end_turn", "label": "结束回合"}) for i in range(1, 11)]
        model = policy_observation(current, action_history=history)["room_history"]
        ui = overlay_view_data(current, "r1", [], history)
        self.assertEqual(ui["recent_actions"], model["steps"][-6:])
        model_lines = render_room_history(model)
        for step in ui["recent_actions"]:
            self.assertIn(history_line(step), model_lines)


if __name__ == "__main__":
    unittest.main()
