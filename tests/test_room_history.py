from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.policy.history import room_history_summary
from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.recording.trajectory_assembler import ActionRecorderEnricher, recorded_history_pending
from sts_agent.runtime.controller import accepted_action_history_item, normalize_state, policy_observation


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
                self.assertEqual(context["steps"][0]["source"], "environment_auto")
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
        self.assertIn("操作对象", prompt)
        self.assertIn("随后观察值", prompt)
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
        self.assertEqual(context["steps"][0]["event_changes"]["revealed_cards"][0]["card"]["name"], "痛击")
        self.assertEqual(context["steps"][1]["event_changes"]["revealed_cards"][0]["position"], 2)
        self.assertEqual(context["steps"][1]["event_changes"]["remaining_attempts"], 4)
        self.assertEqual(context["steps"][0]["details"]["known_card_at_position"]["name"], "痛击")
        self.assertNotIn("禁止泄露", json.dumps(context, ensure_ascii=False))
        prompt = build_scoring_prompt(policy_observation(second, action_history=history), [{"id": "A1", "label": "翻牌"}])
        self.assertIn("本次公开的牌面", prompt)
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


if __name__ == "__main__":
    unittest.main()
