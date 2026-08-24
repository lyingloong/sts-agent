from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import communication_controller as controller  # noqa: E402


class NoBridge:
    def send(self, packet):
        raise AssertionError(f"automatic decision unexpectedly requested terminal input: {packet}")


class FakePolicy:
    def __init__(self, decision):
        self.decision = decision
        self.screens = ("ALL",)

    def rank(self, observation, legal_actions):
        del observation, legal_actions
        return self.decision

    def status_decision(self):
        return self.decision


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


class ConfidenceAutoTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
