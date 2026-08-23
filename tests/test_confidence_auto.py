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


if __name__ == "__main__":
    unittest.main()
