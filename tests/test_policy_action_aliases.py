from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts_agent.config import load_agent_config
from sts_agent.core.action_aliases import model_action_alias
from sts_agent.learning.export_human_sft import export_record
from sts_agent.learning.export_teacher_sft import alias_actions
from sts_agent.learning.sft import encode_sample, validate_sample
from sts_agent.policy.api import ApiPolicyClient
from sts_agent.policy.local_qwen import PROMPT_TEMPLATES, build_scoring_prompt


class PolicyActionAliasTests(unittest.TestCase):
    def test_alias_projection_and_sft_validation_keep_safe_positions(self):
        actions = [
            {"id": f"PLAY:card={index}:target=0", "command": f"PLAY {index} 0", "kind": "play",
             "label": "出牌 打击 -> 敌人 0", "card_index": index, "target": 0,
             "card_uuid": f"private-card-{index}", "execution": {"secret": "private"}}
            for index in (2, 3)
        ]
        actions.append({"id": "POTION_USE:slot=0", "kind": "potion_use", "label": "使用药水",
                        "slot": 0, "index": 0, "target": None, "reward_ref": "R2", "reward_id": "private"})
        aliased, mapping = alias_actions(actions)
        self.assertEqual(len(aliased), 3)
        self.assertEqual(mapping[actions[1]["id"]], "A2")
        self.assertEqual([item["card_index"] for item in aliased[:2]], [2, 3])
        sample = {"sample_id": "r:1", "run_id": "r", "observation": {},
                  "legal_actions": aliased, "target": {"alias": "A2"}}
        _, projected, _, _ = validate_sample(sample)
        self.assertEqual(projected, [model_action_alias(action, f"A{i}") for i, action in enumerate(actions, 1)])
        self.assertEqual(projected[2]["slot"], 0)
        self.assertIsNone(projected[2]["target"])
        self.assertEqual(projected[2]["reward_ref"], "R2")
        for action in projected:
            for private_key in ("action_id", "command", "card_uuid", "execution", "reward_id"):
                self.assertNotIn(private_key, action)

    def test_human_export_preserves_positions_and_rebuilds_the_same_prompt(self):
        row = {"run_id": "r", "step_id": "t", "usable_for_sft": True,
               "training_policy_observation": {"screen_type": "GRID"},
               "training_action": {"id": "CHOOSE:index=0"},
               "training_legal_actions": [{"id": "CHOOSE:index=0", "label": "选择", "index": 0}]}
        sample, reason = export_record(row)
        self.assertEqual(reason, "kept")
        self.assertEqual(sample["legal_actions"][0]["index"], 0)
        self.assertEqual(set(sample["prompt_variants"]), set(PROMPT_TEMPLATES))

        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text.encode("utf-8"))}

        encoded, reason = encode_sample(Tokenizer(), sample, 100000)
        self.assertEqual(reason, "kept")
        self.assertIsNotNone(encoded)

    def test_api_and_sft_use_the_same_action_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_agent_config(mode="baseline_inference")
            config["policy"]["api"]["ssh_tunnel"] = False
            client = ApiPolicyClient(Path(temporary), config)
            self.addCleanup(client.close)
            client.ready = True
            calls = []

            def http(path, body=None):
                calls.append((path, body))
                if path == "/tokenize":
                    return {"count": 100}
                return {"choices": [{"finish_reason": "stop", "message": {"content": "A1"}}]}

            client._http = http
            observation = {"screen_type": "GRID"}
            action = {"id": "CHOOSE:index=0", "command": "CHOOSE 0", "kind": "choose", "index": 0,
                      "label": "选择", "card_uuid": "private-card"}
            with patch("sts_agent.policy.api.build_scoring_prompt", wraps=build_scoring_prompt) as build:
                decision = client.rank(observation, [action], run_id="r")
            self.assertEqual(decision["model_action"], action["id"])
            self.assertEqual(build.call_args.args[1][0], model_action_alias(action, "A1"))
            sample, _ = export_record({"run_id": "r", "step_id": "t", "usable_for_sft": True,
                "training_policy_observation": observation, "training_action": action,
                "training_legal_actions": [action]})
            request = next(body for path, body in calls if path == "/v1/chat/completions")
            self.assertEqual(request["messages"][0]["content"], sample["prompt"])
            self.assertNotIn("private-card", sample["prompt"])


if __name__ == "__main__":
    unittest.main()
