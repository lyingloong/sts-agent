import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.learning.export_human_sft import DEFAULT_MAX_LENGTH, ExportLengthError, build_dataset, export_record, checked_path
from sts_agent.learning.sft import sha256_file, encode_sample, validate_bundle
from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.runtime.controller import semantic_actions
from tests.test_trajectory_assembler import combat_state


class HumanSFTTests(unittest.TestCase):
    def test_late_overflow_cleans_partial_bundle_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            events = [{"type": "run_started", "character": "IRONCLAD", "ascension": 0}]
            for index in (1, 2):
                state = combat_state()
                if index == 2:
                    state["relics"] = [{"id": "LongRelic", "description": "完整描述" * 3000}]
                state_id, tx = f"s:{index}", f"t{index}"
                message = {"recorder_state_id": state_id, "recorder_state_seq": index,
                           "recorder_session": "s", "in_game": True, "ready_for_command": True,
                           "available_commands": ["play", "end"], "game_state": state}
                events.extend([
                    {"type": "state_published", "state_id": state_id, "message": message},
                    {"type": "action_begin", "transaction_id": tx, "before_state_id": state_id, "expected_screen": "NONE"},
                    {"type": "action_accepted", "transaction_id": tx, "action": {"kind": "end_turn", "id": "END_TURN"}},
                ])
            events.append({"type": "run_ended", "reason": "death", "terminal": True})
            event_file, step_file = source / "events.jsonl", source / "steps.jsonl"
            event_file.write_text("".join(json.dumps(dict(e, run_id="r")) + "\n" for e in events), encoding="utf-8")
            step_file.write_text("".join(json.dumps({"transaction_id": f"t{i}", "status": "accepted"}) + "\n"
                                         for i in (1, 2)), encoding="utf-8")
            member = lambda p: {"path": p.name, "sha256": sha256_file(p)}
            (source / "manifest.json").write_text(json.dumps({"schema_version": "actionrecorder-dataset-1",
                "runs": [{"run_id": "r", "metadata": {"outcome": "death"},
                          "events": member(event_file), "trajectory": member(step_file)}]}), encoding="utf-8")
            original_hash = sha256_file(event_file)

            class Tokenizer:
                eos_token_id = 1

                def apply_chat_template(self, messages, **kwargs):
                    return messages[0]["content"]

                def __call__(self, text, **kwargs):
                    if kwargs.get("truncation") is not False:
                        raise AssertionError("export must explicitly disable truncation")
                    return {"input_ids": [2] * len(text)}

            with self.assertRaises(ExportLengthError) as error:
                build_dataset(source, root / "out", tokenizer=Tokenizer())
            self.assertEqual(error.exception.sample_id, "r:t2")
            self.assertFalse((root / "out").exists())
            self.assertFalse(list(root.glob(".out.export-*")))
            self.assertEqual(sha256_file(event_file), original_hash)

    def test_no_character_truncation(self):
        for screen in ("COMBAT", "MAP"):
            prompt = build_scoring_prompt({"screen_type": screen, "details": "长" * 17000 + "末尾标记"}, [{"id": "A1", "label": "确认"}])
            self.assertIn("末尾标记", prompt)
            self.assertNotIn("省略", prompt)

    def test_fine_grained_is_runtime_default(self):
        state = combat_state() | {"screen_type": "REST", "choice_list": ["rest", "smith"], "screen_state": {}}
        actions = semantic_actions({"in_game": True, "available_commands": ["choose"], "game_state": state})
        self.assertIn("CHOOSE:index=1", [a["id"] for a in actions])
        self.assertFalse(any(a.get("kind") == "smith" for a in actions))

    def test_uses_training_view_and_no_after_or_outcome_in_prompt(self):
        row = {"run_id": "r", "step_id": "t", "usable_for_sft": True,
               "training_policy_observation": {"screen_type": "COMBAT", "current_hp": 31},
               "training_action": {"id": "END_TURN"}, "training_legal_actions": [{"id": "END_TURN", "label": "结束回合"}],
               "pre_state": {"secret": "错误状态"}, "post_state": {"secret": "未来结果"}, "run_outcome": "victory"}
        sample, reason = export_record(row)
        self.assertEqual(reason, "kept")
        self.assertEqual(sample["target"]["alias"], "A1")
        self.assertIn("31", sample["prompt"])
        for text in ("错误状态", "未来结果", "victory"):
            self.assertNotIn(text, sample["prompt"])
        # An unfinished animation or an absent post-state cannot change the
        # pre-action prompt, legal actions or supervised target.
        unfinished = dict(row, post_state=None, observation_after=None,
                          raw_message_after=None, stable_after_action=False)
        self.assertEqual(export_record(unfinished), (sample, "kept"))
        without_after = {key: value for key, value in unfinished.items()
                         if key not in {"post_state", "observation_after", "raw_message_after", "stable_after_action"}}
        self.assertEqual(export_record(without_after), (sample, "kept"))
        self.assertEqual(export_record(dict(row, usable_for_sft=False, training_exclusion_reason="cancelled")), (None, "cancelled"))

    def test_bundle_end_to_end_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            state = combat_state()
            message = {"recorder_state_id": "s:1", "recorder_state_seq": 1, "recorder_session": "s",
                       "in_game": True, "ready_for_command": True, "available_commands": ["play", "end"], "game_state": state}
            events = [
                {"type": "run_started", "character": "IRONCLAD", "ascension": 0, "seed": "-123"},
                {"type": "state_published", "state_id": "s:1", "message": message},
                {"type": "action_begin", "transaction_id": "t", "before_state_id": "s:1", "expected_screen": "NONE"},
                {"type": "action_accepted", "transaction_id": "t", "action": {"kind": "end_turn", "id": "END_TURN"}},
                {"type": "run_ended", "reason": "death", "terminal": True}]
            event_file = source / "events.jsonl"
            event_file.write_text("".join(json.dumps(dict(e, run_id="r")) + "\n" for e in events), encoding="utf-8")
            steps_file = source / "steps.jsonl"
            steps_file.write_text(json.dumps({"transaction_id": "t", "status": "accepted"}) + "\n", encoding="utf-8")
            member = lambda p: {"path": p.name, "sha256": sha256_file(p)}
            manifest = {"schema_version": "actionrecorder-dataset-1", "runs": [{"run_id": "r", "metadata": {"outcome": "death"},
                "events": member(event_file), "trajectory": member(steps_file)}]}
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = build_dataset(source, root / "out")
            self.assertEqual(result["dataset"]["samples"], 1)
            self.assertEqual(validate_bundle(root / "out/dataset/human-sft.jsonl", root / "out/manifest.json")["dataset"]["samples"], 1)
            sample = json.loads((root / "out/dataset/human-sft.jsonl").read_text(encoding="utf-8"))
            self.assertTrue((root / "out/previews/NONE.md").is_file())
            self.assertEqual(sample["provenance"]["submission_state_id"], "s:1")
            self.assertNotIn("death", sample["prompt"])
            self.assertIn("这是战斗界面", sample["prompt"])

            class CharacterTokenizer:
                eos_token_id = 1

                def apply_chat_template(self, messages, **kwargs):
                    return "user:" + messages[0]["content"] + "\nassistant:"

                def __call__(self, text, **kwargs):
                    self.assert_no_truncation(kwargs)
                    return {"input_ids": [2] * len(text)}

                def assert_no_truncation(self, kwargs):
                    if kwargs.get("truncation") is True:
                        raise AssertionError("tokenizer must never truncate")

            tokenizer = CharacterTokenizer()
            encoded, status = encode_sample(tokenizer, sample, 100000)
            self.assertEqual(status, "kept")
            self.assertTrue(all(value == -100 for value in encoded["labels"][:-4]))
            self.assertTrue(all(value != -100 for value in encoded["labels"][-4:]))
            with self.assertRaises(ExportLengthError) as overflow:
                build_dataset(source, root / "length_checked", tokenizer=tokenizer,
                              max_length=1, tokenizer_name="test-character-tokenizer")
            self.assertEqual(overflow.exception.sample_id, "r:t")
            self.assertEqual(overflow.exception.limit, 1)
            self.assertEqual(set(overflow.exception.lengths), {"canonical", "state_first", "decision_first"})
            self.assertIn("no truncation", str(overflow.exception))
            self.assertFalse((root / "length_checked").exists())
            self.assertFalse(list(root.glob(".length_checked.export-*")))
            length_checked = build_dataset(source, root / "length_checked", tokenizer=tokenizer,
                                           tokenizer_name="test-character-tokenizer")
            self.assertEqual(length_checked["dataset"]["samples"], 1)
            self.assertEqual(length_checked["max_length"], DEFAULT_MAX_LENGTH)
            self.assertTrue(length_checked["length_checked"])
            self.assertEqual(length_checked["length_policy"], "error_on_overflow_no_truncation")
            self.assertEqual(encode_sample(tokenizer, sample, 1), (None, "overlong"))
            with self.assertRaises(ValueError):
                build_dataset(source, root / "out")
            event_file.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_dataset(source, root / "out2")
            self.assertFalse((root / "out2").exists())
            with self.assertRaisesRegex(ValueError, "invalid dataset member"):
                checked_path(source, {"path": "../outside.jsonl"})

    def test_prompt_drift_rejected_before_encoding(self):
        sample = {"run_id": "r", "sample_id": "s", "observation": {"screen_type": "MAP"},
                  "legal_actions": [{"id": "A1", "label": "继续"}], "target": {"alias": "A1"}, "prompt": "outdated"}
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            encode_sample(None, sample, 4096)


if __name__ == "__main__":
    unittest.main()
