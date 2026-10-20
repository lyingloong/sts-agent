from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.commands.export_baseline_io import export_review, snapshot_records


class ExportBaselineIoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "requests.jsonl"
        self.trace = self.root / "trace.jsonl"
        self.batch = self.root / "batch"
        self.batch.mkdir()
        self.output = self.root / "review"
        self.prompt = "完整输入，不能截断。" * 5000 + "\n```text\n原文中的围栏\n```\n"
        self.records = [self.request(1), self.response(1), self.request(2, "other"), self.response(2),
                        self.request(3), self.response(3), {"event": "stale_action_discarded", "request_id": 3},
                        {"event": "execution_result", "request_id": None, "run_id": "run-2", "step_id": 1,
                         "accepted": True, "decision": {"api": {"request_id": 1}, "source": "baseline_model"}}]
        self.actions = [{"run_id": "run-2", "step_id": 0, "accepted": True,
                         "action": {"label": "环境操作"}, "decision": {"source": "environment_auto"}},
                        {"run_id": "run-2", "step_id": 1, "accepted": True,
                         "action": {"label": "选择防御"}, "decision": {"source": "baseline_model", "api": {"request_id": 1}}},
                        {"run_id": "run-2", "step_id": 2, "accepted": False,
                         "action": {"label": "前进"}, "decision": {"source": "environment_auto"}}]
        self.write_lines(self.source, self.records)
        self.write_lines(self.trace, self.actions)
        result = {"status": "completed", "run_id": "run-2", "seed": "SEED",
                  "summary": {"accepted_actions": 2, "rejected_actions": 1, "terminal": True},
                  "api_performance": {"calls": 2},
                  "artifacts": {"requests": str(self.source), "action_traces": [str(self.trace)]}}
        (self.batch / "results.json").write_text(json.dumps({"results": [result, {"status": "pending", "run_id": "run-3"}]}), encoding="utf-8")

    def write_lines(self, path, records):
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    def request(self, number, run_id="run-2"):
        return {"event": "request", "request_id": number, "run_id": run_id,
                "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
                "request": {"model": "test-model", "messages": [{"role": "user", "content": self.prompt}]},
                "aliases": {"A1": {"label": "选择防御"}}}

    def response(self, number):
        return {"event": "response", "request_id": number, "input_tokens": 99, "latency_ms": 50,
                "response": {"choices": [{"finish_reason": "stop", "message": {"content": "A1\n"}}]}}

    def test_exact_long_original_input_output_and_discard_are_preserved(self):
        result = export_review(self.batch, self.output)
        exported = list(snapshot_records(self.output / "model-io.jsonl"))
        self.assertEqual(result["run_id"], "run-2")
        self.assertEqual([entry["request_id"] for entry in exported], [1, 3])
        self.assertEqual(result["request_dispositions"], {"executed": 1, "discarded": 1})
        self.assertEqual(exported[0]["request_record"]["request"]["messages"][0]["content"], self.prompt)
        self.assertEqual(exported[0]["events"][0]["response"]["choices"][0]["message"]["content"], "A1\n")
        markdown = (self.output / "full-input-output.md").read_text(encoding="utf-8")
        self.assertEqual(markdown.count(self.prompt), 2)
        self.assertIn("````text", markdown)
        before_header = markdown.split("## 请求 3", 1)[0]
        header_line = before_header.count("\n") + 1
        self.assertIn(f"full-input-output.md:{header_line}", (self.output / "index.md").read_text(encoding="utf-8"))

    def test_partial_live_tail_is_ignored_but_complete_malformed_line_is_error(self):
        with self.source.open("ab") as stream:
            stream.write(b'{"event":"request","run_id":"run-3"')
        export_review(self.batch, self.output)
        other_output = self.root / "malformed-review"
        with self.source.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaises(json.JSONDecodeError):
            export_review(self.batch, other_output)
        self.assertFalse(other_output.exists())

    def test_missing_response_wrong_hash_or_trace_count_refuses_export(self):
        for problem in ("response", "hash", "trace"):
            with self.subTest(problem=problem):
                self.write_lines(self.source, self.records)
                self.write_lines(self.trace, self.actions)
                if problem == "response":
                    self.write_lines(self.source, [record for record in self.records if not (record["event"] == "response" and record["request_id"] == 3)])
                elif problem == "hash":
                    wrong = self.request(1)
                    wrong["prompt_sha256"] = "wrong"
                    self.write_lines(self.source, [wrong, *self.records[1:]])
                else:
                    self.write_lines(self.trace, self.actions[:-1])
                with self.assertRaises(ValueError):
                    export_review(self.batch, self.output)
                self.assertFalse(self.output.exists())

    def test_refuses_overwrite_or_pending_run(self):
        with self.assertRaises(ValueError):
            export_review(self.batch, self.output, "run-3")
        export_review(self.batch, self.output)
        with self.assertRaises(ValueError):
            export_review(self.batch, self.output)


if __name__ == "__main__":
    unittest.main()
