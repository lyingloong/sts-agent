from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from sts_agent.config import ConfigError, load_agent_config, validate_config
from sts_agent.policy.api import ApiPolicyClient


class CloudPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "ustc_api.txt").write_text("key: test-key\n", encoding="utf-8")
        self.config = load_agent_config(mode="baseline_api")
        self.client = ApiPolicyClient(self.root, self.config)
        self.addCleanup(self.client.close)
        self.actions = [{"id": "CHOOSE:index=0", "command": "CHOOSE 0", "label": "Take"},
                        {"id": "CHOOSE:index=1", "command": "CHOOSE 1", "label": "Leave"}]
        self.observation = {"screen_type": "EVENT", "current_hp": 80}
        self.response = {"choices": [{"message": {"content": "A2"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 2}}

    def _mock_http(self, content: str = "A2"):
        def http(path, body=None):
            if path == "/models":
                return {"data": [{"id": self.client.model}]}
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 2}}
        self.client._http = Mock(side_effect=http)
        return self.client._http

    def test_profile_selects_hosted_provider_without_tunnel(self):
        api = self.config["policy"]["api"]
        self.assertEqual(api["provider"], "openai_compatible")
        self.assertEqual(api["rpm_limit"], 18)
        self.assertAlmostEqual(self.client.min_interval, 60.0 / 18.0)
        self.assertIsNone(self.client.tunnel)
        self.assertEqual(self.client.base_url, "https://api.llm.ustc.edu.cn/v1")

    def test_request_body_uses_extra_body_not_vllm_fields(self):
        config = copy.deepcopy(self.config)
        config["policy"]["api"]["extra_body"] = {"enable_thinking": False}
        client = ApiPolicyClient(self.root, config)
        self.addCleanup(client.close)
        client._http = self._mock_http()
        decision = client.rank(self.observation, self.actions)
        self.assertEqual(decision["model_action"], "CHOOSE:index=1")
        body = next(call.args[1] for call in client._http.call_args_list if call.args[0] == "/chat/completions")
        self.assertNotIn("chat_template_kwargs", body)
        self.assertIn("enable_thinking", body)
        self.assertFalse(body["enable_thinking"])
        self.assertNotIn("/tokenize", [call.args[0] for call in client._http.call_args_list])

    def test_alias_parsing_tolerates_wrappers_but_not_prose(self):
        for content, expected in (("A2", "CHOOSE:index=1"), ("A2.", "CHOOSE:index=1"),
                                  ('"A2"', "CHOOSE:index=1"), (" `A2` ", "CHOOSE:index=1")):
            with self.subTest(content=content):
                self.client.pending = None
                self.client.ready = True
                self.client._http = self._mock_http(content)
                self.assertEqual(self.client.rank(self.observation, self.actions)["model_action"], expected)
        for content in ("I choose A2", '{"action": "A2"}', "A99"):
            with self.subTest(content=content):
                self.client.pending = None
                self.client.ready = True
                self.client._http = self._mock_http(content)
                decision = self.client.rank(self.observation, self.actions)
                self.assertIsNone(decision["model_action"])
                self.assertIn("暂停", decision["assist_status"])

    def test_missing_api_key_pauses_instead_of_crashing(self):
        (self.root / "ustc_api.txt").unlink()
        decision = self.client.rank(self.observation, self.actions)
        self.assertIsNone(decision["model_action"])
        self.assertIn("api_key_file", decision["assist_reason"])

    def test_throttle_waits_for_the_remaining_interval(self):
        self.client.min_interval = 3.0
        self.client._last_request_at = 100.0
        with patch("sts_agent.policy.api.time.monotonic", side_effect=[101.0, 103.0]), \
                patch("sts_agent.policy.api.time.sleep") as sleep:
            self.client._throttle()
        sleep.assert_called_once_with(2.0)

    def test_transient_status_is_retried_and_client_errors_are_not(self):
        self.client._http = ApiPolicyClient._http.__get__(self.client)
        self.client.ready = True
        self.client.min_interval = 0.0
        self.client.settings["max_retries"] = 1
        self.client.settings["retry_backoff_seconds"] = 1
        self.client.settings["retry_max_backoff_seconds"] = 5
        for code, expected in ((429, 2), (500, 2), (400, 1)):
            with self.subTest(code=code):
                def fail(*args, **kwargs):
                    raise HTTPError("https://example.invalid", code, "error", {"Retry-After": "0"}, io.BytesIO(b"error"))
                self.client.opener = Mock()
                self.client.opener.open.side_effect = fail
                with patch("sts_agent.policy.api.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        self.client._http("/chat/completions", {"model": "x"})
                self.assertEqual(self.client.opener.open.call_count, expected)
                self.assertEqual(sleep.call_count, expected - 1)

    def test_rate_limit_and_auth_settings_are_validated(self):
        for changes in ({"base_url": ""}, {"base_url": "ftp://example"}, {"api_key_file": ""},
                        {"api_key_file": "../key.txt"}, {"rpm_limit": 601}, {"rpm_limit": -1},
                        {"extra_body": []}):
            with self.subTest(changes=changes):
                bad = copy.deepcopy(self.config)
                bad["policy"]["api"].update(changes)
                with self.assertRaises(ConfigError):
                    validate_config(bad)

    def test_manifest_records_provider_and_endpoint_without_secret(self):
        manifest = json.loads((self.client.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["provider"], "openai_compatible")
        self.assertEqual(manifest["endpoint"], "https://api.llm.ustc.edu.cn/v1")
        self.assertNotIn("test-key", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
