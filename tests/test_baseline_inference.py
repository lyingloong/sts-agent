from __future__ import annotations

import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from sts_agent.config import ConfigError, load_agent_config, validate_config
from sts_agent.policy.api import ApiPolicyClient
from sts_agent.policy.serving import launch_command, read_deployment, select_devices
from sts_agent.policy.ssh_tunnel import SshTunnel
from sts_agent.policy.transformers_serving import check_budget, validate_request
from sts_agent.runtime import controller
from sts_agent.ui.overlay import action_panel_mode_info


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_agent_config(mode="baseline_inference")
        self.config["policy"]["api"].update(ssh_tunnel=False, max_retries=0)
        self.client = ApiPolicyClient(self.root, self.config)
        self.addCleanup(self.client.close)
        self.actions = [{"id": "CHOOSE:index=0", "command": "CHOOSE 0", "label": "Take"},
                        {"id": "CHOOSE:index=1", "command": "CHOOSE 1", "label": "Leave"}]
        self.observation = {"screen_type": "EVENT", "current_hp": 80}
        self.response = {"choices": [{"message": {"content": "A2\n"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 3}}
        self.calls = []

        def http(path, body=None):
            self.calls.append((path, body))
            if path == "/v1/models":
                return {"data": [{"id": self.client.model, "root": "/snapshots/" + self.config["policy"]["model"]["revision"]}]}
            if path == "/tokenize":
                return {"count": 100}
            return self.response

        self.client._http = Mock(side_effect=http)

    def logs(self):
        return [json.loads(line) for line in self.client.log_path.read_text(encoding="utf-8").splitlines()]

    def test_profile_override_does_not_change_human_collection(self):
        self.assertEqual(load_agent_config(mode="collect_human")["framework"]["learning_method"], "collect_human")
        self.assertEqual(self.config["policy"]["runtime"], "remote_api")
        self.assertEqual(self.config["policy"]["model"]["base"], "Qwen/Qwen3.8-27B")
        for key in ("teacher", "posthoc_feedback", "prompt_evolution"):
            self.assertFalse(self.config[key]["enabled"])

    def test_invalid_gpu_and_context_configuration(self):
        for changes in ({"devices": [0, 0, 1, 2]}, {"devices": [0, 1]}, {"max_model_len": 32}, {"backend": "invalid"}):
            bad = copy.deepcopy(self.config)
            bad["policy"]["api"]["serving"].update(changes)
            with self.assertRaises(ConfigError):
                validate_config(bad)

    def test_no_adapter_in_baseline(self):
        bad = copy.deepcopy(self.config)
        bad["policy"]["model"]["initial_adapter"] = "old-lora"
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_prewarm_does_not_connect_or_load_weights(self):
        self.client.prewarm()
        self.client._http.assert_not_called()

    def test_alias_requires_fresh_state_before_execution(self):
        first = self.client.rank(self.observation, self.actions, run_id="run-1")
        self.assertEqual(first["model_action"], "CHOOSE:index=1")
        self.assertIsNone(controller.auto_action_for_decision(first, self.actions))
        count = len(self.calls)
        confirmed = self.client.rank(self.observation, self.actions, run_id="run-1")
        self.assertEqual(len(self.calls), count)
        self.assertEqual(controller.auto_action_for_decision(confirmed, self.actions), self.actions[1])
        req = next(item for item in self.logs() if item["event"] == "request")
        self.assertEqual(req["run_id"], "run-1")
        self.assertEqual(req["aliases"]["A2"]["command"], "CHOOSE 1")
        body = req["request"]
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("truncate_prompt_tokens", body)

    def test_stale_state_or_new_episode_discards_old_answer(self):
        for observation, run_id in (({**self.observation, "current_hp": 79}, "run-1"), (self.observation, "run-2")):
            self.client.pending = None
            self.client.rank(self.observation, self.actions, run_id="run-1")
            result = self.client.rank(observation, self.actions, run_id=run_id)
            self.assertFalse(result["api"]["revalidated"])
        self.assertEqual(len([x for x in self.logs() if x["event"] == "stale_action_discarded"]), 2)

    def test_invalid_or_truncated_answer_pauses_without_fallback(self):
        for answer, finish in (("A99", "stop"), ("I choose A2", "stop"), ("A2", "length"), ("{\"action\":\"A2\"}", "stop")):
            self.response["choices"][0] = {"message": {"content": answer}, "finish_reason": finish}
            decision = self.client.rank(self.observation, self.actions)
            self.assertIsNone(decision["model_action"])
            self.assertEqual(decision["model_candidates"], [])
            self.assertIn("暂停", decision["assist_status"])

    def test_context_overflow_is_error_and_never_generates(self):
        self.client.ready = True
        self.client._http = Mock(return_value={"count": 16384})
        decision = self.client.rank(self.observation, self.actions)
        self.assertIn("context overflow", decision["assist_reason"])
        self.assertEqual(self.client._http.call_count, 1)
        self.assertIsNone(self.client.pending)

    def test_wrong_model_is_rejected(self):
        self.client._http = Mock(return_value={"data": [{"id": "wrong-model"}]})
        result = self.client.rank(self.observation, self.actions)
        self.assertIn("does not expose", result["assist_reason"])

    def test_pinned_revision_mismatch_is_rejected(self):
        self.config["policy"]["model"]["revision"] = "b" * 40
        self.client._http = Mock(return_value={"data": [{"id": self.client.model, "root": "/snapshots/" + "a" * 40}]})
        result = self.client.rank(self.observation, self.actions)
        self.assertIn("revision does not match", result["assist_reason"])

    def test_execution_summary_separates_human_intervention(self):
        base = {"run_id": "run-1", "accepted": True, "run_config": {"seed": "ABC"}}
        for source in ("baseline_model", "environment_auto", "human_override"):
            self.client.record_result({**base, "decision": {"source": source}})
        self.client.record_result({**base, "accepted": False})
        self.client.record_result({**base, "terminal": True, "decision": {"source": "baseline_model"},
                                   "post_state": {"screen_type": "GAME_OVER", "act": 2, "floor": 28}})
        summary = json.loads((self.client.output_dir / "summary.json").read_text(encoding="utf-8"))["run-1"]
        self.assertEqual(summary["model_actions"], 2)
        self.assertEqual(summary["human_actions"], 1)
        self.assertEqual(summary["environment_actions"], 1)
        self.assertEqual(summary["rejected_actions"], 1)
        self.assertFalse(summary["won"])

    def test_full_controller_keeps_model_run_in_its_experiment_directory(self):
        menu = {"available_commands": ["start", "state"], "ready_for_command": True, "in_game": False}
        event = {"in_game": True, "ready_for_command": True, "available_commands": ["choose", "state"],
                 "game_state": {"screen_type": "EVENT", "choice_list": ["Take", "Leave"], "seed": 123,
                                "screen_state": {}, "act": 1, "floor": 0}}
        terminal = {"in_game": True, "ready_for_command": True, "available_commands": ["confirm", "state"],
                    "game_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 1, "current_hp": 0}}
        bridge = Mock()
        bridge.receive.return_value = {"type": "input", "text": "s IRONCLAD 0"}
        evolution = Mock(enabled=False)
        with (patch.object(controller, "PROJECT_ROOT", self.root),
              patch.object(controller, "load_agent_config", return_value=self.config),
              patch.object(controller, "TerminalBridge", return_value=bridge),
              patch.object(controller, "PromptEvolutionManager", return_value=evolution),
              patch("sts_agent.policy.api.ApiPolicyClient", return_value=self.client),
              patch.object(controller, "TeacherClient", return_value=Mock(enabled=False, prompt_evolution=evolution)),
              patch.object(controller, "PosthocFeedbackClient", return_value=Mock(enabled=False)),
              patch.object(controller.time, "sleep"),
              patch("sys.argv", ["controller"]),
              patch("sys.stdin", io.StringIO("\n".join(json.dumps(x) for x in (menu, event, event, terminal)) + "\n")),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(controller.main(), 0)
        self.assertEqual(output.getvalue().count("START IRONCLAD 0"), 1)
        traces = list(self.client.output_dir.glob("action-trace-*.jsonl"))
        self.assertEqual(len(traces), 1)
        records = [json.loads(line) for line in traces[0].read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["decision"]["source"], "baseline_model")
        self.assertEqual(records[0]["run_config"]["collection_mode"], "baseline_inference")
        self.assertTrue(records[0]["terminal"])
        self.assertEqual(list((self.root / "data" / "trajectories" / "human").glob("*.jsonl")), [])

    def test_http_transient_retry_but_no_retry_for_bad_request(self):
        self.client._http = ApiPolicyClient._http.__get__(self.client)
        self.client.settings["max_retries"] = 1
        for code, expected in ((429, 2), (400, 1)):
            def fail(*args, **kwargs):
                raise HTTPError("http://localhost", code, "error", {}, io.BytesIO(b"test error"))
            self.client.opener = Mock()
            self.client.opener.open.side_effect = fail
            with patch("sts_agent.policy.api.time.sleep") as sleep:
                with self.assertRaises(RuntimeError):
                    self.client._http("/test", {})
                self.assertEqual(self.client.opener.open.call_count, expected)
                self.assertEqual(sleep.call_count, expected - 1)

    def test_real_http_transport_json_roundtrip(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                raw = json.dumps({"echo": body}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.client.base_url = f"http://127.0.0.1:{server.server_port}"
            result = ApiPolicyClient._http(self.client, "/test", {"中文": "打击"})
            self.assertEqual(result, {"echo": {"中文": "打击"}})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_controller_queries_state_then_executes_without_second_generation(self):
        self.client.rank = Mock(wraps=self.client.rank)
        policy = Mock(runtime="remote_api", screens=("ALL",), api=self.client)
        policy.rank = self.client.rank
        policy.requesting_decision = self.client.requesting_decision
        bridge = Mock(connection=object())
        message = {"in_game": True, "ready_for_command": True, "available_commands": ["choose", "state"],
                   "game_state": {"screen_type": "EVENT", "choice_list": ["Take", "Leave"], "screen_state": {}}}
        with patch.object(controller, "environment_auto_action", return_value=None), patch.object(controller.time, "sleep"), redirect_stdout(io.StringIO()) as stdout:
            first = controller.ask_action(bridge, message, "run-1", policy=policy)
            self.assertEqual(first[0], "STATE")
            second = controller.ask_action(bridge, message, "run-1", policy=policy)
            self.assertEqual(second[0], "CHOOSE 1")
            self.assertEqual(second[2]["source"], "baseline_model")
            self.assertEqual(stdout.getvalue(), "STATE\n")
        bridge.receive.assert_not_called()
        self.assertEqual(len([x for x in self.calls if x[0] == "/v1/chat/completions"]), 1)

    def test_gpu_selection_is_idle_only_and_fixed_once(self):
        candidates = [{"index": i, "free_memory_mib": 23500, "utilization_percent": i * 3} for i in range(8)]
        settings = self.config["policy"]["api"]["serving"]
        self.assertEqual(select_devices(candidates, settings), [0, 1, 2, 3])
        candidates[0]["free_memory_mib"] = 1000
        with self.assertRaises(RuntimeError):
            select_devices(candidates, settings)

    def test_bf16_launch_is_text_only_loopback_without_training(self):
        self.config["policy"]["api"]["serving"]["backend"] = "vllm"
        command = launch_command("/models/snapshots/" + "a" * 40, self.config["policy"])
        for flag, value in (("--dtype", "bfloat16"), ("--tensor-parallel-size", "4"), ("--host", "127.0.0.1"), ("--max-num-seqs", "1")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--language-model-only", command)
        self.assertNotIn("--quantization", command)

    def test_deployment_requires_pinned_snapshot_and_no_quantization(self):
        sha = "a" * 40
        snapshot = self.root / "snapshots" / sha
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        manifest = self.root / "deployment.json"
        content = {"model": self.client.model, "requested_revision": self.config["policy"]["model"]["revision"], "resolved_revision": sha, "snapshot_path": str(snapshot)}
        manifest.write_text(json.dumps(content), encoding="utf-8")
        self.assertEqual(read_deployment(manifest, self.config["policy"])["resolved_revision"], sha)
        (snapshot / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}}', encoding="utf-8")
        with self.assertRaises(RuntimeError):
            read_deployment(manifest, self.config["policy"])

    def test_tunnel_is_loopback_batch_mode_and_no_remote_commands(self):
        tunnel = SshTunnel("gpu-host", 18080, 8000, 30, self.root / "ssh.log")
        cmd = tunnel.command()
        self.assertIn("127.0.0.1:18080:127.0.0.1:8000", cmd)
        self.assertIn("BatchMode=yes", cmd)
        self.assertIn("ExitOnForwardFailure=yes", cmd)
        self.assertEqual(cmd[-1], "gpu-host")
        self.assertIn("预训练基线", action_panel_mode_info({"mode": "baseline_inference"})[0])

    def test_transformers_launch_forwards_config_without_vllm(self):
        command = launch_command("/unused", self.config["policy"], "config/custom.json")
        self.assertEqual(command[1:], ["-m", "sts_agent.policy.transformers_serving", "--config", "config/custom.json"])

    def test_transformers_request_is_fixed_text_only_and_never_truncates(self):
        body = {"model": "test", "messages": [{"role": "user", "content": "choose"}],
                "chat_template_kwargs": {"enable_thinking": False}}
        validate_request(body, "test", False)
        for change in ({"model": "other"}, {"chat_template_kwargs": {"truncation": True}},
                       {"add_generation_prompt": False}, {"messages": []}, {"tools": []}):
            with self.assertRaises(ValueError):
                validate_request({**body, **change}, "test", False)
        validate_request({**body, "max_tokens": 64, "stream": False}, "test", False, completion=True)
        with self.assertRaises(ValueError):
            validate_request({**body, "stream": True}, "test", False, completion=True)
        check_budget(100, 64, 164)
        for budget in (65, True, 0):
            with self.assertRaises(ValueError):
                check_budget(100, budget, 164)

    def test_tunnel_closes_only_its_owned_process(self):
        tunnel = SshTunnel("gpu-host", 18080, 8000, 30, self.root / "ssh.log")
        process = Mock()
        process.poll.return_value = None
        tunnel.process = process
        tunnel.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        process.kill.assert_not_called()
        self.assertIsNone(tunnel.process)

    def test_tunnel_will_not_attach_to_an_occupied_port(self):
        import socket

        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            tunnel = SshTunnel("gpu-host", occupied.getsockname()[1], 8000, 30, self.root / "ssh.log")
            with patch("sts_agent.policy.ssh_tunnel.subprocess.Popen") as popen:
                with self.assertRaises(RuntimeError):
                    tunnel.start()
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
