from __future__ import annotations

import copy
import io
import json
import socket
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sts_agent.commands import evaluate_baseline as batch
from sts_agent.config import load_agent_config
from sts_agent.core.modes import MODE_FRAMEWORKS
from sts_agent.policy.api import ApiPolicyClient
from sts_agent.runtime import controller


def suite(seeds=None):
    return {"schema_version": 1, "suite_id": "test-fixed", "character": "IRONCLAD", "ascension": 0,
            "seeds": seeds or ["1", "2"], "limits": dict(batch.LIMIT_DEFAULTS),
            "output_dir": "data/batches"}


def summary(run_id="run-1", seed="1", terminal=False, won=False):
    return {"run_id": run_id, "accepted_actions": 5, "model_actions": 3, "environment_actions": 2,
            "human_actions": 0, "rejected_actions": 0, "terminal": terminal,
            "won": won if terminal else None, "beat_heart": False if terminal else None,
            "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": seed,
                           "seed_source": "requested", "internal_seed": batch.seed_to_internal(seed)},
            "evaluation": {"autonomous_eligible": terminal, "score_final": terminal,
                           "progress_score": 100 if won else 4, "boss_acts_defeated": [1, 2, 3] if won else [],
                           "highest_floor": 51 if won else 5, "highest_act": 3 if won else 1}}


def packet(directory, *, packet_type="view", attached=False, in_game=False, run=None, ready=True, **extra):
    return {"type": packet_type, "in_game": in_game,
            "baseline": {"protocol_version": 1, "mode": "baseline_inference", "model": "baseline-model",
                         "experiment_dir": str(directory), "can_start_run": not in_game and ready,
                         "ready_for_command": ready, "batch_attached": attached,
                         "run_id": run["run_id"] if run else None, "run_summary": copy.deepcopy(run)},
            **extra}


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeSocket:
    def __init__(self, packets, clock=None):
        self.packets = list(packets)
        self.sent = []
        self.closed = False
        self.shutdown_called = False
        self.timeout = None
        self.clock = clock

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, size):
        if not self.packets:
            return b""
        value = self.packets.pop(0)
        if isinstance(value, tuple):
            seconds, value = value
            self.clock.advance(seconds)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode()

    def sendall(self, data):
        self.sent.append(json.loads(data.decode()))

    def shutdown(self, how):
        self.shutdown_called = how == socket.SHUT_RDWR

    def close(self):
        self.closed = True


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.experiment = self.root / "original-experiment"
        self.experiment.mkdir()
        self.original_manifest = {"mode": "baseline_inference", "model": "baseline-model",
                                  "requested_revision": "rev1", "settings": {"temperature": 0, "thinking": False}}
        (self.experiment / "manifest.json").write_text(json.dumps(self.original_manifest), encoding="utf-8")
        self.clock = FakeClock()
        self.addCleanup(self.temporary.cleanup)

    def report(self, planned=None):
        planned = planned or suite()
        return batch.BatchReport(self.root, planned, self.root / "suite.json", self.root / "agent.json",
                                 "127.0.0.1", 8765, self.clock)

    def state(self, planned=None):
        planned = planned or suite()
        report = self.report(planned)
        return batch.BatchStateMachine(planned, report, "baseline-model", self.clock)

    def run_client(self, packets, planned=None):
        planned = planned or suite()
        report = self.report(planned)
        connection = FakeSocket(packets, self.clock)
        events = []
        def connector(address, timeout):
            self.assertEqual(address, ("127.0.0.1", 8765))
            self.assertEqual(timeout, planned["limits"]["connection_timeout_seconds"])
            return connection
        result = batch.run_batch(planned, report, "baseline-model", "127.0.0.1", 8765,
                                 connector=connector, clock=self.clock, emit=events.append)
        self.assertTrue(connection.closed)
        self.assertTrue(connection.shutdown_called)
        return result, report, connection, events

    def initial(self, state):
        self.assertEqual(state.consume(packet(self.experiment)), [{"type": "baseline_batch_attach", "protocol_version": 1}])
        self.assertEqual(state.consume(packet(self.experiment, attached=True)), [{"type": "input", "text": "s IRONCLAD 0 1"}])

    def test_multiple_seed_tcp_flow_progress_terminal_menu_detach(self):
        first, second = summary(terminal=True), summary("run-2", "2", terminal=True, won=True)
        packets = [{"type": "hello"}, packet(self.experiment), packet(self.experiment),
                   packet(self.experiment, attached=True), packet(self.experiment, attached=True),
                   packet(self.experiment, packet_type="model_update", in_game=True, attached=True, run=summary(), model_pending=True),
                   packet(self.experiment, packet_type="model_update", in_game=True, attached=True, run=first, auto_execute=True),
                   packet(self.experiment, attached=True, run=first),
                   packet(self.experiment, attached=True, run=first),
                   packet(self.experiment, packet_type="model_update", in_game=True, attached=True, run=second),
                   packet(self.experiment, attached=True, run=second),
                   packet(self.experiment, attached=True, run=second),
                   packet(self.experiment, attached=False, run=second)]
        result, report, connection, events = self.run_client(packets)
        self.assertEqual(result, 0)
        self.assertEqual(connection.sent, [{"type": "baseline_batch_attach", "protocol_version": 1},
                                           {"type": "input", "text": "s IRONCLAD 0 1"},
                                           {"type": "input", "text": "s IRONCLAD 0 2"},
                                           {"type": "baseline_batch_detach", "protocol_version": 1}])
        self.assertEqual([row["status"] for row in report.rows], ["completed", "completed"])
        evaluation = json.loads((report.output_dir / "evaluation.json").read_text())
        self.assertEqual(evaluation["progress_score_mean"], 52)
        self.assertEqual(evaluation["win_rate"], .5)
        self.assertEqual(evaluation["technical_aborts"], 0)
        self.assertTrue((report.output_dir / "results.csv").exists())
        self.assertEqual(json.loads((self.experiment / "manifest.json").read_text()), self.original_manifest)
        self.assertTrue(any('"accepted_actions"' in event for event in events))

    def test_terminal_update_never_sends_next_start_until_menu(self):
        state = self.state()
        self.initial(state)
        terminal = packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=summary(terminal=True))
        self.assertEqual(state.consume(terminal), [])
        self.assertEqual(state.consume(terminal), [])
        self.assertEqual(state.phase, "await_menu")
        self.assertEqual(state.report.rows[0]["status"], "pending")
        self.assertEqual(state.consume(packet(self.experiment, attached=True, run=summary(terminal=True))),
                         [{"type": "input", "text": "s IRONCLAD 0 2"}])

    def test_existing_partial_game_refused_without_any_send(self):
        partial = summary()
        partial["run_config"]["partial_run"] = True
        result, report, connection, _ = self.run_client([packet(self.experiment, in_game=True, run=partial)])
        self.assertEqual(result, 1)
        self.assertEqual(connection.sent, [])
        self.assertEqual(report.rows[0]["status"], "technical_abort")
        self.assertEqual(report.rows[1]["status"], "pending")

    def test_stale_previous_summary_does_not_double_start_or_assign_run(self):
        previous = summary("prior", "9", terminal=True)
        state = self.state()
        state.consume(packet(self.experiment, run=previous))
        commands = state.consume(packet(self.experiment, attached=True, run=previous))
        self.assertEqual(len(commands), 1)
        for _ in range(4):
            self.assertEqual(state.consume(packet(self.experiment, attached=True, run=previous)), [])
        self.assertIsNone(state.run_id)

    def test_partial_human_seed_internal_mismatch_and_bad_terminal_fail(self):
        cases = []
        partial = summary(); partial["run_config"]["partial_run"] = True; cases.append(partial)
        human = summary(); human["human_actions"] = 1; cases.append(human)
        wrong_seed = summary(seed="2"); cases.append(wrong_seed)
        wrong_internal = summary(); wrong_internal["run_config"]["internal_seed"] = 999; cases.append(wrong_internal)
        unknown = summary(terminal=True); unknown["won"] = None; cases.append(unknown)
        ineligible = summary(terminal=True); ineligible["evaluation"]["autonomous_eligible"] = False; cases.append(ineligible)
        score = summary(terminal=True, won=True); score["evaluation"]["progress_score"] = 50; cases.append(score)
        bosses = summary(terminal=True); bosses["evaluation"]["boss_acts_defeated"] = [{}]; cases.append(bosses)
        for invalid in cases:
            with self.subTest(invalid=invalid):
                state = self.state()
                self.initial(state)
                with self.assertRaises(batch.BatchError):
                    state.consume(packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=invalid))

    def test_incomplete_menu_cannot_complete_or_advance(self):
        state = self.state()
        self.initial(state)
        with self.assertRaisesRegex(batch.BatchError, "without a matching"):
            state.consume(packet(self.experiment, attached=True, run=summary()))

    def test_pause_error_reflection_malformed_and_disconnect_retain_report(self):
        invalids = [packet(self.experiment, attached=True, in_game=True, run=summary()),
                    packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=summary(), error="bad model"),
                    packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=summary(), decision={"assist_status": "暂停，需人工处理", "assist_reason": "invalid alias"}),
                    {"type": "posthoc_reflection_request"}, b"not-json\n", b"[]\n", b"", OSError("connection failed")]
        for invalid in invalids:
            with self.subTest(invalid=invalid):
                result, report, connection, _ = self.run_client([packet(self.experiment), packet(self.experiment, attached=True), invalid])
                self.assertEqual(result, 1)
                self.assertEqual(len(connection.sent), 2)
                saved = json.loads((report.output_dir / "results.json").read_text())
                self.assertEqual([row["status"] for row in saved["results"]], ["technical_abort", "pending"])
                evaluation = json.loads((report.output_dir / "evaluation.json").read_text())
                self.assertIsNone(evaluation["win_rate"])
                self.assertEqual(evaluation["terminal_runs"], 0)

    def test_bad_metadata_mode_model_version_boolean_summary_and_directory(self):
        for field, invalid in [("mode", "collect_human"), ("model", "different"), ("protocol_version", True),
                               ("can_start_run", 1), ("experiment_dir", "relative"), ("run_id", 4),
                               ("run_summary", {"run_id": "other"})]:
            with self.subTest(field=field):
                state = self.state()
                value = packet(self.experiment)
                value["baseline"][field] = invalid
                with self.assertRaises(batch.BatchError):
                    state.consume(value)
        with self.assertRaises(batch.BatchError):
            self.state().consume({"type": "view", "in_game": False})

    def test_timeouts_max_actions_and_ctrl_c_close_socket(self):
        for name in ("idle_timeout_seconds", "run_timeout_seconds"):
            planned = suite()
            planned["limits"][name] = 1
            self.clock.value = 0
            result, report, _, _ = self.run_client([packet(self.experiment), packet(self.experiment, attached=True),
                                                   (2, TimeoutError("timed out"))], planned)
            self.assertEqual(result, 1)
            self.assertIn("timeout", report.manifest["failure"])
        result, report, connection, _ = self.run_client([packet(self.experiment), packet(self.experiment, attached=True), KeyboardInterrupt()])
        self.assertEqual(result, 130)
        self.assertEqual(report.rows[0]["status"], "technical_abort")
        self.assertNotIn("baseline_batch_detach", [command["type"] for command in connection.sent])
        state = self.state()
        self.initial(state)
        excessive = summary()
        excessive["accepted_actions"] = state.suite["limits"]["max_actions_per_run"]
        with self.assertRaisesRegex(batch.BatchError, "maximum actions"):
            state.consume(packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=excessive))

    def test_disconnect_after_completed_run_preserves_completed_row(self):
        result, report, connection, _ = self.run_client([packet(self.experiment), packet(self.experiment, attached=True),
                                                       packet(self.experiment, attached=True, run=summary(terminal=True)), b""])
        self.assertEqual(result, 1)
        self.assertEqual([row["status"] for row in report.rows], ["completed", "technical_abort"])
        self.assertEqual(connection.sent[-1]["text"], "s IRONCLAD 0 2")
        evaluation = json.loads((report.output_dir / "evaluation.json").read_text())
        self.assertEqual(evaluation["eligible_runs"], 1)
        self.assertEqual(evaluation["technical_aborts"], 1)

    def test_detach_loss_reports_batch_failure_without_losing_terminal_outcome(self):
        planned = suite(["1"])
        result, report, connection, _ = self.run_client([packet(self.experiment), packet(self.experiment, attached=True),
                                                       packet(self.experiment, attached=True, run=summary(terminal=True)), b""], planned)
        self.assertEqual(result, 1)
        self.assertEqual(report.manifest["status"], "technical_abort")
        self.assertEqual(report.rows[0]["status"], "completed")
        self.assertEqual(connection.sent[-1]["type"], "baseline_batch_detach")

    def test_api_performance_joins_run_ids_and_excludes_other_runs(self):
        records = [{"event": "request", "run_id": "other", "request_id": 1},
                   {"event": "response", "request_id": 1, "latency_ms": 9000, "response": {"usage": {"total_tokens": 999}}},
                   {"event": "request", "run_id": "run-1", "request_id": 2},
                   {"event": "transport_error", "error": "retry"},
                   {"event": "response", "request_id": 2, "input_tokens": 10, "latency_ms": 100,
                    "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}},
                   {"event": "stale_action_discarded", "request_id": 2},
                   {"event": "request", "run_id": "run-1", "request_id": 3},
                   {"event": "decision_error", "request_id": 3},
                   {"event": "request", "run_id": "run-1", "request_id": 4},
                   {"event": "response", "request_id": 4, "latency_ms": 300, "response": {}},
                   {"event": "request", "run_id": "other", "request_id": 5},
                   {"event": "transport_error", "error": "exclude"}]
        path = self.experiment / "requests.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        result = batch.api_performance(path, {"run-1"})
        self.assertEqual(result["calls"], 3)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["transport_errors"], 1)
        self.assertEqual(result["discards"], 1)
        self.assertEqual(result["token_usage"]["total_tokens"], 12)
        self.assertEqual(result["latency_ms_p50"], 200)
        self.assertEqual(result["latency_ms_p95"], 290)

    def test_progress_save_does_not_rescan_full_api_log(self):
        state = self.state()
        self.initial(state)
        with patch.object(batch, "api_performance", side_effect=AssertionError("unnecessary source-log scan")):
            for _ in range(4):
                state.consume(packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=summary()))

    def test_opening_attempt_is_not_counted_as_confirmed_run_before_game_progress(self):
        state = self.state()
        self.initial(state)
        evaluation = json.loads((state.report.output_dir / "evaluation.json").read_text())
        self.assertEqual(evaluation["runs_attempted"], 1)
        self.assertEqual(evaluation["runs_started"], 0)
        state.consume(packet(self.experiment, packet_type="model_update", attached=True, in_game=True, run=summary()))
        evaluation = json.loads((state.report.output_dir / "evaluation.json").read_text())
        self.assertEqual(evaluation["runs_started"], 1)

    def test_socket_is_closed_before_abort_report_is_written(self):
        planned = suite()
        report = self.report(planned)
        connection = FakeSocket([packet(self.experiment), packet(self.experiment, attached=True), KeyboardInterrupt()])
        original = report.abort

        def abort(*args, **kwargs):
            self.assertTrue(connection.closed)
            original(*args, **kwargs)

        with patch.object(report, "abort", side_effect=abort):
            result = batch.run_batch(planned, report, "baseline-model", "127.0.0.1", 8765,
                                     connector=lambda *args, **kwargs: connection, emit=lambda message: None)
        self.assertEqual(result, 130)

    def test_real_tcp_controller_packets_attach_two_runs_and_clean_detach(self):
        """Exercise the actual UI bridge/ask_action contract, without a game."""
        bridge = controller.TerminalBridge("127.0.0.1", 0)
        port = bridge.server.getsockname()[1]
        planned = suite()
        report = self.report(planned)
        api = SimpleNamespace(model="baseline-model", output_dir=self.experiment, run_summaries={}, pending=None)
        policy = Mock(runtime="remote_api", screens=("ALL",), api=api)
        policy.status_decision.return_value = {"mode": "baseline_inference"}
        menu = {"in_game": False, "ready_for_command": True, "available_commands": ["start", "state"]}
        terminal = {"in_game": True, "ready_for_command": True, "available_commands": ["confirm", "state"],
                    "game_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 5, "current_hp": 0}}
        errors = []

        def controller_flow():
            run_id = None
            try:
                for number, seed in enumerate(planned["seeds"], 1):
                    command, _, _ = controller.ask_action(bridge, menu, run_id, policy=policy)
                    self.assertEqual(command, f"START IRONCLAD 0 {seed}")
                    run_id = f"run-{number}"
                    api.run_summaries[run_id] = summary(run_id, seed, terminal=True)
                    command, _, _ = controller.ask_action(bridge, terminal, run_id, policy=policy)
                    self.assertEqual(command, "CONFIRM")
                # Menu processing consumes final detach, then waits for a new UI.
                controller.ask_action(bridge, menu, run_id, policy=policy)
            except controller.GameDisconnected:
                pass
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=controller_flow, daemon=True)
        try:
            with patch.object(controller.time, "sleep"):
                worker.start()
                result = batch.run_batch(planned, report, "baseline-model", "127.0.0.1", port,
                                         emit=lambda message: None)
        finally:
            bridge.interrupt()
            worker.join(timeout=2)
            bridge.close()
        self.assertEqual(errors, [])
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, 0)
        self.assertEqual([row["status"] for row in report.rows], ["completed", "completed"])
        self.assertIsNone(bridge.batch_guard.failure)
        policy.rank.assert_not_called()

    def test_controller_error_metadata_stops_even_without_ui_reason(self):
        state = self.state()
        value = packet(self.experiment)
        value["baseline"]["controller_error"] = "protocol mismatch"
        with self.assertRaisesRegex(batch.BatchError, "protocol mismatch"):
            state.consume(value)

    def test_loaded_controller_configuration_must_match_current_profile(self):
        expected = {"model": {"revision": "rev1", "initial_adapter": ""},
                    "api": {"temperature": 0, "thinking": False}}
        report = batch.BatchReport(self.root, suite(), self.root / "suite.json", self.root / "agent.json",
                                   "127.0.0.1", 8765, self.clock, expected_policy=expected)
        report.freeze_experiment(str(self.experiment), "baseline-model")
        manifest = {**self.original_manifest, "settings": {"temperature": 1, "thinking": False}}
        (self.experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(batch.BatchError, "settings.temperature.*restart game/controller"):
            report.freeze_experiment(str(self.experiment), "baseline-model")

    def test_real_baseline_profile_and_api_manifest_pass_preflight_without_adapter(self):
        config = load_agent_config(mode="baseline_inference")
        self.assertEqual(config["policy"]["model"]["initial_adapter"], "")
        client = ApiPolicyClient(self.root, config)
        self.addCleanup(client.close)
        report = batch.BatchReport(self.root, suite(), self.root / "suite.json", self.root / "agent.json",
                                   "127.0.0.1", 8765, self.clock, expected_policy=config["policy"])
        report.freeze_experiment(str(client.output_dir), client.model)
        self.assertIsNone(report.model_manifest["adapter"])
        self.assertEqual(report.manifest["preflight"]["configuration_differences"], [])

    def test_real_mismatch_is_still_rejected_and_diagnostics_survive_abort(self):
        expected = {"model": {"revision": "rev1", "initial_adapter": ""},
                    "api": {"temperature": 0, "thinking": False, "serving": {"max_model_len": 16384}}}
        invalid = {**self.original_manifest, "adapter": "unexpected-lora",
                   "settings": {"temperature": 0, "thinking": False, "serving": {"max_model_len": 8192}}}
        (self.experiment / "manifest.json").write_text(json.dumps(invalid), encoding="utf-8")
        report = batch.BatchReport(self.root, suite(), self.root / "suite.json", self.root / "agent.json",
                                   "127.0.0.1", 8765, self.clock, expected_policy=expected)
        with self.assertRaisesRegex(batch.BatchError, "adapter, settings.serving.max_model_len") as raised:
            report.freeze_experiment(str(self.experiment), "baseline-model")
        report.abort(0, str(raised.exception), None)
        saved = json.loads((report.output_dir / "manifest.json").read_text())
        self.assertEqual(saved["preflight"]["observed_experiment_dir"], str(self.experiment.resolve()))
        self.assertEqual(saved["preflight"]["observed_model_manifest"], invalid)
        self.assertEqual(saved["preflight"]["configuration_differences"], ["adapter", "settings.serving.max_model_len"])
        self.assertIsNone(saved["model_manifest"])  # Invalid settings were never frozen as accepted.

    def test_model_manifest_frozen_and_original_artifact_paths_preserved(self):
        state = self.state()
        self.initial(state)
        run = summary(terminal=True)
        trace = self.experiment / "action-trace-session-run-1-IRONCLAD.jsonl"
        trace.write_text("original", encoding="utf-8")
        state.consume(packet(self.experiment, attached=True, run=run))
        self.assertEqual(state.report.rows[0]["artifacts"]["action_traces"], [str(trace.resolve())])
        self.assertEqual(trace.read_text(), "original")
        manifest = dict(self.original_manifest, settings={"temperature": 1})
        (self.experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(batch.BatchError, "configuration changed"):
            state.report.freeze_experiment(str(self.experiment), "baseline-model")

    def test_suite_strict_validation_and_signed_seed(self):
        invalids = [{"seeds": ["1", "1"]}, {"seeds": ["0"]}, {"seeds": ["O"]}, {"seeds": ["lower"]},
                    {"seeds": ["-1"]}, {"seeds": ["01"]}, {"seeds": [""]},
                    {"seeds": ["Z" * 20]}, {"character": "SILENT"}, {"ascension": True},
                    {"schema_version": True}, {"output_dir": "../escape"}, {"unknown": 1},
                    {"limits": {**batch.LIMIT_DEFAULTS, "max_actions_per_run": 1.5}},
                    {"limits": {**batch.LIMIT_DEFAULTS, "idle_timeout_seconds": float("nan")}}]
        for changes in invalids:
            with self.subTest(changes=changes), self.assertRaises(batch.BatchError):
                batch.validate_suite({**suite(), **changes})
        self.assertEqual(batch.seed_to_internal("1"), 1)
        self.assertEqual(batch.seed_to_internal("Z"), 34)
        # Encode the unsigned representation of Java's minimum long.
        value, encoded = 1 << 63, ""
        while value:
            value, digit = divmod(value, 35)
            encoded = batch.SEED_ALPHABET[digit] + encoded
        self.assertEqual(batch.seed_to_internal(encoded), -(1 << 63))

    def test_dry_run_has_no_connection_report_or_commands(self):
        suite_path = self.root / "suite.json"
        suite_path.write_text(json.dumps(suite()), encoding="utf-8")
        config = {"framework": MODE_FRAMEWORKS["baseline_inference"],
                  "policy": {"runtime": "remote_api", "model": {"base": "baseline-model"}},
                  "server": {"host": "127.0.0.1", "port": 8765}}
        with patch.object(batch, "PROJECT_ROOT", self.root), patch.object(batch, "load_agent_config", return_value=config), \
                patch.object(batch, "BatchReport") as report, patch.object(batch, "run_batch") as runner, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(batch.main(["--suite", str(suite_path), "--dry-run"]), 0)
        report.assert_not_called()
        runner.assert_not_called()
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(json.loads(output.getvalue())["commands"], ["s IRONCLAD 0 1", "s IRONCLAD 0 2"])
        config["framework"] = MODE_FRAMEWORKS["collect_human"]
        with patch.object(batch, "PROJECT_ROOT", self.root), patch.object(batch, "load_agent_config", return_value=config):
            self.assertEqual(batch.main(["--suite", str(suite_path), "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
