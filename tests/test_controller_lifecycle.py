"""Shutdown tests use local pipes/sockets and owned children, never a game/GPU."""

import ctypes
import io
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from ctypes import wintypes
from pathlib import Path
from unittest.mock import Mock, patch

from sts_agent.policy.ssh_tunnel import SshTunnel
from sts_agent.runtime.controller import TerminalBridge
from sts_agent.runtime.lifecycle import GameDisconnected, GameInput
from sts_agent.runtime.owned_process import WindowsChildJob


class LifecycleTests(unittest.TestCase):
    def test_real_controller_exits_when_game_pipe_closes_while_waiting_for_ui(self):
        # No model/network: exercise main(), real stdin pipe, real TCP accept,
        # and interpreter exit (including daemon thread finalization).
        code = """
import sys
from pathlib import Path
from unittest.mock import Mock
from sts_agent.runtime import controller
config = controller.load_agent_config()
controller.PROJECT_ROOT = Path(sys.argv[1])
controller.load_agent_config = lambda *args: config
real_bridge = controller.TerminalBridge
controller.TerminalBridge = lambda *args: real_bridge('127.0.0.1', 0)
evolution = Mock(enabled=False)
controller.PromptEvolutionManager = lambda *args: evolution
policy = Mock(runtime='local', enabled=False, screens=('ALL',))
policy.status_decision.return_value = {'model_candidates': []}
controller.PolicyClient = lambda *args: policy
controller.TeacherClient = lambda *args: Mock(enabled=False, prompt_evolution=evolution)
controller.PosthocFeedbackClient = lambda *args: Mock(enabled=False)
sys.argv = ['controller']
raise SystemExit(controller.main())
"""
        with tempfile.TemporaryDirectory() as directory:
            child = subprocess.Popen(
                [sys._base_executable, "-c", code, directory], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            lines = queue.Queue()
            output = []

            def collect():
                for line in child.stdout:
                    output.append(line)
                    lines.put(line)

            collector = threading.Thread(target=collect, daemon=True)
            collector.start()

            def expect(fragment):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        if fragment in lines.get(timeout=0.1):
                            return
                    except queue.Empty:
                        if child.poll() is not None:
                            break
                self.fail(f"missing {fragment!r}: {output!r}")

            try:
                expect("ready")
                child.stdin.write('{"in_game":false,"ready_for_command":true,"available_commands":["start","state"]}\n')
                child.stdin.flush()
                expect("waiting for terminal client")
                child.stdin.close()
                self.assertEqual(child.wait(timeout=3), 0)
                collector.join(2)
                self.assertIn("controller stopped: CommunicationMod input closed", "".join(output))
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=3)
                if not child.stdin.closed:
                    child.stdin.close()
                collector.join(2)
                child.stdout.close()

    def test_finite_replay_preserves_all_packets_and_runs_synchronously(self):
        source = GameInput(io.StringIO("one\ntwo\n"), Mock())
        self.assertEqual(list(source), ["one\n", "two\n"])
        self.assertEqual(source.call(lambda: threading.get_ident()), threading.get_ident())

    def test_pipe_eof_interrupts_waiting_for_next_packet(self):
        reader, writer = os.pipe()
        with os.fdopen(reader, "r") as stream:
            interrupted = threading.Event()
            source = GameInput(stream, interrupted.set)
            os.write(writer, b"one\n")
            self.assertEqual(next(iter(source)), "one\n")
            os.close(writer)
            self.assertTrue(interrupted.wait(2))
            with self.assertRaises(GameDisconnected):
                next(iter(source))

    def test_input_backlog_stops_instead_of_blocking_shutdown_or_dropping_states(self):
        reader, writer = os.pipe()
        with os.fdopen(reader, "r") as stream:
            source = GameInput(stream, lambda: None)
            try:
                os.write(writer, b"state\n" * 129)
                self.assertTrue(source.stopped.wait(2))
                with self.assertRaisesRegex(GameDisconnected, "backlog"):
                    next(iter(source))
            finally:
                os.close(writer)

    def test_pipe_eof_cancels_blocked_model_without_waiting_for_timeout(self):
        reader, writer = os.pipe()
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        errors = []
        with os.fdopen(reader, "r") as stream:
            source = GameInput(stream, lambda: None)

            def model():
                started.set()
                release.wait(5)
                return "late action"

            def decide():
                try:
                    source.call(model)
                except GameDisconnected as exc:
                    errors.append(exc)
                finally:
                    finished.set()

            worker = threading.Thread(target=decide, daemon=True)
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                os.close(writer)
                self.assertTrue(finished.wait(2))
                self.assertEqual(len(errors), 1)
            finally:
                release.set()
                worker.join(2)

    def test_worker_exception_propagates_without_becoming_a_game_command(self):
        reader, writer = os.pipe()
        with os.fdopen(reader, "r") as stream:
            source = GameInput(stream, lambda: None)
            try:
                with self.assertRaisesRegex(ValueError, "bad model"):
                    source.call(lambda: (_ for _ in ()).throw(ValueError("bad model")))
            finally:
                os.close(writer)
                self.assertTrue(source.stopped.wait(2))

    def check_bridge_interrupt(self, connect):
        bridge = TerminalBridge("127.0.0.1", 0)
        done, errors = threading.Event(), []

        def receive():
            try:
                bridge.receive()
            except GameDisconnected as exc:
                errors.append(exc)
            finally:
                done.set()

        worker = threading.Thread(target=receive, daemon=True)
        worker.start()
        client = None
        try:
            if connect:
                client = socket.create_connection(bridge.server.getsockname(), timeout=2)
                # Reading hello ensures receive() has reached the connected IO.
                self.assertIn(b'"hello"', client.recv(4096))
                client.sendall(b'{"type":')  # no newline: blocking readline
            bridge.interrupt()
            self.assertTrue(done.wait(2))
            self.assertEqual(len(errors), 1)
        finally:
            bridge.close()
            if client:
                client.close()
            worker.join(2)

    def test_game_close_unblocks_accept_without_overlay(self):
        self.check_bridge_interrupt(False)

    def test_game_close_unblocks_partial_overlay_input(self):
        self.check_bridge_interrupt(True)

    def test_overlay_disconnect_does_not_cancel_game_and_can_reconnect(self):
        bridge = TerminalBridge("127.0.0.1", 0)
        try:
            for expected in (None, {"type": "input", "text": "h"}):
                result = []
                worker = threading.Thread(target=lambda: result.append(bridge.receive()), daemon=True)
                worker.start()
                with socket.create_connection(bridge.server.getsockname(), timeout=2) as client:
                    client.recv(4096)
                    if expected is not None:
                        client.sendall(b'{"type":"input","text":"h"}\n')
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(result, [expected])
                self.assertFalse(bridge.interrupted.is_set())
                bridge.close_client()
        finally:
            bridge.close()

    def test_fragmented_utf8_and_multiple_buffered_lines(self):
        bridge = TerminalBridge("127.0.0.1", 0)
        bridge.connection = Mock()
        text = '{"text":"中文"}\n{"text":"h"}\n'.encode("utf-8")
        split = text.index("中文".encode("utf-8")) + 1
        bridge.connection.recv.side_effect = [text[:split], text[split:]]
        try:
            with patch("sts_agent.runtime.controller.select.select", return_value=([bridge.connection], [], [])):
                self.assertEqual(bridge.receive(), {"text": "中文"})
                self.assertEqual(bridge.receive(), {"text": "h"})
            self.assertEqual(bridge.connection.recv.call_count, 2)
        finally:
            bridge.close()

    def test_closed_tunnel_cannot_launch_a_late_child(self):
        with tempfile.TemporaryDirectory() as directory:
            tunnel = SshTunnel("unused", 18080, 8000, 30, Path(directory) / "ssh.log")
            tunnel.close()
            with patch("sts_agent.policy.ssh_tunnel.subprocess.Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "shut down"):
                    tunnel.start()
                popen.assert_not_called()

    def test_guard_attach_failure_cleans_up_owned_process(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()
            tunnel = SshTunnel("unused", port, 8000, 30, Path(directory) / "ssh.log")
            child, job = Mock(), Mock()
            child.poll.return_value = None
            job.attach.side_effect = OSError("job assignment failed")
            with patch("sts_agent.policy.ssh_tunnel.WindowsChildJob", return_value=job), patch(
                "sts_agent.policy.ssh_tunnel.subprocess.Popen", return_value=child
            ):
                with self.assertRaisesRegex(OSError, "job assignment"):
                    tunnel.start()
            child.terminate.assert_called_once()
            job.close.assert_called_once()
            self.assertIsNone(tunnel.log)
            self.assertIsNone(tunnel.process)

    @unittest.skipUnless(os.name == "nt", "Windows job ownership")
    def test_job_close_kills_only_assigned_child(self):
        job = WindowsChildJob()
        child = subprocess.Popen(
            [sys._base_executable, "-c", "import time; time.sleep(60)"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            job.attach(child)
            job.close()
            child.wait(timeout=3)
            self.assertIsNotNone(child.returncode)
        finally:
            job.close()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)

    @unittest.skipUnless(os.name == "nt", "Windows forced owner exit")
    def test_forced_owner_exit_also_kills_assigned_child(self):
        code = (
            "import subprocess, sys; from sts_agent.runtime.owned_process import WindowsChildJob; "
            "job=WindowsChildJob(); "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "creationflags=subprocess.CREATE_NO_WINDOW); job.attach(child); "
            "print(child.pid,flush=True); sys.stdin.readline()"
        )
        owner = subprocess.Popen(
            [sys._base_executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = None
        try:
            pid = int(owner.stdout.readline())
            handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            self.assertTrue(handle)
            owner.kill()  # bypass Python finally entirely
            owner.wait(timeout=3)
            self.assertEqual(kernel.WaitForSingleObject(handle, 3000), 0)
        finally:
            if handle:
                kernel.CloseHandle(handle)
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=3)
            owner.stdin.close()
            owner.stdout.close()


if __name__ == "__main__":
    unittest.main()
