"""An owned, loopback-only SSH forward; never launches a remote model."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path


class SshTunnel:
    def __init__(self, host: str, local_port: int, remote_port: int, timeout: float, log_path: Path):
        self.host, self.local_port, self.remote_port = host, local_port, remote_port
        self.timeout, self.log_path = timeout, log_path
        self.process: subprocess.Popen | None = None
        self.log = None

    def command(self) -> list[str]:
        return [
            "ssh", "-N", "-T", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", f"ConnectTimeout={int(self.timeout)}", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-L",
            f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}", self.host,
        ]

    def start(self) -> None:
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError(f"SSH tunnel exited; inspect {self.log_path}")
            return
        # Do not accidentally connect to an unrelated model already on this port.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", self.local_port))
            except OSError as exc:
                raise RuntimeError(f"local tunnel port {self.local_port} is already occupied") from exc
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self.log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"SSH forward failed; inspect {self.log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", self.local_port), timeout=0.2):
                        return
                except OSError:
                    time.sleep(0.1)
            raise RuntimeError("SSH forward startup timed out")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
