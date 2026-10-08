"""Observe the game pipe even while a decision is waiting on UI or HTTP."""

from __future__ import annotations

import queue
import threading
from typing import Callable, TextIO


class GameDisconnected(Exception):
    """The owning game's input pipe closed; do not reconnect the UI."""


class GameInput:
    def __init__(self, stream: TextIO, interrupt: Callable[[], None]):
        self.stream = stream
        self.interrupt = interrupt
        self.stopped = threading.Event()
        self.reason = "CommunicationMod input closed"
        self.lines: queue.Queue[str] = queue.Queue(maxsize=128)
        # In-memory replay/test streams are finite fixtures, not game pipes.
        try:
            stream.fileno()
            self.live = True
        except (AttributeError, OSError, ValueError):
            self.live = False
        if self.live:
            threading.Thread(target=self._read, name="game-input", daemon=True).start()

    def _read(self) -> None:
        try:
            for line in self.stream:
                if self.stopped.is_set():
                    return
                try:
                    self.lines.put_nowait(line)
                except queue.Full:
                    # Never wait for queue space and thereby miss game exit.
                    # A runaway protocol producer is an error, not permission
                    # to discard/reorder states or guess the next action.
                    self.reason = "CommunicationMod input backlog exceeded 128 packets"
                    return
        except (OSError, ValueError) as exc:
            self.reason = f"CommunicationMod input failed: {exc}"
        finally:
            self.close()

    def close(self) -> None:
        if not self.stopped.is_set():
            self.stopped.set()
            self.interrupt()

    def check(self) -> None:
        if self.stopped.is_set():
            raise GameDisconnected(self.reason)

    def __iter__(self):
        if not self.live:
            yield from self.stream
            return
        while True:
            self.check()
            try:
                yield self.lines.get(timeout=0.1)
            except queue.Empty:
                continue

    def call(self, function, *args, **kwargs):
        """Keep shutdown responsive without waiting for a network timeout.

        Only main writes game commands. An interrupted worker cannot submit a
        late action; it is a daemon so a blocked HTTP read cannot hold exit up.
        """
        if not self.live:
            return function(*args, **kwargs)
        self.check()
        result: queue.Queue = queue.Queue(maxsize=1)

        def run():
            try:
                result.put((True, function(*args, **kwargs)))
            except BaseException as exc:
                result.put((False, exc))

        threading.Thread(target=run, name="game-decision", daemon=True).start()
        while True:
            self.check()
            try:
                success, value = result.get(timeout=0.1)
            except queue.Empty:
                continue
            self.check()
            if success:
                return value
            raise value
