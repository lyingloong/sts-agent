"""Policy backend protocol shared by local, remote and teacher implementations."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import PolicyCandidate


class PolicyBackend(Protocol):
    """Minimal policy surface required by the game runtime."""

    model_id: str

    def prepare(self) -> None:
        """Warm up the backend before the first game action."""

    def rank(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        top_k: int,
    ) -> list[PolicyCandidate]:
        """Return candidates in descending preference."""

    def score(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[PolicyCandidate]:
        """Return one masked score/log-probability per legal action."""

    def close(self) -> None:
        """Release model or network resources."""
