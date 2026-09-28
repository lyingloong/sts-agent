"""StS1 adapter boundary.

CommunicationMod-specific parsing lives in ``sts_agent.runtime.controller`` as
the StS1 runtime orchestrator. This adapter is the dependency-injected game
boundary used by framework code and tests.
"""

from __future__ import annotations

from typing import Any, Callable

from sts_agent.core.contracts import Action, Observation


class Sts1Adapter:
    def __init__(
        self,
        *,
        normalize: Callable[[dict[str, Any]], Observation],
        observe: Callable[..., Observation],
        actions: Callable[..., list[Action]],
        execute: Callable[..., Any],
    ) -> None:
        self._normalize = normalize
        self._observe = observe
        self._actions = actions
        self._execute = execute

    def normalize_message(self, message: dict[str, Any]) -> Observation:
        return self._normalize(message)

    def policy_observation(self, message: dict[str, Any], **kwargs: Any) -> Observation:
        return self._observe(message, **kwargs)

    def legal_actions(self, message: dict[str, Any], **kwargs: Any) -> list[Action]:
        return self._actions(message, **kwargs)

    def execute_action(self, message: dict[str, Any], action: Action, **kwargs: Any) -> Any:
        return self._execute(message, action, **kwargs)
