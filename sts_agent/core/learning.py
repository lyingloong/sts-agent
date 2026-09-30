"""Learning method descriptors used by configuration and experiment tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LearningMethod:
    name: str
    requires_human: bool = False
    updates_parameters: bool = False
    updates_prompt: bool = False


@dataclass(frozen=True, slots=True)
class LearningEntrypoint:
    """Compatibility metadata for the current learning command scripts."""

    method: str
    entrypoint: str
    artifact: str


LEARNING_METHODS = {
    "none": LearningMethod("none"),
    "collect_human": LearningMethod("collect_human", requires_human=True),
    "teacher_sft": LearningMethod("teacher_sft", updates_parameters=True),
    "dagger": LearningMethod("dagger", requires_human=True, updates_parameters=True),
    "prompt_evolution": LearningMethod("prompt_evolution", requires_human=True, updates_prompt=True),
    "online_rl": LearningMethod("online_rl", updates_parameters=True),
}

LEARNING_ENTRYPOINTS = {
    "collect_human": LearningEntrypoint("collect_human", "sts_agent.recording.communication_bridge", "data/trajectories/human"),
    "teacher_sft": LearningEntrypoint("teacher_sft", "sts_agent.learning.sft", "outputs/sft"),
    "dagger": LearningEntrypoint("dagger", "sts_agent.runtime.controller", "data/datasets/dagger"),
    "prompt_evolution": LearningEntrypoint("prompt_evolution", "sts_agent.learning.prompt_evolution", "prompts/characters"),
    "online_rl": LearningEntrypoint("online_rl", "sts_agent.learning.online_rl.worker", "outputs/online_rl"),
}


def learning_entrypoint(method: str) -> LearningEntrypoint | None:
    """Return the canonical command/artifact mapping for a learning method."""

    return LEARNING_ENTRYPOINTS.get(method)
