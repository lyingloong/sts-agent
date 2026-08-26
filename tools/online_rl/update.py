"""Action-level group-relative policy updates for complete StS episodes.

This is intentionally not a wrapper around TRL's text-completion GRPO.  A StS
episode has a different observation after every action, so one group contains
four trajectories from the same initial condition and frozen policy version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policy_server import build_scoring_prompt


@dataclass(frozen=True)
class UpdateConfig:
    max_length: int = 4096
    learning_rate: float = 1e-5
    clip_range: float = 0.2
    update_epochs: int = 1
    max_grad_norm: float = 1.0


class TrainableAliasPolicy:
    """One 4-bit QLoRA policy that alternates between action scoring and update."""

    def __init__(
        self,
        model_name: str,
        adapter: Path,
        config: UpdateConfig,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not adapter.is_dir():
            raise ValueError(f"initial_adapter does not exist: {adapter}")
        if not torch.cuda.is_available():
            raise RuntimeError("online RL worker requires a CUDA GPU")
        self.torch = torch
        self.config = config
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bf16 = bool(torch.cuda.is_bf16_supported())
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            )
            model_kwargs["device_map"] = {"": 0}
        base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if load_in_4bit:
            # The rollout update is one observation at a time. Keeping KV
            # cache enabled avoids recomputing a 4k prefix per legal alias.
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        elif hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        self.model.config.use_cache = True
        self.device = next(self.model.parameters()).device
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("the loaded adapter has no trainable parameters")
        try:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=config.learning_rate)
        except ImportError:
            self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)

    def _prefix_and_suffixes(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> tuple[Any, list[str], list[list[int]]]:
        prompt = build_scoring_prompt(observation, legal_actions)
        try:
            prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        aliases: list[str] = []
        suffixes: list[list[int]] = []
        prefix_list = prefix_ids[0].tolist()
        for action in legal_actions:
            alias = str(action.get("id") or "")
            if not alias or alias in aliases:
                raise ValueError("legal action aliases must be non-empty and unique")
            combined = self.tokenizer(prefix + alias + "\n", add_special_tokens=False)["input_ids"]
            if combined[: len(prefix_list)] != prefix_list:
                raise RuntimeError("tokenizer changed the action scoring prefix boundary")
            suffix = combined[len(prefix_list) :]
            if not suffix:
                raise RuntimeError(f"empty continuation for {alias}")
            aliases.append(alias)
            suffixes.append(suffix)
        longest = len(prefix_list) + max(map(len, suffixes), default=0)
        if longest > self.config.max_length:
            raise ValueError(f"online RL sample exceeds max_length={self.config.max_length}")
        return prefix_ids, aliases, suffixes

    def action_logprobs(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> tuple[list[str], Any]:
        """Return unnormalized alias continuation log probabilities, with gradients."""

        prefix_ids, aliases, suffixes = self._prefix_and_suffixes(observation, legal_actions)
        output = self.model(input_ids=prefix_ids, use_cache=True)
        first_logprobs = self.torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        values = []
        for suffix in suffixes:
            value = first_logprobs[suffix[0]]
            if len(suffix) > 1:
                continuation = self.torch.tensor([suffix[:-1]], dtype=self.torch.long, device=self.device)
                continuation_output = self.model(
                    input_ids=continuation,
                    past_key_values=output.past_key_values,
                    use_cache=False,
                )
                next_logprobs = self.torch.log_softmax(continuation_output.logits[0].float(), dim=-1)
                for position, token_id in enumerate(suffix[1:]):
                    value = value + next_logprobs[position, token_id]
            values.append(value)
        return aliases, self.torch.stack(values)

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.model.eval()
        with self.torch.no_grad():
            aliases, values = self.action_logprobs(observation, legal_actions)
        labels = {str(action.get("id")): str(action.get("label") or action.get("id")) for action in legal_actions}
        return [
            {"action_id": alias, "label": labels[alias], "logprob": float(value)}
            for alias, value in zip(aliases, values.tolist())
        ]

    def update(self, episodes: list[dict[str, Any]], rewards: list[float], sampling_temperature: float = 1.0) -> dict[str, Any]:
        if len(episodes) != len(rewards) or len(episodes) < 2:
            raise ValueError("a group-relative update requires at least two rewarded episodes")
        if sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        scale = math.sqrt(variance)
        advantages = [0.0 if scale < 1e-8 else (reward - mean) / (scale + 1e-8) for reward in rewards]
        step_count = sum(len(episode["steps"]) for episode in episodes)
        if step_count < 1:
            raise ValueError("rollout group has no steps")

        self.model.train()
        losses: list[float] = []
        for _ in range(self.config.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for episode, advantage in zip(episodes, advantages):
                for step in episode["steps"]:
                    aliases, raw_logprobs = self.action_logprobs(step["observation"], step["legal_actions"])
                    try:
                        selected_index = aliases.index(step["selected_alias"])
                    except ValueError as exc:
                        raise ValueError("selected alias disappeared from an online RL step") from exc
                    # The stored behavior logprob includes the rollout
                    # temperature, so apply that same distribution here.
                    current_logprob = self.torch.log_softmax(raw_logprobs / sampling_temperature, dim=0)[selected_index]
                    old_logprob = float(step["old_logprob"])
                    ratio = self.torch.exp(current_logprob - old_logprob)
                    unclipped = ratio * advantage
                    clipped = self.torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantage
                    loss = -self.torch.minimum(unclipped, clipped) / step_count
                    loss.backward()
                    epoch_loss += float(loss.detach())
            grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
            self.optimizer.step()
            losses.append(epoch_loss)
        self.model.eval()
        return {
            "algorithm": "episode_group_relative_ppo",
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": mean,
            "reward_std": scale,
            "steps": step_count,
            "losses": losses,
            "grad_norm": grad_norm,
            "sampling_temperature": sampling_temperature,
        }

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path / "tokenizer")
