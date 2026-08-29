"""Action-level group-relative policy updates for complete StS episodes.

This is intentionally not a wrapper around TRL's text-completion GRPO.  A StS
episode has a different observation after every action, so one group contains
four trajectories from the same initial condition and frozen policy version.
"""

from __future__ import annotations

import inspect
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
    gradient_checkpointing: bool = True
    empty_cache_between_steps: bool = True


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
        # Cache is useful for no-grad action sampling, but it is prohibitively
        # expensive during a 4k-token backward pass. Keep the model itself in
        # the low-memory training configuration; ``score`` opts into cache for
        # its short-lived inference forwards below.
        base.config.use_cache = False
        if load_in_4bit:
            checkpoint_kwargs: dict[str, Any] = {}
            if config.gradient_checkpointing and "gradient_checkpointing_kwargs" in inspect.signature(
                prepare_model_for_kbit_training
            ).parameters:
                # PyTorch 2.9 requires this to be explicit. Non-reentrant
                # checkpointing is the lower-overhead choice for this model.
                checkpoint_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=config.gradient_checkpointing,
                **checkpoint_kwargs,
            )
        elif hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        self.model.config.use_cache = False
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

    def action_logprobs(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        use_cache: bool,
    ) -> tuple[list[str], Any]:
        """Return unnormalized alias continuation log probabilities, with gradients."""

        prefix_ids, aliases, suffixes = self._prefix_and_suffixes(observation, legal_actions)
        output = self.model(input_ids=prefix_ids, use_cache=use_cache)
        first_logprobs = self.torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        values = []
        for suffix in suffixes:
            value = first_logprobs[suffix[0]]
            if len(suffix) > 1:
                continuation = self.torch.tensor([suffix[:-1]], dtype=self.torch.long, device=self.device)
                if use_cache:
                    continuation_output = self.model(
                        input_ids=continuation,
                        past_key_values=output.past_key_values,
                        use_cache=False,
                    )
                    continuation_logits = continuation_output.logits[0]
                else:
                    # Gradient checkpointing is incompatible with retaining a
                    # 4k-token KV cache. Re-evaluate only the few aliases that
                    # have multi-token continuations; A<n> is usually one
                    # token, so this remains rare in practice.
                    continuation_output = self.model(
                        input_ids=self.torch.cat((prefix_ids, continuation), dim=1),
                        use_cache=False,
                    )
                    continuation_logits = continuation_output.logits[0, -len(suffix) : -1]
                next_logprobs = self.torch.log_softmax(continuation_logits.float(), dim=-1)
                for position, token_id in enumerate(suffix[1:]):
                    value = value + next_logprobs[position, token_id]
            values.append(value)
        return aliases, self.torch.stack(values)

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.model.eval()
        with self.torch.no_grad():
            aliases, values = self.action_logprobs(observation, legal_actions, use_cache=True)
        labels = {str(action.get("id")): str(action.get("label") or action.get("id")) for action in legal_actions}
        return [
            {"action_id": alias, "label": labels[alias], "logprob": float(value)}
            for alias, value in zip(aliases, values.tolist())
        ]

    def update(
        self,
        episodes: list[dict[str, Any]],
        rewards: list[float],
        sampling_temperature: float = 1.0,
        *,
        attribution_enabled: bool = False,
        max_abs_advantage: float = 2.0,
    ) -> dict[str, Any]:
        if len(episodes) != len(rewards) or len(episodes) < 2:
            raise ValueError("a group-relative update requires at least two rewarded episodes")
        if sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if max_abs_advantage <= 0:
            raise ValueError("max_abs_advantage must be positive")
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        scale = math.sqrt(variance)
        advantages = [0.0 if scale < 1e-8 else (reward - mean) / (scale + 1e-8) for reward in rewards]
        all_step_count = sum(len(episode["steps"]) for episode in episodes)
        if all_step_count < 1:
            raise ValueError("rollout group has no steps")

        weighted_steps: list[tuple[dict[str, Any], float, float]] = []
        reinforce_steps = 0
        weaken_steps = 0
        if attribution_enabled:
            for episode, advantage in zip(episodes, advantages):
                by_step_id = {int(step["step_id"]): step for step in episode["steps"]}
                # Human direction always controls the sign. Group-relative
                # reward supplies only update strength, preventing a negative
                # note in a low-reward rollout from being accidentally
                # reinforced by a double-negative sign flip.
                magnitude = min(abs(advantage), max_abs_advantage)
                for attribution in episode.get("attributions", []):
                    step = by_step_id.get(int(attribution["step_id"]))
                    if step is None:
                        raise ValueError("attribution references an unknown sampled step")
                    weight = float(attribution["confidence"]) * (float(attribution["severity"]) / 3.0)
                    if weight <= 0 or magnitude <= 0:
                        continue
                    signed_advantage = magnitude if attribution["direction"] == "reinforce" else -magnitude
                    weighted_steps.append((step, signed_advantage, weight))
                    if attribution["direction"] == "reinforce":
                        reinforce_steps += 1
                    else:
                        weaken_steps += 1
        else:
            for episode, advantage in zip(episodes, advantages):
                weighted_steps.extend((step, advantage, 1.0) for step in episode["steps"])

        total_weight = sum(weight for _, _, weight in weighted_steps)
        base_result: dict[str, Any] = {
            "algorithm": "signed_masked_episode_grpo" if attribution_enabled else "episode_group_relative_ppo",
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": mean,
            "reward_std": scale,
            "steps": len(weighted_steps),
            "available_steps": all_step_count,
            "sampling_temperature": sampling_temperature,
            "attribution": {
                "enabled": attribution_enabled,
                "reinforce_steps": reinforce_steps,
                "weaken_steps": weaken_steps,
                "weight_sum": total_weight,
                "max_abs_advantage": max_abs_advantage,
            },
        }
        if not weighted_steps:
            return base_result | {
                "updated": False,
                "skip_reason": "zero_group_reward_std" if scale < 1e-8 else "no_effective_attributions",
                "losses": [],
            }

        self.model.train()
        self.model.config.use_cache = False
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats(self.device)
        losses: list[float] = []
        for _ in range(self.config.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for step, advantage, weight in weighted_steps:
                aliases, raw_logprobs = self.action_logprobs(
                    step["observation"],
                    step["legal_actions"],
                    use_cache=False,
                )
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
                loss = -weight * self.torch.minimum(unclipped, clipped) / total_weight
                loss.backward()
                epoch_loss += float(loss.detach())
                if self.config.empty_cache_between_steps and self.torch.cuda.is_available():
                    # Each action has an independently-sized observation.
                    # Returning freed blocks to CUDA prevents a long map
                    # prompt from fragmenting the later update steps.
                    self.torch.cuda.empty_cache()
            grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
            self.optimizer.step()
            losses.append(epoch_loss)
        self.model.eval()
        result = base_result | {"updated": True, "losses": losses, "grad_norm": grad_norm}
        if self.torch.cuda.is_available():
            result["peak_memory_mib"] = round(
                self.torch.cuda.max_memory_allocated(self.device) / (1024 * 1024),
                1,
            )
        return result

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path / "tokenizer")
