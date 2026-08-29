"""Local Qwen action ranker for Candidate Assist.

The server reads one JSON request per line from stdin and writes one JSON
response per line to stdout.  It never talks to CommunicationMod directly;
the controller remains responsible for legal-action validation and command
compilation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from agent_config import ConfigError, load_agent_config
from gpu_selection import select_configured_gpu


def configure_jsonl_stdio() -> None:
    """Keep JSONL protocol Unicode-safe on Windows pipes."""
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")


def parse_gpu_device(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gpu device must be 'auto' or a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("gpu device must be 'auto' or a non-negative integer")
    return result


def configure_inference_gpu(device: str | int, minimum_free_memory_mib: int) -> dict[str, int] | None:
    """Restrict model loading to one selected GPU before importing Torch."""

    if device == "visible":
        return None
    if minimum_free_memory_mib < 1:
        raise ValueError("minimum_free_memory_mib must be positive")
    selected = select_configured_gpu(device, minimum_free_memory_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    return selected


def compact_json(value: Any, limit: int = 7000) -> str:
    """Serialize observations compactly and bound prompt size."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= limit:
        return text
    # The normalizer already removes hidden fields. Keeping a bounded suffix is
    # preferable to crashing on an unusually verbose event description.
    return text[:limit] + "...<truncated>"


def action_id(action: dict[str, Any]) -> str:
    return str(action.get("id") or action.get("action_id") or "")


def action_label(action: dict[str, Any]) -> str:
    label = action.get("label")
    if label:
        return str(label)
    kind = action.get("kind")
    command = action.get("command")
    return str(kind or command or "")


def action_prompt_lines(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[str]:
    # Route selection is infrequent, and needs the entire visible graph. Other
    # screens stay within the tighter recurring-turn prompt budget.
    observation_limit = 12000 if observation.get("screen_type") == "MAP" else 7000
    lines = [
        "Observation:",
        compact_json(observation, limit=observation_limit),
        "Recent accepted actions are context only; the current screen and legal actions are authoritative.",
        "Legal actions:",
    ]
    for action in legal_actions:
        lines.append(f"- {action_id(action)} | {action_label(action)}")
    return lines


def build_prompt(observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> str:
    lines = [
        "You are a Slay the Spire action ranker.",
        "Rank only the supplied legal action IDs. The IDs are short tokens such as A1 and A2; copy them exactly.",
        "Prefer survival and a strong long-term win chance.",
        f"Return exactly one JSON object with at most {top_k} action IDs in descending preference.",
        'Required format: {"ranked_action_ids":["ACTION_ID_1","ACTION_ID_2"]}',
        "Do not include prose, markdown, commands, or action IDs outside the supplied list.",
    ]
    lines.extend(action_prompt_lines(observation, legal_actions))
    lines.append("JSON:")
    return "\n".join(lines)


def build_scoring_prompt(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> str:
    """Build a prompt whose next completion is exactly one action alias."""
    lines = [
        "You are a Slay the Spire action selector.",
        "Choose exactly one supplied legal action ID that maximizes survival and long-term win chance.",
        "Output only that exact action ID, followed by a newline. Do not output JSON, prose, or a command.",
    ]
    lines.extend(action_prompt_lines(observation, legal_actions))
    # Keep the continuation on the next line so aliases such as A1 and A10
    # are scored with an explicit newline boundary rather than prefix matching.
    lines.append("Selected action ID:")
    return "\n".join(lines) + "\n"


def ranked_ids_from_output(text: str, legal_actions: list[dict[str, Any]], top_k: int) -> list[str]:
    """Parse generated JSON and defensively keep only current legal IDs."""
    legal_ids = [action_id(action) for action in legal_actions]
    legal_set = set(legal_ids)
    generated: list[str] = []
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            items = value.get("ranked_action_ids", []) if isinstance(value, dict) else []
            if isinstance(items, list):
                generated = [str(item) for item in items]
        except json.JSONDecodeError:
            pass
    # A small fallback makes imperfect JSON harmless while still requiring an
    # exact legal ID; sort by occurrence to preserve model preference order.
    if not generated:
        positions = []
        for candidate in legal_ids:
            match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])", text)
            if match:
                positions.append((match.start(), candidate))
        generated = [candidate for _, candidate in sorted(positions)]
    result: list[str] = []
    for candidate in generated:
        canonical_id = candidate
        if canonical_id not in legal_set:
            # Small models often omit the target suffix for a single-target
            # card (for example, "play:0" instead of "play:0:0"). Accept
            # that shorthand only when it resolves to exactly one legal ID.
            matching_ids = [legal_id for legal_id in legal_ids if legal_id.startswith(f"{candidate}:")]
            if len(matching_ids) == 1:
                canonical_id = matching_ids[0]
        if canonical_id in legal_set and canonical_id not in result:
            result.append(canonical_id)
        if len(result) >= top_k:
            break
    return result


class MockRanker:
    """Deterministic ranker used to verify controller/terminal integration."""

    def rank(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        del observation
        # Keep END_TURN as a useful fallback, but prefer concrete actions in
        # the mock so Candidate Assist can be tested without model downloads.
        ordered = sorted(
            legal_actions,
            key=lambda action: (action_id(action) == "END_TURN", action_id(action)),
        )
        result = []
        for rank, action in enumerate(ordered[:top_k], start=1):
            result.append(
                {
                    "rank": rank,
                    "action_id": action_id(action),
                    "label": action_label(action),
                    "score": None,
                }
            )
        return result

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return deterministic but non-uniform values for protocol tests."""
        ordered = self.rank(observation, legal_actions, len(legal_actions))
        # These are unnormalized log weights. The controller applies the only
        # normalization that matters: over the current legal action set.
        return [
            {
                "action_id": item["action_id"],
                "label": item["label"],
                "logprob": -2.0 * rank,
            }
            for rank, item in enumerate(ordered)
        ]


class QwenRanker:
    def __init__(self, model_name: str, load_in_4bit: bool, local_files_only: bool, adapter: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_source = model_name
        if local_files_only and not Path(model_name).is_dir():
            # Passing a Hub model ID to recent Transformers can still trigger
            # metadata requests during tokenizer setup, despite
            # local_files_only=True. Resolve it to the cached snapshot first
            # so game-time inference is strictly offline.
            from huggingface_hub import snapshot_download

            model_source = snapshot_download(repo_id=model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device.type == "cuda":
            model_kwargs["dtype"] = torch.float16
            if load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["device_map"] = "auto"
                except (ImportError, Exception) as exc:
                    raise RuntimeError(
                        "4-bit loading is required for the local 4GB GPU; install a CUDA-compatible bitsandbytes build "
                        "or start with --no-load-in-4bit only on a machine with enough VRAM."
                    ) from exc
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=local_files_only,
            **model_kwargs,
        )
        if not model_kwargs.get("device_map"):
            self.model.to(self.device)
        if adapter:
            adapter_path = Path(adapter)
            if not adapter_path.is_dir():
                raise RuntimeError(f"LoRA adapter directory does not exist: {adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("install peft to load a LoRA adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), is_trainable=False)
        self.model.eval()
        # Qwen ships sampling defaults. They are irrelevant for deterministic
        # candidate ranking and otherwise cause a warning on every turn.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model_device = next(self.model.parameters()).device

    def rank(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not legal_actions:
            return []
        user_prompt = build_prompt(observation, legal_actions, top_k)
        tokenizer = self.tokenizer
        try:
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older Transformers versions do not expose Qwen3's
            # enable_thinking option, but still provide the chat template.
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model_device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                do_sample=False,
                # The response is a compact JSON list of A1/A2-style IDs.
                # A short cap prevents a malformed generation from holding up
                # the terminal on the 4GB local GPU.
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_length = encoded["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True).strip()
        ranked_ids = ranked_ids_from_output(generated, legal_actions, top_k)
        actions_by_id = {action_id(action): action for action in legal_actions}
        result = []
        for rank, candidate_id in enumerate(ranked_ids, start=1):
            action = actions_by_id[candidate_id]
            result.append(
                {
                    "rank": rank,
                    "action_id": action_id(action),
                    "label": action_label(action),
                    "score": None,
                }
            )
        return result

    def _chat_prefix(self, user_prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score every exact legal alias with one shared prompt forward pass.

        Scores are the log probability of ``A<n>`` followed by a newline. The
        newline is important: it prevents A1 from being treated as a prefix of
        A10. The controller later normalizes these scores only over legal
        actions, which is the desired masked policy distribution.
        """
        if not legal_actions:
            return []
        tokenizer = self.tokenizer
        prefix = self._chat_prefix(build_scoring_prompt(observation, legal_actions))
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prefix_ids = prefix_ids.to(self.model_device)
        prefix_length = prefix_ids.shape[1]

        continuation_ids: dict[str, list[int]] = {}
        for action in legal_actions:
            alias = action_id(action)
            combined_ids = tokenizer(prefix + alias + "\n", add_special_tokens=False)["input_ids"]
            prefix_as_list = prefix_ids[0].tolist()
            if combined_ids[:prefix_length] != prefix_as_list:
                raise RuntimeError("tokenizer changed the scoring prompt boundary")
            suffix = combined_ids[prefix_length:]
            if not suffix:
                raise RuntimeError(f"empty token continuation for action {alias}")
            continuation_ids[alias] = suffix

        with self.torch.inference_mode():
            prefix_output = self.model(input_ids=prefix_ids, use_cache=True)
            first_logprobs = self.torch.log_softmax(prefix_output.logits[0, -1].float(), dim=-1)
            result: list[dict[str, Any]] = []
            for action in legal_actions:
                alias = action_id(action)
                suffix = continuation_ids[alias]
                logprob = float(first_logprobs[suffix[0]].item())
                if len(suffix) > 1:
                    continuation = self.torch.tensor([suffix[:-1]], device=self.model_device)
                    continuation_output = self.model(
                        input_ids=continuation,
                        past_key_values=prefix_output.past_key_values,
                        use_cache=False,
                    )
                    next_logprobs = self.torch.log_softmax(continuation_output.logits[0].float(), dim=-1)
                    for position, token_id in enumerate(suffix[1:]):
                        logprob += float(next_logprobs[position, token_id].item())
                result.append(
                    {
                        "action_id": alias,
                        "label": action_label(action),
                        "logprob": logprob,
                    }
                )
        return result


def serve(args: argparse.Namespace) -> int:
    configure_jsonl_stdio()
    # Hugging Face libraries read this process-local setting. The user-facing
    # source of truth remains config/agent.json or an explicit CLI override.
    os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        selected_gpu = configure_inference_gpu(args.gpu_device, args.minimum_free_memory_mib)
    except Exception as exc:
        print(f"policy server GPU selection failed: {exc}", file=sys.stderr, flush=True)
        return 2
    ranker: MockRanker | QwenRanker
    if args.mock:
        ranker = MockRanker()
        model_id = "mock-candidate-ranker"
    else:
        try:
            ranker = QwenRanker(args.model, args.load_in_4bit, args.local_files_only, args.adapter or None)
            model_id = args.model if not args.adapter else f"{args.model}+{Path(args.adapter).name}"
        except Exception as exc:
            print(f"policy server failed to load {args.model}: {exc}", file=sys.stderr, flush=True)
            if args.debug:
                traceback.print_exc(file=sys.stderr)
            return 2

    print(
        json.dumps(
            {"type": "ready", "model_id": model_id, "endpoint": args.endpoint, "gpu": selected_gpu},
            ensure_ascii=False,
        ),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Any = None
        try:
            request = json.loads(line)
            if request.get("type") == "shutdown":
                break
            request_type = request.get("type")
            if request_type not in {"rank", "score"}:
                raise ValueError("expected request type rank or score")
            observation = request.get("observation")
            actions = request.get("legal_actions")
            if not isinstance(observation, dict) or not isinstance(actions, list):
                raise ValueError("observation and legal_actions are required")
            legal_actions = [item for item in actions if isinstance(item, dict) and action_id(item)]
            top_k = max(1, min(int(request.get("top_k", args.top_k)), len(legal_actions) or 1))
            if request_type == "rank":
                candidates = ranker.rank(observation, legal_actions, top_k)
                response = {
                    "type": "ranking",
                    "request_id": request.get("request_id"),
                    "model_id": model_id,
                    "candidates": candidates,
                }
            else:
                response = {
                    "type": "scoring",
                    "request_id": request.get("request_id"),
                    "model_id": model_id,
                    "action_scores": ranker.score(observation, legal_actions),
                }
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"type": "error", "request_id": request.get("request_id") if isinstance(request, dict) else None, "error": str(exc)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.debug:
                traceback.print_exc(file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    try:
        policy = load_agent_config()["policy"]
    except ConfigError as exc:
        raise SystemExit(f"invalid agent configuration: {exc}") from exc
    parser = argparse.ArgumentParser(description="Qwen ranker for StS Candidate Assist")
    parser.add_argument("--model", default=policy["model"]["base"])
    parser.add_argument(
        "--adapter",
        default=policy["model"]["initial_adapter"],
        help="local LoRA adapter directory to attach to the base model",
    )
    parser.add_argument("--endpoint", default=policy["model"]["hub_endpoint"])
    parser.add_argument("--top-k", type=int, default=policy["request"]["top_k"])
    parser.add_argument(
        "--gpu-device",
        type=parse_gpu_device,
        default="visible",
        help="auto or physical GPU index; visible leaves CUDA_VISIBLE_DEVICES unchanged",
    )
    parser.add_argument("--minimum-free-memory-mib", type=int, default=1)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认只从本地 Hugging Face 缓存加载；请先运行 download_qwen.py",
    )
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=policy["development"]["mock"],
        help="不加载模型，用于测试协议和终端交互",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=policy["development"]["debug"],
        help="将模型服务异常完整写入 stderr",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(serve(parse_args()))
