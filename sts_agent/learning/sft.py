"""Train the first alias-action QLoRA policy from a checked SFT bundle.

The trainer uses the exact selector prompt consumed by
``sts_agent.policy.local_qwen``:
an already-routed public observation plus the current dynamic alias table, with
one alias such as ``A3`` as the supervised completion. The controller remains
the sole authority for compiling aliases into game commands.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from sts_agent.runtime import controller as communication_controller
from sts_agent.policy import local_qwen as policy_server
from sts_agent.paths import PROJECT_ROOT


def configure_nccl_for_consumer_rtx() -> None:
    """Make Accelerate usable on consumer RTX 40-series multi-GPU hosts.

    Accelerate checks the physical GPU topology even when a caller exposes one
    device through CUDA_VISIBLE_DEVICES. RTX 4090 hosts need these NCCL paths
    disabled unless the launcher has already supplied an explicit choice.
    """

    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return {"commit": None, "dirty": None}
    if revision.returncode or status.returncode:
        return {"commit": None, "dirty": None}
    return {"commit": revision.stdout.strip() or None, "dirty": bool(status.stdout.strip())}


def discover_bundle_manifest(dataset_path: Path) -> Path | None:
    """Find the release manifest for ``<bundle>/dataset/teacher-sft.jsonl``."""

    candidate = dataset_path.parent.parent / "manifest.json"
    return candidate if candidate.is_file() else None


def validate_bundle(dataset_path: Path, manifest_path: Path | None) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"dataset manifest must be an object: {manifest_path}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"dataset manifest has no dataset section: {manifest_path}")
    expected = dataset.get("sha256")
    actual = sha256_file(dataset_path)
    if not isinstance(expected, str) or expected.lower() != actual:
        raise ValueError(
            f"dataset hash mismatch: manifest={expected!r} actual={actual}. "
            "Re-transfer the bundle and rerun sha256sum -c SHA256SUMS."
        )
    expected_samples = dataset.get("samples")
    if expected_samples is not None and (not isinstance(expected_samples, int) or expected_samples < 1):
        raise ValueError("dataset manifest has an invalid dataset.samples")
    return manifest


def validate_sample(sample: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    observation = sample.get("observation")
    legal_actions = sample.get("legal_actions")
    target = sample.get("target")
    run_id = sample.get("run_id")
    if not isinstance(observation, dict) or not isinstance(legal_actions, list) or not isinstance(target, dict):
        raise ValueError(f"invalid SFT sample structure: {sample.get('sample_id', '<unknown>')}")
    target_alias = target.get("alias")
    if not isinstance(target_alias, str) or not target_alias:
        raise ValueError(f"sample has no target alias: {sample.get('sample_id', '<unknown>')}")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"sample has no run_id: {sample.get('sample_id', '<unknown>')}")
    aliases = []
    legal_aliases = set()
    for action in legal_actions:
        if not isinstance(action, dict):
            continue
        alias = policy_server.action_id(action)
        if not alias or alias in legal_aliases:
            continue
        legal_aliases.add(alias)
        aliases.append({"id": alias, "label": policy_server.action_label(action)})
    if target_alias not in legal_aliases:
        raise ValueError(f"target alias is absent from legal actions: {sample.get('sample_id', '<unknown>')}")
    sample_id = str(sample.get("sample_id") or f"{run_id}:unknown")
    return observation, aliases, target_alias, run_id + "\x00" + sample_id


def route_trace_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Mirror live controller routing without mutating the durable trace."""

    return communication_controller.route_policy_observation(copy.deepcopy(observation))


def render_chat_prefix(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_sample(
    tokenizer: Any,
    sample: dict[str, Any],
    max_length: int,
    prompt_template: str = "canonical",
) -> tuple[dict[str, list[int]] | None, str]:
    """Encode one response-only Causal LM sample or return an explicit drop reason."""

    observation, legal_actions, target_alias, identity = validate_sample(sample)
    routed = route_trace_observation(observation)
    user_prompt = policy_server.build_scoring_prompt(routed, legal_actions, prompt_template)
    variants = sample.get("prompt_variants")
    if isinstance(variants, dict) and isinstance(variants.get(prompt_template), str):
        if variants[prompt_template] != user_prompt:
            raise ValueError(f"prompt variant differs from live renderer; rebuild the dataset: {identity}")
        user_prompt = variants[prompt_template]
    if prompt_template == "canonical" and sample.get("prompt") is not None and sample["prompt"] != user_prompt:
        raise ValueError(f"prompt differs from exported preview; rebuild the dataset: {identity}")
    prefix = render_chat_prefix(tokenizer, user_prompt)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_alias + "\n", add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = [*target_ids, int(tokenizer.eos_token_id)]
    if not target_ids:
        raise ValueError(f"empty target tokenization: {identity}")
    input_ids = [*prefix_ids, *target_ids]
    if len(input_ids) > max_length:
        # Do not truncate from either side: doing so can remove state fields or
        # legal aliases and create a prompt that is impossible at inference.
        return None, "overlong"
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prefix_ids) + target_ids,
    }, "kept"


def _seed_group(sample: dict[str, Any]) -> str:
    config = sample.get("run_config")
    config = config if isinstance(config, dict) else {}
    seed = next((config.get(key) for key in ("display_seed", "seed_display")
                 if config.get(key) not in (None, "")), None)
    if seed is None and config.get("internal_seed") not in (None, ""):
        seed = communication_controller.display_seed_from_internal(config["internal_seed"])
    if seed is None:
        seed = next((config.get(key) for key in ("seed", "game_seed")
                     if config.get(key) not in (None, "")), None)
    if seed is None or str(seed).strip().casefold() in {"random", "unknown", "none", "-"}:
        return "run:" + str(sample.get("run_id") or sample.get("sample_id") or "unknown")
    # ActionRecorder's legacy run metadata may contain the signed internal
    # Java long rather than the player's base-35 display seed. Canonicalize
    # those values so alternate encodings of one seed cannot cross the split.
    seed_text = str(seed).strip()
    if re.fullmatch(r"-?\d{15,}", seed_text):
        decoded = communication_controller.display_seed_from_internal(seed_text)
        if decoded is not None:
            seed_text = decoded
    seed_key = "".join(ch for ch in seed_text.upper() if ch.isalnum())
    return f"seed:{seed_key}"


def split_by_run(samples: list[dict[str, Any]], validation_runs: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Split complete seed families, falling back to run IDs if seed is absent."""
    run_ids = sorted({str(sample.get("run_id") or "") for sample in samples if sample.get("run_id")})
    if not run_ids:
        raise ValueError("at least one complete run is required")
    if validation_runs == 0:
        return samples, [], []
    groups = {_seed_group(sample) for sample in samples}
    if len(groups) < 2:
        raise ValueError("at least two independent seed groups are required for validation")
    if not 1 <= validation_runs < len(groups):
        raise ValueError(f"validation seed groups must be from 1 to {len(groups) - 1}, got {validation_runs}")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    validation_groups = set(shuffled[:validation_runs])
    train = [sample for sample in samples if _seed_group(sample) not in validation_groups]
    validation = [sample for sample in samples if _seed_group(sample) in validation_groups]
    validation_run_ids = sorted({str(sample.get("run_id")) for sample in validation})
    return train, validation, validation_run_ids


class EncodedDataset:
    def __init__(self, items: list[list[dict[str, list[int]]]], sample_ids: list[str] | None = None) -> None:
        self.items = items
        self.epoch = 0
        self.offsets = [
            int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % max(1, len(variants))
            for value, variants in zip(sample_ids or [str(i) for i in range(len(items))], items)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        variants = self.items[index]
        if len(variants) == 1:
            return variants[0]
        return variants[(self.epoch + self.offsets[index]) % len(variants)]


@dataclass
class ResponseOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        maximum = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            padding = maximum - len(item["input_ids"])
            input_ids.append([*item["input_ids"], *([self.pad_token_id] * padding)])
            attention_mask.append([*item["attention_mask"], *([0] * padding)])
            labels.append([*item["labels"], *([-100] * padding)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def encode_split(
    tokenizer: Any,
    samples: list[dict[str, Any]],
    max_length: int,
    prompt_templates: tuple[str, ...] = ("canonical",),
) -> tuple[EncodedDataset, Counter[str]]:
    encoded: list[list[dict[str, list[int]]]] = []
    encoded_ids: list[str] = []
    stats: Counter[str] = Counter()
    for sample in samples:
        choices: list[dict[str, list[int]]] = []
        failed: list[str] = []
        for template in prompt_templates:
            item, status = encode_sample(tokenizer, sample, max_length, template)
            if item is None:
                failed.append(f"{template}:{status}")
            else:
                choices.append(item)
        if failed:
            stats["overlong_template_group"] += 1
            continue
        stats["kept"] += 1
        encoded.append(choices)
        encoded_ids.append(str(sample.get("sample_id") or sample.get("run_id") or len(encoded_ids)))
    if not encoded:
        raise ValueError("no samples remain after length filtering")
    return EncodedDataset(encoded, encoded_ids), stats


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def summarize_epoch_losses(log_history: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Aggregate Trainer fixed-step training-loss windows by epoch.

    Each loss is the mean of a logging interval. The result is a train-only
    diagnostic, not a replacement for held-out gameplay evaluation.
    """

    windows: dict[int, list[float]] = {}
    for item in log_history:
        if not isinstance(item, dict) or "loss" not in item or "epoch" not in item:
            continue
        try:
            epoch_position = float(item["epoch"])
            loss = float(item["loss"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(epoch_position) or not math.isfinite(loss):
            continue
        # A loss logged at exactly epoch 1.0 belongs to the first epoch.
        epoch = max(1, math.ceil(epoch_position))
        windows.setdefault(epoch, []).append(loss)
    return [
        {
            "epoch": epoch,
            "logged_windows": len(values),
            "mean_logged_loss": sum(values) / len(values),
            "last_logged_loss": values[-1],
        }
        for epoch, values in sorted(windows.items())
    ]


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Train a response-only QLoRA StS action selector")
    parser.add_argument("--dataset", type=Path, required=True, help="teacher-sft.jsonl from a verified transfer bundle")
    parser.add_argument("--manifest", type=Path, help="bundle manifest.json; autodetected beside dataset when omitted")
    parser.add_argument("--output", type=Path, required=True, help="new output directory for the LoRA adapter and run manifest")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Hub ID or local base-model directory")
    parser.add_argument("--model-revision", default="main", help="recorded Hugging Face revision when --model is a Hub ID")
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="completed LoRA adapter to continue from; starts a new run with fresh optimizer state",
    )
    parser.add_argument("--max-length", type=int, default=4096, help="drop samples above this full prompt length; never truncate them")
    parser.add_argument(
        "--prompt-templates",
        default="canonical,state_first,decision_first",
        help="comma-separated equivalent templates rotated per sample and epoch",
    )
    parser.add_argument(
        "--validation-seed-groups", "--validation-runs",
        dest="validation_runs",
        type=int,
        default=3,
        help="number of whole seed groups held out for validation; set 0 to train on every run",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--max-train-samples", type=int, help="bounded smoke test before a full run")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise ValueError(f"dataset does not exist: {args.dataset}")
    if args.manifest is not None and not args.manifest.is_file():
        raise ValueError(f"manifest does not exist: {args.manifest}")
    if args.output.exists() and any(args.output.iterdir()) and args.resume_from_checkpoint is None:
        raise ValueError(f"output directory is non-empty: {args.output}; choose a new run name or pass --resume-from-checkpoint")
    if args.initial_adapter is not None:
        if not args.initial_adapter.is_dir():
            raise ValueError(f"initial_adapter does not exist: {args.initial_adapter}")
        if not (args.initial_adapter / "adapter_config.json").is_file():
            raise ValueError(f"initial_adapter is not a completed PEFT adapter: {args.initial_adapter}")
    if args.initial_adapter is not None and args.resume_from_checkpoint is not None:
        raise ValueError("initial_adapter and resume_from_checkpoint are mutually exclusive")
    if args.max_length < 128:
        raise ValueError("max_length must be at least 128")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if args.validation_runs < 0:
        raise ValueError("validation_runs must be zero or positive")
    for name in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "logging_steps",
        "save_total_limit",
        "dataloader_workers",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")


def training_arguments(training_args_cls: Any, args: argparse.Namespace, bf16: bool, has_evaluation: bool) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output),
        "run_name": args.output.name,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "group_by_length": True,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "bf16": bf16,
        "fp16": not bf16,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.dataloader_workers,
        "report_to": [],
        "remove_unused_columns": False,
    }
    signature = inspect.signature(training_args_cls).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch" if has_evaluation else "no"
    else:
        kwargs["evaluation_strategy"] = "epoch" if has_evaluation else "no"
    if "gradient_checkpointing_kwargs" in signature:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return training_args_cls(**kwargs)


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    manifest_path = args.manifest or discover_bundle_manifest(args.dataset)
    bundle_manifest = validate_bundle(args.dataset, manifest_path)
    samples = list(read_jsonl(args.dataset))
    if bundle_manifest is not None:
        expected = bundle_manifest.get("dataset", {}).get("samples")
        if isinstance(expected, int) and expected != len(samples):
            raise ValueError(f"dataset sample count mismatch: manifest={expected} actual={len(samples)}")
    prompt_templates = tuple(item.strip() for item in args.prompt_templates.split(",") if item.strip())
    if not prompt_templates or any(item not in policy_server.PROMPT_TEMPLATES for item in prompt_templates):
        raise ValueError(f"prompt_templates must be selected from {policy_server.PROMPT_TEMPLATES}")
    if len(set(prompt_templates)) != len(prompt_templates):
        raise ValueError("prompt_templates must not contain duplicates")
    train_samples, eval_samples, validation_run_ids = split_by_run(samples, args.validation_runs, args.seed)
    if args.max_train_samples is not None:
        if args.max_train_samples < 1:
            raise ValueError("max_train_samples must be positive")
        train_samples = train_samples[: args.max_train_samples]

    configure_nccl_for_consumer_rtx()
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("install requirements/train.txt and a CUDA PyTorch build before training") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this trainer requires an NVIDIA GPU")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device is visible")
    bf16 = bool(torch.cuda.is_bf16_supported())
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset, train_encode_stats = encode_split(
        tokenizer, train_samples, args.max_length, prompt_templates=prompt_templates
    )
    eval_dataset: EncodedDataset | None = None
    eval_encode_stats: Counter[str] = Counter()
    if eval_samples:
        eval_dataset, eval_encode_stats = encode_split(tokenizer, eval_samples, args.max_length)

    model_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.initial_adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.initial_adapter), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    resolved_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") if isinstance(tokenizer.init_kwargs, dict) else None

    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest_path = args.output / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "schema_version": "sts-sft-run-0.1",
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "repository": git_metadata(PROJECT_ROOT),
        "bundle": {
            "manifest_path": str(manifest_path) if manifest_path else None,
            "dataset_path": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "bundle_manifest": bundle_manifest,
        },
        "model": {
            "id": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_model_revision,
            "resolved_tokenizer_revision": resolved_tokenizer_revision,
            "load_in_4bit": args.load_in_4bit,
            "bf16": bf16,
        },
        "initial_adapter": str(args.initial_adapter) if args.initial_adapter is not None else None,
        "arguments": serializable_args(args),
        "split": {
            "unit": "character_ascension_seed_group",
            "train_runs": len({str(sample["run_id"]) for sample in train_samples}),
            "validation_runs": validation_run_ids,
            "train_seed_groups": sorted({_seed_group(sample) for sample in train_samples}),
            "validation_seed_groups": sorted({_seed_group(sample) for sample in eval_samples}),
            "prompt_templates": list(prompt_templates),
            "raw_train_samples": len(train_samples),
            "raw_validation_samples": len(eval_samples),
            "encoded_train_samples": len(train_dataset),
            "encoded_validation_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "train_encoding": dict(train_encode_stats),
            "validation_encoding": dict(eval_encode_stats),
        },
    }
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    class _PromptTemplateEpochCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            train_dataset.set_epoch(int(state.epoch or 0))
            return control

    trainer = Trainer(
        model=model,
        args=training_arguments(TrainingArguments, args, bf16, has_evaluation=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ResponseOnlyCollator(tokenizer.pad_token_id),
        callbacks=[_PromptTemplateEpochCallback()],
    )
    train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    metrics = dict(train_result.metrics)
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output / "tokenizer")
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    epoch_loss_summary = {
        "schema_version": "sts-sft-epoch-loss-0.1",
        "kind": "mean_of_trainer_logging_windows",
        "logging_steps": args.logging_steps,
        "warning": "train-only diagnostic; do not select a policy from this without gameplay evaluation",
        "epochs": summarize_epoch_losses(list(trainer.state.log_history)),
    }
    (args.output / "epoch_loss_summary.json").write_text(
        json.dumps(epoch_loss_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_manifest["status"] = "completed"
    run_manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    run_manifest["metrics"] = metrics
    run_manifest["epoch_loss_summary"] = epoch_loss_summary
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    try:
        return train(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
