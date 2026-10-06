"""Remote-only preparation and launch of the official BF16 baseline.

Run manually on the Linux GPU host, from a separate serving environment.
Neither this entry point nor the client changes remote Git files or trains.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from sts_agent.config import load_agent_config
from sts_agent.paths import PROJECT_ROOT
from sts_agent.policy.gpu import query_gpus


def select_devices(candidates: list[dict], settings: dict) -> list[int]:
    eligible = [g for g in candidates if g["free_memory_mib"] >= settings["minimum_free_memory_mib"]
                and g["utilization_percent"] <= settings["maximum_gpu_utilization_percent"]]
    requested = settings["devices"]
    if requested == "auto":
        selected = sorted(eligible, key=lambda g: (g["utilization_percent"], -g["free_memory_mib"], g["index"]))[:settings["tensor_parallel_size"]]
        devices = [g["index"] for g in selected]
    else:
        devices = list(requested)
        if not set(devices).issubset({g["index"] for g in eligible}):
            raise RuntimeError("requested GPUs are missing, busy, or below the free-memory threshold")
    if len(devices) != settings["tensor_parallel_size"]:
        raise RuntimeError(f"need {settings['tensor_parallel_size']} idle GPUs; availability: {candidates}")
    return devices


def launch_command(model_path: str, policy: dict) -> list[str]:
    api, serving = policy["api"], policy["api"]["serving"]
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
           "--served-model-name", policy["model"]["base"], "--host", "127.0.0.1", "--port", str(api["remote_port"]),
           "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--tensor-parallel-size", str(serving["tensor_parallel_size"]),
           "--max-model-len", str(serving["max_model_len"]), "--max-num-seqs", "1",
           "--gpu-memory-utilization", str(serving["gpu_memory_utilization"]),
           "--language-model-only", "--reasoning-parser", "qwen3", "--generation-config", "vllm",
           "--enable-prefix-caching", "--disable-custom-all-reduce"]
    if serving["enforce_eager"]:
        cmd.append("--enforce-eager")
    return cmd


def read_deployment(path: Path, policy: dict) -> dict:
    if not path.exists():
        raise RuntimeError("model has not been prepared; run this command with --prepare first")
    deployment = json.loads(path.read_text(encoding="utf-8"))
    model = policy["model"]
    if deployment.get("model") != model["base"] or deployment.get("requested_revision") != model["revision"]:
        raise RuntimeError("deployment does not match configured model/revision; prepare it explicitly again")
    if not re.fullmatch(r"[0-9a-f]{40}", deployment.get("resolved_revision", "")):
        raise RuntimeError("deployment must pin an exact Hugging Face commit SHA")
    snapshot = Path(deployment["snapshot_path"])
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise RuntimeError("prepared model snapshot is missing")
    if snapshot.name != deployment["resolved_revision"]:
        raise RuntimeError("snapshot directory does not match the pinned revision")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if model_config.get("quantization_config"):
        raise RuntimeError("the BF16 baseline must not use a pre-quantized checkpoint")
    return deployment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/agent.json")
    parser.add_argument("--prepare", action="store_true", help="explicitly download and pin the official checkpoint; do not start service")
    parser.add_argument("--dry-run", action="store_true", help="validate snapshot/GPUs and print launch command without allocating CUDA memory")
    args = parser.parse_args()
    config = load_agent_config(config_file=args.config, mode="baseline_inference")
    policy = config["policy"]
    model = policy["model"]
    settings = policy["api"]["serving"]
    deployment_path = PROJECT_ROOT / settings["deployment_file"]
    try:
        if args.prepare:
            from huggingface_hub import HfApi, snapshot_download

            info = HfApi(endpoint=model["hub_endpoint"]).model_info(model["base"], revision=model["revision"])
            if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
                raise RuntimeError("Hugging Face did not return an exact checkpoint revision")
            snapshot = snapshot_download(repo_id=model["base"], revision=info.sha,
                                         endpoint=model["hub_endpoint"], cache_dir=str(PROJECT_ROOT / "models" / "hub"))
            deployment = {"model": model["base"], "requested_revision": model["revision"],
                          "resolved_revision": info.sha, "snapshot_path": snapshot,
                          "prepared_at": datetime.now(timezone.utc).isoformat()}
            deployment_path.parent.mkdir(parents=True, exist_ok=True)
            deployment_path.write_text(json.dumps(deployment, indent=2), encoding="utf-8")
            print(json.dumps(deployment, indent=2))
            return 0
        deployment = read_deployment(deployment_path, policy)
        devices = select_devices(query_gpus(), settings)
        command = launch_command(deployment["snapshot_path"], policy)
        versions = {name: importlib.metadata.version(name) for name in ("vllm", "transformers", "torch", "huggingface-hub")}
        if tuple(int(x) for x in versions["transformers"].split(".")[:2]) < (5, 8):
            raise RuntimeError("Qwen3.8 serving requires transformers >= 5.8; update the separate serving environment")
        print(f"Selected physical GPUs (fixed for service lifetime): {devices}", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        if os.name == "nt":
            raise RuntimeError("start the vLLM service on the Linux GPU host, not Windows")
        deployment.update({"devices": devices, "command": command, "versions": versions, "serving": settings,
                           "started_at": datetime.now(timezone.utc).isoformat()})
        deployment_path.write_text(json.dumps(deployment, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        if settings["disable_nccl_p2p"]:
            env["NCCL_P2P_DISABLE"] = "1"
        if settings["disable_nccl_ib"]:
            env["NCCL_IB_DISABLE"] = "1"
        os.execvpe(command[0], command, env)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"baseline serving failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
