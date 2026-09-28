"""Select a physical GPU before importing CUDA-aware libraries."""

from __future__ import annotations

import subprocess
from typing import Any


def validate_gpu_selector(device: Any, minimum_free_memory_mib: Any, *, name: str = "gpu") -> None:
    if device != "auto" and (isinstance(device, bool) or not isinstance(device, int) or device < 0):
        raise ValueError(f"{name}.device must be 'auto' or a non-negative integer")
    if (
        isinstance(minimum_free_memory_mib, bool)
        or not isinstance(minimum_free_memory_mib, int)
        or minimum_free_memory_mib < 1
    ):
        raise ValueError(f"{name}.minimum_free_memory_mib must be a positive integer")


def query_gpus() -> list[dict[str, int]]:
    """Read physical GPU availability without initializing CUDA."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot query GPU availability with nvidia-smi: {exc}") from exc
    candidates: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            candidates.append(
                {
                    "index": int(fields[0]),
                    "free_memory_mib": int(fields[1]),
                    "utilization_percent": int(fields[2]),
                }
            )
        except ValueError:
            continue
    if not candidates:
        raise RuntimeError("nvidia-smi returned no parseable GPU availability rows")
    return candidates


def select_gpu(candidates: list[dict[str, int]], minimum_free_memory_mib: int) -> dict[str, int]:
    """Choose the least-utilized GPU that has enough free VRAM."""

    eligible = [item for item in candidates if item["free_memory_mib"] >= minimum_free_memory_mib]
    if not eligible:
        summary = ", ".join(
            f"GPU {item['index']}: {item['free_memory_mib']} MiB free, {item['utilization_percent']}% util"
            for item in candidates
        )
        raise RuntimeError(
            f"no GPU has at least {minimum_free_memory_mib} MiB free; observed: {summary or 'none'}"
        )
    return min(eligible, key=lambda item: (item["utilization_percent"], -item["free_memory_mib"], item["index"]))


def select_configured_gpu(device: str | int, minimum_free_memory_mib: int) -> dict[str, int]:
    """Select automatically or validate a requested physical GPU."""

    candidates = query_gpus()
    if device == "auto":
        return select_gpu(candidates, minimum_free_memory_mib)
    selected = next((item for item in candidates if item["index"] == device), None)
    if selected is None:
        raise RuntimeError(f"configured GPU {device} does not exist")
    if selected["free_memory_mib"] < minimum_free_memory_mib:
        raise RuntimeError(
            f"configured GPU {device} has only {selected['free_memory_mib']} MiB free; "
            f"requires at least {minimum_free_memory_mib} MiB"
        )
    return selected
