"""Pre-download a Qwen policy model from Hugging Face for Candidate Assist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent_config import ConfigError, load_agent_config


def configure_windows_system_proxy() -> str | None:
    """Expose the active Windows proxy to Hugging Face Python clients.

    Requests-based libraries do not consistently consume WinINET proxy
    settings.  Clash Verge writes its HTTP proxy to this registry location,
    so use it only when the caller has not explicitly supplied proxy vars.
    """
    if os.name != "nt" or any(os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")):
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
    except OSError:
        return None
    if not enabled or not isinstance(proxy_server, str) or not proxy_server.strip():
        return None
    entries = {}
    for item in proxy_server.split(";"):
        if "=" in item:
            protocol, value = item.split("=", 1)
            entries[protocol.strip().lower()] = value.strip()
        else:
            entries.setdefault("default", item.strip())
    value = entries.get("https") or entries.get("http") or entries.get("default")
    if not value:
        return None
    proxy_url = value if "://" in value else f"http://{value}"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    return proxy_url


def parse_args() -> argparse.Namespace:
    try:
        policy = load_agent_config()["policy"]
    except ConfigError as exc:
        raise SystemExit(f"invalid agent configuration: {exc}") from exc
    parser = argparse.ArgumentParser(description="从 Hugging Face 官方预下载 Candidate Assist 使用的 Qwen 模型")
    parser.add_argument("--model", default=policy["model"])
    parser.add_argument("--endpoint", default=policy["endpoint"])
    parser.add_argument("--force", action="store_true", help="重新检查并下载缺失/更新文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Must be set before importing huggingface_hub so every Hub request uses
    # the requested endpoint.
    os.environ["HF_ENDPOINT"] = args.endpoint
    # hf_xet currently does not reliably inherit Windows system proxy settings.
    # Standard HTTPS downloads do, which is more dependable with Clash.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    proxy_url = configure_windows_system_proxy()
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model} from {args.endpoint}", flush=True)
    if proxy_url:
        print(f"Using Windows system proxy: {proxy_url}", flush=True)
    path = snapshot_download(
        repo_id=args.model,
        resume_download=True,
        force_download=args.force,
    )
    weight_file = Path(path) / "model.safetensors"
    if not weight_file.is_file():
        raise RuntimeError(
            f"model snapshot is missing {weight_file.name}: {path}. "
            "Run this command again with --force after checking the network connection."
        )
    print(f"Model ready in cache: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
