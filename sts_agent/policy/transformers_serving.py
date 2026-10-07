"""Pinned, text-only baseline API using Transformers' own generation handler.

The upstream CLI does not expose the snapshot SHA or /tokenize needed by the
game client. This adapter adds those checks; it does not implement generation.
Heavy dependencies are imported only on the remote serving host. Private HF
serving interfaces are deliberately pinned to Transformers 5.17.0.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import time
import uuid
from contextlib import asynccontextmanager

from sts_agent.config import load_agent_config
from sts_agent.paths import PROJECT_ROOT
from sts_agent.policy.serving import read_deployment


def validate_request(body: dict, model: str, thinking: bool, *, completion: bool = False) -> None:
    """Reject template changes/truncation and inputs outside this text-only API."""
    if body.get("model") != model:
        raise ValueError("request model does not match the fixed baseline")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
        or not isinstance(m.get("content"), str) or set(m) - {"role", "content"}
        for m in messages
    ):
        raise ValueError("messages must contain text-only system/user/assistant entries")
    if body.get("chat_template_kwargs", {}) not in ({}, {"enable_thinking": thinking}):
        raise ValueError("chat_template_kwargs must match configured enable_thinking; no truncation")
    allowed = {"model", "messages", "chat_template_kwargs"}
    if completion:
        allowed.update({"stream", "temperature", "max_tokens"})
        if body.get("stream", False) is not False:
            raise ValueError("baseline API supports non-streaming requests only")
    else:
        allowed.add("add_generation_prompt")
        if body.get("add_generation_prompt", True) is not True:
            raise ValueError("tokenize must include the generation prompt")
    if set(body) - allowed:
        raise ValueError(f"unsupported request fields: {sorted(set(body) - allowed)}")


def check_budget(count: int, output_tokens: int, maximum: int) -> None:
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if count + output_tokens > maximum:
        raise ValueError(f"context overflow: {count} input + {output_tokens} output > {maximum}; no truncation")


def create_app(policy: dict, deployment: dict):
    if importlib.metadata.version("transformers") != "5.17.0":
        raise RuntimeError("the Transformers serving adapter requires transformers==5.17.0")
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from transformers.cli.serving.chat_completion import ChatCompletionHandler
    from transformers.cli.serving.model_manager import ModelManager
    from transformers.cli.serving.utils import GenerationState

    api = policy["api"]
    public_model = policy["model"]["base"]
    snapshot = deployment["snapshot_path"]
    maximum = api["serving"]["max_model_len"]
    manager = ModelManager(device="auto", dtype="bfloat16", trust_remote_code=False,
                           attn_implementation="sdpa", model_timeout=-1, force_model=snapshot)
    state = GenerationState(continuous_batching=False, compile=False)
    handler = ChatCompletionHandler(model_manager=manager, generation_state=state,
                                    chat_template_kwargs={"enable_thinking": api["thinking"]})
    model, processor = manager.load_model_and_processor(manager.process_model_name(snapshot))
    modality = manager.get_model_modality(model, processor=processor)
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            state.shutdown()
            manager.shutdown()

    app = FastAPI(lifespan=lifespan)

    def token_count(body: dict) -> int:
        messages = handler.get_processor_inputs_from_messages(body["messages"], modality)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tools=None, return_tensors="pt",
            return_dict=True, tokenize=True, load_audio_from_video=False,
            enable_thinking=api["thinking"], truncation=False,
        )
        return int(inputs["input_ids"].shape[-1])

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"object": "model", "id": public_model,
                "created": int(time.time()), "owned_by": "Qwen", "root": snapshot,
                "max_model_len": maximum, "backend": "transformers"}]}

    @app.post("/tokenize")
    async def tokenize(body: dict):
        try:
            validate_request(body, public_model, api["thinking"])
            async with lock:
                count = await asyncio.to_thread(token_count, body)
            check_budget(count, api["max_output_tokens"], maximum)
            return {"count": count, "max_model_len": maximum}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/chat/completions")
    async def completion(body: dict):
        try:
            validate_request(body, public_model, api["thinking"], completion=True)
            body = dict(body)
            body.setdefault("max_tokens", api["max_output_tokens"])
            body.setdefault("temperature", api["temperature"])
            body.setdefault("stream", False)
            async with lock:
                count = await asyncio.to_thread(token_count, body)
                check_budget(count, body["max_tokens"], maximum)
                body["model"] = snapshot
                response = await handler.handle_request(body, str(uuid.uuid4()))
            content = json.loads(response.body)
            content["model"] = public_model
            return JSONResponse(content, status_code=response.status_code)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/agent.json")
    args = parser.parse_args()
    policy = load_agent_config(config_file=args.config, mode="baseline_inference")["policy"]
    deployment = read_deployment(PROJECT_ROOT / policy["api"]["serving"]["deployment_file"], policy)
    app = create_app(policy, deployment)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=policy["api"]["remote_port"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
