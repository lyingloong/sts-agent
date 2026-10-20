"""Export a completed batch run's exact recorded model I/O for inspection.

No prompt reconstruction, tokenization, training-data conversion, truncation,
network access, or game commands. Original controller artifacts stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter
from pathlib import Path

from sts_agent.paths import PROJECT_ROOT


def snapshot_records(path: Path):
    """Bound reads to the file size at entry while other runs keep appending.

    Only a non-newline-terminated final fragment of this snapshot is ignored.
    Export completeness is checked against the completed batch's request count.
    Malformed complete lines always raise instead of silently dropping data.
    """
    limit = path.stat().st_size
    with path.open("rb") as source:
        while source.tell() < limit:
            line = source.readline(limit - source.tell())
            if not line.endswith(b"\n"):
                return
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL record in {path}")
                yield value


def code_block(value: str, language: str = "") -> list[str]:
    fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`{3,}", value)), default=0))
    return [fence + language, value, fence, ""]


def collect_requests(path: Path, run_id: str) -> list[dict]:
    requests: dict[int, dict] = {}
    active_request = None
    for record in snapshot_records(path):
        event, request_id = record.get("event"), record.get("request_id")
        if event == "request":
            active_request = request_id
            if record.get("run_id") == run_id:
                if type(request_id) is not int or request_id in requests:
                    raise ValueError("invalid or duplicate request ID")
                requests[request_id] = {"request_id": request_id, "request_record": record,
                                        "events": [], "execution_results": []}
            continue
        if event == "execution_result" and record.get("run_id") == run_id:
            request_id = ((record.get("decision") or {}).get("api") or {}).get("request_id")
            if request_id in requests:
                requests[request_id]["execution_results"].append(record)
        elif event == "transport_error" and request_id is None:
            request_id = active_request
        if request_id in requests:
            requests[request_id]["events"].append(record)
    return list(requests.values())


def disposition(entry: dict) -> str:
    executions = entry["execution_results"]
    if any(record.get("accepted") is True for record in executions):
        return "executed"
    if executions:
        return "rejected"
    if any(record.get("event") == "stale_action_discarded" for record in entry["events"]):
        return "discarded"
    if any(record.get("event") == "decision_error" for record in entry["events"]):
        return "error"
    return "unresolved"


def export_review(batch_report: Path, output: Path, run_id: str | None = None) -> dict:
    batch_report, output = batch_report.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"output already exists; refusing overwrite: {output}")
    results = json.loads((batch_report / "results.json").read_text(encoding="utf-8"))["results"]
    completed = [row for row in results if row["status"] == "completed" and (run_id is None or row["run_id"] == run_id)]
    if not completed:
        raise ValueError("requested run is not a completed batch result")
    row = completed[-1]
    run_id = row["run_id"]
    artifacts = row["artifacts"]
    entries = collect_requests(Path(artifacts["requests"]), run_id)
    if len(entries) != row["api_performance"]["calls"]:
        raise ValueError("recorded request count does not match completed batch result")
    responses = {}
    for entry in entries:
        responses[entry["request_id"]] = [item for item in entry["events"] if item.get("event") == "response"]
        if len(responses[entry["request_id"]]) != 1:
            raise ValueError(f"request {entry['request_id']} lacks one complete original response")
        messages = entry["request_record"]["request"]["messages"]
        if not messages or any(not isinstance(item.get("content"), str) for item in messages):
            raise ValueError("expected complete text messages; refusing to reconstruct input")
        if len(messages) == 1:
            expected_hash = entry["request_record"].get("prompt_sha256")
            actual_hash = hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()
            if expected_hash and expected_hash != actual_hash:
                raise ValueError("recorded prompt hash mismatch")
        entry["disposition"] = disposition(entry)
    traces = [Path(path) for path in artifacts["action_traces"]]
    if len(traces) != 1:
        raise ValueError("expected one original action trace for completed run")
    actions = list(snapshot_records(traces[0]))
    if any(record.get("run_id") != run_id for record in actions):
        raise ValueError("trace contains another run")
    accepted = sum(record.get("accepted") is True for record in actions)
    rejected = sum(record.get("accepted") is False for record in actions)
    if (accepted != row["summary"]["accepted_actions"] or rejected != row["summary"]["rejected_actions"]
            or len(actions) != accepted + rejected):
        raise ValueError("action trace count does not match completed batch result")
    counts = Counter(entry["disposition"] for entry in entries)
    report = {"run_id": run_id, "seed": row["seed"], "summary": row["summary"],
              "api_performance": row["api_performance"], "source_artifacts": artifacts,
              "request_dispositions": dict(counts), "input_source": "recorded request.messages, not reconstructed",
              "truncated": False, "accepted_trace_actions": accepted}
    output.mkdir(parents=True)
    with (output / "model-io.jsonl").open("w", encoding="utf-8") as destination:
        for entry in entries:
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (output / "review-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# 完整模型输入输出：{run_id}", "", f"种子：`{row['seed']}`。请求数：{len(entries)}。", "",
             "所有输入直接取当时请求的 `messages`，所有回答直接取原始 API response；不重建、不截断。",
             "`discarded` 是因状态变化丢弃、未执行的回答，不应当作实际动作。自动环境动作不调用模型。",
             "JSONL 中的 `aliases`、state key 等是控制器元数据，不属于模型输入；实际输入仅看 `request.messages`。", "",
             "## 游戏动作索引", "", "| step_id | 幕/楼层 | 来源 | 动作 | 接受 | 请求 ID |",
             "| --- | --- | --- | --- | --- | --- |"]
    for action in actions:
        state, decision = action.get("pre_state") or {}, action.get("decision") or {}
        label = str((action.get("action") or {}).get("label") or (action.get("action") or {}).get("command"))
        lines.append(f"| {action.get('step_id')} | {state.get('act')}/{state.get('floor')} | {decision.get('source')} "
                     f"| {label.replace('|', '/').replace(chr(10), ' ')} | {action.get('accepted')} "
                     f"| {(decision.get('api') or {}).get('request_id', '')} |")
    request_lines = {}
    for entry in entries:
        request_id = entry["request_id"]
        request_lines[request_id] = sum(part.count("\n") + 1 for part in lines) + 2
        request = entry["request_record"]["request"]
        response_event = responses[request_id][0]
        response = response_event["response"]
        content = response["choices"][0]["message"].get("content", "")
        chosen = entry["request_record"].get("aliases", {}).get(content.strip(), {})
        lines.extend(["", f"## 请求 {request_id} · {entry['disposition']}", "",
                      f"回答对应动作：{chosen.get('label') or chosen.get('command') or '无法解析'}", "",
                      f"输入 token：{response_event.get('input_tokens')}；延迟：{response_event.get('latency_ms')} ms。", "",
                      "### 请求参数（messages 在下方完整展示）", ""])
        lines.extend(code_block(json.dumps({k: v for k, v in request.items() if k != "messages"}, ensure_ascii=False, indent=2), "json"))
        for message in request["messages"]:
            lines.extend([f"### 完整输入 · {message['role']}", ""])
            lines.extend(code_block(message["content"], "text"))
        lines.extend(["### 原始完整 API 输出", ""])
        lines.extend(code_block(json.dumps(response, ensure_ascii=False, indent=2), "json"))
        lines.extend(["### 执行／复核记录", ""])
        lines.extend(code_block(json.dumps(entry["events"], ensure_ascii=False, indent=2), "json"))
    full_path = output / "full-input-output.md"
    full_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = [f"# 输入输出复盘：{run_id}", "", f"种子：`{row['seed']}`；请求数：{len(entries)}；处理结果：`{dict(counts)}`。", "",
             "每个链接跳到完整输入输出文件的对应请求。动作索引在该文件开头。", "",
             "| 请求 | 处理结果 | 输入 token | 回答 |", "| --- | --- | --- | --- |"]
    for entry in entries:
        request_id = entry["request_id"]
        response_event = responses[request_id][0]
        answer = response_event["response"]["choices"][0]["message"].get("content", "")
        index.append(f"| [请求 {request_id}](<{full_path.as_posix()}:{request_lines[request_id]}>) | {entry['disposition']} "
                     f"| {response_event.get('input_tokens')} | {answer.strip().replace('|', '/').replace(chr(10), ' ')} |")
    (output / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return {"output": str(output), **report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or PROJECT_ROOT / "data" / "reviews" / "baseline" / f"review-{uuid.uuid4().hex[:8]}"
    result = export_review(args.batch_report, output, args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
