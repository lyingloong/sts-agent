"""Minimal CommunicationMod observer.

The process must keep stdout reserved for the CommunicationMod protocol.
Diagnostic output is written to stderr and raw messages are appended to a
JSONL file. The probe does not send game actions, so it is safe for the first
manual state-capture test.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from sts_agent.paths import PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    output_dir = PROJECT_ROOT / "data" / "raw" / "communication_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "messages.jsonl"

    # CommunicationMod waits for this exact readiness handshake.
    print("ready", flush=True)

    with output_path.open("a", encoding="utf-8") as output:
        for line in sys.stdin:
            raw = line.rstrip("\r\n")
            if not raw:
                continue

            record: dict[str, object] = {
                "received_at": utc_now(),
                "raw": raw,
            }
            try:
                record["message"] = json.loads(raw)
            except json.JSONDecodeError as exc:
                record["parse_error"] = str(exc)

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            # Never print diagnostics to stdout: it is the command channel.
            if "parse_error" in record:
                print("received a non-JSON CommunicationMod message", file=sys.stderr)

    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
