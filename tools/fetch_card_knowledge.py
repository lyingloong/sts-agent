"""Download local StS1 card, relic, and potion knowledge for model fallback."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


SOURCE_URLS = {
    "cards": "https://raw.githubusercontent.com/nkhoit/spire-archive/main/data/sts1/cards.json",
    "relics": "https://raw.githubusercontent.com/nkhoit/spire-archive/main/data/sts1/relics.json",
    "potions": "https://raw.githubusercontent.com/nkhoit/spire-archive/main/data/sts1/potions.json",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "knowledge"


def download(kind: str, source_url: str) -> int:
    request = urllib.request.Request(source_url, headers={"User-Agent": "sts-agent-knowledge"})
    with urllib.request.urlopen(request, timeout=30) as response:
        entities = json.load(response)
    if not isinstance(entities, list) or not entities:
        raise RuntimeError(f"{kind} knowledge source returned no entries")
    output_path = OUTPUT_DIR / f"sts1_{kind}.json"
    compact = {
        "schema_version": "0.2",
        "source": source_url,
        kind: {str(entity["id"]).upper(): entity for entity in entities if isinstance(entity, dict) and entity.get("id")},
    }
    output_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count = len(compact[kind])
    print(f"wrote {count} {kind} to {output_path}")
    return count


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind, source_url in SOURCE_URLS.items():
        download(kind, source_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
