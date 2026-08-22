"""Download a local StS1 card knowledge snapshot for terminal/model fallback."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


SOURCE_URL = "https://raw.githubusercontent.com/nkhoit/spire-archive/main/data/sts1/cards.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "knowledge" / "sts1_cards.json"


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "sts-agent-card-knowledge"})
    with urllib.request.urlopen(request, timeout=30) as response:
        cards = json.load(response)
    if not isinstance(cards, list) or not cards:
        raise RuntimeError("card knowledge source returned no cards")
    compact = {
        "schema_version": "0.1",
        "source": SOURCE_URL,
        "cards": {str(card["id"]).upper(): card for card in cards if isinstance(card, dict) and card.get("id")},
    }
    OUTPUT_PATH.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(compact['cards'])} cards to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
