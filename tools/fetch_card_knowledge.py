"""Fetch a versioned, full StS1 static-knowledge snapshot for this project."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/nkhoit/spire-archive"
SOURCE_BASE_URL = "https://raw.githubusercontent.com/nkhoit/spire-archive/main/data/sts1"
SOURCE_COMMIT = "14e05c09dc388f0171eb215f058bd2995da645b7"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "knowledge"

# These are every JSON resource in the source's StS1 root as of the pinned
# project snapshot. Localizations are a separate set of text translations, not
# additional game mechanics, but are retained for inspection and future use.
KNOWLEDGE_FILES = (
    "achievements.json",
    "blights.json",
    "card_powers.json",
    "cards.json",
    "characters.json",
    "events.json",
    "keywords.json",
    "mechanics.json",
    "monsters.json",
    "orbs.json",
    "potions.json",
    "powers.json",
    "relic_values.json",
    "relics.json",
    "stances.json",
)
LOCALIZATION_FILES = tuple(f"localization/{language}.json" for language in (
    "de",
    "en",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "th",
    "tr",
    "zh",
))

# The upstream parser has a few known StS1 text defects. Keep every local
# correction in this table so the tracked snapshot and future refreshes are
# transparent and reproducible. Do not replace this with broad text stripping:
# legitimate game descriptions contain numbers and repeated terms.
ENTITY_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "cards": {
        # Impulse was an early Defect beta card. Its upstream record has an
        # empty description, so retain the historical base/upgraded wording.
        "IMPULSE": {
            "description": "Use the Passive effect of all your Channeled Orbs. Exhaust.",
            "upgrade": {"description": "Use the Passive effect of all your Channeled Orbs."},
        },
    },
    "powers": {
        "BARRICADE": {"description": "Block is not removed at the start of your turn."},
        "BLUR": {"description": "Block is not removed at the start of your next turn."},
        "BUFFER": {"description": "Prevent the next time you would lose HP."},
        "BURST": {"description": "Your next Skill is played twice this turn."},
        "COLLECT": {"description": "At the start of your next turn, put a Miracle+ into your hand."},
        "CONSERVE": {"description": "Energy is conserved this turn."},
        "CORPSEEXPLOSIONPOWER": {"description": "On death, deal damage equal to its Max HP to ALL enemies."},
        "DARK_EMBRACE": {"description": "Whenever a card is Exhausted, draw 1 card."},
        "DOUBLE_TAP": {"description": "Your next Attack is played twice this turn."},
        "DRAW_REDUCTION": {"description": "Draw 1 less card next turn."},
        "DUPLICATIONPOWER": {"description": "Your next card is played twice this turn."},
        "ECHO_FORM": {"description": "The first card you play each turn is played twice."},
        "EQUILIBRIUM": {"description": "Retain your hand this turn."},
        "EVOLVE": {"description": "Whenever you draw a Status card, draw 1 card."},
        "FREEATTACKPOWER": {"description": "Ignore energy cost on the next Attack you play."},
        "LOOP": {"description": "At the start of your turn, trigger the passive ability of your next Orb."},
        "MAGNETISM": {"description": "At the start of your turn, add X random Colorless card to your hand."},
        "MAYHEM": {"description": "At the start of your turn, play the top card of your draw pile."},
        "PHANTASMAL": {"description": "Deal Double Damage next turn."},
        "REBOUND": {"description": "The next card you play this turn is placed on top of your draw pile."},
        "THEBOMB": {"description": "At the end of 3 turns, deal massive damage to ALL enemies."},
    },
    "potions": {
        "ATTACKPOTION": {"description": "Choose 1 of 3 random Attack cards to add to your hand, it costs 0 this turn."},
        "COLORLESSPOTION": {"description": "Choose 1 of 3 random Colorless cards to add to your hand, it costs 0 this turn."},
        "DUPLICATIONPOTION": {"description": "This turn, your next card is played twice."},
        "LIQUIDMEMORIES": {"description": "Choose a card in your discard pile and return it to your hand. It costs 0 this turn."},
        "POWERPOTION": {"description": "Choose 1 of 3 random Power cards to add to your hand, it costs 0 this turn."},
        "SKILLPOTION": {"description": "Choose 1 of 3 random Skill cards to add to your hand, it costs 0 this turn."},
        "SMOKEBOMB": {"description": "Escape from a non-boss combat. Receive no rewards."},
        "SWIFT_POTION": {"description": "Draw 3 cards."},
    },
}


def fetch_json(source_url: str) -> Any:
    """Read source JSON through curl, which is more reliable on this Windows host."""
    result = subprocess.run(
        [
            "curl.exe",
            "--fail",
            "--location",
            "--http1.1",
            "--retry",
            "1",
            "--retry-all-errors",
            "--connect-timeout",
            "10",
            "--max-time",
            "20",
            "--silent",
            "--show-error",
            source_url,
        ],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not download {source_url}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"source returned invalid JSON: {source_url}") from exc


def normalize_entity(kind: str, entity: dict[str, Any]) -> dict[str, Any]:
    """Apply documented source corrections to one entity."""
    result = dict(entity)
    item_id = str(result.get("id") or "").upper()
    override = ENTITY_OVERRIDES.get(kind, {}).get(item_id)
    if override is None:
        return result
    for field, value in override.items():
        if field == "upgrade" and isinstance(value, dict):
            result["upgrade"] = dict(result.get("upgrade") or {}) | value
        else:
            result[field] = value
    return result


def normalized_payload(relative_path: str, payload: Any) -> Any:
    """Normalize entity collections and apply their explicit corrections."""
    kind = Path(relative_path).stem
    if kind not in ENTITY_OVERRIDES:
        return payload
    entities = payload.get(kind, payload) if isinstance(payload, dict) else payload
    if isinstance(entities, dict):
        entities = list(entities.values())
    if not isinstance(entities, list):
        raise RuntimeError(f"{relative_path} does not contain an entity list")
    return [normalize_entity(kind, entity) if isinstance(entity, dict) else entity for entity in entities]


def output_path(relative_path: str) -> Path:
    return OUTPUT_DIR / f"sts1_{relative_path}"


def write_snapshot(relative_path: str, payload: Any) -> int:
    if not isinstance(payload, (list, dict)):
        raise RuntimeError(f"{relative_path} snapshot contains an unsupported JSON value")
    destination = output_path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count = len(payload)
    print(f"wrote {count} entries to {destination.relative_to(PROJECT_ROOT)}", flush=True)
    return count


def write_manifest() -> None:
    manifest = {
        "schema_version": "1.0",
        "source_repository": SOURCE_REPOSITORY,
        "source_ref": "main",
        "source_commit": SOURCE_COMMIT,
        "source_base_url": SOURCE_BASE_URL,
        "files": [f"sts1_{path}" for path in (*KNOWLEDGE_FILES, *LOCALIZATION_FILES)],
        "local_entity_overrides": ENTITY_OVERRIDES,
        "notes": "Static snapshot is tracked in Git. Runtime fields from CommunicationMod remain authoritative.",
    }
    path = OUTPUT_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest to {path.relative_to(PROJECT_ROOT)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize the tracked StS1 knowledge snapshot")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="apply local corrections and write the manifest from the files already in data/knowledge",
    )
    return parser.parse_args()


def fetch_all(files: tuple[str, ...]) -> dict[str, Any]:
    """Fetch all files before touching the current snapshot on disk."""
    payloads: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_json, f"{SOURCE_BASE_URL}/{relative_path}"): relative_path
            for relative_path in files
        }
        for future in as_completed(futures):
            relative_path = futures[future]
            payloads[relative_path] = normalized_payload(relative_path, future.result())
    return payloads


def existing_all(files: tuple[str, ...]) -> dict[str, Any]:
    """Read and normalize a fully downloaded snapshot without using the network."""
    payloads: dict[str, Any] = {}
    for relative_path in files:
        path = output_path(relative_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot finalize missing or invalid snapshot file: {path}") from exc
        payloads[relative_path] = normalized_payload(relative_path, payload)
    return payloads


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = (*KNOWLEDGE_FILES, *LOCALIZATION_FILES)
    payloads = existing_all(files) if args.finalize_existing else fetch_all(files)
    for relative_path in files:
        write_snapshot(relative_path, payloads[relative_path])
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
