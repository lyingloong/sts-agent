"""Readable reward relationships for a caller-owned text display copy.

Control observations retain reward identities. This final display projection
uses identities only to resolve public relationships, then removes those keys.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


_IDENTIFIER_FIELDS = {
    "reward_id", "reward_ref", "linked_reward_id", "linked_reward_ref",
    "source_reward_id", "source_reward_ref",
}
_TYPE_NAMES = {
    "RELIC": "遗物", "CARD": "卡牌奖励", "POTION": "药水奖励", "GOLD": "金币奖励",
    "STOLEN_GOLD": "被偷的金币", "SAPPHIRE_KEY": "蓝宝石钥匙",
    "EMERALD_KEY": "翡翠钥匙", "RUBY_KEY": "红宝石钥匙",
}


def _tokens(item: dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    return {str(item[key]) for key in fields if isinstance(item.get(key), (str, int))
            and not isinstance(item[key], bool) and str(item[key])}


def _kind(item: dict[str, Any], fallback: str) -> str:
    value = str(item.get("reward_type") or fallback).upper()
    if value:
        return value
    for child, child_kind in (("relic", "RELIC"), ("card", "CARD"), ("potion", "POTION")):
        if isinstance(item.get(child), dict):
            return child_kind
    return ""


def _label(item: dict[str, Any], fallback: str) -> str:
    kind = _kind(item, fallback)
    name = item.get("name")
    if not isinstance(name, str) or not name:
        for child in ("relic", "card", "potion"):
            entity = item.get(child)
            if isinstance(entity, dict) and isinstance(entity.get("name"), str) and entity["name"]:
                name = entity["name"]
                break
    category = _TYPE_NAMES.get(kind, "奖励")
    if kind.endswith("_KEY") and kind in _TYPE_NAMES:
        label = category
    elif isinstance(name, str) and name:
        label = name if category == "奖励" or name == category else f"{category}：{name}"
    else:
        label = category
    if type(item.get("choice_index")) is int:
        label += f"（选择位置={item['choice_index']}）"
    elif type(item.get("reward_index")) is int:
        # An existing current reward position is useful; it is not a new ID or
        # a guessed command index for filtered/legacy reward lists.
        label += f"（当前奖励索引={item['reward_index']}）"
    return label


def _relationships(screen: dict[str, Any]) -> None:
    entries: list[tuple[dict[str, Any], str]] = []
    for field, kind in (("rewards", ""), ("relics", "RELIC")):
        values = screen.get(field)
        if isinstance(values, list):
            entries.extend((item, kind) for item in values if isinstance(item, dict))
    identities: dict[str, set[int]] = defaultdict(set)
    for index, (item, _) in enumerate(entries):
        for token in _tokens(item, ("reward_id", "reward_ref")):
            identities[token].add(index)
    existing = screen.get("reward_relationships")
    descriptions = list(existing) if isinstance(existing, list) else []
    pairs: set[tuple[int, int]] = set()
    for index, (item, kind) in enumerate(entries):
        linked = _tokens(item, ("linked_reward_id", "linked_reward_ref"))
        if not linked:
            continue
        matches = set().union(*(identities.get(token, set()) for token in linked)) - {index}
        if len(matches) == 1:
            target = next(iter(matches))
            pair = tuple(sorted((index, target)))
            if pair in pairs:
                continue
            pairs.add(pair)
            left, right = (item, kind), entries[target]
            if _kind(right[0], right[1]) == "SAPPHIRE_KEY":
                left, right = right, left
            text = (f"{_label(*left)}与{_label(*right)}二选一；"
                    "领取其中一项会放弃另一项。")
        else:
            # A consumed or unresolvable linked reward has no reliable current
            # name/position. Retain exclusivity without guessing its identity.
            text = f"{_label(item, kind)}与另一项关联奖励二选一；另一项当前无法确定。"
        if text not in descriptions:
            descriptions.append(text)
    if descriptions:
        screen["reward_relationships"] = descriptions


def project_reward_text(view: dict[str, Any]) -> dict[str, Any]:
    """Mutate a display copy only; preserve positions, names and control aliases.

    Call after copying the durable observation and before removing display
    metadata. A second call retains the same semantic relationships.
    """
    screen = view.get("screen_state")
    if isinstance(screen, dict):
        _relationships(screen)

    def remove_identifiers(value: Any) -> None:
        if isinstance(value, dict):
            for key in _IDENTIFIER_FIELDS:
                value.pop(key, None)
            for child in value.values():
                remove_identifiers(child)
        elif isinstance(value, list):
            for child in value:
                remove_identifiers(child)

    remove_identifiers(view)
    return view
