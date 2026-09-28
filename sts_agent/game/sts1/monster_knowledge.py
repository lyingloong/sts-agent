"""Static, public StS1 monster knowledge used for future-boss planning."""

from __future__ import annotations

import json
from typing import Any

from sts_agent.game.sts1.item_knowledge import normalize_item_key
from sts_agent.paths import PROJECT_ROOT


KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "sts1_monsters.json"
_CACHE: dict[str, dict[str, Any]] | None = None

# The upcoming boss is shown globally, so the action model receives a planning
# guide rather than a move table. These are curated paraphrases of the strategy
# or response sections in the Huiji Wiki entries documented in
# ``doc/BOSS_SUMMARIES.md``. They cover what to seek before the boss and how to
# preserve resources for the key windows. Detailed move tables remain
# combat-only through ``monster_profile`` and must not be duplicated in
# ``act_boss_profile``.
BOSS_STRATEGIES: dict[str, str] = {
    "THEGUARDIAN": "需要攻防均衡，而不是只堆输出或只堆格挡。路线、选牌和商店应优先补足稳定格挡、虚弱或力量削减，以及能集中打出高伤害的组件：攻击形态的高额单次或多段伤害要能持续接住，形态转换前后又要有一轮爆发快速压过阈值。进入防御形态后，锋利外甲会惩罚低价值、反复的小攻击，除非这些攻击能带来足够收益，否则应等待更好的输出窗口。",
    "SLIMEBOSS": "这是对卡组爆发伤害能力的检查，而不只是平均每回合伤害。路线和商店应重视高质量攻击、力量或易伤、费用或能量以及爆发药水；面对 Boss 时先把血量控制在接近分裂阈值但不要过早越过，利用它准备的回合完成铺垫，再在一回合内压过阈值。越过阈值时 Boss 剩余血量越低，分裂后的两只史莱姆压力通常越小；缺乏爆发的慢速卡组需要谨慎选择路线和营火。",
    "HEXAGHOST": "需要长线输出，同时也要有尽快结束战斗的能力。Boss 会持续向牌组加入灼伤，战斗拖长会降低抽牌质量并挤占防御，因此路线与选牌应同时补充稳定的持续伤害、过牌和可靠防御，而不能只依赖一次性爆发。若当前输出足够，进入 Boss 前可以考虑保持较低生命值，因为早期地狱火伤害与当前生命值有关；在能安全击杀时，营火通常可优先锻造而非休息。其多段攻击使减力量、荆棘和反伤具有额外价值，但仍要以能活过高压回合为前提。",
    "CHAMP": "需要较高的输出上限、稳定防御和半血转阶段规划。它在低于半血时会清除减益并强化，所以不要在自身无法防御或核心牌尚未启动时随意把 Boss 压过阈值；可以先准备一回合防御或爆发，再用集中输出进入并尽快处理强化阶段。路线和商店应优先补齐力量成长、虚弱或易伤、过牌与高效格挡，不能只依赖一个持续减益维持防线。",
    "THECOLLECTOR": "前置重点是同时处理 Boss 和火炬头的能力，尤其要有群体输出或高效集火。路线和商店应优先补充能快速清理小怪、削弱敌方力量，并在召唤或究极减益后维持防御的牌和药水；不要让火炬头存活太久并在强化后叠加伤害。是否先杀 Boss 还是小怪要看当前意图、群攻和牌组，避免机械地固定目标。",
    "BRONZEAUTOMATON": "需要处理铜球的能力，同时保留足够的单体和群体输出。铜球会凝滞牌组中的高价值牌，路线和商店应重视能快速清理铜球、穿透格挡或在多敌人场景发挥作用的组件；利用本体压力较低或非攻击的回合处理铜球，不要让铜球和本体的高压攻击叠加。进入 Boss 前还要检查是否能在超能光束前建立足够防御或完成击杀，并尽量避免把唯一的关键牌暴露在凝滞风险中。",
    "AWAKENEDONE": "需要在能力牌的即时价值与 Boss 的力量成长之间权衡。路线和选牌不必完全放弃能力牌，但应优先选择能立即带来生存或输出收益、且不依赖长时间叠层的能力；没有必要的能力不要为了启动体系而早打。两阶段共用资源，进入 Boss 前要有足够的持续输出、爆发、过牌和防御，并计划第一阶段邪教徒的处理时机，避免第一阶段耗尽药水或核心牌后无法应对第二阶段。",
    "DONU": "这是一个会随着回合数增加而变难的双 Boss 战。通常应优先压制 Donu 的力量成长，因为战斗越长两名敌人的攻击压力都会放大；但也不能因此忽略 Deca 的格挡和晕眩。路线和商店应重视群攻或多目标收益、稳定格挡和持续伤害，并根据当前牌组判断何时集火 Donu、何时先清理或削弱 Deca。",
    "DECA": "Deca 的格挡和晕眩会把战斗拖成长线资源战，并与 Donu 的力量成长互相放大。通常可以把 Donu 作为优先击杀目标，但如果当前卡组特别怕晕眩或无法及时处理 Deca，就应保留群攻、过牌和不依赖状态牌的输出，避免只盯单一目标而被格挡和状态牌拖垮。Boss 前要准备同时处理两名敌人的防御回合，并根据实时意图调整集火顺序。",
    "TIMEEATER": "需要同时管理 12 张牌计数、Boss 的成长和半血回复窗口。路线和商店应倾向于高效牌、过牌和可控爆发，避免只为了花光能量而打低价值牌；当前计数接近 12 时，要提前决定是结束回合还是用高价值牌跨过阈值，并为下一回合保留防御或爆发。Boss 血量接近半血时，尽量在能快速完成后续输出的窗口压过阈值，避免它回复后重新进入长线消耗。",
    "CORRUPTHEART": "需要完整的长线构筑和较高上限：既要有稳定格挡或免伤应对多段与重击，也要有持续输出、力量削减、反伤或其他减少多段压力的手段。路线、商店和营火要优先保证能在死亡律动和伤害上限下连续多个回合生存，不能只靠单回合爆发；同时要考虑伤害上限，避免把大量低伤害攻击浪费在同一回合。强化回合要谨慎使用会使力量削减失效的手段，药水和关键能力应留给真正的高压回合。",
}


def _load() -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {}
    try:
        payload = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _CACHE
    entities = payload.get("monsters", payload) if isinstance(payload, dict) else payload
    if not isinstance(entities, list):
        return _CACHE
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        for value in (entity.get("id"), entity.get("name")):
            key = normalize_item_key(value)
            if key:
                _CACHE[key] = entity
    return _CACHE


def boss_profile(value: Any) -> dict[str, Any] | None:
    """Return a global planning strategy for one known Act boss.

    This field is intentionally a strategic guide for pre-boss planning, not
    a prediction of the next intent. Current combat intent, damage, HP, block,
    and Powers remain the responsibility of the live game state and the
    combat-only static monster knowledge.
    """
    key = normalize_item_key(value)
    entry = _load().get(key)
    if not isinstance(entry, dict) or str(entry.get("type") or "").casefold() != "boss":
        return None
    boss_id = normalize_item_key(entry.get("id")) or key
    result = {
        field: entry.get(field)
        for field in ("id", "name", "act")
        if entry.get(field) is not None
    }
    result["strategy"] = BOSS_STRATEGIES.get(
        boss_id,
        "这是路线和卡组规划的参考，不是硬性规则。根据当前卡组、生命、遗物、药水、路线和战斗中的实时意图，提前准备稳定防御、持续输出和处理高压阶段所需的资源。",
    )
    result["source"] = "huiji_wiki_curated_strategy"
    experience = entry.get("experience")
    if isinstance(experience, list) and experience:
        result["experience"] = experience[:3]
    return result


def monster_profile(value: Any) -> dict[str, Any] | None:
    """Return static reference mechanics for one known combat enemy.

    This is deliberately separate from the live enemy fields. ``moves`` lists
    possibilities from the knowledge snapshot; the current intent, damage,
    HP, block, and Powers remain sourced from CommunicationMod.
    """
    key = normalize_item_key(value)
    entry = _load().get(key)
    if not isinstance(entry, dict):
        return None
    moves: list[dict[str, Any]] = []
    for move in entry.get("moves", []):
        if not isinstance(move, dict):
            continue
        item = {
            field: move.get(field)
            for field in ("id", "name", "intent", "damage", "damage_ascension", "hits", "block")
            if move.get(field) is not None
        }
        if item:
            moves.append(item)
    result = {
        field: entry.get(field)
        for field in (
            "id",
            "name",
            "type",
            "act",
            "min_hp",
            "max_hp",
            "min_hp_ascension",
            "max_hp_ascension",
        )
        if entry.get(field) is not None
    }
    result["moves"] = moves
    result["source"] = "static_knowledge"
    experience = entry.get("experience")
    if isinstance(experience, list) and experience:
        result["experience"] = experience[:3]
    return result
