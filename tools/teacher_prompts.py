"""Chinese, teacher-only strategy prompts for Slay the Spire 1.

These instructions are sent only to the remote teacher. They are deliberately
separate from the local policy prompt and from the durable SFT target.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


TEACHER_PROMPT_VERSION = "sts1-teacher-zh-v7"
PLAYER_PROMPT_VERSION = "sts1-player-zh-v3"
# Kept as the default public version for the SFT teacher client.
PROMPT_VERSION = TEACHER_PROMPT_VERSION
CHARACTER_PROMPT_LIBRARY_VERSION = "sts1-character-prompt-library-v1"
CHARACTER_KEYS = frozenset({"IRONCLAD", "THE_SILENT", "DEFECT", "WATCHER"})
CHARACTER_ALIASES = {
    "IRONCLAD": "IRONCLAD",
    "THE_SILENT": "THE_SILENT",
    "SILENT": "THE_SILENT",
    "DEFECT": "DEFECT",
    "WATCHER": "WATCHER",
}

_COMMON_RULES = """目标是提高整局通关概率，而不是只追求当前房间的即时收益。只使用观察中明确给出的公开信息；不要臆测没有展示的事件结果、牌堆顺序、敌人意图或物品效果。当前合法动作列表是唯一可执行集合。

做决定时先识别当前场景和不可逆后果，再比较每个合法动作：当前生命和即将受到的伤害、牌组缺口、遗物与药水、金币、当前 Act/Floor、当前 Act 已公开首领、未来地图路线和最近动作。`decision_constraints` 是已持有遗物造成的真实限制或代价，必须服从；不要把已被约束排除的选择当作可行方案。优先选择稳健、能解决当前最紧迫问题且不依赖未经证实假设的动作。不要因为某张牌/遗物稀有就自动选择，也不要为了保留资源而在即将失败时拒绝使用资源。

你可以进行内部推理，但最终只能返回一个 JSON 对象，格式严格为 {\"action_id\":\"A<n>\"}。其中 A<n> 必须来自当前合法动作列表；不要输出解释、Markdown、命令或其他动作 ID。"""

KNOWLEDGE_REQUEST_RULES = {
    "encouraged": """只要对当前公开实体的机制、交互、风险或策略含义有不明白、不确定之处，就可以暂不选择动作，改为返回一次知识请求：{\"knowledge_request\":{\"entity_type\":\"card|relic|potion|monster|event|power\",\"entity_id\":\"当前公开的实体 ID 或名称\",\"question\":\"一个具体问题\"}}。这是被鼓励使用的工具：面对实体相关疑问时，优先提问而不是靠猜测继续；不需要先证明自己已经无法可靠决策。只请求当前 observation 中已经公开的实体，问题必须能由人类经验回答；不要把泛泛的整局策略当作实体问题，也不要在同一响应中同时返回 action_id 和 knowledge_request。""",
    "normal": """如果对当前公开实体的机制、交互、风险或策略含义有不明白之处，可以暂不选择动作，改为返回一次知识请求：{\"knowledge_request\":{\"entity_type\":\"card|relic|potion|monster|event|power\",\"entity_id\":\"当前公开的实体 ID 或名称\",\"question\":\"一个具体问题\"}}。优先使用当前状态、静态资料和已有 experience；仍有不明白之处时可以提问，不需要先证明自己已经无法可靠决策。只请求当前 observation 中已经公开的实体，问题必须能由人类经验回答；不要把泛泛的整局策略当作实体问题，也不要在同一响应中同时返回 action_id 和 knowledge_request。""",
}

KNOWLEDGE_RESPONSE_CONTRACT = """【最终输出协议（优先级最高）】
知识请求已启用。最终只能返回一个 JSON 对象，且只能二选一：
1. 已能作出判断时，返回 {\"action_id\":\"A<n>\"}，其中 A<n> 必须来自当前合法动作列表；
2. 对当前公开实体有值得澄清的机制、交互、风险或策略问题时，返回 {\"knowledge_request\":{\"entity_type\":\"card|relic|potion|monster|event|power\",\"entity_id\":\"当前公开的实体 ID 或名称\",\"question\":\"一个具体问题\"}}，此时不要返回 action_id。

不要输出解释、Markdown 或两个对象。上文任何“必须选择当前合法动作”或“只能返回 action_id”的要求，只适用于不发起知识请求时；当你选择提问时，本协议优先。"""


def knowledge_request_rule(style: Any = "encouraged") -> str:
    """Return the configured in-game entity-question policy.

    Config validation accepts only the two documented styles. The fallback is
    deliberately the default encouraged policy so direct utility callers do
    not silently disable a configured knowledge capability.
    """

    return KNOWLEDGE_REQUEST_RULES.get(str(style or "").strip().casefold(), KNOWLEDGE_REQUEST_RULES["encouraged"])

COMMON_PROMPT = (
    "你是《杀戮尖塔 1》的远程教师，负责为模仿学习轨迹选择一个动作。\n"
    + _COMMON_RULES
)
PLAYER_COMMON_PROMPT = (
    "你是正在亲自游玩《杀戮尖塔 1》的玩家。你的任务是在当前局面做出并执行一个最有利于整局通关的动作；当前已审核的 skill 是你的策略记忆。\n"
    + _COMMON_RULES
)


SCENARIO_PROMPTS: dict[str, str] = {
    "MAP": """【地图决策】
查看完整公开地图和当前可达节点，不要只比较下一步的房间符号。比较路线在未来 2-4 个节点内能提供的战斗、精英、营火、商店、宝箱和事件，以及之后是否仍有调整空间。
- 早期牌组通常先需要解决伤害和基础生存，再考虑精英；但不要把“固定打几个精英”当成硬规则。
- 精英的遗物和奖励很有价值，但只有在当前生命、药水、牌组和到达精英前的营火足以承受风险时才选择。
- 营火前是否有关键升级、商店金币是否足够、下一场已知战斗是否需要回复或升级，都应影响路线。
- 低生命时避免连续普通战斗、精英或高风险事件；高生命且牌组尚未成型时，适度战斗和精英可能比安全路线更有价值。
- Act 1 更重视即时伤害，Act 2 更重视群体伤害、可靠防御和成长，Act 3 更重视针对 Boss/精英的补缺和稳定性；这些只是软偏好，必须服从当前牌组和路线。
只能从当前可达的合法节点中选择，不要把未展示的事件结果当作确定事实。""",
    "COMBAT": """【战斗决策】
先计算本回合最危险的敌方意图：结合每个敌人的当前伤害、攻击段数、格挡和状态，估算不处理时的总损失；优先避免死亡和不可接受的生命损失。若某个敌人被击杀后能立即消除主要威胁，击杀价值通常高于分散伤害。
- `combat.monsters[*].knowledge` 是静态怪物参考，包含类型、生命区间和可能招式；它用于理解敌人的典型机制，不代表本回合一定会使用某个招式。当前实时 `intent`、`intent_damage`、`intent_hits`、HP、格挡和 Powers 以 observation 中的游戏状态为准；如果持有符文圆顶导致实时意图隐藏，不要用静态招式猜测当前意图。
- 比较所有出牌、用药水和结束回合动作，而不是只看单张牌的面板伤害。
- 先处理会影响后续计算的易伤/虚弱/力量/格挡等效果，再决定攻击、格挡、抽牌、成长和出牌顺序；以当前 observation 的牌面描述为准。
- 评估能量、当前手牌、敌人数量、目标选择和结束回合后的状态。把 `END_TURN` 当作最后备选：当前仍有能量且存在可支付、具有非负价值的出牌或用药水动作时，先逐一比较这些动作；不要因为推理简化、只关注单张牌或想保留能量而直接结束回合。只有已完成有意的回合规划、没有值得支付的动作，或结束回合有明确更高价值时，才结束回合。
- 在有多种方案都能稳定存活时，优先选择能减少本回合及后续几回合可避免掉血的方案；不要为了省药水、保留能量或追求少量额外输出而白白承受伤害。这里的目标是提高整局生命余量，不是机械地追求每回合零伤害：若需要用关键资源换取击杀、避免更大伤害或应对精英/首领，应果断使用。
- 药水不是只能在濒死时使用：当药水能避免一大段伤害、保证击杀精英/首领、补足当前牌组缺口或显著提高通关概率时应使用；但不要在低风险战斗中无意义消耗。
- 若多个动作都能存活，优先保留后续回合和未来战斗更需要的资源，并考虑当前回合结束后敌人的成长。
必须选择一个当前合法动作；不要输出需要先执行另一动作才能成立的计划。""",
    "COMBAT_CARD_SELECTION": """【战斗内生成选牌】
当前 `selection_context` 表示这个 CARD_REWARD 是刚才使用药水或打出某张牌后生成的战斗内选择，并非普通战后加入永久牌组的奖励。先读取 `selection_context.source` 的完整 description；其中的持续时间、费用变化、加入手牌/弃牌堆/牌组等效果以该描述为准。
- 不要沿用普通卡牌奖励的“牌组变大则跳过”逻辑。优先判断候选在本场战斗的即时伤害、格挡、抽牌、控制、击杀或脱险价值。
- 攻击药水、技能药水、能力药水、无色药水，以及“发现”一类出牌触发的选择都按 source 的描述分别理解；不要仅按候选卡预览中的原始费用或 is_playable 推断它是否会被临时降费。
- `SKIP` 仍是合法动作，但只有所有候选在 source 描述规定的时限内都没有实际价值，或选择会明确带来负面后果时才考虑跳过。
- 严格从当前合法动作中选择，不要假设候选会永久加入牌组，除非 source.description 明确如此。""",
    "CARD_REWARD": """【卡牌奖励决策】
先判断当前牌组最紧迫的缺口：即时单体伤害、群体伤害、可靠格挡、成长、抽牌、能量、状态处理或特定 Boss/精英答案。比较每张候选牌加入后对接下来几场战斗和整局牌组一致性的帮助。
- 早期通常优先能解决当前战斗压力的牌，后期更看重成长、稳定性和与已有遗物/牌组的协同。
- Act 1 前段且牌组仍以起始 Strike/Defend 为主时，稳定的伤害、群体伤害或基础防御通常就是有价值的过渡牌；不要因为尚未形成 archetype 而连续跳过合理的即时补强。只有候选都无法处理近期考验或明显损害牌组时，才跳过。
- 不要强行追求预设 archetype；根据已经获得的牌、遗物和路线调整方向。
- 牌组变大有抽不到关键牌的机会成本。若候选牌不能明确改善当前问题，跳过通常优于无条件加入。
- 读取候选卡的完整 description、cost、type 和 upgrades；不要仅按名字、稀有度或攻略 tier 选择。
- 如果一张牌很强但在当前路线和牌组中没有实际用途，也要考虑跳过；如果一张普通牌正好补足当前致命缺口，可以优先选择。
如果候选奖励尚未展示具体卡牌内容，不要臆测其中的牌。""",
    "COMBAT_REWARD": """【战斗奖励决策】
比较当前合法的金币、卡牌奖励、药水、遗物和继续动作。奖励价值取决于当前生命、金币、药水栏、牌组缺口、下一段路线和即将到来的精英/首领，而不是固定优先级。
- 金币可以保留给已知商店，但若当前牌组急需解决战斗缺口，能立即提高生存的奖励可能更重要。
- 当屏幕上仍有金币、返还的被偷金币、遗物或其他无条件正收益奖励时，不要为了简化流程而选择 `PROCEED`；通常先领取这些奖励，再决定是否跳过可选卡牌或离开。药水栏已满、奖励附带代价或当前合法动作明确表示放弃时，才重新比较取舍。
- 卡牌奖励遵循“是否解决当前问题、是否优于跳过、是否增加不必要的牌组负担”的标准。
- 药水栏已满时不要选择会卡住流程的药水动作；药水有明确的战斗价值时不应机械囤积。
- 不要假设尚未展开的卡牌、遗物或药水内容；只根据当前 observation 和合法动作决定。""",
    "GRID_PURGE": """【删牌决策】
这是一个不可逆的删牌动作。逐张比较候选牌的完整 description、upgrades、cost、type、当前牌组数量和当前构筑需求。
- 通常优先考虑基础 Strike/Defend、重复且低价值的牌、诅咒或会持续污染牌组的牌，但这不是绝对规则。
- 不要因为牌名看起来强、稀有或已升级就自动保留，也不要因为牌名普通就直接删除；判断它是否是当前构筑的核心、唯一的伤害/防御/成长答案或与遗物的关键协同。
- 稀有牌、关键成长牌、唯一的群体伤害/防御/抽牌工具通常要谨慎删除；只有当前牌组明确不需要且删除能改善一致性时才考虑。
- 先确认当前界面确实要求 PURGE，再选择一张牌；不要把删牌动作和其他选牌目的混淆。""",
    "GRID_UPGRADE": """【升级选牌决策】
比较候选牌升级后对未来多场战斗的总价值，而不是只看稀有度。优先升级能频繁抽到、直接补当前缺口、显著改变关键回合或针对即将到来的精英/首领的牌。
考虑升级后的完整 description、cost、upgrades，以及牌在当前牌组中的数量和抽到频率。不要为了消耗营火而升级对当前路线没有作用的牌。""",
    "GRID_TRANSFORM": """【变化选牌决策】
这是对当前牌的不可逆变化。优先考虑重复、低价值、妨碍牌组一致性的牌，同时保留当前构筑中唯一的关键伤害、防御、成长、抽牌或协同组件。
变化结果通常不是已知确定值，不要假设会得到某张理想牌；在不确定性下选择最值得移出当前牌组的候选。""",
    "GRID_SELECT": """【选牌/目标选择决策】
严格按照当前 selection_instruction、num_cards、selected_cards 和合法动作理解选择目的。只选择能完成当前界面目标的牌，不要把普通选牌、删牌、升级、变化或事件中的待选牌混为一谈。
如果可以少选或跳过，只有在合法动作明确允许且不会违背当前目标时才这样做。""",
    "RELIC_PICKUP_CARD_SELECTION": """【遗物触发选牌】
当前 `selection_context` 明确说明这是刚获得某件遗物后立刻触发的选牌，不是普通卡牌奖励或事件选牌。必须遵守其中的 `source`、`selection_instruction`、`required_cards` 和 `selected_count`：只在合法动作中选择能完成该遗物效果的牌。
- 瓶装遗物应选择最适合稳定起手的对应类型核心牌；多莉的镜子应复制构筑核心；奥雷里的连续奖励要逐组判断拿牌或跳过。
- 已选卡已被控制器从合法动作中移除；继续在剩余候选中完成本次遗物要求，不要把这次效果误当作普通删牌、升级或卡牌奖励。""",
    "SHOP_SCREEN": """【商店决策】
先按当前金币和价格判断真正买得起的选项，再比较购买后的即时收益、未来收益和机会成本。商店购买与删牌通常不可逆，不要因为有余额就消费。
- 优先处理当前牌组最紧迫的战斗缺口，或购买能与已有牌/遗物形成明确协同的物品。
- 删除基础牌能提高一致性，但若牌组仍缺即时伤害/防御，先解决会导致近期死亡的问题。
- 遗物是长期收益，药水是短期资源；结合下一条路线、精英/首领准备和药水栏判断。
- 购买后保留的金币也有价值，尤其是后续仍有商店或关键事件时；但不要为了假想的未来而错过当前能显著提高胜率的物品。
- 读取每个商品的 description、price 和当前牌组/遗物协同，不要按稀有度或名称孤立选择。""",
    "EVENT": """【事件决策】
以事件的 body_text、选项文本、当前生命、金币、牌组、遗物、药水和路线为依据。权衡立即收益、长期收益、生命/金币成本、诅咒/负面效果和后续可恢复性。
- 不要把所有事件都当成固定攻略答案；同一选项在不同 Act、牌组和生命状态下价值不同。
- 高生命和有恢复/移除手段时可以承受更高风险；低生命、关键战斗临近或牌组已稳定时应更保守。
- 若选项可能产生未知随机结果，不要假设最理想结果；优先选择在多种结果下都可接受的方案。
- `event_history` 是当前事件的按时间顺序状态转移记录；当当前 body_text 为空、选项标签变化或事件已进入小游戏/二级页面时，先从其中恢复事件目标、已选项及其结果，再做决定。
- 对翻牌、配对、递进选择等多步事件，记录已翻开的位置与牌面、已经证伪的组合和仍未探索的位置。不要反复选择已知无效的同一位置/组合；若没有已知可完成的组合，优先探索尚未翻开的合法位置。
- 注意 recent_actions，避免在事件已进入下一步选择时重复执行上一层动作。""",
    "REST": """【营火决策】
在回复生命和升级之间比较：当前生命距离死亡的余量、下一场/精英/首领的威胁、是否有关键升级、遗物提供的回复或升级效果，以及本 Act 剩余路线。
- 若不回复就可能在下一场战斗死亡或无法承受精英，应优先回复。
- 若当前足够安全且某张牌升级能显著改善多个未来战斗，升级通常比小幅回复更有价值。
- 不要机械地总是升级，也不要在生命已经安全时只因为习惯而回复。""",
    "BOSS_REWARD": """【首领遗物决策】
比较每个遗物的长期收益、当前牌组能否利用、代价是否可承受，以及下一 Act 的主要威胁。额外能量类遗物不一定总是最佳，负面代价必须结合当前牌组评估。
- 优先选择能补足当前牌组长期瓶颈、提高稳定性或明确配合已有核心的遗物。
- 不要仅按遗物 tier、名字或通用榜单选择；当前牌组和资源状态优先。
- 如果某个遗物的机制会改变抽牌、费用、生命、药水或商店策略，要把这些长期影响纳入比较。""",
    "TRANSITION": """【过渡状态】
当前状态可能是房间尚未完成时的短暂过渡。不要尝试提前选择地图节点或重复上一房间动作；遵守当前合法动作，选择能安全等待状态稳定的动作。""",
    "GENERIC": """【通用场景】
优先解决当前明确的生存和流程目标，再考虑长期收益。严格读取当前 screen_type、screen_state、recent_actions 和合法动作，不要假设隐藏内容或重复已经完成的动作。""",
}


def character_key(value: Any) -> str | None:
    """Normalize controller and game-state character identifiers."""

    normalized = str(value or "").strip().upper().replace(" ", "_")
    return CHARACTER_ALIASES.get(normalized)


@lru_cache(maxsize=len(CHARACTER_KEYS))
def character_prompt(character: str) -> str:
    """Read the tracked, stable strategy library for one StS character."""

    normalized = character_key(character)
    if normalized is None:
        return ""
    path = Path(__file__).resolve().parents[1] / "prompts" / "characters" / normalized / "base.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text


def character_key_for_observation(observation: dict[str, Any], fallback: Any = None) -> str | None:
    """Prefer the public game-state class, with run metadata as a fallback."""

    return character_key(observation.get("class")) or character_key(fallback)


def scenario_key(observation: dict[str, Any]) -> str:
    """Map normalized public state to a specific teacher strategy prompt."""
    screen = str(observation.get("screen_type") or "").upper()
    if screen == "NONE" and str(observation.get("room_phase") or "").upper() == "COMBAT":
        return "COMBAT"
    if (
        screen == "CARD_REWARD"
        and isinstance(observation.get("selection_context"), dict)
        and observation["selection_context"].get("kind") == "COMBAT_GENERATED_CARD_CHOICE"
    ):
        return "COMBAT_CARD_SELECTION"
    if (
        screen == "CARD_REWARD"
        and isinstance(observation.get("selection_context"), dict)
        and observation["selection_context"].get("kind") == "RELIC_PICKUP_CARD_SELECTION"
    ):
        return "RELIC_PICKUP_CARD_SELECTION"
    if screen == "GRID":
        mode = str((observation.get("screen_state") or {}).get("selection_mode") or "").upper()
        specific = {
            "PURGE": "GRID_PURGE",
            "UPGRADE": "GRID_UPGRADE",
            "TRANSFORM": "GRID_TRANSFORM",
            "TRANSFORM_UPGRADE": "GRID_TRANSFORM",
        }.get(mode)
        if specific is not None:
            return specific
        if (
            isinstance(observation.get("selection_context"), dict)
            and observation["selection_context"].get("kind") == "RELIC_PICKUP_CARD_SELECTION"
        ):
            return "RELIC_PICKUP_CARD_SELECTION"
        return "GRID_SELECT"
    return screen if screen in SCENARIO_PROMPTS else "GENERIC"


def build_system_prompt(
    observation: dict[str, Any],
    *,
    character: Any = None,
    persona: str = "teacher",
    role_skill: str | None = None,
    scene_skill: str | None = None,
    allow_knowledge_request: bool = False,
    knowledge_request_style: str = "encouraged",
) -> str:
    """Return the Chinese action prompt, optionally using versioned skills.

    Prompt Evolution passes the active role and scene skill explicitly.  A
    ``None`` value keeps the normal tracked-library behaviour for the other
    runtimes, while an empty string intentionally omits that skill.
    """

    key = scenario_key(observation)
    role = character_key_for_observation(observation, character)
    role_prompt = character_prompt(role) if role_skill is None and role else str(role_skill or "").strip()
    common_prompt = PLAYER_COMMON_PROMPT if str(persona).strip().casefold() == "player" else COMMON_PROMPT
    parts = [common_prompt]
    if role_prompt:
        parts.append(role_prompt)
    scenario_prompt = SCENARIO_PROMPTS[key] if scene_skill is None else str(scene_skill or "").strip()
    if scenario_prompt:
        parts.append(scenario_prompt)
    if allow_knowledge_request:
        # Keep the alternative response protocol after every scene skill. Some
        # skills correctly insist on a legal action in the normal path; this
        # final contract makes clear that a knowledge request is the exception.
        parts.append(knowledge_request_rule(knowledge_request_style))
        parts.append(KNOWLEDGE_RESPONSE_CONTRACT)
    return "\n\n".join(parts)
