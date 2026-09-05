# Act Boss 应对画像

`act_boss_profile.strategy` 会在已知当前 Act Boss 时全局提供给动作模型。它是用于路线、选牌、商店、营火、药水保留和战斗资源分配的规划画像，而不是逐回合的行动说明。运行时文本以 [`tools/monster_knowledge.py`](../tools/monster_knowledge.py) 的 `BOSS_STRATEGIES` 为准。

内容根据灰机 Wiki 的[首领图鉴](https://sts.huijiwiki.com/wiki/%E9%A6%96%E9%A2%86)及各 Boss 条目的“策略”或“应对策略”整理。建议本身不是硬规则，必须结合当前卡组、HP、遗物、药水、路线和实时战斗意图判断。

逐回合行动规律、伤害数值、意图、HP 和 Powers 不放在这个全局字段中。Boss 进入实际战斗后，`combat.monsters[].knowledge` 才会给出静态招式知识，CommunicationMod 的实时字段仍是当前局面的唯一事实来源。

| Boss | Boss 前的核心准备与策略侧重 | 来源 |
| --- | --- | --- |
| The Guardian | 重视攻防均衡：补齐稳定格挡、虚弱或力量削减，以及可集中打出的爆发伤害。攻击形态需要能稳定接住高压回合；防御形态则避免低价值小攻击被锋利外甲惩罚，并利用爆发跨过形态转换阈值。 | [守护者](https://sts.huijiwiki.com/wiki/%E5%AE%88%E6%8A%A4%E8%80%85) |
| Slime Boss | 重点寻找爆发伤害、力量/易伤、能量和爆发药水。进入 Boss 后将其血量停在接近分裂阈值，利用准备回合铺垫，再一回合压过阈值，能降低分裂后的压力。 | [史莱姆老大](https://sts.huijiwiki.com/wiki/%E5%8F%B2%E8%8E%B1%E5%A7%86%E8%80%81%E5%A4%A7) |
| Hexaghost | 同时需要长线输出、过牌和可靠防御，并尽快结束避免灼伤污染牌组。若输出足够，Boss 前可保持较低生命值，并在营火优先锻造；减力量、荆棘和反伤对多段攻击很有价值。 | [六火亡魂](https://sts.huijiwiki.com/wiki/%E5%85%AD%E7%81%AB%E4%BA%A1%E9%AD%82) |
| The Champ | 补齐输出上限、稳定防御、过牌和高效减益。应规划半血转阶段，不在防御不足或核心牌未启动时轻易压过阈值，并将爆发和资源留给强化阶段。 | [第一勇士](https://sts.huijiwiki.com/wiki/%E7%AC%AC%E4%B8%80%E5%8B%87%E5%A3%AB) |
| The Collector | 需要群攻或有效集火，并能在小怪、强化和究极减益后维持防御。具体先击杀 Boss 还是小怪取决于实时意图、群攻和牌组，不能固定目标。 | [收藏家](https://sts.huijiwiki.com/wiki/%E6%94%B6%E8%97%8F%E5%AE%B6) |
| Bronze Automaton | 准备清理铜球的手段、单体与群体输出，并能在超能光束前建立防御或完成击杀。要考虑高价值牌被凝滞的节奏损失，利用低压回合处理铜球。 | [铜制机械人偶](https://sts.huijiwiki.com/wiki/%E9%93%9C%E5%88%B6%E6%9C%BA%E6%A2%B0%E4%BA%BA%E5%81%B6) |
| Awakened One | 权衡能力牌的即时收益和给 Boss 加力量的代价；为双阶段保留持续输出、爆发、过牌和防御，避免第一阶段耗尽药水或核心资源。 | [觉醒者](https://sts.huijiwiki.com/wiki/%E8%A7%89%E9%86%92%E8%80%85) |
| Donu / Deca | 这是会随时间升级的双 Boss 战：通常优先压制 Donu 的力量成长，同时准备双目标攻防、群攻或多目标收益、稳定格挡和持续伤害；是否改为处理 Deca 取决于晕眩和格挡对当前卡组的威胁。 | [八体＆甜圈](https://sts.huijiwiki.com/wiki/%E5%85%AB%E4%BD%93%EF%BC%86%E7%94%9C%E5%9C%88) |
| Time Eater | 需要高效牌、过牌和可控爆发；将 12 张牌计数视为资源，临近阈值时为下一回合留防御或爆发，并在可以继续压制时处理半血回复窗口。 | [时间吞噬者](https://sts.huijiwiki.com/wiki/%E6%97%B6%E9%97%B4%E5%90%9E%E5%99%AC%E8%80%85) |
| Corrupt Heart | 需要完整长线构筑：以稳定格挡或免伤应对多段和重击，以持续输出、减力量或反伤降低多段压力，并为死亡律动和伤害上限规划药水、关键牌及整局资源。 | [腐化之心](https://sts.huijiwiki.com/wiki/%E8%85%90%E5%8C%96%E4%B9%8B%E5%BF%83) |

若某个版本的 CommunicationMod 没有识别到 Boss，`act_boss_profile` 会缺省；实时战斗状态仍可正常使用。
