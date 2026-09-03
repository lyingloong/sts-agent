# 场景化动作提示词

更新时间：2026-08-25

## 目的

远程教师和 Prompt 演化动作模型共用 `tools/teacher_prompts.py` 中的中文场景策略。教师 SFT 请求使用“远程教师”身份；Prompt 演化的实际动作请求使用“正在游玩的玩家”身份，并在末尾追加当前角色已接受的 skill。提示词只进入远程请求，不进入本地 Qwen 的 observation、SFT target 或 GRPO 输入；动作模型仍然只能从 controller 提供的合法动作中选择一个 `A<n>`。

教师 SFT 当前版本：`sts1-teacher-zh-v5`；Prompt 演化动作版本：`sts1-player-zh-v1`。

## 场景路由

| 状态 | 策略框架 |
| --- | --- |
| `MAP` | 全图路线、未来 2-4 个节点、精英/营火/商店风险收益 |
| `NONE + room_phase=COMBAT` | 计算来袭伤害、击杀威胁、出牌顺序、药水时机 |
| `CARD_REWARD` | 牌组缺口、下一项考验、加入与跳过的机会成本；Act 1 的过渡牌价值 |
| `CARD_REWARD + selection_context` | 药水或已打出卡牌生成的战斗内选牌，读取来源物品描述，不按永久奖励处理 |
| `COMBAT_REWARD` | 金币、卡牌、药水、遗物和继续动作的整局价值；未领取无条件收益时通常不离开 |
| `GRID/PURGE` | 不可逆删牌、基础牌与核心牌的区分 |
| `GRID/UPGRADE` | 升级对未来多场战斗的频率和影响 |
| `GRID/TRANSFORM` | 在未知结果下移除最弱候选 |
| `SHOP_SCREEN` | 价格、即时缺口、长期协同、保留金币 |
| `EVENT` | 选项文本、生命/金币成本、未知结果风险，以及当前事件的状态转移历史 |
| `REST` | 回复与升级的生存价值比较 |
| `BOSS_REWARD` | 长期遗物收益、牌组兼容性和负面代价 |

## 设计依据

这些提示词不是卡牌 tier list，也不是固定路线脚本。它们抽取了 StS 攻略中较稳定的原则：先根据当前牌组解决下一项考验，地图提前看路线并安排营火，精英奖励与生命风险一起评估，牌组奖励允许跳过，商店购买考虑机会成本，删牌和升级读取具体机制而不是只看稀有度。可参考 [Spire Codex 新手指南](https://github.com/ptrlrd/spire-codex/blob/main/data/guides/new-players-guide.md)、[Slay the Spire Strategy Guide](https://www.slaythespire.gg/guides/strategy) 和 [地图风险评估指南](https://www.ludo.guide/guide/slay-the-spire/pathing-risk-assessment/map-generation-and-branching)。

近期 StS LLM agent 工作也采用按 combat、map、shop、event 等上下文切换专用 prompt，而不是一个全局 prompt；相关实验报告指出，专用上下文提示可以被视作内部工具，且 LLM 更擅长长期策略概念，精确短期战术更容易出错。因此本项目把战斗提示写成“先处理致命威胁”的检查流程，把场外提示写成“当前缺口 + 下一项考验 + 不可逆后果”的比较流程。参考 [How LLMs Slay the Spire 2](https://openreview.net/pdf/521037328af4df7687b86343bc0ed0bddcc441fb.pdf) 和 [The Spire Agent](https://github.com/HZhangM/The-Spire-Agent)。

## 不写成硬规则的内容

- “Act 1 必须打两个精英”只能作为软偏好，必须服从生命、牌组、药水和路线。
- “稀有牌一定好”“商店一定删 Strike”“营火一定升级”都不能作为固定策略。
- 未展示的事件结果、卡牌奖励内容、抽牌顺序和敌人意图不能由提示词臆测。
- 提示词不替 controller 做合法性判断，也不生成游戏命令。
- 金币、药水、遗物和 `SKIP`/`PROCEED` 都保留为合法动作；提示词只表达通常的价值判断，不通过 controller 硬编码策略。

## 迭代方法

每次只改一个场景的 prompt，并在已有轨迹中抽取关键状态进行回放比较。优先评测：

1. 有可用药水且存在高来袭伤害时是否使用药水。
2. `PURGE` 是否保留稀有/升级/唯一功能牌，优先删除低价值基础牌。
3. 卡牌奖励是否能根据牌组缺口选择或跳过。
4. 商店是否同时考虑购买价格、牌组缺口和药水栏。
5. 地图是否查看后续路径，而不是只选择眼前节点。
6. 攻击/技能/能力/无色药水和“发现”等效果生成的选牌，是否依据来源描述而非按永久奖励跳过。
7. 多步事件是否根据 `event_history` 避免重复翻开已知无效位置。
