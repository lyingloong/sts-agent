# 细粒度语义动作

`semantic_actions()` 默认 `fine_grained=True`：模型选择一次游戏动作，控制器执行一次
对应命令。营火、商店和战斗内选牌不会被自动合并为一个宏动作。

## 动作与训练答案

- 出牌：`PLAY:card=<1起手牌序号>:target=<0起敌人序号>`；无目标时省略 target。
- 结束回合：`END_TURN`。
- 药水：`POTION_USE:slot=...`、`POTION_DISCARD:slot=...`。
- 地图、事件、奖励、营火入口、商店商品、选牌：`CHOOSE:index=<0起序号>`。
- 确认、继续、返回、跳过：`CONFIRM / PROCEED / RETURN / CANCEL / SKIP / LEAVE`。

`id` 是当前观察中的动作身份，`command` 是 CommunicationMod 执行协议。模型 SFT 使用
当前列表的 A1…An 别名，不要求模型生成命令、UUID 或鼠标坐标；别名不能跨状态重用。

营火：选择锻造 → 选择卡牌 → 确认 → 继续。
商店：打开删牌 → 选择卡牌 → 确认（仅当游戏实际要求确认）。

是否出现确认由当前游戏状态决定，不假定所有选择都有相同按钮流程。

## 原始记录与策略过滤

ActionRecorder 协议0.5保存独立的细粒度动作事务及完整原始状态。其通用数据集保留
取消、返回、拒绝与状态缺失，不做训练样本筛选。原始 available_commands/choice_list
不是本模块枚举出的模型合法动作。

记录还原显式使用 `include_recorded_choices=True`，保留人类真实操作中可能出现的
重新打开奖励、取消已选牌等；运行时仍保留苏祖、灵体外质、重复选择、商店重入等策略过滤。
细粒度开关不再顺带关闭这些过滤。地图不提供药水动作。

SFT 导出使用还原后的 training_* 字段并应用训练步骤筛选。排队的出牌/结束回合用
精确执行前状态；无法验证的尝试保留在审计轨迹，不伪造为有效样本。
完整流程见 [HUMAN_SFT_DATASET.md](HUMAN_SFT_DATASET.md)。

## 自动动作与历史宏动作

controller 现有环境自动化不变：自动动作仍按实际动作记录，不把多步选择合并。
地图只有一个合法目的地时可以自动前进，判断以完整合法动作列表为准（包含飞鞋效果）。

历史 `CAMPFIRE:SMITH:card=...`、`SHOP:PURGE:card=...` 和展开执行辅助代码仍存在，
仅显式 `fine_grained=False` 的调用会生成宏动作。新的人类 SFT 及默认运行时不使用它们。
