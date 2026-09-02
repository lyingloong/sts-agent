# 通信协议与数据格式

## CommunicationMod 配置

CommunicationMod 启动外部进程，进程 stdout 只能用于协议。控制器启动时发送 `ready`，随后接收 JSON 状态并输出协议命令；诊断写入 stderr 和项目日志。

```properties
command=D\:/miniconda3/python.exe D\:/Code/spire-mod/sts-agent/tools/communication_controller.py
runAtGameStart=true
```

协议支持的关键命令包括 `START`、`PLAY`、`END`、`POTION`、`CHOOSE`、`PROCEED`、`RETURN` 和 `STATE`。详细定义以 [CommunicationMod README](https://github.com/ForgottenArbiter/CommunicationMod) 为准。

## START 命令

控制器把终端输入转换为 CommunicationMod 命令：

```text
s                         -> START IRONCLAD 0
s ABC123                  -> START IRONCLAD 0 ABC123
s SILENT 5 ABC123        -> START THE_SILENT 5 ABC123
s DEFECT 10              -> START DEFECT 10
s WATCHER 20 ABC123      -> START WATCHER 20 ABC123
```

角色别名：`IRONCLAD`、`SILENT`/`THE_SILENT`、`DEFECT`、`WATCHER`。进阶范围为 0-20；CommunicationMod 的 `START` 种子必须是字母数字。游戏运行状态中的 `game_state.seed` 可能带前导 `-`，但该值不能直接作为 `START` 参数。旧版 `s [seed]` 兼容为 Ironclad A0。

## 轨迹记录

每局有独立的 `run-events` 和 `action-trace` 文件，文件名包含 session、局序号、角色、进阶和种子，例如：

```text
action-trace-20260823-120000-run-002-THE_SILENT-A5-seed-4QTKQ7VBMFX47.jsonl
run-events-20260823-120000-run-002-THE_SILENT-A5-seed-4QTKQ7VBMFX47.jsonl
```

`run-events-*.jsonl` 记录单局生命周期，例如：

```json
{"event":"run_start","run_id":"...","run_config":{"character":"THE_SILENT","ascension":5,"seed":"ABC123","seed_source":"requested","internal_seed":null},"command":"START THE_SILENT 5 ABC123"}
```

未指定种子时，控制器先写 `seed_source: "random"`，收到第一条游戏运行状态后，从 CommunicationMod 的 `game_state.seed` 记录内部数值，并按 StS1 的 `0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ` 编码（省略 `O`）还原显示种子。`run_seed_observed` 事件同时包含 `display_seed` 和 `internal_seed`，后续轨迹的 `run_config.seed` 使用显示种子，`run_config.internal_seed` 保留协议原始值。种子字段用于复现实验；它们不会进入模型的规范化 observation。

每个动作的单局 `action-trace-*.jsonl` 至少包含：

```json
{
  "schema_version": "0.1",
  "session_id": "...",
  "run_id": "...",
  "run_config": {"character":"IRONCLAD","ascension":0,"seed":"ABC123"},
  "step_id": 12,
  "pre_state": {},
  "legal_actions": [],
  "action": {},
  "post_state": {},
  "accepted": true,
  "stable_after_action": true,
  "terminal": false
}
```

在线纠正式采集时，轨迹可升级为 `schema_version: "0.2"` 并增加可选的 `decision` 字段。该字段同时保存 `model_action`、top-k `model_candidates`、`candidate_source`、`executed_action`、`source`（模型接受/人类覆盖/fallback）和 `overridden`，用于区分模型建议与人类最终标签。零样本 Qwen 第一版只生成给定 `legal_actions` 中的 action ID JSON 列表，controller 会再次过滤；后续的置信度路由会以可选 `action_scores` 和 `confidence` 字段保存受约束的全动作打分及自动/人工路由结果。完整字段、训练集导出和交互流程见[在线纠正式模仿学习方案](ONLINE_DAGGER.md)。旧的 `action` 字段继续保留，以兼容现有回放和清洗脚本。

`pre_state`/`post_state` 是规范化状态，去除抽牌顺序、UUID 和其他不应作为玩家观察输入的字段。`messages-*.jsonl` 仍按 controller session 保存，只用于调试和重新清洗。随机种子未指定时，轨迹文件先使用 `seed-random`，收到内部 `game_state.seed` 后还原出显示种子并自动改名。

`START` 的种子参数仍必须是字母数字。`game_state.seed` 的负号只是 Java 有符号 64 位内部表示；控制器不会把这个数值直接发送给 `START`，而是把它保存为 `internal_seed` 并解码成可输入的显示种子。

规范化状态中的两个重要字段：

- `map.nodes`：当前 Act 已公开的完整地图节点及其 `x/y/symbol/children/parents` 关系；`map.current_node` 和 `map.next_nodes` 仍保留作快捷字段。
- `combat.monsters[*].intent_damage`、`intent_hits`：从 CommunicationMod 的当前意图字段映射而来，用于表达敌人当前显示的攻击伤害和段数。
- `act_boss` / `act_boss_profile`：当前 Act 地图公开的首领名称，以及从版本化怪物知识库提取的简洁血量和招式预览；它们会保留在地图外的策略观察中，供路线、拿牌和战斗资源规划使用。
- `decision_constraints`：当前遗物带来的长期规则，例如苏祖禁止获得药水、灵体外质禁止获得金币、添水的本回合出牌余量，以及营火/宝箱限制。能由状态确定为无效的药水或金币奖励会直接从 `legal_actions` 移除；其余限制仅作为显式上下文保留。持有符文圆顶时，敌人当前意图字段也会从模型观察和终端视图隐藏。
- `combat.player.powers`、`combat.monsters[*].powers`：保留 CommunicationMod 当前的 `id/name/amount/type`，并补充静态 `description`。description 优先来自 `sts1_powers.json`；缺失时才查询 `sts1_card_powers.json`，不使用关键词库。
- `combat.hand[*]`：使用扁平卡牌 schema：`id/name/type/cost/description/upgrades/is_playable`。运行时非空字段优先；静态知识只补缺失字段，不再输出嵌套 `effect`。当前 CommunicationMod 版本未稳定提供实时 `damage/block/magic_number`，因此不把它们纳入训练 schema。
- `recent_actions`：模型观察和轨迹状态中保留当前局最近 6 个已被 CommunicationMod 确认且状态稳定的动作，包含执行前后界面、动作类型、规范 `action_id` 和简短语义。协议错误或尚未确认的动作不会进入历史；新局从空历史开始。
- `screen_type=GRID`：若当前是事件/营火等触发的选牌界面，`screen_state.selection_mode` 和 `selection_instruction` 会明确表示 `PURGE`（移除）、`TRANSFORM`（变化）或 `UPGRADE`（升级）等目的。`for_purge` 等原始布尔字段仍保留作兼容。
- `screen_type=TRANSITION`：CommunicationMod 返回地图界面但 `room_phase` 仍不是 `COMPLETE` 时的保护态。该状态不包含可训练的地图选择动作，只允许等待刷新或返回当前房间。
- `screen_name=SETTINGS` 且 `is_screen_up=true`：表示设置/暂停覆盖层挡住了当前房间。此时 CommunicationMod 通常只暴露 `KEY`、`WAIT`、`STATE` 等通用命令，controller 将其显示为 `KEY:CANCEL`，终端输入 `x` 会发送 `KEY Cancel`；关闭覆盖层后才恢复战斗动作。
- 药水栏已满或持有苏祖时，controller 不会把 `COMBAT_REWARD` 中的 `POTION` 或商店中已识别的药水商品列为合法动作。持有灵体外质时，`COMBAT_REWARD` 中的金币也会被移除。CommunicationMod 对这类操作可能不返回错误或状态变化；终端会说明对应原因。
- `COMBAT_REWARD` 在请求模型前按“遗物、金币、药水、卡牌奖励入口”的顺序自动推进：遗物和金币必取；药水仅在有空位时必取；卡牌奖励只自动打开，选牌或跳过仍交给模型。首次进入 `SHOP_ROOM` 也会自动打开商人界面，购买和移除仍是模型决策。每个这类动作都以 `decision.source=environment_auto` 正常写入 trace 和 recent_actions，但不会进入教师 SFT 或在线 RL 的策略更新；`BOSS_REWARD` 的遗物选择始终保留为策略决策。
- 启用 `posthoc_feedback.enabled` 且远端 `attribution.enabled` 后，在线 RL 的每个终局会额外产生本地 `data/reflections/` 与 `data/attributions/` artifact。`posthoc_feedback.mode="human"` 时前者保存人类的可强化/应弱化复盘；`mode="auto"` 时保存自动复盘标记，教师根据账本独立给出双向候选。后者保存本地校验过的 `step_id/direction/confidence/severity`；只有后者的已应用字段进入远端 episode。教师原始 completion 与思考不写入 artifact 或更新协议。详情见[终局反思归因 GRPO](POSTHOC_ATTRIBUTION_GRPO.md)。

Candidate Assist 由 controller 在 CommunicationMod 启动握手前预热独立的 `tools/policy_server.py`。模型、端口、超时和模式全部由 `config/agent.json` 管理；模型必须先通过 `tools/download_qwen.py` 从 `policy.model.hub_endpoint` 预下载。脚本会在没有显式代理变量时读取 Windows 系统代理。游戏运行时 policy server 只从本地缓存加载，以免网络问题阻塞回合。默认对所有游戏内场景请求候选；可将 `policy.request.screens` 配置为 `["COMBAT"]` 收窄。`MAP` 才发送完整公开 `map.nodes`、当前位置和可选节点；战斗、事件、商店、奖励、休息和地图过渡态都移除地图。controller 维护显式环境记忆 `run_memory.visited_shop_rooms`。首次观察到 `SHOP_SCREEN` 时按 `(Act, Floor)` 记录该商店；之后同层 `SHOP_ROOM` 从合法动作中移除 `CHOOSE shop`，只保留 `PROCEED` 等离开动作，人工输入也不能绕过。它把其余真实合法动作映射为本请求内短别名 `A1/A2/...`，policy server 返回至多三个别名，controller 再映射、校验并编译 CommunicationMod 命令。这避免小模型复制长的卡牌/目标 ID 时产生空候选。终端 `a` 接受首选，`pick <序号>` 选择其他候选；裸数字不被候选逻辑截获，保留给游戏界面自己的选项。模型无候选、超时或加载失败时，`decision` 会记录 `assist_status` 和 `assist_reason`，终端显示原因后退回人工操作。将 `policy.enabled` 设为 `false` 可关闭，`policy.development.mock=true` 可在不加载模型时测试协议。

设置 `policy.confidence_auto.enabled=true` 后，controller 不再请求生成式 top-k 排序，而请求 policy server 为每个当前合法别名计算精确续写 `A<n>\\n` 的条件 `logprob`。controller 仅在这一个合法集合内 softmax，完整 `action_scores` 按规范 action ID 写入 trace；`confidence` 保存首选/次选概率、margin、entropy、`min_probability` 和路由结果。默认 `min_probability=0.85`；首选满足 `p1 >= min_probability` 即执行 `source=model_auto`。margin 与 entropy 用于校准记录，不作为第二道门槛。评分缺项、格式异常、超时或上一动作返回协议错误时，自动执行被禁用并回退人工，不能绕过 controller 的合法动作表。

当前已增加本地静态知识库 fallback：卡牌统一为 `id/name/type/cost/description/upgrades/is_playable`，药水统一为 `id/name/description/target/can_use`，遗物统一为 `id/name/description`，并只在 `counter >= 0` 时附加遗物计数。运行时非空字段覆盖静态值；静态知识只用于补全缺失字段，不再进入嵌套 `effect`。商店商品保留额外的 `price`。对事件选项与普通奖励，controller 只补全 CommunicationMod 已附带具体物品 `id/name` 的实体；裸 `reward_type` 不会泄漏尚未公开的随机奖励。战斗卡牌奖励被跳过后，controller 以 `(Act, Floor)` 记录到 `run_memory.declined_card_rewards` 并从父 `COMBAT_REWARD` 动作表移除同一张卡牌奖励入口，防止协议导航造成重开同一奖励的循环。静态知识库由 `tools/fetch_card_knowledge.py` 从 [spire-archive StS1 数据](https://github.com/nkhoit/spire-archive)生成，`data/knowledge/` 是受 Git 跟踪的版本化快照；上游来源、文件清单和本项目的显式文本修复均记录在该目录的 `manifest.json` 与 `README.md`。刷新快照后必须审查 Git diff，运行时 CommunicationMod 字段始终优先。

## 数据使用规则

- 只把动作已接受、状态稳定的 step 放入正式纠正式数据集；`executed_action` 是监督标签，`model_action` 用于统计错误和 DAgger 重采样。
- 中途退出、错误响应、状态不稳定的 run 保留在 raw 目录，标记为调试数据。
- train/validation/test 按 `run_id` 或 `seed` 切分，不能按动作随机切分。
- 不把 raw JSON 直接喂给模型；先经过版本化 normalizer 和 action compiler。
