# 在线纠正式模仿学习方案

更新时间：2026-08-23

## 1. 目标与边界

目标是减少“人类完整手动游玩”的数据采集成本：模型在每个决策点先提出一个合法动作，人类确认或改选，游戏继续运行，系统同时保存模型建议和人类最终动作。之后用这些访问到的状态继续训练模型。

这条路线是 DAgger（Dataset Aggregation）风格的在线纠正式模仿学习，不是完全无监督强化学习，也不要求一开始就做完整的 SFT。人类最终执行的动作仍然是监督标签，但只需要处理模型暴露出的困难状态；模型不会在一局游戏中更新权重。

第一版范围：

- 只使用 CommunicationMod 的结构化状态，不加入视觉输入。
- 先使用一个 Qwen 小骨干，按 `screen_type` 做高层/战斗路由，不同时驻留多个完整模型。
- 先训练离散动作选择器，不做 token 级 PPO 或在线 policy gradient。
- 训练在局与局之间进行；游戏运行时只加载冻结的推理模型。

## 2. 为什么选择在线纠正

StS1 每局包含数百个动作，胜负奖励非常稀疏，且一旦早期路线或战斗动作出错，后续状态会快速偏离人类轨迹。完全跳过冷启动直接做 RL 会同时遇到探索、长时信用分配和样本效率问题，在单机单卡上很难稳定。

在线纠正保留了人类的强先验，又把人类时间集中到模型最容易犯错的状态：

```text
模型观察状态 -> 提议合法动作 -> 人类接受/纠正 -> 游戏执行
       ^                                  |
       |---------- 记录状态与纠正标签 -----|
```

人类评分可以作为附加信号，但不应替代纠正动作。建议战斗中优先要求“改选具体动作”，地图、事件、商店等长程选择再允许 1-5 分评分或简短原因。

## 3. 三种运行模式

### 3.1 Shadow：只观察，不影响游戏

人类照常输入，模型并行读取同一个 `pre_state` 并给出建议。记录 top-1/top-k、合法性、延迟和人类动作，但不把建议发送给游戏。

用途是验证状态序列、动作编译和模型延迟。建议先跑 5-10 局，不产生交互负担。

### 3.2 Candidate Assist：模型给出多个候选，人类选择（当前实现）

这是更推荐的第一版交互。模型给出至多三个 action ID 组成的 JSON 排序列表，controller 只保留当前 `legal_actions` 中的候选，终端显示 top-3。这样可以统计模型对专家动作的 `recall@k`，且模型无法直接发出 CommunicationMod 命令。

- `Enter` 或 `a`：接受第一候选；
- `pick <序号>`：选择显示的候选；裸 `1/2/3` 始终保留给事件、奖励和商店的游戏选项；
- `more`：展开其余合法动作；
- 原有 `p/e/u/d/<序号>/c/x`：仍可直接输入，视为人类覆盖；
- `q`：只断开终端，不结束 controller。

候选动作不合法、推理超时、状态不稳定或模型没有生成可解析的 ID 时，自动降级为纯人工输入。模型永远不能绕过 controller 的合法动作检查。第一版不把零样本生成概率伪装成可靠置信度；校准的 reject/fallback 分数属于后续训练目标。

这仍属于在线纠正式模仿学习：人类选择的候选可以直接作为正标签；人类没有选择的候选提供相对偏好约束。人类仍可以输入候选列表之外的动作，以覆盖模型遗漏。

### 3.3 置信度路由 Assist（实验性实现）

Candidate Assist 收集到足够的“模型首选是否被人类接受”记录后，可升级为单首选动作的选择性自动化。controller 仍提供完整合法动作集合，policy server 不自由生成长的 CommunicationMod 命令，而是在这个有限集合上给每个动作打分：

```text
observation + legal_actions
        |
constrained action scoring
        |
top-1 action, top-2 action, p1, p2, margin, entropy
        |
high confidence -> auto execute
low confidence  -> human confirm or override
```

动作在本请求内继续使用 `A1/A2/...` 短别名，controller 保存其到规范 `action_id` 的映射。当前实验实现对每个合法别名计算精确 `A<n>\\n` 续写的 teacher-forced `logprob`，并只在完整合法 ID 的集合内 softmax；换行避免 `A1` 与 `A10` 发生前缀混淆。这样得到的是当前动作集上的相对分布，能正确处理多敌人目标等分支，也不会把长命令的格式概率误当策略偏好。

初始阈值只作为安全闸门，不作为真实正确率解释。`config/agent.json` 中的 `policy.confidence_auto.enabled=true` 启用自动执行，规则为 `p1 >= min_probability`（默认 `0.85`）；不开启时保持原有 Candidate Assist。删除了 `p1-p2` 下限，因为在同一合法动作集合归一化后，`p1 >= 0.85` 已隐含 `p2 <= 0.15`。阈值必须按真实交互数据校准：在 `(p1, margin, entropy, screen_type)` 分桶中统计模型首选被人类接受的比例，选择能达到目标接受率（建议至少 97%）的 `p1` 阈值。没有足够校准样本前，默认回退到人工确认。

自动化的边界：

- 一个 `policy_version` 在一局内冻结；绝不在回合之间即时更新权重。
- 低置信度、无合法首选、非法动作、超时、连续重复状态和 watchdog 命中时，立即交给人类或安全中止。
- 只把“自动执行”理解为减少操作量，不把模型置信度当作环境奖励或胜率证明。

### 3.4 多策略候选（可选）

如果希望引入多 agent，可以让同一个 Qwen 骨干以不同角色提示词产生候选，再由轻量 ranker 合并：

- 战斗：`aggressive`（输出/击杀）、`defensive`（生存/格挡）、`resource`（药水与能量管理）；
- 战斗外：`route`（路线收益）、`deck`（卡组协同）、`risk`（低血量和精英风险）。

这些角色不必对应多个常驻模型。3050上建议共享一个4-bit骨干，串行使用不同提示词或 adapter；候选生成后统一去重并用合法动作 mask 过滤。只有在外部GPU上训练完成后，才考虑比较不同 adapter 的 ensemble。多策略的价值是增加候选多样性，不应让多个模型直接绕过统一的动作编译器。

### 3.5 DAgger：周期性聚合并更新

每完成一批 assist 局，将模型访问到的状态追加到训练缓冲区，用人类最终动作作为标签训练 LoRA。新模型通过固定种子回归集和少量新局验证后，再切换到下一批 assist。

第一阶段不在游戏进程内实时改权重，避免训练阻塞协议、改变同一局的策略分布或造成难以回放的行为。

## 4. 组件设计

```text
StS1 + CommunicationMod
        |
communication_controller.py
  normalize_state / legal-action mask / trace writer
        |  localhost JSON RPC
policy_server.py（独立进程，Qwen 4-bit 推理）
        |
human_terminal.py（显示建议，接收确认或纠正）
```

建议新增独立的 `policy_server.py` 或同等功能的模型进程，不把 Transformers 推理循环塞进 CommunicationMod controller。这样模型卡顿时可以超时并回退人工，controller 仍保持协议安全。

默认动作输出采用受约束的候选 ID 排序，而不是让模型自由生成 CommunicationMod 命令。启用置信度自动执行时，`candidates` 会扩展为全体合法动作的受约束分数，仍由 controller 编译命令：

```json
{
  "action_id": "PLAY_CARD:2:0",
  "candidates": [
    {"action_id": "PLAY:card=2:target=0", "score": null},
    {"action_id": "END_TURN", "score": null},
    {"action_id": "PLAY:card=4:target=0", "score": null}
  ],
  "candidate_source": "shared_backbone",
  "confidence": null,
  "model_id": "qwen3-0.6b-combat-lora-v001",
  "prompt_version": "obs-v1",
  "latency_ms": 840
}
```

`action_id` 必须来自 controller 提供的 `legal_actions`。模型只负责排序，命令字符串由现有 action compiler 生成。

## 5. 轨迹格式、数据集与奖励

当前 `action-trace` 可以继续使用，增加一个可选的 `decision` 字段即可兼容旧数据。置信度自动执行会写入新增字段；它们仍保持可选，兼容历史 Candidate Assist 轨迹：

```json
{
  "schema_version": "0.3",
  "run_id": "...",
  "step_id": 12,
  "pre_state": {},
  "legal_actions": [],
  "decision": {
    "mode": "confidence_assist",
    "model_action": "PLAY_CARD:2:0",
    "model_candidates": ["PLAY_CARD:2:0", "END_TURN"],
    "action_scores": [
      {"action_id": "PLAY_CARD:2:0", "logprob": -0.31, "probability": 0.81},
      {"action_id": "END_TURN", "logprob": -1.72, "probability": 0.10}
    ],
    "confidence": {
      "top_probability": 0.81,
      "second_probability": 0.10,
      "margin": 0.71,
      "entropy": 0.62,
      "threshold_version": "confidence-v1",
      "route": "human_confirm"
    },
    "executed_action": "END_TURN",
    "source": "human_override",
    "overridden": true,
    "override_reason": "保留能量等待防御",
    "policy_version": "qwen3-0.6b-lora-v003"
  },
  "action": {},
  "post_state": {},
  "accepted": true,
  "stable_after_action": true,
  "terminal": false
}
```

字段规则：

- `model_action` 是模型首选的规范 `action_id`，没有模型建议时为 `null`；
- `executed_action` 是真正送入游戏的 `action_id`，训练标签使用它；
- `source` 取 `model_auto`、`human_confirm`、`human_override`、`fallback` 或 `human_only`；
- `overridden` 表示模型建议是否被人类替换；
- `override_reason` 可选，短文本即可，不要求每步填写；
- `model_candidates` 保存显示或对比用的 top-k；`action_scores` 是当前全体合法动作的归一化分数，仅在置信度路由启用后记录；
- `confidence` 记录路由可复现所需的统计量和阈值版本。它们用于校准，不应被当作真实胜率；
- `accepted=false` 的协议错误记录保留，但不进入训练集。

每局的 `run-events` 增加 `policy_version`、`mode`、`override_count` 和结束原因。训练集按 `run_id`/种子切分，不能按动作随机切分。

原始 `action-trace` 是唯一事实来源，不直接喂给训练。每次数据构建从它导出以下互相可追溯的数据集：

| 数据集 | 一条样本 | 主要来源 | 初始训练用途 |
|---|---|---|---|
| `sft.jsonl` | `observation`、`legal_actions`、`executed_action`、样本权重 | 人类覆盖、人类确认；经整局结果筛选的自动动作 | 行为克隆 / LoRA SFT |
| `preference.jsonl` | `observation`、`legal_actions`、`chosen_action`、`rejected_action` | 人类覆盖时的最终动作与模型首选 | DPO 或 pairwise ranking |
| `calibration.jsonl` | `p1`、`p2`、margin、entropy、模型首选是否被接受、screen 类型 | 每个有模型决策的稳定 step | 设定自动执行阈值 |
| `trajectory.jsonl` | 按局排列的状态、执行动作、旧策略分数、来源、奖励和终局结果 | 全部稳定的已接受 step | 错误分析、后续离线 RL / value 训练 |

构建器必须同时保留动态别名表和规范 `action_id`。SFT/DPO 的输入可使用“动作表 + 别名”，目标为当步别名；跨样本统计、重放和去重一律使用规范 `action_id`。这样训练不会把 `A1` 错当作跨状态恒定语义。

样本权重默认遵循：

```text
human_override > human_confirm > model_auto + successful outcome
```

自动动作不是天然正确标签。自动运行但失败的轨迹保留在 `trajectory.jsonl` 和失败分析集中，不能直接加入 SFT；只有满足预先定义的整局/层级结果筛选条件时，才以较低权重作为伪标签加入。人工覆盖产生的 `(chosen_action, rejected_action)` 是最有价值的偏好对；若人类确认模型首选，记录为正向校准样本和普通 SFT 样本，但不凭空构造负例。

奖励按来源和时间尺度分别保存，而不是要求玩家逐步打分：

- 自动环境信号：胜负、到达 Act/Floor、战斗胜负、HP 变化、金币/遗物/卡牌变化；
- 可选人工信号：仅在关键动作或楼层结束时给 `-1/0/+1`，并标记范围为 `step`、`floor` 或 `run`；
- 终局信号：通关、死亡、主动退出、watchdog 中止，必须区分，不能把退出当失败奖励。

这些奖励先用于筛选自动轨迹和错误分析。只有在 SFT、偏好数据、评测和策略稳定后，才把版本化的轨迹用于离线 RL；不在当前游戏运行时在线更新权重。

## 6. 训练目标

### 6.1 第一阶段：候选排序与后训练动作打分

对每个规范化状态和合法动作集合训练一个动作选择器：

```text
输入：observation + legal_actions
标签：executed_action，或人类选择产生的候选相对顺序
在线推理：零样本 Qwen 输出合法 ID 列表；后训练损失：masked cross-entropy、pairwise ranking loss 或 listwise loss
```

训练出的 scorer 只在合法动作集合内计算 loss；这比让 Qwen 生成任意文本更容易稳定，也能直接复用现有 action compiler。`human_override=true` 的样本建议提高 2-3 倍采样权重，因为它们正是模型的失败状态。候选集训练的关键指标是专家动作是否进入 top-k，而不只是 top-1 是否正确。

### 6.2 第二阶段：置信度和价值辅助任务

先以 `calibration.jsonl` 分桶校准规则阈值；数据足够后，可加入是否被纠正、动作后短期结果、是否进入战斗失败/胜利等标签，训练一个轻量 value/action scorer。它用于拒答和 fallback，不直接决定最终动作。

### 6.3 第三阶段：离线 RL，再考虑在线 RL

当至少有稳定的 assist 策略和可复现评测后，才使用胜负、生命损失、金币、路线进度等奖励做 CQL/IQL 类离线 RL 或 preference optimization。此时仍保留人类动作数据作为行为约束。

不建议当前阶段做：

- 无人类标签的“无监督 RL”；没有任务奖励时无法定义赢法；
- 同一局内更新 Qwen 权重；会破坏回放和数据分布；
- 在 4GB 显存上同时驻留 combat/high-level 两个完整大模型。

## 7. 数据量与人工成本估计

按一局约 100-300 个决策估算，实际数量随战斗长度和路线变化：

| 阶段 | 建议数据 | 人类负担 | 目的 |
|---|---:|---:|---|
| Shadow | 5-10 局 | 与正常游玩相同 | 检查状态、动作映射、延迟 |
| Candidate Assist v0 | 5 局或 500-1500 步 | 每步选择/纠正 | 验证候选覆盖和交互成本 |
| Assist v1 | 累计 20-40 局 | 重点纠正困难状态 | 形成第一版可用策略 |
| DAgger 迭代 | 每轮 5-10 局 | 随 override 率下降 | 覆盖模型访问分布 |

不要把“完整胜局数量”作为唯一门槛。更有用的指标是：模型 top-1 接受率、top-3 覆盖率、纠正率、非法动作率、每步延迟和 Act 进度。第一版目标可以设为：

- 非法动作率 `0%`；
- Assist 平均推理延迟 `<2s`；
- 战斗专家动作 `recall@3 >80%`，否则先改善候选生成而不是训练 ranker；
- 战斗 top-1 接受率 `>50%`，top-3 覆盖率 `>80%`；
- 连续 3 个固定种子可无人干预通过 Act 1，之后再扩展。

## 8. 显存与机器分工

本机实测 Qwen3-0.6B 4-bit policy server 加载后额外占用约 `1.3 GiB` 显存，短候选生成整卡约为 `3.34 GiB / 4 GiB`（当时桌面基线约 `1.98 GiB`）。因此，若 StS1 本身接近 `2.5 GiB`，理论余量只有约 `0.2 GiB`，可用于短测但不应视为稳定运行配置。不要在本机同时加载多个模型，也不要边玩边训练；出现 OOM 或明显卡顿时，应将 policy server 放到外部 GPU。

推荐分工：

- 本机 3050：游戏、controller、policy server、4-bit 推理、轨迹写盘；
- 外部 RTX 5050：定期 LoRA/QLoRA 训练、验证和模型导出；
- 两台机器：按局同步 `action-trace` 和版本化 observation/action schema，不同步 raw 全量日志，除非排错。

每轮训练只替换一个 adapter；切换前用固定种子回归，失败则保留旧版本。模型文件、训练缓存和数据目录继续由 `.gitignore` 排除。

## 9. 分阶段实现清单

### P0：协议扩展

1. controller 在每个 view 中附带 `decision_context`：`run_id`、`step_id`、规范化状态、合法动作和当前 mode。
2. 增加 `policy_server.py` 的超时、断线和版本握手。
3. 终端增加 `a/accept` 和 `m <id>`，保留现有人工命令。
4. trace 写入 `decision` 字段，旧的 `action` 字段保持不变。

### P1：Shadow 与 Assist

1. 先用规则/随机合法动作验证协议，再接 Qwen；
2. Shadow 运行 5-10 局并生成统计报告；
3. Assist 默认覆盖所有游戏内场景；仅 `MAP` 提供完整地图，其余场景裁剪地图，并以 controller 的合法动作约束防止重复进入同层商店；
4. 任何推理异常自动回退 human-only，并在轨迹中记录原因。

### P2：第一轮 DAgger

1. 按 `screen_type`、角色、进阶和是否纠正分层采样；
2. 训练 combat adapter，再训练共享 high-level adapter；
3. 用固定种子比较旧模型、新模型和纯人工上限；
4. 只有 override 率和非法动作率改善后才扩大自动化范围。

### P3：置信度半自动与评测（实验性实现，待校准）

1. 已实现受约束全动作 scoring，并保存 `action_scores`、p1、margin、entropy 和阈值；
2. 先在人工确认下收集校准集，再让达到目标接受率的低风险动作自动执行；
3. 增加 watchdog、超时、状态重复检测和回退动作；
4. 统计按角色/A0-A20/Act/房间类型拆分，避免总平均掩盖战斗崩溃；
5. 从原始轨迹导出 SFT、偏好、校准和按局 trajectory 数据集；之后再评估是否值得加入离线 RL。

## 10. 当前决策

当前不需要重新收集大量纯人类 SFT。现阶段继续用 `Shadow -> Candidate Assist -> DAgger` 验证候选覆盖和交互闭环；置信度路由已作为显式开关实现，但自动执行仍需积累模型首选与人类最终动作的配对记录后校准。若零样本 Qwen 的候选 `recall@3` 过低，再补充少量人类冷启动数据，而不是直接进入在线 RL。

参考： [DAgger](https://arxiv.org/abs/1011.0686)、[TAMER](https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-ICDL08-knox.html)、[PEBBLE](https://arxiv.org/abs/2106.05091)。
