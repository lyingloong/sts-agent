# 终局反思归因 GRPO

该模式用于在线 RL 的第二阶段：每局游戏结束后，人类可写两段简短自然语言复盘，或让强教师基于完整动作账本自行复盘；教师把结论定位到少量已经执行的模型动作，远端只对这些动作更新 LoRA。它不替代现有的终局奖励，也不把教师思考过程写入训练数据。

## 目标与边界

常规 episode-level GRPO 会把一局的同一个组相对优势分配给全轨迹的每个动作。对于 StS 这种长程决策，这会把明显错误的操作与正常操作一起更新。

本模式保留 K=4 的同起点、同冻结策略 rollout 组和终局奖励，但让人类给出两个互补的字段：

- `可强化之处`：本局哪些关键操作值得保留。
- `应弱化之处`：本局哪些关键操作应降低概率。

教师只返回已有 `step_id` 的结构化归因。它不能执行游戏动作、不能引入新的监督动作，也不能引用未出现在轨迹中的隐藏信息。环境自动推进（遗物、金币、药水、打开卡牌奖励和进入商店）仍保留在 trace/history，但不属于策略样本，不能被归因。

## 使用前配置

必须同时打开本机和训练机两个开关。若本机开启而远端未开启，controller 会在启动时停用 online RL，防止新的 attribution 被旧 worker 静默当成全轨迹更新。

本机 `config/agent.json` 添加或覆盖以下字段。API key 文件仍使用被忽略的本地文件；可复用现有的阿里兼容端点与 key 文件。

```json
"posthoc_feedback": {
  "enabled": true,
  "mode": "human",
  "base_url": "https://.../compatible-mode/v1",
  "model": "deepseek-v4-pro",
  "api_key_file": "ali_api.txt",
  "timeout_seconds": 180.0,
  "max_retries": 1,
  "temperature": 0.0,
  "max_tokens": 2048,
  "max_input_chars": 96000,
  "max_reflection_chars": 4000,
  "max_candidates_per_direction": 10,
  "max_applied_per_direction": 8,
  "min_confidence": 0.75
}
```

训练机的被忽略文件 `config/online_rl.json` 必须包含：

```json
"attribution": {
  "enabled": true,
  "max_abs_advantage": 2.0,
  "min_tied_advantage": 0.2
}
```

`config/online_rl.example.json` 已包含这段示例。将本地代码通过 Git 同步到训练机后，仅修改训练机的被忽略 `config/online_rl.json`，不要从本机直接改远端工作树。

## 运行流程

1. 正常以 `policy.runtime: "remote_online_rl"` 启动游戏与 Overlay。
2. 每个 rollout 的策略动作仍由冻结的远端 adapter 自动采样和执行。
3. `mode="human"` 时，`GAME_OVER` 或 `VICTORY` 后 Overlay 弹出 `ROLLOUT REFLECTION`，其中按时间顺序显示本局全部已确认动作（`step_id`、Act-Floor、场景、模型/自动来源和动作语义）。两栏均可留空；点击“跳过本局”也可继续。`mode="auto"` 时不弹窗，教师直接根据终局状态、完整动作账本和各步骤公开动态状态独立生成双向归因。
4. controller 保存人工文字或自动复盘标记，然后使用 DeepSeek 生成每个方向至多 10 个候选归因。
5. 本地校验 `step_id`、方向、置信度、严重度和冲突；仅保留每方向最高置信度的 8 个、且 `confidence >= 0.75` 的动作。
6. 完整 episode 与筛选后的归因上传。第 4 局结束时，同一远端 GPU 进行一次 QLoRA 更新并保存新版本。

教师返回后，Overlay 会显示一个非阻塞的 `TEACHER ATTRIBUTION` 窗口，其中列出实际通过校验、会参与更新的动作（`step_id`、动作、置信度、严重度和简短证据），以及过滤数量和原因摘要。它不显示教师思考或原始响应。

`mode="human"` 下，若人类只填写其中一栏，该栏是唯一允许的归因方向；教师在空栏方向给出的候选会被拒绝，不能凭自身判断补出正或负反馈。`mode="auto"` 下没有这一限制，教师可在严格置信度过滤前分别提出正、负候选；这不是独立真值，建议每组定期人工审计 artifact，避免教师和策略共同偏差被自我强化。

自动模式的教师提示还要求将终局结果只作为复盘范围而非逐步标签：候选必须给出当时合法替代动作或公开状态中的直接后果，不能仅因整局死亡、低楼层、`END_TURN`、`SKIP` 或牌的名称归因。它通常每个方向只返回 0-3 条；`max_candidates_per_direction` 仍是硬上限，只有多个相互独立且证据明确的问题才应超过该数量。

文本客户端也支持该流程：终局时依次输入 `可强化>` 和 `应弱化>`；两行都为空即跳过。

### 教师归因输入

`posthoc-attribution-v3` 对每个 `online_policy_sample` 步骤发送所选 alias/动作标签、该步完整 alias 合法动作表，以及动态公开状态。战斗状态包括手牌的 `id/name/type/cost/upgrades/is_playable`、能量/格挡/Power、敌人血量/格挡/意图/Power、牌堆计数和最近动作；非战斗状态保留公开的选项、奖励、商店、事件历史和牌组信息。首领档案与遗物造成的决策约束也随动态状态发送；地图图结构每个 Act 只完整发送一次，后续地图步骤通过 `map_id` 引用并保留当前位置和可达节点。

卡牌、药水、遗物和 Power 的 `description`、flavor/effect 等由本地静态知识库补全的字段不会发送。这避免跨步骤重复静态文本，同时不损失归因所需的实时数值与可选动作。默认 `max_input_chars=96000`；达到上限时仍可在本机配置中提高，配置校验允许至 `262144`。

## 数据产物

以下目录均为本机运行数据，已被 Git 忽略：

```text
data/reflections/reflection-<run_id>.json
data/attributions/attribution-<run_id>.json
data/trajectories/action-trace-<run_id>....jsonl
```

`reflection` 文件保存人类两段文字或 `mode="auto"` 自动复盘标记、终局摘要和提交时间。`attribution` 文件保存教师模型/延迟/用量元数据、已应用与被拒绝的候选和拒绝原因。两类文件都不保存教师原始 completion、思考内容或链式推理。远端 episode 只带如下最小更新字段：

```json
{
  "step_id": 18,
  "direction": "weaken",
  "confidence": 0.91,
  "severity": 3
}
```

本地 artifact 额外保存短 `evidence`、选中 alias 与动作标签，便于审计，不会传给更新器。

## 更新公式

对同组终局奖励 `R_i` 先计算标准 GRPO 相对优势：

```text
g_i = (R_i - mean(R)) / (std(R) + 1e-8)
m_i = min(abs(g_i), max_abs_advantage)
```

如果一组轨迹的终局奖励完全相同，`std(R)` 为零，普通组相对优势没有方向。
归因模式下，对已经通过校验的教师关键动作使用配置项
`attribution.min_tied_advantage` 作为 `m_i`；未被归因的动作仍不更新。普通
（无归因）模式在这种情况下继续跳过更新。

对于 episode `i` 中由人类/教师选中的动作 `t`：

```text
d_t = +1  (reinforce)  或  -1  (weaken)
w_t = confidence_t * severity_t / 3
A_hat_t = d_t * m_i

L = - sum_t w_t * min(r_t * A_hat_t, clip(r_t, 1-eps, 1+eps) * A_hat_t) / sum_t w_t
```

其中 `r_t = exp(logpi_current - logpi_behavior)`，行为策略与当前更新都使用 rollout 的同一采样温度。关键点是 `d_t` 独立决定正负：一条“应弱化”反馈总是降低该动作概率；低回报 rollout 不会造成双负号反向强化。组内奖励只控制更新强度 `m_i`。

默认 `max_abs_advantage=2.0`，`severity` 取 1/2/3。更新仍使用既有的 PPO clip、学习率、梯度裁剪和单 GPU 4-bit QLoRA 设置。

### 归因纠错项

对通过校验的高置信度归因，worker 还在上述 PPO 项上加入局部纠错损失；这部分不依赖本组终局奖励差异，仍按 `confidence * severity / 3` 加权。动作概率始终在当时完整的合法 alias 集合上归一化，并使用 rollout 的同一采样温度。

```text
reinforce:  L_ce = -sum_good w * reinforce_ce_weight * log pi(a_good | s) / sum_all w
weaken:     L_ul = -sum_bad w * weaken_unlikelihood_weight * log(1 - pi(a_bad | s)) / sum_all w
L_total = L_ppo + L_ce + L_ul
```

`reinforce` 项提高已执行的好动作。`weaken` 项提高所有其他合法动作的总概率质量，因此即使坏动作已经接近 100% 概率，仍有足够梯度将其压低；若当时只有一个合法动作，则它没有可替代项，unlikelihood 项跳过。默认两个系数均为 `0.25`，可在远端 `online_rl.json` 的 `update` 中分别设为 `0` 关闭。

## 跳过与失败语义

- 人类跳过、教师请求失败、所有候选低置信度或相互冲突：该 episode 的 `attributions=[]`。
- 一个完整 K 组没有任何有效归因：worker 正常关闭组并返回 `group_completed_no_update`。普通模式或归因模式下没有有效动作时，终局奖励标准差近零也会跳过；归因模式若有有效动作，即使奖励打平，也只用 `min_tied_advantage` 更新这些动作。
- 部分 episode 有有效归因：只更新那些动作。人类跳过的 episode 不会获得隐含正/负标签。
- 远端服务、网络或上传失败：原始 trace、reflection 和 attribution 均保留在本机；controller 不会伪造已完成更新。

## 观察与调参

先固定角色、进阶和种子，观察至少数个完整 K 组。重点记录每组的 `reward_std`、`steps`、`available_steps`、`reinforce_steps`、`weaken_steps`、`weight_sum`、`skip_reason`，以及 `ppo_losses`、`reinforce_ce_losses`、`weaken_unlikelihood_losses`。初期建议每局只写 1-3 条最关键的正/负问题，即使教师可产生至多 10 条候选，也不要把每一步都标成训练信号。

`max_candidates_per_direction=10` 是教师候选上限，`max_applied_per_direction=8` 是最终更新上限；真正有效数量还会受 `min_confidence` 和冲突校验限制。若教师定位过于宽泛，先提高 `min_confidence` 至 `0.8`，而不是提高更新 epoch 或学习率。
