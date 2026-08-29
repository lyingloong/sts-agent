# 终局人类反思归因 GRPO

该模式用于在线 RL 的第二阶段：每局游戏结束后，人类只写两段简短自然语言复盘，强教师把复盘定位到少量已经执行的模型动作，远端只对这些动作更新 LoRA。它不替代现有的终局奖励，也不把教师思考过程写入训练数据。

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
  "base_url": "https://.../compatible-mode/v1",
  "model": "deepseek-v4-pro",
  "api_key_file": "ali_api.txt",
  "timeout_seconds": 180.0,
  "max_retries": 1,
  "temperature": 0.0,
  "max_tokens": 2048,
  "max_input_chars": 48000,
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
  "max_abs_advantage": 2.0
}
```

`config/online_rl.example.json` 已包含这段示例。将本地代码通过 Git 同步到训练机后，仅修改训练机的被忽略 `config/online_rl.json`，不要从本机直接改远端工作树。

## 运行流程

1. 正常以 `policy.runtime: "remote_online_rl"` 启动游戏与 Overlay。
2. 每个 rollout 的策略动作仍由冻结的远端 adapter 自动采样和执行。
3. `GAME_OVER` 或 `VICTORY` 后，Overlay 弹出 `ROLLOUT REFLECTION`，其中按时间顺序显示本局全部已确认动作（`step_id`、Act-Floor、场景、模型/自动来源和动作语义）。两栏均可留空；点击“跳过本局”也可继续。
4. controller 保存人类文字，然后使用 DeepSeek 生成每个方向至多 10 个候选归因。
5. 本地校验 `step_id`、方向、置信度、严重度和冲突；仅保留每方向最高置信度的 8 个、且 `confidence >= 0.75` 的动作。
6. 完整 episode 与筛选后的归因上传。第 4 局结束时，同一远端 GPU 进行一次 QLoRA 更新并保存新版本。

若人类只填写其中一栏，该栏是唯一允许的归因方向；教师在空栏方向给出的候选会被拒绝，不能凭自身判断补出正或负反馈。

文本客户端也支持该流程：终局时依次输入 `可强化>` 和 `应弱化>`；两行都为空即跳过。

## 数据产物

以下目录均为本机运行数据，已被 Git 忽略：

```text
data/reflections/reflection-<run_id>.json
data/attributions/attribution-<run_id>.json
data/trajectories/action-trace-<run_id>....jsonl
```

`reflection` 文件保存人类两段文字、终局摘要和提交时间。`attribution` 文件保存教师模型/延迟/用量元数据、已应用与被拒绝的候选和拒绝原因。两类文件都不保存教师原始 completion、思考内容或链式推理。远端 episode 只带如下最小更新字段：

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

对于 episode `i` 中由人类/教师选中的动作 `t`：

```text
d_t = +1  (reinforce)  或  -1  (weaken)
w_t = confidence_t * severity_t / 3
A_hat_t = d_t * m_i

L = - sum_t w_t * min(r_t * A_hat_t, clip(r_t, 1-eps, 1+eps) * A_hat_t) / sum_t w_t
```

其中 `r_t = exp(logpi_current - logpi_behavior)`，行为策略与当前更新都使用 rollout 的同一采样温度。关键点是 `d_t` 独立决定正负：一条“应弱化”反馈总是降低该动作概率；低回报 rollout 不会造成双负号反向强化。组内奖励只控制更新强度 `m_i`。

默认 `max_abs_advantage=2.0`，`severity` 取 1/2/3。更新仍使用既有的 PPO clip、学习率、梯度裁剪和单 GPU 4-bit QLoRA 设置。

## 跳过与失败语义

- 人类跳过、教师请求失败、所有候选低置信度或相互冲突：该 episode 的 `attributions=[]`。
- 一个完整 K 组没有任何有效归因，或四局终局奖励标准差近零：worker 正常关闭组，但返回 `group_completed_no_update`，不保存 adapter、不递增策略版本；下一组继续使用当前 adapter。
- 部分 episode 有有效归因：只更新那些动作。人类跳过的 episode 不会获得隐含正/负标签。
- 远端服务、网络或上传失败：原始 trace、reflection 和 attribution 均保留在本机；controller 不会伪造已完成更新。

## 观察与调参

先固定角色、进阶和种子，观察至少数个完整 K 组。重点记录每组的 `reward_std`、`steps`、`available_steps`、`reinforce_steps`、`weaken_steps`、`weight_sum` 和 `skip_reason`。初期建议每局只写 1-3 条最关键的正/负问题，即使教师可产生至多 10 条候选，也不要把每一步都标成训练信号。

`max_candidates_per_direction=10` 是教师候选上限，`max_applied_per_direction=8` 是最终更新上限；真正有效数量还会受 `min_confidence` 和冲突校验限制。若教师定位过于宽泛，先提高 `min_confidence` 至 `0.8`，而不是提高更新 epoch 或学习率。
