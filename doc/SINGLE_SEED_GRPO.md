# 单种子轨迹级 GRPO 方案

更新时间：2026-08-23

## 状态

这是后续在 RTX 5050 上推进的训练方案，当前不实现训练器。现有 `tools/single_seed_solver.py` 只完成了固定种子的自动 rollout：它接受 controller 提供的模型 top-1 候选、写入 `source=model_auto` 的 action trace，并在 `run-events` 中记录终局结果。它还没有全动作概率、温度采样、GRPO loss、LoRA adapter 加载或自动切换 checkpoint。

目标是先让一个模型反复求解固定 seed，验证“采样、回报、更新、评测、回滚”的完整闭环。固定 seed 的成功允许过拟合，不能作为泛化能力结论；未见 seed 评测从第一轮起就必须保留。

此处的“轨迹级 GRPO”是工程上的简称：一组从同一初始 seed 出发的完整轨迹用相对回报计算优势，并将该优势应用于轨迹内的离散合法动作。它不是直接将面向单个文本 completion 的标准 GRPO trainer 原样用于 StS。

## 1. 一轮训练

一轮内模型参数绝不变化；只有整组 rollout 都结束后，才训练并产生下一个策略版本。

```text
冻结 policy vN 和 LoRA adapter
        |
固定同一 character / ascension / seed，温度采样 K=4 局
        |
全局动作轨迹 + 环境回报 + 旧策略概率
        |
组内 advantage、一次受限 LoRA 更新
        |
policy vN+1 -> 固定评测 -> 保留或回滚
```

`K=4` 是第一版固定值。四局使用相同初始 seed，但因为每一步从合法动作分布采样而进入不同游戏分支。一个策略版本在整个 K 局中冻结，防止组内回报来自不同策略而失去可比性。

## 2. 全局动作空间

训练覆盖每个已接受、状态稳定的游戏决策，不筛选为仅战斗：

- 战斗：出牌、目标、药水使用/丢弃、结束回合；
- 地图：所有可选节点；
- 战斗奖励、卡牌奖励、Boss 奖励、事件、商店、营火、手牌选择和确认/跳过；
- 终局返回等导航动作保留在原始轨迹中，但不计入策略奖励 loss。

每一个策略训练样本是一个独立的决策，而不是将完整一局拼接为超长上下文：

```json
{
  "observation": {},
  "legal_actions": [
    {"alias": "A1", "action_id": "..."},
    {"alias": "A2", "action_id": "..."}
  ],
  "sampled_alias": "A2",
  "sampled_action_id": "...",
  "old_logprob": -0.84,
  "old_probability": 0.31,
  "policy_version": "v012"
}
```

别名只在当前状态有效；重放、去重和指标一律使用规范 `action_id`。训练输入继续遵守实际玩家可见信息：完整公开地图只在地图选择屏提供，其他界面保持当前局部观察，不额外拼接历史或隐藏随机信息。

全场景训练不等于允许战斗样本淹没稀有界面。训练器会在一个 epoch 内使用所有稳定决策，并以 `screen_type` 做权重或分层 batch，使商店、事件、地图和营火仍有梯度贡献；这不是筛除任何场景。

在 5050 上也应先使用紧凑、版本化的结构化表示，而不是原样拼接冗长 JSON。地图的图结构不能删除，但应压缩字段和文本；卡牌的关键运行时数值与静态 `effect` 字段保留，重复说明文字可限长。推理和训练必须使用同一 observation schema。

## 3. 受约束采样与记录

现有候选生成必须先升级为全合法动作打分。controller 为当前状态生成 `A1/A2/...`；policy server 对每个别名计算条件 log probability，并只在完整合法集合内归一化：

```text
pi(a | state, legal_actions) = masked_softmax(score(a) / temperature)
```

rollout 按该分布采样，不能始终接受 top-1。温度只服务于训练时的探索：

- 起步：`T=0.6-1.0`；
- 已出现稳定高回报轨迹后：`T=0.2-0.4`；
- 固定评测和展示：`T=0`，贪心选择最高分动作。

每一步写入全体合法动作分数或至少可重建归一化分布、采样动作的 `old_logprob`、temperature、随机种子、policy version 和 action alias 映射。没有这些旧策略信息，无法正确计算 PPO/GRPO ratio，只能退化为奖励加权 SFT。

## 4. 回报与轨迹级 GRPO loss

每条轨迹先得到可审计的标量回报。第一版奖励以完成目标为主，密集项只帮助区分同批次的失败局：

```text
R = 100 * victory
  + 15 * boss_kills
  + 5 * elite_kills
  + 2 * normal_combat_wins
  + 0.2 * max_floor
  - 0.05 * hp_lost
  - 10 * death
```

主动退出、controller 协议错误和 watchdog 中止不是普通死亡；它们单独标记，不进入该组的 RL loss。奖励权重属于配置，必须随每轮结果写入 manifest，防止不同实验不可比较。

同组四局的相对优势为：

```text
A_i = (R_i - mean(R_1...R_K)) / (std(R_1...R_K) + 1e-6)
```

第 `i` 条轨迹的每个策略动作 `t` 使用该轨迹优势，初版 loss 为带合法动作 mask 的 PPO/GRPO 形式：

```text
ratio_i,t = exp(log pi_new(a_i,t | s_i,t) - old_logprob_i,t)

L_policy = -sum_i,t min(ratio_i,t * A_i,
                          clip(ratio_i,t, 1-eps, 1+eps) * A_i)
L_total = L_policy + beta * KL(pi_new || pi_vN)
```

KL reference 固定为本轮 rollout 使用的 `vN`。当四局回报相同，优势接近零，训练器应记录“无相对信号”并跳过策略更新，而不是凭失败轨迹任意漂移。后续可加入折扣 return-to-go 或 value baseline 改善信用分配，但不是第一版前置条件。

## 5. 训练批次与 optimizer step

一个 `optimizer step` 是一次 LoRA 参数更新，不是一局游戏也不是一次游戏动作：

```text
若干决策样本 forward/backward
-> 累积梯度
-> optimizer.step()
-> LoRA 权重更新一次
```

例如四局共有 600 个稳定策略决策，`micro_batch=1`、`gradient_accumulation=4` 时，完成一个覆盖所有样本的 epoch 约需 `600 / 4 = 150` 个 optimizer steps。全局动作训练不适用任意固定的“只训练 5-15 steps”；第一版按一轮数据做一个 epoch，并设置安全上限，例如 `max_optimizer_steps=200`。

更新由一轮 `K=4` 触发，但训练 batch 不应只含四条最新轨迹。使用“最新组 + 有上限的近期回放 + 该 seed 的较优历史轨迹”，并给最新数据较高权重。回放缓存只能保存与当前 observation/action schema 兼容的数据。

## 6. 数据与实验产物

原始 `action-trace` 和 `run-events` 仍是事实来源。训练管线新建下列可忽略的生成目录：

```text
data/rollouts/                 # 每次求解器运行的 manifest
data/datasets/rl-rollouts/     # 标准化后的逐步训练样本与组元数据
outputs/grpo/                  # batch 指标、评测和 checkpoint 选择结果
checkpoints/                   # LoRA adapters 和可回滚元数据
```

每个 batch manifest 至少保存：`batch_id`、seed/角色/进阶、`policy_version`、K、temperature、奖励配置、每局 run_id/return/终局原因、训练前后指标、是否接受新 checkpoint 和回滚原因。

## 7. 评测与回滚

训练数据和评测数据必须分离。每产生 `vN+1` 后：

1. 对训练 seed 用 `T=0` 跑至少 3 局，比较平均回报、最高层数和通关率；
2. 对一组从未进入训练的 seed 用 `T=0` 各跑一局，检查明显泛化退化；
3. 只有达到预设接受规则才标记 `vN+1` 为 active，否则继续保留 `vN` 并记录回滚。

固定 seed 的表现用于确认闭环有效，不能代替未见 seed 指标。所有评测都使用冻结 checkpoint，不与探索 rollout 混合。

## 8. 硬件与推进顺序

本机 RTX 3050 4GB 上，StS1 与 4-bit Qwen 推理已接近显存上限，不能并行执行反向传播训练。即使关闭游戏后勉强进行短上下文 QLoRA，也会受到上下文长度、速度和 Windows 显存碎片限制；不把它作为可靠训练方案。

本阶段暂停实现训练器，等待 RTX 5050。拿到 5050 后先确认实际显存，再确定 LoRA rank、上下文长度、batch size 和 optimizer。预期流程为：

```text
3050 或 5050 采集 K=4 rollout
-> 5050 训练 LoRA
-> 导出 adapter
-> policy server 加载新 adapter
-> 下一组 rollout
```

首批实现顺序：

1. 实测确认 `single_seed_solver.py` 的全局自动 rollout、终局和 watchdog；
2. 实现全合法动作 scoring、温度采样、old logprob 和 schema；
3. 实现 K=4 batch manager 与奖励/manifest 导出；
4. 实现 5050 上的 masked trajectory-GRPO LoRA trainer；
5. 实现 adapter 加载、策略版本握手、固定评测和回滚。

在第 2-5 项完成前，不应把当前自动求解器描述为“会自动训练”或“已经使用 GRPO”。
