# 教师 SFT 与 GRPO 路线

更新时间：2026-08-26

## 目标

使用远程强模型生成冷启动动作示范，训练本地 Qwen3-0.6B 学会完整 StS1 的结构化动作接口；随后仅以游戏环境回报对 Qwen 做 GRPO。教师不是奖励模型，不是最终真值，也不参与 GRPO rollout。

这条路线的前提是：教师轨迹足以让 Qwen 输出稳定的合法动作；游戏环境能对整局优劣给出可靠比较；训练种子与评测种子严格隔离。满足这些前提时，GRPO 可以纠正教师的局部错误，理论上不受教师上限约束；不满足时，模型只会复刻或放大教师偏差。

当前已有 30 局完整教师轨迹，可导出 6,417 条样本；全部为 Ironclad A0 且均以死亡结束。它们用于 `sft_base_v0` 的接口冷启动和训练管线验证，不作为策略质量的基准，更不能直接开始 GRPO。

## 已实现：教师采集

`teacher.enabled=true` 且 `policy.enabled=false` 时，controller 在每个游戏内稳定状态执行：

```text
公开结构化 observation + 当前 legal_actions
        -> 短别名 A1/A2/... -> 远程教师
        <- {"action_id":"A<n>"}
        -> controller 合法性复核 -> CommunicationMod 命令
        -> action-trace (source=teacher_auto)
```

教师只能看到 `normalize_state` 产生的公开状态；种子、抽牌顺序、对象 UUID 和 API 密钥不进入 prompt。教师使用 `tools/teacher_prompts.py` 中的中文场景化策略提示词：地图、战斗、卡牌奖励、战斗奖励、删牌、升级、变化、商店、事件、营火、首领遗物和过渡态分别采用不同决策框架。策略提示词只发送给远程教师，不进入本地 Qwen prompt、SFT observation 或 target。每条已接受 trace 额外记录教师模型 ID、prompt 版本、请求延迟、尝试次数、finish reason 和 token 用量。默认开启 thinking；Overlay 的 `MODEL / ACTION` 窗口会在教师模式下显示有界的思考内容和最终输出，本地 Qwen 模式仍使用候选/置信度界面。教师回复原文及 reasoning 只存在于当前内存中的 UI 决策，不写入 action trace 或 SFT 数据。

单步 API 失败、超时、解析失败或返回未知别名时，该步进入人工 fallback；不能自动重试或猜测游戏命令。教师使用 `screens=["ALL"]` 时覆盖地图、战斗、事件、商店、奖励、休息和牌组选择等全局动作，不按游戏部分筛选。

密钥文件默认为项目根目录的 `deepseek_api.txt`，仅接受 `key: ...` 或一行裸 key，且已被 Git 忽略。密钥轮换后才可启用此模式。`max_tokens` 默认 512，因为思考内容也会消耗生成预算，过小会在输出动作 JSON 前截断；它不控制输入状态。`max_input_chars` 默认 32000，控制公开 observation JSON 的字符上限，设为 `0` 可关闭该截断。提供方代理是否支持 `deepseek-v4-pro`、thinking 和 JSON mode 需要通过一次真实请求确认。当前 USTC 网关在 `thinking=true` 时会因 `response_format=json_object` 不返回 `reasoning_content`，因此客户端会自动优先 thinking，改为依赖提示词和本地解析器验证 JSON 动作；若关闭 thinking，才会发送 JSON response format。思考字段的具体名称可能是 `reasoning_content`、`reasoning` 或 `thinking`，客户端兼容这三种形式，解析器仍只接受唯一合法别名。

实时教师运行可以将 `teacher.thinking` 设为 `false`，并将 `timeout_seconds` / `max_retries` 调低；这会取消深层推理阶段，而不是保证模型把思考压缩成更短文本。复杂地图决策可能有质量损失，离线 SFT 采集再使用 thinking 模式更合适。

## SFT 数据契约

原始事实来源仍是每局独立的 `action-trace-*.jsonl`。运行：

```powershell
python tools\export_teacher_sft.py
```

导出器仅保留同时满足以下条件的 step：

- `accepted=true` 且 `stable_after_action=true`；
- `decision.source=teacher_auto`；
- 动作仍在该 step 保存的 `legal_actions` 中。

结果写入 `data/datasets/teacher-sft.jsonl`，一条样本为：

```json
{
  "schema_version": "sts-teacher-sft-0.1",
  "sample_id": "<run_id>:<step_id>",
  "observation": {"...": "动作前公开状态"},
  "legal_actions": [
    {"id": "A1", "label": "...", "action_id": "PLAY:card=1:target=0"}
  ],
  "target": {"alias": "A1", "action_id": "PLAY:card=1:target=0"},
  "source": "teacher_auto",
  "teacher": {"model_id": "...", "latency_ms": 0}
}
```

`A1` 不是跨状态恒定语义，因而输入必须始终包含同一步的别名动作表。统计、去重、回放和环境执行使用规范 `action_id`。中途退出、人工接管和协议异常数据仍保留在 trace 中用于排错，但不会进入教师 SFT 文件。

教师数据是噪声标签，不能等权视为专家。第一轮 SFT 应按完整局结果分层或加权：胜局和深层失败局优先，早期崩溃、循环、超时和异常局排除。教师 explanation 不应训练为目标，目标只保留一个动作别名。

## Qwen SFT

从 `Qwen/Qwen3-0.6B` 做 QLoRA SFT。输入为上面的 observation 和动态别名动作表，输出只为 `target.alias` 加换行。训练时保留 controller 的合法动作 mask；部署仍由 controller 将别名映射到真实命令。

保存的第一个可用 checkpoint 为 `sft_base`。它既是 GRPO 的 rollout 初始策略，也是 KL 参考策略。先在固定验证种子上检查合法动作率、卡死率、平均层数和 Act 进度，不能只看 SFT loss。

当前样本可以先训练一轮 `sft_base_v0`，验证 alias 输出、动作合法率和未见种子上的运行稳定性。之后继续收集不同角色、不同进阶和更深 Act 的完整局；目标仍是 100-200 局和数万动作样本。教师质量不足时，应按整局结果加权、加入人工纠正，而不是立即扩大 GRPO。

## GRPO 设计

GRPO 不在一局中更新参数。每次迭代冻结 rollout checkpoint，并以同一 `(角色, 进阶, 显示 seed)` 采样 `K=4` 条完整局轨迹：

```text
选 8 个训练初始条件
  -> 每个条件从同一 checkpoint 跑 4 局完整游戏
  -> 组内比较 4 个回报，计算相对 advantage
  -> 汇总 32 局后做 1-2 个训练 epoch
  -> 固定种子验证，通过才产生下一批 rollout
```

对组内回报使用：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

每个实际动作保存旧策略在当前合法动作集合上的 log probability。长时程版本应进一步保存每步势函数奖励并使用 reward-to-go，避免把整局终局信号等量归因给所有动作。训练实现需要严格对合法别名集合归一化，不能将任意自由文本生成的概率当作策略概率。

初始回报使用版本化、难以投机的终局指标：

```text
R = 100 * victory
  + 12 * defeated_act_bosses
  + 0.25 * final_floor
  + 2 * final_hp / max_hp
  - 5 * timeout_or_protocol_abort
```

不直接奖励金币、卡牌名或单次购买，避免局部奖励劫持。`quit` 与 `game_over` 必须区分；主动退出的轨迹不参与 GRPO。奖励系数先通过固定种子消融验证，后续才可引入势函数差分奖励。

## 评测与防过拟合

种子按局切分，绝不按动作随机切分：

- `train`: 用于教师 SFT 与 GRPO；
- `validation`: 选 checkpoint、早停和奖励消融；
- `test`: 永不训练，只汇报最终结果。

每个 checkpoint 记录通关率、平均和中位层数、Act Boss 击杀率、非法动作率、watchdog/超时率、每步延迟，并分别按角色、进阶、Act 和房间类型拆分。固定单种子求解可以用于调试，但不能作为泛化结论。

## 机器分工

3050 只承担游戏、CommunicationMod、controller、远程教师请求和 trace 写盘；教师 API 不占本地模型显存。Qwen SFT、离线评测和 GRPO 更新放在远程 8x 4090。对当前 Qwen3-0.6B 和 6,417 条样本，第一轮使用单张 4090 QLoRA；其余 GPU 用于并行消融和评测，而不是强行 DDP。游戏仍在本机，因此 rollout 要先把冻结 adapter 取回本机运行，再把新轨迹 bundle 回传训练机。

当前仓库已实现教师采集、SFT 导出和训练 bundle，尚未实现 Qwen SFT trainer、GRPO rollout worker、奖励器、远程推理 endpoint 和 checkpoint 同步器。它们应作为独立训练模块实现，不放进 `communication_controller.py`。数据 release、环境和 checkpoint 契约见 [TRAINING_ON_4090.md](TRAINING_ON_4090.md)。
