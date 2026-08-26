# 实施计划

更新时间：2026-08-26

## 目标

构建一个可重复采集、可检查、可回放的 StS1 结构化 Agent 系统。近期目标不是论文结果，而是获得一个能稳定执行游戏操作、可持续迭代的数据与训练闭环，为后续自媒体展示和研究工作打基础。

## 已完成

- StS1 + CommunicationMod 外部 Python controller，支持四角色、A0-A20、固定/随机显示种子和多局连续采集。
- 结构化 observation、规范 action ID、合法动作编译、单局 action trace、运行事件与隐藏信息隔离。
- 本地静态知识快照：卡牌、药水、遗物、怪物、事件、powers 等；运行时字段优先。
- 人工终端、透明 Overlay、Candidate Assist、受约束置信度自动执行。
- OpenAI 兼容远程教师采集，按 screen 注入中文策略提示词，输出经过合法动作验证。
- 教师 SFT 导出器与 checksummed transfer bundle。当前 30 局完整轨迹可导出 6,417 条样本。

## 当前约束

- 现有 30 局均为 Ironclad A0，全部死亡；23 局未进入 Act 2，7 局进入 Act 2，最高到 Act 2 Boss。
- 数据可用于第一轮接口/行为冷启动，不能作为策略质量或通关能力的结论。
- SFT trainer、评测 runner、GRPO rollout worker 和远程推理 endpoint 尚未实现。
- 游戏环境仍在本地 Windows；远程 8x 4090 只承担训练与离线计算。

## 下一阶段

1. 固化当前数据 release，按局切分 train/validation/test，并训练 `sft_base_v0`。
2. 在未见种子上评测合法动作率、卡死率、平均层数、Act 进度与延迟。
3. 用模型辅助或人工纠正补充深层、不同角色和不同进阶的高质量状态。
4. 实现独立训练模块：结构化 observation 压缩、alias action SFT、checkpoint/run manifest、固定种子评测。
5. 当 SFT 基线和评测稳定后，再实现冻结策略的 K=4 GRPO rollout 与组内更新。

## 模型与资源策略

本机 3050 负责游戏和采集。第一轮 Qwen3-0.6B SFT 放在远程 4090：先使用单卡 QLoRA，不为当前小模型强行 DDP。8 卡优先用于并行消融、评测和后续多游戏实例 rollout。详细的环境、数据 release 和 checkpoint 约定见 [训练机与迁移指南](TRAINING_ON_4090.md)。

高层与战斗策略先共享一个 backbone，通过 `screen_type`、结构化状态和合法动作表路由；不在冷启动阶段引入参数级 MoE。只有在单模型基线、数据规模和评测都证明存在明确瓶颈后，才比较 adapter 或 scorer 分工。

## 验收标准

- 数据 release 可由 hash 校验，且能从 trace 重建训练样本。
- 训练样本按完整 run/seed 切分，验证或测试轨迹不参与训练。
- policy 输出经 controller 约束后合法动作率为 100%。
- checkpoint 有源码 commit、数据 hash、基础模型 revision、参数和评测结果。
- 自动运行出现协议错误、重复状态、超时或非法输出时可回退，而非卡住游戏。

协议字段见 [PROTOCOL.md](PROTOCOL.md)，教师数据和 GRPO 设计见 [TEACHER_SFT_GRPO.md](TEACHER_SFT_GRPO.md)。
