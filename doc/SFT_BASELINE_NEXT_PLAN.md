# 人类示范 SFT 基线：实现状态与实验路径

历史记录（2026 年 10 月）。SFT 工具和用法见[数据集指南](HUMAN_SFT_DATASET.md)，本页的研究计划已停止。

更新：2026-10-04。本文记录已落地的工程路线与尚未由真实数据证明的部分，不把计划写成实验结果。

2026-10-06 补充：先用官方 Qwen3.8-27B BF16 做未微调的完整游戏基线，
以区分模型原生能力与接入/状态/输出格式问题。实现、配置与部署步骤见
[BASELINE_INFERENCE.md](BASELINE_INFERENCE.md)。截至 2026-10-07，远端已部署并生成过游戏动作；
尚无完整自动对局结果。最新会话暴露的采集快照/命令应答混淆已修复，仍待实机复测。
评测改为 Orak-style 进度分为主、通关率为辅，见 [BASELINE_EVALUATION.md](BASELINE_EVALUATION.md)。

## 目标

以《杀戮尖塔 1》A0 铁甲战士作为第一个 baseline：玩家在游戏内游玩，ActionRecorder 留存完整的
原始事件和状态；Recorder 工具筛选对局；sts-agent 将可训练决策转换为单步 SFT 样本；一个模型负责
所有场景，输出当前合法动作 ID；最后以完整对局进度分、真实终局和通关率评测。

首版目标是第三幕 Boss 通关，击败心脏单独统计。人工胜利轨迹说明该局最终通关，不代表其中每步都
最优，也不能称为完美专家演示。评测需报告种子数、通关数、置信区间、人工接管、非法动作、卡死及延迟。

## 当前已实现的代码路径

- 游戏内原始记录、状态契约和对局筛选：独立仓库 `../ActionRecorder`；协议见其
  [状态契约](https://github.com/lyingloong/sts-action-recorder/blob/master/docs/STATE_SCHEMA.md) 与 [数据集提取](https://github.com/lyingloong/sts-action-recorder/blob/master/docs/DATASET_EXTRACTION.md)。
- 原始事件到下游训练轨迹：`sts_agent/recording/communication_bridge.py`、
  `trajectory_assembler.py`。
- 人类 SFT 样本导出：`sts_agent/learning/export_human_sft.py`；细节见
  [HUMAN_SFT_DATASET.md](HUMAN_SFT_DATASET.md)。
- 统一运行时与训练渲染：`sts_agent/runtime/controller.py`、
  `sts_agent/policy/local_qwen.py`。
- QLoRA 训练、种子族切分及每 epoch 的 prompt 布局轮换：`sts_agent/learning/sft.py`。
- 开发者 prompt 示例：[`POLICY_PROMPT_PREVIEW.md`](POLICY_PROMPT_PREVIEW.md)。

## 输入事实与知识边界

卡牌、遗物、药水、Power 的效果说明，以及敌人/事件的当前公开状态，直接从游戏状态读取。
ActionRecorder 在原始状态中保存游戏对象描述；卡牌动态变量按当前对象缓存值展开，并保留原文、来源、
未解析变量和多段伤害数组。
sts-agent 只白名单投影游戏实际返回的字段；不会再加载本地静态实体知识库、经验字段或 Boss 攻略。
静态实体 JSON、知识请求、经验编辑器和下载工具已经从本仓库移除。

这是一个明确的信息源取舍：纯 CommunicationMod 如果不给某实体效果文本，模型就只能依赖自身参数
记忆和其他当前公开状态。未知值保留未知；不允许按名字从旧 JSON、外网或策略文档偷偷补齐。
游戏给出的实时数值优先，控制器不能重复叠加力量、虚弱等效果。Recorder 补丁仍需实际游戏验收，
源码层能取到字段不等于所有模组卡牌/tooltip 都已完整覆盖。

地图全图只在 MAP 状态进入模型；非地图状态过滤掉地图。敌人当前意图和牌面实时数据来自状态消息；
不注入离线 Boss 档案或攻略文本。历史动作保留在当前策略输入中，事件历史有独立的多步上下文。

## Prompt 与监督目标

所有场景共用一个模型，监督标签是当前合法动作表中的 alias，例如 `A2`。输入自然语言化，包含通用
玩家指令、场景提示、路由后的公开状态、动作历史和合法动作；只对回答 alias 与结束标记计算 loss，
不训练模型生成显式推理。实时推理、SFT 和动作评分共用 `build_scoring_prompt`，防止 train/serve 偏差。

目前提供三种确定性等价布局：`canonical`、`state_first`、`decision_first`。相同决策不会扩增为三行，
训练时每个 epoch 按样本 ID 轮换布局，验证固定使用 canonical。三者仅改变状态、指令、场景提示的排列，
实体、数字、合法动作和答案不变。它是提示格式鲁棒性增强，不应包装成新数据或新策略知识。

指定 tokenizer 时导出默认完整 token 上限为 8192，可显式调高；任一样本/模板超限则报错终止整个导出，绝不截断或按长度静默排除，失败不发布半成品目录。导出预览逐场景、逐模板保存。
对自然语言是否比 JSON 更省 token，要按实际 Qwen tokenizer 的长度报告验证，不能凭感觉宣称。

## 训练数据质量与切分

- 原始事件、通用细粒度轨迹、可用决策样本和模板布局是不同计数口径，不能把三模板当三倍数据。
- 取消、未执行、缺失/错配执行前状态等步骤由导出器按 `usable_for_sft` 和明确排除原因处理；不从动作后状态倒推正确标签。
- 训练/验证先按种子族切分，而不是随机切动作或单纯按 run ID。显示种子先规范化，重复局归在同一组；种子确实缺失才退回 run ID，并在实验记录中披露。
- 人工仅一人采集时应透明说明示范者数、角色/A0 范围和采集过程；不同游戏版本、语言及模组要作为数据 provenance 记录。
- SFT 训练先按目标模型 tokenizer 测完整输入长度，再确定 max length、batch/累积、epoch 和学习率。LoRA/QLoRA 是参数更新方式，不是 SFT 目标本身。

## 下一步实验（这是剩余实验，不是待补代码阶段）

1. 用新 Mod 在真实游戏中验收完整状态描述和细粒度动作，覆盖战斗、药水选择、GRID/HAND_SELECT、商店、营火、事件、Boss 奖励、存档继续和终局；监测帧率/输入延迟。
2. 按用户选择筛选 A0 铁甲战士完整胜利局；保留原始死亡/中断轨迹，作为采集调试材料，不混入胜利训练集。
3. 导出 SFT 数据并审阅各场景预览和排除报告；统计独立 run/seed 数、场景分布、重复种子、token 长度和标签覆盖。
4. 小规模训练检查端到端加载、adapter 保存/恢复、训练输入与实际推理一致；之后再用选定的 8B Qwen 系列做正式训练。模型尺寸和 epoch 数需依数据量及验证 loss 决定。
5. 在训练期间完全留出的种子上做完整自动运行评估；公布同基础模型未微调、SFT 单模板和 SFT 多布局的对照。配置、自动化环境动作和人工干预都要同口径披露。

这些实验尚未在本次仓库改动中实际运行；当前不能据代码完整性声称模型已能通关或数据集已经达到大规模。

## 研究定位

合理的首个贡献可以是：一个可审计的、覆盖完整 StS1 对局的人类细粒度动作数据集，以及在真实完整局通关率
上评测的轻量策略 baseline。是否有研究含金量取决于对局数和独立种子覆盖、状态/动作可靠性、可发布性、模型
相对未微调基线的增益和复现材料。单人、单角色、A0 数据要如实界定，不宜称“大型通用数据集”或宣称首个
完整 agent。Prompt 布局变体和运行时描述是合理工程设计与可做消融的因素，单独而言不是强创新主张。

参考：

- [Orak 论文](https://openreview.net/forum?id=H1ncX6O6Yh)
- [Orak STS 环境与状态渲染](https://github.com/krafton-ai/Orak/blob/release/src/mcp_game_servers/slay_the_spire/game/slay_the_spire_env.py)
- [Orak 实体解析代码](https://github.com/krafton-ai/Orak/tree/release/src/mcp_game_servers/slay_the_spire/game/spire)
- [Orak 公开数据集](https://huggingface.co/datasets/KRAFTON/Orak)
