# 从 ActionRecorder 数据集构建人类 SFT

## 职责

ActionRecorder 的 `tools/extract_dataset.py` 负责按角色、进阶、胜负筛选原始对局，输出
`actionrecorder-dataset-1`。sts-agent 只负责重建下游 observation、合法动作、动作标签与训练
prompt；不会再查找或加载本地卡牌、遗物、药水、Power、怪物、事件知识库。

动作前状态里的实体描述应由游戏运行时提供。缺失信息保持 null/未知，不根据实体 ID 猜效果。
ActionRecorder 原始 JSONL 保留完整源字段；下游适配器只把公开且决策相关的字段投影给模型。

## 导出

在 sts-agent 根目录使用项目 Python 环境：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.learning.export_human_sft --input ../ActionRecorder/datasets/a0-ironclad-v1 --output data/datasets/human-a0-v1
```

输出目录必须不存在。不指定 tokenizer 时完整保留输入，明确提示尚未校验 token 长度。
正式导出建议指定目标模型 tokenizer，统计实际 chat template 下各 prompt 版本的完整长度：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.learning.export_human_sft --input ../ActionRecorder/datasets/a0-ironclad-v1 --output data/datasets/human-a0-v1-checked --tokenizer models/base/Qwen3-0.6B --max-length 8192
```

tokenizer 只从本地加载，不自动下载。指定 tokenizer 时默认 `--max-length 8192`，可显式调高；
长度包含 chat template、完整输入、答案和 EOS。任一训练模板超限会报 error 并终止整个导出，
列出样本 ID、各模板 token 数和上限；不截断，也不因超长而悄悄排除样本。调高上限后重新导出。
只有全部处理成功才发布目标目录，失败时不留下半成品数据集，也不覆盖已有目录。
取消、未执行、关联缺失等质量排除仍记录在 `excluded-steps.jsonl`，与长度超限错误区分。
manifest 中 `length_checked` 标记是否校验，`length_policy` 为 `error_on_overflow_no_truncation`。

导出上限不代表训练一定能装入显存；训练时应按目标 tokenizer 和数据长度设置匹配的上下文上限，
不能以截断状态或动作列表来降低显存占用。

## 转换规则

1. 校验 Recorder manifest 和输入文件哈希；只读取数据集目录内的文件。
2. 离线重放事件，通过 `CommunicationBridgeRecorder` / `trajectory_assembler.py` 重建状态关联、
   细粒度动作和最近动作历史。无需启动游戏、TCP 服务、教师 API 或 GPU。
3. 使用 `training_policy_observation`、`training_legal_actions`、`training_action`。普通动作取提交前
   状态；被执行跟踪的排队动作取确切执行前状态，不能用最近状态或动作后状态猜标签。
4. 仅导出 `usable_for_sft=true` 且目标能映射到合法动作的样本；取消、未执行、关联缺失等记录到
   排除报告。终局、未来动作和动作后状态不拼进当前输入。
5. 所有场景共用一个单步策略，输出合法动作别名（如 `A2`）。牌升级/删除等遵从数据里真实的细粒度
   入口、选牌、确认步骤；转换时不把它们合并成房间级动作。
6. prompt 由 `policy.local_qwen.build_scoring_prompt` 统一生成，地图全貌只在 MAP 显示。每个决策保留
   同一组公开事实和标签，并生成 `canonical`、`state_first`、`decision_first` 三种等价布局；它们只
   调整信息顺序，不新增/删除状态事实，不复制成三条训练样本。

### 动画与动作后状态

当前 SFT 样本为「动作前公开状态 + 合法动作列表 → 人类实际执行的动作」，不使用该动作的
后状态。`post_state`、`observation_after` 缺失或 `stable_after_action=false` 均不单独排除样本，
也不要求人类或模型判断动画是否结束。原始日志和整理轨迹中的后状态保留用于调试；未经结算
的后状态不能当作完整效果或奖励证据。

这是记录的已知局限：连续操作时后状态可能仍显示旧值，也可能包含下一动作的影响。例如
领取绿钥匙后立即拾取药水，关联的后状态可能仍显示 `emerald=false`，而稍后的真实状态已
变为 true。`resolution=next_decision` 只表示关联在下一决策边界关闭，不代表结算完成。
这些快照不会被静默改写成“最终结果”，使用边界见 [轨迹格式说明](DATA_AND_TRACE_SCHEMA.md#人类采集轨迹的动作后状态)。

需要区分视觉动画和实际执行：快速连出的牌可能先排队，后续还可能被游戏取消。因此仍要保留
执行跟踪，使用确切执行前状态，确认 `execution_status=executed` 并验证动作合法。不能仅凭
点击/提交记录生成正例。升级/删牌等取消重选仍按选择事务校验最终确认，不依赖动画完成。

历史只包含当前决策之前已发生的操作；当前动作的后状态、结果与未来步骤不会作为输入。

## 输出

```text
human-a0-v1/
  manifest.json                 数据、来源、代码哈希与 prompt 模板版本
  dataset/human-sft.jsonl        单步样本、完整 prompt 及等价 prompt_variants
  quality_report.json           对局/场景/排除原因和各模板字符、token 长度
  excluded-steps.jsonl          被排除动作及原因
  previews/<screen>--<template>.md
  trajectories/run-0000/        重建后的下游轨迹与 bridge 诊断记录
```

`sample_id/run_id/provenance` 可追溯原动作及提交/执行状态 ID；`run_outcome` 只作为元数据。
训练只监督目标动作别名与结束标记，输入 token 的 label 为 -100，不生成虚构思考。
使用 `manifest.json` 的哈希检查确保传输后数据未改变。

## 训练

训练入口：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.learning.sft --dataset data/datasets/human-a0-v1/dataset/human-sft.jsonl --output outputs/sft/run-name --model <base-model>
```

训练默认在同一种子族内分组，避免同一种子重试局的相邻步骤泄漏到训练和验证两边。
`--validation-seed-groups N`（兼容别名 `--validation-runs N`）按种子族留出验证；设为 0 时不留验证集。
只有随机/未知种子时，代码退回按 run ID 分组，此时无法排除未记录的同种子重试。
每轮训练中，每条决策从三种布局中确定性轮换一种，不增加样本数或优化步数；验证集使用 canonical。

训练 manifest 记录所用模板、切分的种子组、数据和模型信息。不同模板只验证对格式变化的鲁棒性，
不能替代独立种子上的完整对局评测。loss 和 token 准确率也不能代替实际通关率。
