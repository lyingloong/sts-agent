# 命令入口

在克隆的仓库根目录用虚拟环境执行 `python -m pip install -e .`。激活虚拟环境后可使用表中的短命令，也可用同一解释器执行 `python -m`。下表带参的命令需将占位符替换为实际路径。

| 用途 | 模块入口 | 安装后的命令 |
| --- | --- | --- |
| 离线查看或验证模式配置 | `python -m sts_agent.commands.check_config` | `sts-agent-config` |
| CommunicationMod controller | `python -m sts_agent.runtime.controller` | `sts-agent-controller` |
| Overlay | `python -m sts_agent.ui.overlay` | `sts-agent-overlay` |
| 文本客户端 | `python -m sts_agent.ui.terminal` | `sts-agent-terminal` |
| 本地 Qwen JSONL 服务 | `python -m sts_agent.policy.local_qwen` | `sts-agent-policy` |
| 下载基座模型 | `python -m sts_agent.commands.download_model` | `sts-agent-download-model` |
| 运行自建基线模型服务 | `python -m sts_agent.policy.serving` | `sts-agent-serve-baseline` |
| 连接并检查模型 API | `python -m sts_agent.policy.api` | `sts-agent-check-baseline` |
| CommunicationMod 只读探针 | `python -m sts_agent.commands.communication_probe` | 无 |
| 自建或托管模型的固定种子评测 | `python -m sts_agent.commands.evaluate_baseline` | `sts-agent-evaluate-baseline` |
| 合并多个已完成批次为同一 cohort | `python -m sts_agent.commands.merge_batches <批次目录>...` | `sts-agent-merge-batches` |
| 导出已完成基线局的原始输入输出复盘 | `python -m sts_agent.commands.export_baseline_io --batch-report <批次目录>` | 无 |
| CommunicationMod 状态/动作事务桥 | `python -m sts_agent.recording.communication_bridge --config config/agent.json` | `sts-agent-communication-bridge` |
| 导出教师 SFT JSONL | `python -m sts_agent.learning.export_teacher_sft` | `sts-agent-export-sft` |
| 导出 ActionRecorder 人类 SFT | `python -m sts_agent.learning.export_human_sft` | `sts-agent-export-human-sft` |
| 构建迁移 bundle | `python -m sts_agent.learning.training_bundle` | `sts-agent-bundle` |
| LoRA SFT | `python -m sts_agent.learning.sft` | `sts-agent-train-sft` |
| 远端在线 RL worker | `python -m sts_agent.learning.online_rl.worker` | `sts-agent-online-rl` |
| 本机 Codex MCP 服务 | `python -m sts_agent.codex_mcp` | `sts-agent-codex-mcp` |
| 打印合成局面的完整提示词预览 | `python -m sts_agent.commands.preview_policy_prompt` | 无 |

首次设置建议先运行 `sts-agent-config --list` 查看模式，再运行 `sts-agent-config --mode baseline_api` 查看选中模式的配置摘要。
`--all` 可检查全部模式，`--config PATH` 可检查其他配置文件。这些检查可在游戏关闭时离线完成。
若某个模式缺少必填项，`--all` 会显示模式名并以非零状态退出。例如，默认示例未指定在线 RL 使用的已训练 adapter，仍可正常使用 `collect_human`。
实际连接托管服务前先配好自己的模型与凭据，再运行 `sts-agent-check-baseline --mode baseline_api`。运行条件见[服务配置](BASELINE_INFERENCE.md)。
