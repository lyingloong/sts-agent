# 命令入口

安装 `pip install -e .` 后可以使用短命令；未安装 console scripts 时，使用表中的 `python -m` 形式。所有命令均从仓库根目录运行。

| 用途 | 模块入口 | 安装后的命令 |
| --- | --- | --- |
| CommunicationMod controller | `python -m sts_agent.runtime.controller` | `sts-agent-controller` |
| Overlay | `python -m sts_agent.ui.overlay` | `sts-agent-overlay` |
| 文本客户端 | `python -m sts_agent.ui.terminal` | `sts-agent-terminal` |
| 本地 Qwen JSONL 服务 | `python -m sts_agent.policy.local_qwen` | `sts-agent-policy` |
| 下载基座模型 | `python -m sts_agent.commands.download_model` | `sts-agent-download-model` |
| 刷新静态知识 | `python -m sts_agent.commands.fetch_knowledge` | `sts-agent-fetch-knowledge` |
| CommunicationMod 只读探针 | `python -m sts_agent.commands.communication_probe` | 无 |
| ActionRecorder 人类轨迹接收与整理 | `python -m sts_agent.recording.action_recorder --config config/agent.json` | `sts-agent-record-human` |
| 导出教师 SFT JSONL | `python -m sts_agent.learning.export_teacher_sft` | `sts-agent-export-sft` |
| 构建迁移 bundle | `python -m sts_agent.learning.training_bundle` | `sts-agent-bundle` |
| LoRA SFT | `python -m sts_agent.learning.sft` | `sts-agent-train-sft` |
| 远端在线 RL worker | `python -m sts_agent.learning.online_rl.worker` | `sts-agent-online-rl` |

`teacher.py`、`prompt_evolution.py`、`posthoc_feedback.py` 和知识模块是库组件，由 controller 或学习流程调用，不是独立命令。不要为调用内部函数重新增加根目录脚本。
