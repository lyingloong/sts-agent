# 迁移说明

旧版根目录 `tools/*.py` 已移除，源码统一进入 `sts_agent/`。不提供旧路径兼容层。

## 新机器

```bash
uv venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r requirements/train.txt
```

随后修改 `config/agent.json` 中的模型路径、SSH 主机和 API key 文件。轨迹、训练 bundle、模型 adapter 与在线 RL 输出按实验需要单独迁移。

## 入口替换

| 旧入口 | 新入口 |
| --- | --- |
| `tools/communication_controller.py` | `python -m sts_agent.runtime.controller` |
| `tools/overlay_terminal.py` | `python -m sts_agent.ui.overlay` |
| `tools/human_terminal.py` | `python -m sts_agent.ui.terminal` |
| `tools/policy_server.py` | `python -m sts_agent.policy.local_qwen` |
| `tools/train_sft.py` | `python -m sts_agent.learning.sft` |
| `tools/online_rl_worker.py` | `python -m sts_agent.learning.online_rl.worker` |

CommunicationMod 配置、远端启动命令和本地自动化均应更新。验证方式见根目录 README。
