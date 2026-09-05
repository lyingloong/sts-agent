# 运行模式配置

这些文件是基于 `tools/agent_config.py` 默认值的严格稀疏配置：未写出的共享字段使用默认值，写出的字段覆盖默认值。这样每种模式只维护自己的开关和关键参数，不再复制整份配置。

推荐直接修改 `config/agent.json` 的 `active_mode`，这样 controller 和客户端天然使用同一配置。旧的分文件启动方式仍兼容：

```powershell
python tools\communication_controller.py --config config\modes\candidate_assist.json
python tools\overlay_terminal.py --config config\modes\candidate_assist.json
```

可用模式：

| 文件 | 用途 |
| --- | --- |
| `manual.json` | 人工操作，不启动模型 |
| `candidate_assist.json` | 本机 Qwen Candidate Assist，达到置信度阈值时自动执行 |
| `teacher_sft.json` | DeepSeek 教师自动决策，收集 SFT 轨迹 |
| `prompt_evolution.json` | 教师驱动的 Prompt 演化；首次可随机种子，随后固定真实游戏种子比较候选 |
| `remote_inference.json` | 远端只推理；请把 `initial_adapter` 改成远端实际 adapter |
| `remote_online_rl.json` | 远端在线 RL；请按实验修改 adapter 和 `online_rl.json` |

`remote_inference.json` 和 `remote_online_rl.json` 中的 Linux 路径是当前训练机示例，迁移到其他机器时只需在本地配置中改路径。密钥仍放在项目根目录的 `ustc_api.txt`，不会进入 Git。

教师模式默认对网络错误重试 3 次，并使用 5、10、20 秒退避（按远程 RPM=20 设置）；可在 `teacher_sft.json` 或 `prompt_evolution.json` 中调整 `max_retries`、`retry_backoff_seconds` 和 `retry_max_backoff_seconds`。Prompt Evolution 还会强制动作和候选生成请求使用 `temperature=0`。

`config/agent.json` 是默认唯一入口；修改 `active_mode` 后重启 controller 和客户端即可切换模式。分文件配置仅用于兼容旧命令或临时隔离实验。

`knowledge_experience` 不是一种独立的游戏决策模式。启用后，它作为教师动作决策的带外能力：教师通常选择动作；对某个当前公开实体的机制或策略依据有疑问时，也可直接返回一次 `knowledge_request`。完成“人类回答 -> DeepSeek 整理 -> 人类审核”后，控制器回到同一个游戏状态重新请求动作。
