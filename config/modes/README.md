# 运行模式配置

这些文件是基于 `tools/agent_config.py` 默认值的严格稀疏配置：未写出的共享字段使用默认值，写出的字段覆盖默认值。这样每种模式只维护自己的开关和关键参数，不再复制整份配置。

启动时给 controller 和客户端传入同一个配置文件：

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

`config/agent.json` 继续保留，作为兼容旧启动命令 `python tools\\...` 的当前活动配置。新实验建议显式指定模式文件，避免误用上一种模式。
