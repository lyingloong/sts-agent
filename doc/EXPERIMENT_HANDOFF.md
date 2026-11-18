# A0 铁甲战士 30 种子基线（历史批次）

2026-11-18 已停止该批次。托管模型的固定 10 局全部完成。扩展到 30 个种子时复用了这 10 局，第 11 个种子在第 4 层中断，扩展批次新增终局数为 0。
以下保留当时的配置和操作记录，供排查历史结果。其他使用者可按[固定种子评测说明](BASELINE_EVALUATION.md)选择自己的配置。

## 0. 一分钟版

```powershell
# 从仓库根目录运行。历史批次复用了 10 局，新检出仓库的复用数量取决于本机数据。
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline `
  --suite config/evaluation/a0_ironclad_30.json --dry-run

# 正式跑；前台运行，别关窗口
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline `
  --suite config/evaluation/a0_ironclad_30.json
```

预计 20 局约 4 小时（夜间实测 11.6 分钟/局），傍晚高峰可能慢 2–4 倍。
跑完在 `data/experiments/baseline_batches/a0-ironclad-fixed30-v1-<时间>-<后缀>/`。

## 1. 分工

- **人**：启动游戏（Steam → ModTheSpire），保证 Steam 在线；游戏崩溃时重启。
- **agent**：启动批次、盯进度、异常时停下并报告、结束后汇总结果。
- 游戏必须保持运行且停在**普通主菜单**；批次运行期间不要点游戏。

## 2. 启动前检查（逐条打勾）

```powershell
cd D:\Code\spire-mod\sts-agent

# 1) controller 在监听
Get-NetTCPConnection -LocalPort 8765 -State Listen

# 2) 当前模型与限速
.\.venv\Scripts\python.exe -c "from sts_agent.config import load_agent_config; c=load_agent_config(); a=c['policy']['api']; print(c['policy']['model']['base'], a['provider'], a['rpm_limit'])"
# 期望：qwen3.8-chat openai_compatible 18

# 3) 没有别的客户端占用：Overlay 和终端必须关闭
Get-Process | Where-Object { $_.ProcessName -match 'python|pythonw' } | Select-Object Id, ProcessName
```

游戏必须停在主菜单。若在局内、在 Daily/Trial、或有 Overlay 连着，批次会直接拒绝。

## 3. 启动

前台开一个终端跑，输出会逐局打印：

```json
{"run": "11/30", "phase": "await_run", "seed": "2E68S0D9KAXX4", "accepted_actions": 37, ...}
```

`run` 的分母是目标 30 局，分子是当前种子的序号（前 10 个被复用，所以从 11 开始）。

不要用 `Start-Job` 或关掉终端；进程挂了游戏会停在原地，批次客户端断开后 controller
会在下一个安全点暂停。真要挂后台，用独立终端窗口并保持打开。

## 4. 看进度

### 4.1 批次输出目录

```powershell
$dir = Get-ChildItem data\experiments\baseline_batches -Directory -Filter "a0-ironclad-fixed30-v1-*" |
  Sort-Object Name -Descending | Select-Object -First 1
$dir.FullName
```

目录里：

| 文件 | 内容 | 更新时机 |
| --- | --- | --- |
| `results.csv` | 每局一行：状态、分数、楼层、动作数、token、延迟 | 每局结束 |
| `results.json` | 同上，含完整 summary 与复用来源 | 每局结束 |
| `evaluation.json` | 当前已完成局的聚合（均值、标准差、各幕通过率、胜率） | 每局结束 |
| `manifest.json` | 批次状态、失败原因、复用了哪些局 | 状态变化 |
| `suite.json` | 冻结的 30 个种子与预算 | 创建时 |

### 4.2 一条命令看进度

```powershell
$dir = Get-ChildItem data\experiments\baseline_batches -Directory -Filter "a0-ironclad-fixed30-v1-*" |
  Sort-Object Name -Descending | Select-Object -First 1
$rows = Get-Content "$($dir.FullName)\results.csv" | ConvertFrom-Csv
$done = $rows | Where-Object { $_.status -eq 'completed' }
"{0}: {1}/{2} completed" -f $dir.Name, $done.Count, $rows.Count
$done | Select-Object index, seed, progress_score, highest_floor, highest_act | Format-Table -AutoSize
if ($done) { "mean so far = {0:N2}" -f (($done | Measure-Object progress_score -Average).Average) }
```

建议每 10–15 分钟看一次，不要频繁读 `requests.jsonl`（单局 4 MB 以上）。

## 5. 正常结束

- 30/30 `completed`，`manifest.json` 的 `status` 为 `completed`，`failure` 为空。
- `evaluation.json` 里 `progress_score_mean`、`progress_score_std`、`act_clear_rates`、`win_rate` 齐全。
- 汇报时给出：批次目录、30 局均值与标准差、各幕通过率、通关率、总耗时、
  异常局（technical_abort/人工接管）以及 `api_performance` 的调用数和延迟。

## 6. 异常处理

批次遇到任何错误都会**停止整批**并保留已完成局，不会自动重试整局或跳过坏种子。
修复后重新执行同一条命令即可：已完成的种子会自动复用，只补缺的。

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `controller model/settings differ from current config` | 游戏/controller 加载的是旧配置 | 重启游戏 |
| `batch must connect at a ready main menu` | 不在主菜单，或 Overlay 占用了连接 | 回主菜单，关 Overlay |
| `controller already has a batch attached` | 另一个客户端还连着 | 关掉它再重试 |
| `controller paused: ...` | 模型暂停（超时、连续错误、输入异常） | 看实验目录 `requests.jsonl` 的 `decision_error`，修好后重跑 |
| `run timeout while waiting...` | 单局超过 7200 秒 | 记录种子与服务延迟，重跑该批 |
| `429` / `503` | 网关限流或上游抖动 | 客户端自动退避重试；连续失败才停 |
| `controller disconnected` | 游戏崩溃或 controller 退出 | 重启游戏，重跑（复用已完成的局） |

不要通过改配置、点击游戏或重启 controller 来“绕开”报错；先看批次目录里的
`manifest.json` 和实验目录的 `requests.jsonl`。

## 7. 约束

- 不要修改代码、配置或测试集；这批实验要的是可复现的固定条件。
- 不要开第二个 controller、不要同时跑两个批次。
- 批次运行期间不要点游戏、不要开 Overlay、不要切换模式。
- 数据目录是 git 忽略的，不要提交轨迹或日志。

## 8. 其他模型的配置

只改一个字段，然后**必须重启游戏**：

```jsonc
// config/modes/baseline_api.json
"policy": {"model": {"base": "qwen3.8-flash"}}   // 换成目标模型
```

使用 `--dry-run` 查看所选模型已完成和待运行的种子；不同模型各自保留独立结果。
