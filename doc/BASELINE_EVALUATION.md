# StS1 基线评测：进度分与完整对局

更新：2026-10-12。baseline_inference 已接入本文的进度分、Boss 计数和跨局统计，并提供固定 10 种子批量入口。
评分是只读评测，不影响动作选择，不等于 RL 训练奖励。

## Orak 的口径

[Orak 论文 Appendix L.3](https://arxiv.org/html/2506.03610#A12.SS3) 定义：

```text
S = 已清楼层数 + (50 / 3) × 已击败的幕末 Boss 数
```

这是 benchmark 自定义分，不是游戏结算界面的 score。论文按前三幕的 50 个已清楼层和
3 个 Boss 定义满分 100。死在第 43 层时，楼层部分是 42；若此前击败两个 Boss，
总分约为 75.33。只到达某层不等于清掉该层。

论文 Appendix L.1/L.4 的实验是 A0 铁甲战士、固定种子、完整解锁，重复 3 次；
温度 0、repetition penalty 1、最多 200 次环境交互。模型只负责战斗出牌/结束回合和
卡牌奖励选择，地图、事件、药水等由规则代理处理。一次模型输出可以包含多条动作，
所以其 200 次交互不等价于本项目的 200 个细粒度游戏动作。

还有一处需要明确：[公开 release 环境代码的 evaluate](https://github.com/krafton-ai/Orak/blob/release/src/mcp_game_servers/slay_the_spire/game/slay_the_spire_env.py#L165)
在死亡时返回 floor-1，胜利时返回 floor，未在这个函数中加入论文的 Boss 奖励。
因此不能只调用这个函数就声称复现了论文公式；目前也不能断言整个评测后处理没有加分。
本项目显式采用论文公式，命名为 Orak-style progress score，而非宣称完整复现 Orak 协议。

## 我们采用什么

首个实验仍限定 A0 铁甲战士、原版游戏机制、完整解锁情况一致；一个模型负责完整对局。
保留框架现有自动动作，并冻结其配置，让未微调模型与微调模型使用同一套规则。

| 指标 | 用途 |
| --- | --- |
| Orak-style 进度分，平均值与标准差 | 主指标；区分全部未通关时的实际进步 |
| 已清楼层、击败前三幕 Boss 数 | 展开主指标，避免同一个总分掩盖策略差异 |
| 第三幕通关率、第一/二幕通过率 | 辅助结果指标，不再只报告一个二元通关率 |
| 心脏通关率 | 单独的第四幕任务，不混进前三幕 100 分 |
| 完整自主对局比例、人工接管/技术中止比例 | 分清策略能力与框架可靠性 |
| 非法输出、命令拒绝、旧状态回答丢弃 | 分清格式错误、执行错误和状态变化 |
| 每步请求延迟 p50/p95、每局 token/耗时 | 衡量实时性与成本 |

计分实现位于 sts_agent/evaluation/progress.py，版本 orak_style_act3_v1：

- 进入第 N 层时暂记已清 N-1 层；源 room_phase=COMPLETE 且非死亡时可记第 N 层完成。
  死亡终局按 N-1 结算，不采信残留 COMPLETE 标记。
- Boss 击败按幕去重：活着的 MonsterRoomBoss 完成、BOSS_REWARD、进入下一幕可以证明通关。
  不看怪物名字或 HP=0，所以觉醒者变身、事件 Boss 和重复消息不额外加分。
- 楼层部分上限 50。确认第三幕胜利或进入第四幕后记为 50 层、3 Boss，总分 100，
  不让引擎终局 floor=51 或第四幕楼层把主指标抬到 100 以上。
- 保留 highest_floor、highest_act、terminal_floor 和原始 terminal_state，不覆盖引擎楼层。
  进入第四幕后即使死于心脏，前三幕进度仍是 100；won 表示实际整局胜利，act3_cleared
  表示第三幕通过，beat_heart 表示心脏胜利，三者不是同一个字段。
- 缺失楼层时不造 0 分；缺失终局必要信息时不标记 score_final。终局锁定后不被返回菜单覆盖。

目前计分针对标准三幕游戏，首个实验使用 A0 铁甲战士；A20 双 Boss 或改变楼层结构的模组
不属于此版验收范围。真实游戏的房间完成/跨幕边界仍需实机复测。

未完成、局中接入、人工接管的轨迹必须标明；不应与完整无干预结果混为同一个平均分。
技术中止可报告截至中止的进度，但不能把它假装成游戏死亡或最终得分，也不能只删掉失败
尝试让结果变好。公布全部启动次数和各类结果数，同时报告策略完成局统计与端到端可靠性。
自动环境动作不算人工干预，但必须披露规则及数量。

同样使用这个分数，不代表我们的结果能直接和 Orak 表格比较：模型负责的动作范围、
种子、信息内容、交互粒度、自动规则和预算都不同。直接对标需要另做 Orak 协议实验。

## 固定 10 种子批量测评

入口为 `sts_agent.commands.evaluate_baseline`。它只是本机测评协调客户端，
不选择局内动作、不训练、不部署或重载远端模型。局内仍由现有 controller、
同一模型和同一套环境自动规则处理。

测试集位于 [`config/evaluation/a0_ironclad_10.json`](../config/evaluation/a0_ironclad_10.json)：
`a0-ironclad-fixed10-v1`，A0 铁甲战士，10 个不同的固定种子，各跑一次。
种子来自 `random.Random(20261012).getrandbits(63)` 的前十个结果，再转换为游戏显示种子；
没有依据游玩成绩筛选。已用本机游戏的 Java `SeedHelper.getLong/getString` 往返验证，
并确认不是特殊 Trial 种子。

普通固定种子游戏提供完整涅奥祝福，不受上一局是否到达第一幕 Boss 影响。
这个结论已核查本机游戏/CommunicationMod 的实际逻辑；脚本不会改动角色历史或存档。
但种子不会统一解锁进度或心脏开放条件：测试前应确认普通主菜单，保持相同游戏存档、
解锁状态、语言、模组及 jar 版本。不要从 Daily/Trial/Endless 或正在进行的游戏接入。
这些环境条件目前需要操作者确认，不应声称客户端已自动验证全部隐藏设置。

### 运行

1. 准备模型后端：远端 vLLM 用 `baseline_inference`，没有 GPU 时用托管的 `baseline_api`；
   本机 `agent.json.active_mode` 必须与之一致。模型与 API 参数仍只在对应模式配置中维护，
   测试集不重复定义它们。
2. 重启游戏/controller，使新增批量协议生效，停在普通主菜单。
3. 关闭 Overlay 和终端客户端。controller 只接一个客户端，批量脚本占用该连接；
   不要另开一个 controller，也不要在测评期间点击游戏。
4. 在本机仓库根目录执行：

```powershell
# 只校验并显示计划：不联网、不写结果、不开局
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline --dry-run

# 单种子冒烟（先用它验证链路，约 20 分钟）
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline --suite config/evaluation/a0_ironclad_smoke.json

# 完整运行 10 局
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline
```

托管 API 受请求频率限制（USTC 为 20 RPM）。客户端按请求起点计间隔，推理耗时计入其中，
只有响应快于 3 秒时才补足差额；实测单局 15–20 分钟，10 种子约 3 小时，远端 vLLM 约 12 分钟/局。
首次调用可能是冷启动，可先运行 `python -m sts_agent.policy.api --mode baseline_api` 预热。

可用 `--config`、`--suite` 指定其他配置文件，`--host/--port` 覆盖 controller 连接地址。
已经在局内时脚本直接拒绝接管，不放弃/重置现有游戏。
启动时会比较 controller 已加载的模型版本与 API 设置；真正不一致时，错误信息列出
具体字段（如 `settings.temperature`），并在批次 manifest 的 `preflight` 中保留收到的
实验配置与差异列表。配置里的空 adapter 路径 `""` 与 manifest 的 `null` 都表示未加载 adapter，
不会仅因表示形式不同而要求重启。

串行流程：连接并校验模式/模型/协议 → 注册批量监督 → 指定种子开局 →
接收进度与模型状态 → 核对本局种子/角色/终局 → 等正常流程返回主菜单 → 下一种子。
重复菜单消息不会重复开局，终局动画中的状态也不能直接触发下一局。
最后一局回到主菜单后解除监督并退出，不再开第 11 局。

默认预算在测试集的 `limits` 中设置：连接 30 秒、无状态消息 600 秒、
每局从开局到返回菜单最多 3600 秒、每局最多 4000 个已接受游戏动作。
只读 `STATE` 和 API 调用不算游戏动作，环境自动动作计入预算并单独统计。
这是防止失联/死循环的技术停止预算，不是用截短轨迹充当完整游戏。
预算在批次前调整并冻结，不要根据某个种子的结果临时放宽。

模型输出错误、输入超限、控制器暂停、种子不符、未知/不合格终局、预算耗尽或
连接中断都会停止整批，保留前面已完成的局和剩余计划。不随机回退、不自动重试整局、
不跳过坏种子；API 原有的有限网络重试仍保留。
`Ctrl+C` 同样保存部分报告，并先关闭客户端连接；controller 在下一个安全决策点暂停，
不会因批量客户端消失而继续自动游玩。已经提交的动作或正在播放的动画不能撤销。
之后应先检查原因再人工恢复，不支持自动断点续跑。

### 批次输出

每批创建新的 `data/experiments/baseline_batches/<suite_id>-<时间>-<后缀>/`：

- `suite.json`：冻结的种子列表和预算。
- `manifest.json`：测试集哈希、controller 地址、原始实验目录、模型/SHA/prompt/设置快照，以及批次状态与失败原因。
- `results.json` / `results.csv`：每个计划种子对应的 run ID、实际内部种子、耗时、得分、幕/楼层、胜负、模型/环境/人工动作数、拒绝数、API/token/延迟统计和原始轨迹路径。
- `evaluation.json`：合格完成局的平均进度分/样本标准差、各幕通过率、胜率及请求统计；同时列出计划、已启动、完成、未启动和技术中止情况。

计划行状态为 `pending`、`completed` 或 `technical_abort`；`attempted_at` 表示开始提交开局，
`started_at` 表示收到新的对局 ID。未收到游戏确认的开局尝试不能声称已成功开始游戏。
技术中止不是游戏死亡，不向平均分加入 0，也不进入完成局胜率的分母。
它可能保留中止前的暂时得分，应结合 `score_final` 和状态解释。
因此部分完成报告必须同时展示技术中止与未执行数量，不能只引用完成局的平均成绩。
若所有局已完成但正常解除连接失败，批次仍会标明失败原因，不丢掉已完成的局。

原 controller 的 `data/experiments/baseline/<会话>/` 中仍保存完整 prompt、API 回答、
逐局消息/动作轨迹；批量脚本不改写它们。批次统计只关联本批的 run ID，不混入先前手动测试。
模型请求原文日志较大，统计仅在局结束等边界读取，避免每次进度刷新都扫描全部日志。

10 个固定种子适合当前初测和后续模型的配对比较，但不是充分的泛化或稳定胜率证据。
正式报告应扩大预先冻结的测试集、报告不确定性，训练种子不能进入正式测试集。

## 结果文件

baseline 会话目录中：

- manifest.json 保存 evaluation_metric=orak_style_act3_v1。
- summary.json 仍以 run_id 为键，每局新增 evaluation：progress_score、cleared_floors、
  bosses_defeated、boss_acts_defeated、score_final、act3_cleared 和 autonomous_eligible 等。
  模型、自动和人工动作计数保持原有字段。
- evaluation.json 是当前会话汇总，包含 runs_started、terminal_runs、unfinished_runs、
  human_intervened_runs、partial_runs、eligible_runs，以及 progress_score_mean/std、
  act_clear_rates、win_rate、heart_win_rate。

平均分、样本标准差和比率只使用 autonomous_eligible=true：真实终局、分数可结算、
无 controller 记录的人工动作、非 partial_run。只有一局时标准差为 0；没有符合条件的局
时平均分和比率为 null，不伪装成 0。未完成或人工参与的局仍保留各自分数和排除计数。
心脏胜率当前以这些合格局全部为分母，不是“进入第四幕后”的条件胜率。
主菜单 START 时注册对局，即使之后未收到有效游戏状态也会留下未完成计数。
汇总只覆盖同一个 controller 会话；跨会话/不同角色或进阶的实验不能直接混算。

controller 只向计分器传正常可决策观察，不传只读采集快照。终局即使没有 pending 动作
也会记录。游戏中直接点击的人工干预不能仅凭 controller 完全检测，因此评测时不要手动操作；
UI/终端提交的人工动作会明确排除该局。计分不改变 RL 奖励或人类采集格式。

## 本次协议排查与修复

排查对象：data/experiments/baseline/20261007-145610-41a617c9。
该会话有模型/环境/人工动作，未记录真实终局，不能当成完整自主游戏能力样本。

Recorder 在动作入口、队列执行前、效果结算时主动调用 CM 发布器，以取得精确采集状态。
这些消息复用脚本管道，原 ready_for_command 有时仍为 true。旧 controller 把它们当成
待执行命令的应答，提前写 accepted 并推进下一步；后续错误就可能配到错误的命令。
这足以解释一部分重复领取、过早提交和“暂停，需人工处理”，不能据此归因于模型不会玩。

修复分工：

- Recorder 增加 recorder_snapshot_only 和 recorder_snapshot_kind，保留原 readiness、完整状态与 ID。
  标记只在同步采集发布期间生效，异常或嵌套调用后恢复，不改变正常 CM 的可决策发布时机。
- controller 照常保存采集消息，但不让其确认 pending、复核回答或触发策略。
  人类采集 bridge 仍保留它们，动作/状态按明确 ID 匹配。
- 命令错误后先发 STATE，拿最新观察再交给玩家；不使用错误包之前的旧界面。
  ask_action 只选择动作，main 统一写出 STATE 和动作，避免恢复命令漏发/重复发送。
- GAME_OVER 按 screen_state.victory 区分死亡/胜利；不能只看界面枚举判断失败。

必须同时使用新版 Recorder jar 和 Python controller，并重启游戏/controller。
仅更新 Python 无法可靠区分旧 Recorder 的无标记采集消息。不通过增加等待时间掩盖此问题。
CM 没有完整的命令 request ID，因此这个修复也不是事务锁；模型运行期间仍不要手动点击游戏。

本地回归测试覆盖采集消息插入、单一 STATE 写出、错误后最新界面接管、恢复命令及终局胜负；
Java smoke 覆盖标记作用域和实际补丁签名/编译。批量协议另覆盖多局串行、重复消息、失联/超时/暂停、
种子与终局校验、正常解除连接。固定 10 种子批次已于 2026-10-12 完整执行（10/10 终局，
平均进度分 27.33，通关率 0）；托管 API 模式的单局与批次链路尚待实机验证。
