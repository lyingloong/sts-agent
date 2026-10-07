# StS1 基线评测：进度分与完整对局

更新：2026-10-08。baseline_inference 已接入本文的进度分、Boss 计数和跨局统计。
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

## 下一次测试

1. 先用 1–2 局确认本次协议修复：没有采集快照触发的新动作、没有同一动作提前确认和错误错配。
2. 再做一组预先确定的种子，初测可用 5 局观察失败类型，正式比较建议至少 20 个不同种子。
   这些是实验规模建议，不是已有结果或统计充分性的保证。
3. 冻结模型 SHA、prompt、自动规则、解锁/模组/进阶设置和游戏动作预算。
   不照搬 Orak 的 200 次交互；本项目细粒度动作应单独设足以完成完整对局的预算。
4. 后续模型使用同一组种子进行配对比较，主看进度分的变化，辅看通关率、Boss 通过率与延迟。
   训练用种子不得进入正式测试集。

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
Java smoke 覆盖标记作用域和实际补丁签名/编译。构建已安装本机 jar，但真实游戏复测尚未进行。
