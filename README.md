# StS Agent

针对《Slay the Spire 1》的结构化游戏 Agent 原型。当前目标是先用 CommunicationMod 建立可靠的人类轨迹采集链路，再训练小型 Qwen 模型做分层决策和无人值守运行。

## 当前状态

- 游戏：StS1 + ModTheSpire + BaseMod + CommunicationMod
- 采集方式：`communication_controller.py` 作为 CommunicationMod 外部进程，`human_terminal.py` 通过本机 TCP 提供输入
- 当前不需要额外编写 Java Mod；原生鼠标动作记录仍是后续工作
- 状态输入使用 CommunicationMod 的结构化 JSON，并在轨迹中移除抽牌顺序、UUID 和随机种子等隐藏信息
- 地图决策保留当前 Act 的完整公开地图图结构，而不是只有下一层节点
- 战斗状态保留敌人当前意图的伤害/多段攻击数值；卡牌效果同时使用 CommunicationMod 可见字段和本地静态知识库
- 终端和模型状态支持本地 StS1 卡牌知识库兜底，运行时卡牌字段优先于静态字段
- 支持原版四角色：Ironclad、Silent、Defect、Watcher；支持 A0-A20 和可选种子
- 数据、模型和运行日志默认不提交 Git

## 快速开始

### 1. 配置 CommunicationMod

在 CommunicationMod 的 `config.properties` 中将 `command` 指向本项目的控制器，例如：

```properties
command=D\:/miniconda3/python.exe D\:/Code/spire-mod/sts-agent/tools/communication_controller.py
runAtGameStart=true
```

路径按本机 Python 和项目目录修改。重启 ModTheSpire 后，CommunicationMod 会自动启动控制器。

### 2. 启动终端客户端

```powershell
cd D:\Code\spire-mod\sts-agent
python tools\human_terminal.py
```

终端客户端支持 `--no-clear` 保留历史输出、`--no-color` 禁用 ANSI 颜色；游戏中可用 `s` 或 `r` 请求刷新，`h`/`help` 在本地显示帮助，不会发送游戏动作。

首次使用或卡牌知识库缺失时执行一次：

```powershell
python tools\fetch_card_knowledge.py
```

该命令将本地 StS1 卡牌快照写入 `data/knowledge/`；目录已被 Git 忽略。知识库来自按游戏 JAR 解析的第三方数据，更新游戏版本后应重新下载并检查差异。

### 3. 开始游戏

在终端输入：

```text
s                         Ironclad A0，随机种子
s ABC123                  Ironclad A0，固定种子 ABC123
s SILENT 5 ABC123        Silent A5，固定种子
s DEFECT 10              Defect A10，随机种子
s WATCHER 20 ABC123      Watcher A20，固定种子
```

角色名也接受 `THE_SILENT`。旧的 `s [seed]` 写法继续可用。角色、进阶和种子会写入 `run-events-*.jsonl`，轨迹中的每条动作也带 `run_config`。

## 终端动作

```text
p <手牌序号> [敌人序号]   出牌；单个存活敌人时可以省略目标
e                        结束回合
u <药水槽位> [敌人序号]  使用药水
d <药水槽位>             丢弃药水
<序号> / m <序号>        选择事件、奖励等选项
c                        确认或继续
x                        取消、返回或跳过
s                        游戏中请求当前状态；主菜单中开始游戏
h                        显示帮助
q                        断开当前终端，controller 保持运行
```

终端客户端每次刷新状态、帮助或错误都会清屏。`h`/`help` 在客户端本地重绘当前状态并显示帮助，`s`/`r` 在游戏中请求最新状态；`--no-clear` 和 `--no-color` 可关闭清屏或颜色。`p 1` 在单敌人战斗中会自动解析为实际的 `PLAY 1 0`；多敌人时必须显式指定目标。CommunicationMod 的 `START` 种子只允许字母数字；运行状态中的 `game_state.seed` 是内部有符号整数，不是界面显示的种子字符串。

控制器会同时记录 `run_config.seed`（可输入的显示种子）、`run_config.internal_seed`（协议原始数值），并在随机开局时按 StS1 的 `0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ` 编码（省略 `O`）将内部值还原为显示种子，再用显示种子命名轨迹文件。内部值只用于诊断和校验，不应直接填入 `START`。

同一个游戏进程可以连续采集多局：回到主菜单后再次输入 `s` 即可开始下一局，controller 会为每局分配新的 `run_id`。输入 `q` 只断开当前终端客户端，不会关闭 controller；重新运行 `python tools\\human_terminal.py` 即可重新连接。要停止 controller，需要结束 ModTheSpire/CommunicationMod 外部进程。

## 输出文件

控制器会自动创建以下目录：

```text
data/raw/communication_controller/messages-<session>.jsonl  # 原始协议消息
data/trajectories/action-trace-<session>-run-<n>-<character>-A<ascension>-seed-<seed>.jsonl
                                                               # 单局状态-动作-结果轨迹
data/trajectories/run-events-<session>-run-<n>-<character>-A<ascension>-seed-<seed>.jsonl
                                                               # 单局生命周期和种子事件
```

原始消息按 controller session 保存；轨迹和 run-events 按单局保存。随机种子在第一条游戏状态中出现前，文件名暂用 `seed-random`，收到 `game_state.seed` 后还原出显示种子并自动改名。正式训练主要使用单局 `action-trace`；原始消息只用于排错和重新规范化，不能直接作为模型输入。中途退出的局应标记为调试数据，不进入正式 SFT 集。

地图和战斗信息说明：地图上的已生成节点在游戏界面中是公开的，因此会保留完整节点、坐标和连接关系。敌人状态会尽量保留 `move_adjusted_damage`、`move_hits` 等当前意图数值。CommunicationMod 并不保证返回每张卡牌的完整动态描述；动态值不会仅凭卡牌 ID 推断，只有明确标记来源的版本化静态知识库才作为 fallback。

CommunicationMod 偶尔会在事件房间尚未完成时短暂返回 `screen_type=MAP`、`room_phase=EVENT`。controller 将其标记为 `screen_type=TRANSITION`，不会暴露地图节点选择；终端此时只允许 `x` 返回或 `s` 刷新，避免提前进入下一个问号房间导致事件重复显示。

卡牌信息采用两层来源：如果 CommunicationMod 返回当前卡牌的 `damage`、`block`、`magic_number` 或描述，使用 `effect.source=runtime_over_static`；否则使用 `effect.source=static_knowledge`。静态 `effect` 包含卡牌类型、稀有度、颜色、目标、费用、基础数值、关键字、标签和完整升级字段。静态值不包含力量、易伤、遗物等实时修正，模型仍应读取当前战斗状态。

## 文档

- [实施计划与训练路线](doc/PLAN.md)
- [通信协议与数据格式](doc/PROTOCOL.md)

## 验证

```powershell
python -m py_compile tools\communication_controller.py tools\human_terminal.py
python tools\human_terminal.py --help
```

完整游戏测试需要启动 ModTheSpire；代码测试不会替代实际协议测试。

## 设计边界

第一阶段只做结构化状态 Agent，不做视觉识别，不把 Qwen 嵌入游戏 Mod，不在一局内更新模型权重。SFT 用于冷启动，后续通过 shadow、assist 和 DAgger 收集模型访问到的困难状态。

CommunicationMod 的协议和限制见其[官方仓库](https://github.com/ForgottenArbiter/CommunicationMod)。
