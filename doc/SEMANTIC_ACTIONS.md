# 语义动作层

`sts-agent` 的模型动作空间采用“语义动作”，Communication Mod 的命令只作为执行协议。模型不需要学习界面按钮、列表下标和多步确认的偶然细节。

## 两层边界

```text
观察状态 -> semantic_actions() -> 模型选择一个语义动作
                                      |
                                      v
                         Communication Mod 原始命令
```

动作记录中的 `id` 是训练和策略接口的稳定令牌，`command` 是当前版本 Communication Mod 能执行的命令。二者不能混用：策略只输出 `id`，控制器只在本地把它解析为 `command`。

## 当前语义动作

### 营火

在 `REST` 界面，模型看到：

- `CAMPFIRE:REST`：休息
- `CAMPFIRE:SMITH:card=<uuid>`：锻造指定卡牌
- `CAMPFIRE:RECALL`：获得红钥匙
- `CAMPFIRE:LIFT`、`CAMPFIRE:DIG`：对应营火选项（出现时才提供）

锻造动作的底层流程是：选择营火锻造 -> 在牌堆中选择指定卡牌 -> 确认。控制器通过 `execution` 元数据自动发送后两个命令；轨迹中只保留一次 `CAMPFIRE:SMITH:card=<uuid>` 决策，不把 GRID 和确认按钮伪装成新的策略决策。

### 商店

结构化商店状态完整时，模型看到：

- `SHOP:BUY_CARD:<id-or-uuid>`
- `SHOP:BUY_POTION:<id>`
- `SHOP:BUY_RELIC:<id>`
- `SHOP:PURGE:card=<uuid>`

删牌动作同样由控制器自动完成“打开删牌 -> 选择指定卡牌 -> 确认”。药水栏满、苏祖、灵体外质等现有合法性过滤仍在语义动作生成前执行。

如果旧版 Communication Mod 没有提供足够的结构化商品信息，控制器保留 `CHOOSE:index=<n>` 兼容路径，避免错误猜测商品身份。这只影响旧协议，不改变新协议下的动作定义。

## 轨迹约定

一个语义动作对应一个训练决策：

- `pre_state`：模型选择前的稳定状态；
- `legal_actions`：语义动作列表；
- `action`：模型选择的语义动作；
- `command`：该动作的第一条底层命令；
- `action.execution`：仅供控制器执行多步流程的内部状态；
- `post_state`：整个语义动作完成后的稳定状态。

中间的自动命令不会再次请求模型，也不会新增 SFT/GRPO 决策步。若执行器找不到目标卡牌或确认界面异常，动作不会被伪造完成，控制器会停留在当前状态等待人工处理。

人工采集要求 ActionRecorder 事件格式 0.2。Mod 在进入锻造或删牌流程时缓存初始状态，在游戏实际升级或移除卡牌时直接输出 `CAMPFIRE:SMITH:card=<uuid>` 或 `SHOP:PURGE:card=<uuid>`，随后补充动作后状态。recording 接收端直接按 ID 匹配初始合法动作，不依赖中间选牌、确认事件，也不再归并点击序列。

`decision_started`、`decision_detail` 和 `decision_cancelled` 留在 Mod 原始 JSONL 中用于复盘，不生成策略决策或动作历史。取消操作不会生成成功的锻造/删牌动作。若 Mod 没有捕获起始状态，它会保存实际完成动作并将初始状态设为 null，接收端标记 `usable_for_sft: false`。旧版细粒度日志需单独迁移。

## 扩展原则

新增场景时先定义稳定的语义动作 ID，再实现一个小型执行器适配器。不要把鼠标坐标、按钮位置或暂时的列表下标写入提示词。对于纯粹的动画等待、继续按钮和确认按钮，应由执行器处理；只有具有策略意义的选择才进入 `legal_actions`。
