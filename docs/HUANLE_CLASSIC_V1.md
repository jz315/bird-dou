# 欢乐斗地主经典规则 v1 规格

## 状态与兼容性

本文件冻结 `huanle_classic_v1` 的产品规则输入，来源是《BIRD-Dou
欢乐斗地主规则与全系统修复计划》v2.0 第 1 节。它是 R001 的规格和金标
corpus，不是当前引擎已经支持的 profile，也不改变 `douzero_post_bid` 或
`canonical_full` 的语义。

`douzero_post_bid` 继续是 DouZero 对齐的 post-bid 基准。`canonical_full`
继续按 schema v1 读取，且在本规格落地后应标为 `legacy_experimental`。
新 profile 必须使用新的规则、状态、观察和事件 schema；禁止以 v1
`canonical_full` 状态静默解释成欢乐规则。

所有可执行实现必须满足本文的 `MUST`/`MUST NOT` 条款。每条冻结条款在
[`../tests/rules/huanle_classic_v1/frozen_rules.json`](../tests/rules/huanle_classic_v1/frozen_rules.json)
中有正例和反例；牌型条款在
[`../tests/rules/huanle_classic_v1/move_goldens.json`](../tests/rules/huanle_classic_v1/move_goldens.json)
中有正例和反例。尚未由用户冻结的选择只能出现在“必须配置化的差异”中，
不能通过代码默认值猜测。

## 冻结的比赛流程

<!-- rule:HCV1-DEAL-001 -->

### HCV1-DEAL-001：物理牌与初始分配

实现 **MUST** 使用一副 54 张物理牌；三名玩家各得到 17 张，余下 3 张为
底牌。任意合法状态都 **MUST** 满足 54 张牌守恒。

<!-- rule:HCV1-BOTTOM-001 -->

### HCV1-BOTTOM-001：底牌可见性和归属

地主确定前，底牌 **MUST NOT** 出现在任一玩家的公开 Observation 中。地主
确定后，3 张底牌 **MUST** 加入地主手牌并向所有玩家公开。

<!-- rule:HCV1-REVEAL-001 -->

### HCV1-REVEAL-001：发牌前明牌

每位玩家在发牌前 **MUST** 有“明牌开始”或“不明牌”的显式决策。选择明牌
开始的倍率 **MUST** 为 ×5，且该决定不可撤销。

<!-- rule:HCV1-REVEAL-002 -->

### HCV1-REVEAL-002：发牌中明牌与首明牌者

引擎 **MUST** 支持逐轮发牌后的“现在明牌/继续收牌”决策，并按 authoritative
事件顺序记录首位明牌者。发牌中倍率 **MUST** 从显式配置的
`reveal.factor_by_cards_received[0..17]` 查得；实现 **MUST NOT** 在代码中
猜测 ×4/×3 的收牌张数分界。

<!-- rule:HCV1-REVEAL-003 -->

### HCV1-REVEAL-003：明牌公开与公共倍率

已明牌玩家的剩余手牌 **MUST** 向所有观察者公开并随出牌更新；未明牌玩家的
剩余手牌 **MUST NOT** 泄漏。多人明牌时公共明牌倍率 **MUST** 取全部明牌事件
的最大值，而不是相乘。

<!-- rule:HCV1-CALL-001 -->

### HCV1-CALL-001：初始叫地主

Calling 阶段的初始动作集合 **MUST** 仅含“叫地主”和“不叫”。任一玩家成功
叫地主后，Calling **MUST** 立即结束，并以该玩家为 provisional candidate
进入 Robbing；不得继续让其他玩家追加叫地主。

<!-- rule:HCV1-CALL-002 -->

### HCV1-CALL-002：首叫顺序

存在明牌者时，首位明牌者 **MUST** 获得首叫权；不存在明牌者时，首叫权
**MUST** 来自当前 deal attempt 已保存的随机候选座位。

<!-- rule:HCV1-CALL-003 -->

### HCV1-CALL-003：全员不叫分支

全员不叫且存在首位明牌者时，该首位明牌者 **MUST** 直接成为地主。全员不叫
且没有明牌者时，当前 attempt **MUST** 记录为 all-pass，并在同一 match 中
自动开始新的确定性 deal attempt；它 **MUST NOT** 被伪装成整个 match 的终局。

<!-- rule:HCV1-ROB-001 -->

### HCV1-ROB-001：不叫者失去资格

初始 Calling 中做过“不叫”的玩家 **MUST NOT** 获得 Rob 或 PassRob 合法动作。

<!-- rule:HCV1-ROB-002 -->

### HCV1-ROB-002：抢地主队列

Robbing **MUST** 使用显式 candidate、eligible、acted 和循环队列。每名有资格
玩家最多一次真实 Rob/PassRob 决策；成功 Rob **MUST** 把 candidate 改为当前
玩家。被别人抢走 candidate 的原 caller/旧 candidate，是否可在其尚未使用的
机会中反抢，只能由 `robbing.caller_can_reclaim` 显式配置决定。

<!-- rule:HCV1-ROB-003 -->

### HCV1-ROB-003：抢地主倍率

每一次成功 Rob **MUST** 使公共抢地主因子乘以 2。PassRob **MUST NOT** 改变
candidate 或该因子。

<!-- rule:HCV1-POST-001 -->

### HCV1-POST-001：地主解析后的阶段顺序

地主确定后，流程 **MUST** 依次执行：底牌加入地主手牌并公开；可选的
PostBottomReveal；逐玩家 Doubling；地主先出 CardPlay。若启用收底牌后明牌，
该事件倍率 **MUST** 为 ×2，且地主资格由显式规则字段控制。

<!-- rule:HCV1-DOUBLE-001 -->

### HCV1-DOUBLE-001：逐玩家加倍资格

加倍 **MUST** 是逐玩家状态。地主可加倍的前提是三家余额均越过房间阈值；
农民可加倍的前提是自己与地主余额均越过阈值。无资格玩家的合法动作集合
**MUST NOT** 含 Double，只能为 Decline 或显式系统未加倍事件。

<!-- rule:HCV1-SETTLE-001 -->

### HCV1-SETTLE-001：逐对结算

公共因子由 base unit、最大明牌倍率、成功抢地主次数、炸弹/王炸、春天和反春
构成。设地主为 L、农民为 F1/F2，单对 stake **MUST** 为
`common × double[L] × double[Fi]`；地主胜时 L 收取两对 stake，两个农民各自
损失本对 stake，农民胜时符号反转。实现 **MUST NOT** 用一个全局 boolean
double multiplier 替代逐对结算。两个农民 raw payoff 可以不同，但总 payoff
**MUST** 为零。

<!-- rule:HCV1-PLAY-001 -->

### HCV1-PLAY-001：出牌顺序与结束

地主 **MUST** 先出，之后按固定座位循环。任一玩家手牌归零时比赛 **MUST**
立即结束；地主归零为地主胜，否则农民胜。

<!-- rule:HCV1-PLAY-002 -->

### HCV1-PLAY-002：标准非癞子牌型

本 profile **MUST** 支持标准非癞子单、对、三、三带一、三带对、顺子、连对、
三顺、飞机带单、飞机带对、四带二单、四带两对、炸弹和王炸；**MUST NOT**
实现癞子。具体合法性和比较由 `HCV1-MOVE-*` 金标冻结。

## 牌型与比较金标

<!-- rule:HCV1-MOVE-001 -->

### HCV1-MOVE-001：顺子边界

顺子 **MUST NOT** 包含 2 或任一王。

<!-- rule:HCV1-MOVE-002 -->

### HCV1-MOVE-002：连对下限

连对 **MUST** 至少含 3 对连续对子。

<!-- rule:HCV1-MOVE-003 -->

### HCV1-MOVE-003：三顺下限

三顺 **MUST** 至少含 2 组连续三张。

<!-- rule:HCV1-MOVE-004 -->

### HCV1-MOVE-004：四带二不是炸弹

四带二 **MUST NOT** 享有炸弹优先级。

<!-- rule:HCV1-MOVE-005 -->

### HCV1-MOVE-005：飞机主体比较

飞机 **MUST** 只比较主体，不比较附件。

<!-- rule:HCV1-MOVE-006 -->

### HCV1-MOVE-006：王炸最大

王炸 **MUST** 高于所有非王炸牌；普通炸弹不得击败王炸。

<!-- rule:HCV1-MOVE-007 -->

### HCV1-MOVE-007：飞机单翅配置

飞机带单的同点数单翅膀复用 **MUST NOT** 被硬编码；它必须由
`card_play.airplane.single_attachments` 配置。

<!-- rule:HCV1-MOVE-008 -->

### HCV1-MOVE-008：四带二单附件配置

四带二单的同点数附件复用 **MUST NOT** 被硬编码；它必须由
`card_play.four_with_two.single_attachments` 配置。

## 必须配置化的差异

下列项尚未冻结具体值。R002 的 RuleConfig v2 已将它们显式表达；profile
冻结、解析和运行时均 **MUST** 拒绝缺失值，且不得提供隐式默认值。每项在
[`../tests/rules/huanle_classic_v1/unresolved_config.json`](../tests/rules/huanle_classic_v1/unresolved_config.json)
中有正例和反例。

<!-- config:HCV1-CONFIG-DEAL-REVEAL-SCHEDULE -->

| ID | 必需配置键 | 未冻结的选择 |
|---|---|---|
| HCV1-CONFIG-DEAL-REVEAL-SCHEDULE | `reveal.factor_by_cards_received[0..17]` | 发牌中 ×4 与 ×3 的已收牌张数分界 |
| HCV1-CONFIG-AIRPLANE-SINGLE-WINGS | `card_play.airplane.single_attachments` | 飞机单翅是否可由一对拆成两个同点数单牌 |
| HCV1-CONFIG-FOUR-TWO-SINGLE-WINGS | `card_play.four_with_two.single_attachments` | 四带二单的两张附件是否可同点数 |
| HCV1-CONFIG-SPRING | `settlement.spring` | 春天、反春是否启用及其倍率 |
| HCV1-CONFIG-SCORE-CAP | `settlement.score_cap` 与 `settlement.bean_cap_policy` | 得分/欢乐豆封顶与破产保护 |
| HCV1-CONFIG-CALLER-RECLAIM | `robbing.caller_can_reclaim` | 原 caller 被抢后是否可反抢 |

<!-- config:HCV1-CONFIG-AIRPLANE-SINGLE-WINGS -->
<!-- config:HCV1-CONFIG-FOUR-TWO-SINGLE-WINGS -->
<!-- config:HCV1-CONFIG-SPRING -->
<!-- config:HCV1-CONFIG-SCORE-CAP -->
<!-- config:HCV1-CONFIG-CALLER-RECLAIM -->

## R001 corpus 边界

R001 只验证规格和金标的完整性。R002 新增的
`tests/rules/huanle_classic_v1/parser_fixture_v2.yaml` 只用于验证 schema、显式
字段、版本隔离和规则哈希；其中六个未冻结选择是测试值，**MUST NOT** 当作
可部署 profile。正式 `configs/rules/huanle_classic_v1.yaml` 仍必须等待相应
规则金标与引擎实现通过验收后才可冻结。

R002 不声称 Rust 引擎已实现欢乐流程，也不以 JSON corpus 或 parser fixture
代替 R003–R010 的规则、序列化、replay、Web 和 differential 验收。任何实现
提交都必须引用相应 case ID，并将其从“规格金标”提升为真实引擎测试。

R003 已实现独立的 `MatchStateV2` / `DealAttemptStateV2` 协调层：每个 attempt
保留子 seed、物理 deck、随机首叫候选、每个已接受的 `GameActionV2`、动作计数和 all-pass 摘要；all-pass
自动生成下一次确定性发牌，且完整 decision history 可重放为相同 match state。
它不实现明牌、叫抢、加倍、出牌或结算；这些阶段只能由后续 ticket 的
authoritative 状态机向该协调层报告已验证结果。
