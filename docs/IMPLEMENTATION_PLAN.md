# BIRD-Dou 斗地主 AI 研发实施规格书

**版本：** v1.0  
**日期：** 2026-07-22  
**用途：** 交给编码 AI、研究工程师或多人团队，按阶段实现一个可复现、可评测、可扩展的顶级斗地主 AI。  
**工作名称：** BIRD-Dou（Belief-constrained, Information-set-distilled, Role-adaptive, Distributional DouDizhu）

---

## 0. 如何使用本文件

本文件不是一份“把所有新技术一次性堆进去”的愿望清单，而是一份带有依赖关系和验收闸门的实施规格书。

执行规则：

1. 严格按 M0 → M10 的顺序推进。
2. 每个里程碑必须通过验收标准，才能进入下一阶段。
3. 每次只引入一个主要变量，必须保留可切换的旧实现做消融实验。
4. 规则正确性优先于模型强度，模型强度优先于功能数量。
5. 所有“更先进”的模块都必须通过同规则、同牌局、同座位轮换的统计评测证明有效。
6. 不允许把对手真实手牌泄漏给实战 Actor；完整手牌只能用于训练期 Teacher、Critic 和监督标签。
7. 任何硬剪枝都必须先证明对教师最优动作的召回率足够高。
8. 所有超参数写入配置文件，不允许散落在代码里。

推荐把本文件放入仓库：

```text
docs/IMPLEMENTATION_PLAN.md
```

并在每个任务开始前让编码 AI 先读取本文件和对应模块文档。

---

# 1. 项目目标、范围和边界

## 1.1 最终目标

构建一个完整斗地主 AI 系统，覆盖：

```text
发牌
→ 叫地主 / 抢地主
→ 确定地主与底牌
→ 可选加倍
→ 出牌
→ 春天 / 反春 / 炸弹等计分
→ 复盘、评测和模型训练
```

核心目标包括：

1. 在“确定地主后的标准出牌环境”中稳定超过 DouZero 基线。
2. 在完整叫牌环境中超过“固定叫牌器 + 强出牌器”的组合。
3. 明确建模隐藏牌不确定性，而不是只使用其他玩家手牌并集。
4. 训练时利用完整手牌指导学习，执行时严格只使用玩家可见信息。
5. 两个农民共享团队目标，但保留上家、下家的策略差异。
6. 同时优化胜率与实际得分，而不是只预测一个终局标量。
7. 具备高吞吐自博弈、统一评测、消融实验和可复现训练能力。

## 1.2 第一阶段明确不做的内容

在 M0～M6 阶段，不做：

- 癞子斗地主；
- 四人斗地主；
- 平台自动识牌、鼠标控制、外挂接入；
- 语言模型直接出牌；
- 全程在线树搜索；
- 未经验证的复杂奖励塑形；
- 一开始就把合法动作剪到 Top-K；
- 一开始就重写叫牌、出牌、搜索和模型全部模块。

## 1.3 两套规则环境

必须同时支持两套规则配置。

### A. `douzero_post_bid`

用途：与 DouZero 完全对齐，作为研究基准。

特征：

- 地主已经确定；
- 地主 20 张牌，两位农民各 17 张；
- 地主先出；
- 不含叫牌、抢地主和加倍；
- 奖励可选 WP、ADP、logADP；
- 出牌规则必须与 DouZero 兼容。

### B. `canonical_full`

用途：完整游戏训练。

要求通过 `RuleConfig` 明确配置：

- 叫分制或抢地主制；
- 最高叫分；
- 底牌是否公开；
- 是否允许加倍；
- 炸弹、王炸倍率；
- 春天、反春规则；
- 四带二的具体合法性；
- 流局重发规则；
- 得分封顶规则。

禁止在代码中默认假定某个平台规则，所有差异都由配置控制。

---

# 2. 技术路线依据

本项目吸收但不机械复制以下路线：

| 研究/项目 | 应吸收的部分 | 不直接照搬的部分 |
|---|---|---|
| DouZero | 规则引擎枚举合法动作；对每个 `(信息集, 动作)` 评分；并行自博弈；DMC 终局监督 | 原始平坦特征、最近 15 手 LSTM、单标量输出 |
| DouZero+ | 隐藏牌预测；Coach/Teacher 思想；叫牌扩展 | 逐点数独立 5 分类且缺少全局牌数约束 |
| PerfectDou | 完美信息训练、不完美信息执行；Privileged Critic；PPO/GAE | 训练代码未完全开放，需独立实现；不能把教师真实状态动作直接生硬蒸馏给学生 |
| OADMCDou | Oracle 信息退火；限制过大更新；训练稳定性 | 不把 Oracle 输入保留到执行期 |
| DouRN / ResNet 版本 | 结构化卷积/残差网络可以改善原始 MLP | 不把 15 个有序点数简单当普通图像处理 |
| AlphaDou | 完整叫牌；胜率与条件得分分开预测；候选动作筛选 | 不在模型尚未成熟时进行激进硬剪枝 |
| OMODMC | 最少拆牌和动作筛选能提高训练效率 | 最少拆牌只是先验，不等价于最优策略 |
| CAPRE_DMC | 农民合作信用分配；多源隐藏信息预测 | QMIX 的单调分解不一定适合所有农民协作场景 |
| 完整斗地主并发多阶段训练 | 叫牌和出牌共同训练；先胜率后得分 | 不用弱固定叫牌器长期污染出牌训练分布 |
| DouRD | 地主和农民应允许不同结构；注意力效果存在角色差异 | 不把单篇实验结论硬编码成“地主永远不能用注意力” |
| IMPALA / V-trace | Actor 与 Learner 解耦时修正策略滞后 | 不强制替换 DMC，必须和 DMC 做公平对照 |
| COMA | 中央 Critic 和反事实基线用于团队信用分配 | 斗地主是轮流决策，需要改造成顺序决策版本 |
| ReBeL / Student of Games | 公共 belief、信息集一致搜索的重要性 | 斗地主是三人非严格两人零和，不能直接声称理论收敛保证 |

**重要说明：** BIRD-Dou 是一个工程和研究方案。各组成思想有文献依据，但它们的组合效果必须通过消融实验验证，不能预先宣称必然达到 SOTA。

---

# 3. 成功标准

## 3.1 正确性标准

必须满足：

- 与 DouZero 基准规则的合法动作集合完全一致；
- 状态转移、过牌、轮次重置、炸弹计数和胜负判定一致；
- 玩家 Observation 不包含不可见手牌；
- 同一信息集对应的 Student 输出不随真实隐藏牌改变；
- 隐藏牌模型输出永远满足牌数守恒；
- 所有评测可由随机种子完全复现。

## 3.2 强度标准

分三级：

### Gate A：基线复现

- 官方 DouZero 权重能通过新环境运行；
- 对固定牌局，新旧环境产生相同合法动作与结果；
- 自训练模型稳定超过 random 和 RLCard 规则基线。

### Gate B：出牌模型改进

在 `douzero_post_bid` 环境中：

- 对 DouZero-ADP 和 DouZero-WP 做座位平衡、同牌配对评测；
- 主要指标的 95% 置信区间下界高于基线；
- 地主、上家农民、下家农民分别报告，不允许只报整体平均。

### Gate C：完整游戏

在 `canonical_full` 环境中：

- 与基线叫牌器 + DouZero/PerfectDou 出牌器组合进行完整对局；
- 同时报告胜率、原始平均分、得分分布和尾部风险；
- 不允许只靠极少量高倍炸弹局提高均分，而显著损害普通局胜率。

## 3.3 工程标准

- 所有核心模块有单元测试；
- Rust 规则环境相对 Python DouZero 有明确加速；
- 训练可断点恢复；
- checkpoint 带规则版本、特征版本、模型版本和 Git commit；
- 评测结果包含置信区间；
- 无静默 NaN、无无界队列、无 Actor 死锁；
- 模型输入输出 schema 有版本号。

---

# 4. 总体架构

```text
┌─────────────────────────────── Rust 游戏环境 ───────────────────────────────┐
│ 发牌 / 叫牌 / 出牌 / 计分 / 合法动作 / 序列化 / 回放 / 批量 step             │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ Observation + legal actions
                                     ▼
┌─────────────────────────────── 特征与批处理层 ──────────────────────────────┐
│ 点数特征、完整历史事件、标量特征、候选动作、ragged offsets、隐藏牌标签       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     ▼
┌──────────────────────────── BIRD-Dou 神经网络 ─────────────────────────────┐
│ Rank Mixer ─┐                                                               │
│ History GRU ├─ 角色门控 ─ Belief CRF ─ State Latent                         │
│ History Attn┘                                                               │
│ Legal Action Encoder ─ Post-hand Encoder ─ Cross Attention                  │
│                                                                             │
│ 输出：Policy / Win / Score Distribution / MC-Q / Auxiliary                  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
             ┌───────────────────────┼─────────────────────────┐
             ▼                       ▼                         ▼
     DMC / Hybrid Trainer     Privileged Teacher       Farmer Team Critic
             │                       │                         │
             └─────────────── 自博弈 Actor-Learner ────────────┘
                                     │
                                     ▼
                         Arena / Cross-play / Ablation
```

---

# 5. 仓库结构

```text
bird-dou/
├─ Cargo.toml
├─ pyproject.toml
├─ README.md
├─ docs/
│  ├─ IMPLEMENTATION_PLAN.md
│  ├─ RULES.md
│  ├─ FEATURE_SCHEMA.md
│  ├─ MODEL_ARCHITECTURE.md
│  ├─ TRAINING.md
│  ├─ EVALUATION.md
│  └─ LICENSE_AUDIT.md
├─ crates/
│  ├─ ddz-core/                 # 牌、动作、状态、回放、序列化
│  ├─ ddz-rules/                # 牌型检测、动作生成、规则配置
│  ├─ ddz-batch/                # 批量环境
│  ├─ ddz-search/               # 手牌分解、残局求解、belief rollout
│  └─ ddz-pyo3/                 # Python 绑定
├─ python/birddou/
│  ├─ __init__.py
│  ├─ schemas.py                # Tensor schema/dataclass
│  ├─ features/
│  │  ├─ encoder.py
│  │  ├─ history.py
│  │  └─ action_features.py
│  ├─ belief/
│  │  ├─ cardinality_crf.py
│  │  ├─ sampler.py
│  │  └─ losses.py
│  ├─ models/
│  │  ├─ rank_mixer.py
│  │  ├─ history_encoder.py
│  │  ├─ action_encoder.py
│  │  ├─ role_adapters.py
│  │  ├─ bird_dou.py
│  │  ├─ privileged_teacher.py
│  │  └─ baseline_douzero.py
│  ├─ rl/
│  │  ├─ dmc.py
│  │  ├─ vtrace.py
│  │  ├─ hybrid.py
│  │  ├─ losses.py
│  │  ├─ replay.py
│  │  └─ learner.py
│  ├─ actors/
│  │  ├─ actor_worker.py
│  │  ├─ inference_server.py
│  │  └─ shared_queue.py
│  ├─ league/
│  │  ├─ population.py
│  │  ├─ matchmaking.py
│  │  └─ snapshot.py
│  ├─ eval/
│  │  ├─ arena.py
│  │  ├─ paired_deals.py
│  │  ├─ metrics.py
│  │  ├─ bootstrap.py
│  │  └─ baselines.py
│  └─ cli/
│     ├─ train.py
│     ├─ evaluate.py
│     ├─ generate_dataset.py
│     └─ inspect_replay.py
├─ configs/
│  ├─ rules/
│  ├─ model/
│  ├─ train/
│  └─ eval/
├─ tests/
│  ├─ rust/
│  ├─ python/
│  ├─ differential/
│  ├─ golden_replays/
│  └─ performance/
├─ scripts/
│  ├─ reproduce_douzero.sh
│  ├─ train_belief.sh
│  ├─ train_cardplay.sh
│  ├─ train_full_game.sh
│  └─ run_crossplay.sh
└─ artifacts/                   # 默认不提交大模型，仅保存 manifest
```

---

# 6. Rust 游戏引擎规格

## 6.1 牌表示

花色不参与斗地主牌型比较，但发牌和回放仍建议保留 54 张实体牌 ID。

```rust
pub type CardId = u8;     // 0..53
pub type RankId = u8;     // 0..14，顺序：3..A,2,小王,大王
pub type Seat = u8;       // 0..2
pub type RankCounts = [u8; 15];
```

提供两种表示：

- `CardId`：用于洗牌、发牌和可审计回放；
- `RankCounts`：用于规则判断、动作生成和模型特征。

必须提供经过测试的互转函数。

## 6.2 动作表示

```rust
pub enum MoveKind {
    Pass,
    Single,
    Pair,
    Triple,
    TripleWithSingle,
    TripleWithPair,
    Straight,
    PairStraight,
    TripleStraight,
    AirplaneWithSingles,
    AirplaneWithPairs,
    FourWithTwoSingles,
    FourWithTwoPairs,
    Bomb,
    Rocket,
}

pub struct Move {
    pub kind: MoveKind,
    pub cards: RankCounts,
    pub main_rank: RankId,
    pub chain_len: u8,
    pub total_cards: u8,
}

pub enum GameAction {
    Bid(BidAction),
    Double(DoubleAction),
    Play(Move),
}
```

动作必须规范化：

- 相同牌张组合只能产生一个动作；
- 翅膀顺序不应导致重复动作；
- `main_rank`、`chain_len` 定义必须写入 `RULES.md`；
- 所有合法动作按稳定顺序排序，保证固定种子完全复现。

推荐排序键：

```text
phase
→ move kind
→ total cards
→ chain length
→ main rank
→ cards 计数字典序
```

## 6.3 状态表示

```rust
pub struct GameState {
    pub rule_config_id: u32,
    pub phase: Phase,
    pub current_player: Seat,
    pub landlord: Option<Seat>,
    pub hands: [RankCounts; 3],
    pub bottom_cards: RankCounts,
    pub played_cards: [RankCounts; 3],
    pub cards_left: [u8; 3],
    pub last_non_pass: Option<Move>,
    pub last_non_pass_player: Option<Seat>,
    pub consecutive_passes: u8,
    pub bid_state: BidState,
    pub multiplier_exp: u8,
    pub bomb_count: u8,
    pub spring_state: SpringState,
    pub history: Vec<GameEvent>,
    pub terminal: bool,
    pub raw_payoff: [i32; 3],
}
```

## 6.4 玩家 Observation

```rust
pub struct Observation {
    pub schema_version: u32,
    pub phase: Phase,
    pub observer: Seat,
    pub role: Role,
    pub own_hand: RankCounts,
    pub public_played: [RankCounts; 3],
    pub public_bottom_cards: RankCounts,
    pub unknown_pool: RankCounts,
    pub cards_left: [u8; 3],
    pub current_player: Seat,
    pub landlord: Option<Seat>,
    pub last_non_pass: Option<Move>,
    pub consecutive_passes: u8,
    pub bid_history: Vec<BidEvent>,
    pub history: Vec<PublicEvent>,
    pub multiplier_exp: u8,
    pub bomb_count: u8,
}
```

**隐私不变量：**

对于任意两个完整状态 `s1`、`s2`，如果它们对玩家 `p` 产生相同信息集，则：

```rust
observe(s1, p) == observe(s2, p)
```

必须写自动测试：在不改变玩家自己的牌和公共历史的前提下交换另外两家的隐藏牌，Observation 必须逐字节相同。

## 6.5 核心 API

```rust
pub trait DdzEnv {
    fn reset(&mut self, seed: u64, rules: &RuleConfig) -> Observation;
    fn observe(&self, player: Seat) -> Observation;
    fn legal_actions(&self) -> Vec<GameAction>;
    fn step(&mut self, action: &GameAction) -> StepResult;
    fn is_terminal(&self) -> bool;
    fn serialize_state(&self) -> Vec<u8>;
    fn deserialize_state(bytes: &[u8]) -> Result<GameState>;
}
```

搜索模块还需：

```rust
fn apply_in_place(&mut self, action: &GameAction) -> UndoToken;
fn undo(&mut self, token: UndoToken);
```

禁止在搜索中反复深拷贝大型历史向量。

## 6.6 合法动作生成

不允许枚举手牌所有子集。按牌型模板生成：

```text
单牌：count >= 1
对子：count >= 2
三张：count >= 3
炸弹：count == 4
王炸：小王和大王同时存在
顺子：3..A 的连续区间，每点至少 1，长度 >= 5
连对：3..A 的连续区间，每点至少 2，长度 >= 3
飞机：3..A 的连续区间，每点至少 3，长度 >= 2
带牌：在扣除主体后，从剩余牌中组合附件
```

附件枚举必须去重，并严格遵守 `RuleConfig`。

跟牌时：

1. 生成与目标相同类型、相同结构长度且主点更大的动作；
2. 加入炸弹；
3. 必要时加入王炸；
4. 加入 Pass；
5. 若目标是王炸，只允许 Pass；
6. 两家连续 Pass 后，下一位获得自由领出权。

## 6.7 引擎测试

至少包含：

- 每种牌型的正例和反例；
- 所有边界顺子；
- 2 和大小王不得进入顺子；
- 飞机翅膀去重；
- 炸弹压普通牌、炸弹互压、王炸；
- 连续 Pass 后轮次重置；
- 出完最后一手立即终局；
- 叫牌流局；
- 加倍、炸弹、春天计分；
- 序列化后状态完全一致；
- `step → undo` 恢复逐字节一致；
- 随机游戏永远不出现负牌数或牌总数变化。

使用 Rust `proptest` 生成随机手牌和动作序列。

---

# 7. Python 绑定与批量环境

## 7.1 PyO3 接口

暴露：

```python
class PyDdzEnv:
    def reset(seed: int, rule_config: dict) -> Observation
    def legal_actions() -> list[Action]
    def step(action: Action) -> StepResult
    def observe(player: int) -> Observation
    def serialize() -> bytes

class PyBatchDdzEnv:
    def reset(seeds: np.ndarray) -> BatchObservation
    def legal_actions_packed() -> PackedActions
    def step_packed(action_indices: np.ndarray) -> BatchStepResult
```

## 7.2 批量环境要求

- 一个 Python Actor 持有多个 Rust 环境；
- Rust 内部并行或顺序批处理均可，先保证确定性；
- 不为每个小数组创建大量 Python 对象；
- Observation 和 Actions 尽量直接返回 NumPy 连续缓冲区；
- 使用整数紧凑存储，进入 GPU 前再转 embedding。

---

# 8. 特征协议

所有特征 schema 必须有版本号，修改维度时必须升级版本。

## 8.1 Ragged Batch

```python
@dataclass
class RaggedBatch:
    rank_categorical: Tensor      # [B, 15, Cr_cat], int64
    rank_numeric: Tensor          # [B, 15, Cr_num], float32
    history_rank_counts: Tensor   # [B, T, 15], uint8/int64
    history_meta: Tensor          # [B, T, Ch], int64
    history_mask: Tensor          # [B, T], bool
    scalars: Tensor               # [B, Cs], int64/float32

    action_rank_counts: Tensor    # [M, 15], uint8/int64
    post_hand_counts: Tensor      # [M, 15], uint8/int64
    action_meta: Tensor           # [M, Ca], int64/float32
    action_state_index: Tensor    # [M], int64
    action_offsets: Tensor        # [B+1], int64

    chosen_action_flat_index: Tensor  # [B], int64
```

其中：

- `B`：决策状态数量；
- `M`：这些状态的合法动作总数；
- 第 `i` 个状态的动作区间是 `[offsets[i], offsets[i+1])`。

## 8.2 点数特征

每个点数包含：

```text
rank_id
own_count
unknown_count
played_count_by_relative_seat_0
played_count_by_relative_seat_1
played_count_by_relative_seat_2
last_non_pass_count
public_bottom_count
is_straight_eligible
```

使用 embedding 表示离散数量，不把 0～4 简单当连续实数。

## 8.3 历史事件特征

每个事件包含：

```text
phase
relative_actor
is_pass
is_play
is_bid
is_double
move_kind
main_rank
chain_len
wing_kind
total_cards
cards_left_after
multiplier_exp_after
trick_index
position_in_trick
rank_counts[15]
```

出牌历史应覆盖完整一局。初始 `T_max=96`，超过时保留：

```text
全部叫牌事件
+ 最早关键事件摘要
+ 最近出牌事件
```

一般斗地主不应频繁超过该长度；如果超过，记录监控指标后再调整。

## 8.4 候选动作特征

每个动作包含：

```text
rank_counts[15]
post_hand_counts[15]
move_kind
main_rank
chain_len
wing_kind
total_cards
is_pass
is_bomb
is_rocket
empties_hand
leaves_one_card
breaks_bomb_count
breaks_pair_count
min_groups_after
number_of_min_decompositions_capped
```

`min_groups_after` 来自手牌最少拆牌模块，仅作为特征；M0～M6 不用于硬剪枝。

---

# 9. BIRD-Dou 模型架构

## 9.1 默认规模

```yaml
d_model: 256
rank_blocks: 4
rank_attention_every: 2
history_gru_layers: 2
history_transformer_layers: 3
history_heads: 8
action_blocks: 2
role_adapter_dim: 64
score_quantiles: 11
```

目标总参数：20M～35M。先做中型模型，禁止直接扩到数亿参数。

## 9.2 Rank Token Encoder

15 个点数分别形成 Token：

```python
rank_token = concat(
    rank_embedding,
    own_count_embedding,
    unknown_count_embedding,
    three_played_count_embeddings,
    last_move_count_embedding,
    bottom_count_embedding,
    straight_eligible_embedding,
)
rank_token = linear(rank_token)  # -> d_model
```

## 9.3 Rank Mixer

每个残差块：

```text
RMSNorm
→ depthwise Conv1D kernel=3
 + depthwise Conv1D kernel=5
→ pointwise SwiGLU
→ residual
```

每两个块加入一次带相对点数偏置的多头自注意力。

目的：

- 卷积捕获相邻点数的顺子、连对和飞机；
- 全局注意力捕获炸弹、王炸和远距离控制关系；
- 保留点数顺序，不把牌面当普通无序集合。

## 9.4 历史双编码器

### GRU 分支

```text
Event Embedding → 2-layer GRU → last valid hidden
```

### Transformer 分支

```text
Event Embedding
→ causal Transformer Encoder
→ last valid token
```

### 角色门控

```python
g = sigmoid(role_bias[role] + gate_mlp(concat(gru_h, attn_h, scalar_h)))
history_h = g * gru_h + (1 - g) * attn_h
```

作用：让模型自己学习地主更依赖递归记忆还是注意力，农民亦然，而不是人工固定。

推理服务可为 Transformer 历史维护 KV cache；第一版可先完整重算，保证正确性后再优化。

## 9.5 角色适配

共享主干，但加入：

```text
role embedding：landlord / farmer
seat embedding：landlord / landlord_up / landlord_down
role adapter：低维瓶颈残差模块
role-specific LayerNorm scale/bias
role-specific output head
```

两个农民共享大部分参数，保留上家和下家的小型 Adapter。

## 9.6 状态表示

```python
rank_tokens = RankMixer(rank_tokens)
rank_mean = masked_mean(rank_tokens)
rank_max = max(rank_tokens)
scalar_h = ScalarEncoder(scalars)
history_h = RoleGatedHistoryEncoder(history)
pre_belief_state = MLP([rank_mean, rank_max, history_h, scalar_h])
```

然后通过 Belief CRF 产生隐藏牌表示，再融合：

```python
state_h = MLP([pre_belief_state, belief_pool])
```

## 9.7 Action Encoder

对所有合法动作并行：

```python
action_rank_h = encode_rank_counts(action_rank_counts)
post_hand_h = encode_rank_counts(post_hand_counts)
action_meta_h = ActionMetaEncoder(action_meta)
state_for_action = state_h[action_state_index]
query = MLP([action_rank_h, post_hand_h, action_meta_h, state_for_action])
```

让动作 Query 对对应状态的 15 个 `rank_tokens` 做 Cross-Attention：

```python
context = CrossAttention(
    query=query.unsqueeze(1),
    key_value=rank_tokens[action_state_index],
)
action_h = MLP([query, context])
```

不建议让每个动作重复注意完整历史序列，历史已压缩进 `state_h`，否则候选动作多时显存开销过高。

## 9.8 合法动作集合上下文

每个状态内计算：

```text
segment_mean(action_h)
segment_max(action_h)
segment_logsumexp(policy_prelogit)
```

再拼回每个动作：

```python
action_h2 = MLP([action_h, set_mean[state_id], set_max[state_id]])
```

这让动作评分具有“相对当前其他合法动作”的信息，同时保持近似线性复杂度。

## 9.9 输出头

第一版必须输出：

```python
policy_logit      # 行为策略
win_logit         # P(win | I,a)
score_if_win      # 胜利条件下的期望得分或 log 得分
score_if_loss     # 失败条件下的期望得分或 log 得分
mc_q              # DouZero 风格终局 Monte Carlo 回报
turns_to_finish   # 辅助任务
```

稳定后增加条件分位数：

```python
score_win_quantiles   # [M, 11]
score_loss_quantiles  # [M, 11]
```

评测动作值：

```python
p_win = sigmoid(win_logit)
expected_score = p_win * E(score_if_win) + (1 - p_win) * E(score_if_loss)
```

不同模式：

```text
WP 模式：argmax p_win
Score 模式：argmax expected_score
Risk 模式：argmax expected_score - lambda * downside_risk
```

---

# 10. 约束隐藏牌模型

这是项目的核心研究模块。

## 10.1 两个隐藏容器的情况

当前玩家知道自己的手牌和公共牌，因此未知牌分布在另外两名玩家之间。

对每个点数 `r`：

- 未知池总数：`u_r`；
- 隐藏玩家 A 还剩 `N_A` 张；
- 隐藏玩家 B 还剩 `N_B` 张；
- `N_A + N_B = sum(u_r)`。

令：

```text
x_r = 点数 r 中属于玩家 A 的张数
```

则玩家 B 自动拥有：

```text
u_r - x_r
```

网络输出：

```text
score[r, k]，k = 0..u_r
```

联合分布：

```math
P(x|I) ∝ exp(Σ_r score[r,x_r]) · 1[Σ_r x_r=N_A]
```

## 10.2 前向动态规划

```python
neg_inf = -1e30
forward = full((16, N_A + 1), neg_inf)
forward[0, 0] = 0.0

for r in range(15):
    for used in range(N_A + 1):
        for k in range(u[r] + 1):
            if used + k <= N_A:
                forward[r + 1, used + k] = logaddexp(
                    forward[r + 1, used + k],
                    forward[r, used] + score[r, k],
                )

log_z = forward[15, N_A]
```

真实分配标签为 `x_true`，损失：

```math
L_belief = logZ - Σ_r score[r, x_true_r]
```

## 10.3 边缘概率

实现后向 DP，计算：

```text
P(x_r=k | I)
E[x_r]
Var[x_r]
Entropy[x_r]
```

融合回策略网络的特征：

```text
玩家 A 每个点数的期望、方差、熵
玩家 B 每个点数的期望、方差、熵
关键牌概率：2、小王、大王、炸弹概率
```

## 10.4 采样合法完整手牌

从后向 DP 依次采样 `x_r`，保证：

- 每种牌数量正确；
- 玩家总牌数正确；
- 采样状态与公共历史一致。

用于：

- 信息集一致教师蒸馏；
- 残局 belief rollout；
- belief 校准评测。

## 10.5 叫牌阶段的三容器扩展

叫牌时未知牌可能分布在：

```text
隐藏玩家 A
隐藏玩家 B
三张底牌
```

使用二维容量 DP：

```text
dp[rank][count_A][count_B]
```

第三容器数量由守恒决定。

复杂度很小，可后续实现；M5 只要求出牌阶段两容器 CRF。

## 10.6 Belief 监督数据

每个自博弈决策点保存：

```text
玩家合法 Observation
真实隐藏手牌分配
剩余张数容量
真实下一步对手动作
当前策略版本
```

训练数据必须来自混合策略：

```text
随机策略
规则策略
DouZero
当前模型
历史 checkpoint
```

避免 belief 只适应某一种自博弈风格。

## 10.7 Belief 验收

必须通过：

- 守恒违规率严格为 0；
- 小规模牌堆中 DP 与暴力枚举 `logZ` 一致；
- 抽样分布与精确边缘一致；
- NLL 显著优于只按剩余张数均匀分配；
- 关键牌概率有校准曲线；
- 把隐藏牌真实标签打乱后性能明显下降，证明模型不是读取泄漏字段。

---

# 11. Privileged Teacher 与信息集一致蒸馏

## 11.1 Teacher

Teacher 可以看到：

```text
三家完整手牌
完整游戏状态
当前合法动作
```

Teacher 结构可复用 Action Encoder，但增加：

```text
每个玩家的真实手牌 Rank Tokens
完整状态交互层
```

Teacher 输出：

```text
Q_teacher(full_state, action)
Policy_teacher(full_state)
Win / Score heads
```

## 11.2 Student

Student 只能看到：

```text
自己的牌
公共历史
各家剩余牌数
公开底牌
Belief 分布
合法动作
```

执行和评测时绝不传入真实隐藏手牌。

## 11.3 为什么不能直接蒸馏真实状态教师

同一个信息集 `I` 可能对应多个隐藏状态，教师在不同隐藏状态下可能给出相互冲突的动作。学生无法区分这些状态。

因此不能直接使用：

```text
真实隐藏状态 s 的教师动作 → Student(I)
```

## 11.4 信息集一致蒸馏

对一个 Student 信息集：

1. 从 Belief CRF 采样 `K` 个完整合法状态；
2. 实际真实状态作为一个额外样本加入训练；
3. Teacher 对每个状态的所有合法动作评分；
4. 对隐藏状态求期望；
5. 再生成 Student 的软目标。

```math
Q_bar(I,a) = (1/K) Σ_k Q_teacher(s_k,a)
```

```math
pi_teacher(a|I) = softmax(Q_bar(I,a)/temperature)
```

```math
L_KD = KL(pi_teacher || pi_student)
```

还可增加：

```math
L_value_KD = Huber(Q_student(I,a), Q_bar(I,a))
```

初始实现：

```yaml
belief_samples_k: 4
teacher_temperature: 0.5
stop_gradient_through_belief_for_kd: true
```

稳定后增加到 8 或动态 K。

## 11.5 Oracle Dropout

Teacher/Privileged Critic 训练中可随机遮蔽部分真实隐藏牌，形成从完美信息到不完美信息的课程：

```text
早期：完整真实手牌
中期：随机遮蔽部分点数
后期：真实手牌 + belief 表示混合
```

Student 始终不读取 Oracle 输入。

## 11.6 泄漏检查

必须编写测试：

- 固定 Student Observation；
- 更换真实隐藏状态；
- Student 输出逐元素相同；
- Teacher 输出允许变化；
- 保存模型时 Student checkpoint 不应包含需要真实手牌的输入接口。

---

# 12. 训练算法

实现三种可切换模式：

```text
TrainerMode.DMC
TrainerMode.VTRACE
TrainerMode.HYBRID
```

不能预先假定 V-trace 一定优于 DMC。

## 12.1 DMC 模式

沿用 DouZero 思路：

- Actor 打完整局；
- 每个角色本局做出的所有动作共享终局回报；
- 对 `mc_q` 做 Huber/MSE；
- 行为策略按 `mc_q`、胜率或混合目标选动作。

优点：简单、稳定、强基线。  
缺点：回报方差大，信用分配粗。

## 12.2 V-trace 模式

Actor 保存行为策略概率 `mu(a_t|I_t)`，Learner 计算当前策略 `pi`。

```math
rho_t = min(rho_bar, pi(a_t|I_t)/mu(a_t|I_t))
```

```math
c_t = min(c_bar, pi(a_t|I_t)/mu(a_t|I_t))
```

使用标准 V-trace target。斗地主终局任务初始设置：

```yaml
gamma: 1.0
rho_bar: 1.0
c_bar: 1.0
```

必须记录策略版本差和重要性权重分布。

## 12.3 Hybrid 模式

主策略使用 V-trace Actor-Critic，同时保留终局 Monte Carlo 监督：

```math
L = L_policy + c_v L_value + c_mc L_mc_q + ...
```

默认起始权重，仅作为实验初值：

```yaml
policy_coef: 1.0
value_coef: 0.5
mc_q_coef: 0.25
win_coef: 0.25
score_coef: 0.10
belief_coef: 0.20
kd_coef: 0.0          # 蒸馏阶段逐步升高
entropy_coef: 0.01
aux_coef: 0.05
```

全部写入配置并可独立关闭。

## 12.4 奖励

保留原始奖励和训练奖励两套值。

```text
raw_reward：真实平台计分
wp_reward：胜 +1，负 -1
score_train_reward：对原始得分做稳定变换
```

推荐：

```python
score_train_reward = sign(raw_score) * log2(1 + abs(raw_score))
```

再按训练数据统计进行缩放。禁止直接让极端炸弹倍数无限支配梯度。

## 12.5 多阶段目标

完整游戏训练：

```math
R = (1-lambda) R_wp + lambda R_score
```

阶段：

```text
Stage 0：lambda=0，先学会赢
Stage 1：小比例加入分数
Stage 2：胜率和得分并重
Stage 3：主要优化实际分数，但继续监控胜率
```

进入下一阶段以评测指标为条件，不按固定训练步数硬切换。

## 12.6 分布式得分头

第一版用：

```text
P(win)
E(score | win)
E(score | loss)
```

后续用 11 个 quantile，采用 quantile Huber loss。

## 12.7 辅助任务

可用：

- 当前手牌最少还需几次出完；
- 选择动作后最少组合数；
- 下一位玩家是否会 Pass；
- 下一位玩家动作类型；
- 当前局是否存在一手直接走完；
- 隐藏关键牌概率；
- 队友在若干轮内走完的概率。

辅助任务只影响表示学习，不直接修改游戏奖励。

---

# 13. 农民协作和信用分配

## 13.1 共享 Actor

```text
Farmer Shared Encoder
+ landlord_up / landlord_down seat embedding
+ seat-specific adapter
+ seat-specific output head
```

避免完全独立模型重复学习，也避免强行完全相同。

## 13.2 Centralized Team Critic

训练时 Critic 可看完整状态，输出当前农民每个合法动作的团队价值：

```math
Q_team(s,a)
```

反事实基线：

```math
b(s,I) = Σ_a pi(a|I) Q_team(s,a)
```

优势：

```math
A_cf(s,a_taken) = Q_team(s,a_taken) - b(s,I)
```

这相当于问：

```text
在队友和地主状态不变的情况下，当前农民这一步比他自己的其他合法选择好多少？
```

## 13.3 不使用手工合作奖励

第一版禁止加入：

```text
压队友扣分
给队友让牌加分
消耗地主大牌加分
```

这些规则容易产生奖励漏洞。先使用真实团队终局奖励 + Centralized Critic + 反事实优势。

## 13.4 反事实 rollout 数据

只在少量高价值状态中生成：

1. 选择一个农民决策点；
2. 对 Top-N 替代动作从完整状态继续 rollout；
3. 估计动作差值；
4. 作为 Centralized Critic 或 Student 的额外监督。

不要对每一步所有动作做完整 rollout，成本过高。

---

# 14. 自博弈系统

## 14.1 单机推荐分工

结合 V100 + P100：

```text
V100：主 Learner，混合精度训练
P100：Actor 批量推理、Teacher 推理或评测
CPU：Rust 批量环境和合法动作生成
```

Belief CRF 的 log-sum-exp 建议保留 FP32。

## 14.2 Actor 架构

```text
多个 Python Actor 进程
每个进程持有若干 Rust 环境
→ 收集待决策 Observation
→ 发送给 GPU Inference Server
→ 得到动作分布/动作价值
→ 采样动作并 step
→ 终局后打包轨迹
→ 写入共享内存队列
```

初始配置可从小规模开始：

```yaml
actor_processes: 8
envs_per_actor: 32
max_inference_states: 128
max_inference_actions: 8192
unroll_length: 32
```

这些是启动值，不是性能结论。

## 14.3 Inference Server

需求：

- 根据动作总数而非仅状态数控制 batch；
- 微批等待上限可配置；
- 返回每个状态内部的 segment softmax；
- 支持多个模型版本；
- 返回 `policy_version`；
- 队列有上限和背压；
- Actor 断开后不造成永久等待。

## 14.4 轨迹记录

```python
@dataclass
class Transition:
    serialized_state: bytes
    observer: int
    chosen_action: bytes
    behavior_logprob: float
    policy_version: int
    reward: float
    done: bool
    raw_score: int

@dataclass
class EpisodeMeta:
    seed: int
    rules_hash: str
    model_versions: tuple[int, int, int]
    winner: str
    raw_payoff: tuple[int, int, int]
```

Learner 可从序列化状态重新生成 Observation 和合法动作，避免轨迹存储全部候选动作造成巨大内存。若成为瓶颈，再增加 Actor 侧压缩特征缓存。

## 14.5 League

模型池包含：

```text
当前主模型
若干历史主模型
地主 exploiter
农民 exploiter
DouZero / PerfectDou / 规则基线
```

匹配策略：

```text
大多数对局：当前模型自博弈
一部分：对历史 checkpoint
一部分：对专门 exploiter
少量：对固定公开基线
```

保存 checkpoint 的条件：

- 相对当前 champion 的 paired cross-play 达到门槛；
- 没有严重角色退化；
- 数值稳定；
- belief 校准未崩溃。

---

# 15. 完整叫牌训练

## 15.1 叫牌模型输入

复用：

- 17 张初始手牌 Rank Encoder；
- 当前叫牌历史；
- 座位；
- 规则模式；
- 对未发底牌的三容器 belief；
- 当前合法叫牌动作。

## 15.2 叫牌动作评分

仍使用候选动作条件评分：

```text
不叫 / 叫1 / 叫2 / 叫3
或
不抢 / 抢
```

输出的含义是：

```text
采取该叫牌动作后，在当前出牌策略下整局最终胜率与得分
```

## 15.3 初始化

1. 先冻结强 Cardplay 模型；
2. 采样大量初始 17 张手牌；
3. 对每种叫牌动作和可能底牌进行 Monte Carlo 模拟；
4. 生成叫牌监督标签；
5. 训练初始 Bid Head；
6. 再进入联合训练。

## 15.4 并发联合训练

每局完整运行：

```text
叫牌模型决定地主
→ Cardplay 模型出牌
→ 终局回报同时更新叫牌和出牌模型
```

必须监控由叫牌模型导致的训练数据分布：

- 地主初始牌力分布；
- 各叫分比例；
- 流局率；
- 不同叫分下的胜率和均分；
- 是否退化为几乎总叫或总不叫。

---

# 16. 动作 Proposal 与剪枝

M9 之前禁止启用硬剪枝。

## 16.1 Proposal 网络

使用便宜特征预测候选动作排名：

```text
状态摘要
+ 动作结构
+ 出牌后手牌最少组合数
```

## 16.2 保留规则

Top-K 外永久保护：

```text
Pass
所有炸弹
王炸
直接出完动作
能阻止对手直接出完的动作
随机探索动作
Teacher 明确高价值动作
```

## 16.3 启用条件

在独立验证集上：

- Teacher 最优动作 Top-K 召回率达到预设高门槛；
- 直接出完动作召回率 100%；
- 炸弹与王炸召回率 100%；
- 开启剪枝后的 paired 评测不显著下降；
- 训练中固定比例状态仍运行完整动作集，防止 Proposal 自我强化错误。

---

# 17. 残局搜索

这是可选增强，不是前期依赖。

## 17.1 触发条件

```text
总剩余牌数低于阈值
或任意玩家剩余牌数 <= 5
或出现炸弹/王炸关键决策
或 belief entropy 很低
```

## 17.2 Root-consistent Belief Rollout

1. 从约束 Belief 中采样 K 个完整状态；
2. 根节点对所有采样状态强制比较同一组合法动作；
3. 对每个根动作，在各采样状态中继续 rollout；
4. 使用当前策略、Teacher 或浅层搜索评估；
5. 聚合期望、胜率和风险；
6. 根动作不能因不同隐藏样本而采用不同选择。

这是务实的近似方法，不宣称具有两人零和公共 belief 搜索的理论保证。

## 17.3 精确解算

当隐藏牌分配唯一或进入完全信息调试模式时：

- 使用 `apply/undo`；
- 置换表缓存；
- 对地主 vs 农民团队做终局求解；
- 输出强制胜负和最短结束步数。

## 17.4 搜索蒸馏

搜索结果写入离线数据：

```text
信息集
候选动作
搜索访问分布
搜索价值
belief 样本摘要
```

定期蒸馏回无搜索模型，降低线上计算量。

---

# 18. 评测系统

## 18.1 牌局配对

所有正式对比必须：

- 使用固定随机种子牌局集；
- 同一牌局轮换座位；
- 记录叫牌结果和角色；
- 对每副牌做 paired 统计；
- 不允许 A 和 B 各自重新随机发不同牌。

## 18.2 评测规模

按用途：

```text
Smoke：小规模，仅检查程序正确
Quick Gate：快速淘汰明显差模型
Research Gate：足够牌局 + 95% CI
Final：运行到置信区间宽度达到预设阈值
```

不要把固定局数当唯一标准，最终以置信区间精度为准。

## 18.3 指标

### 游戏表现

```text
地主胜率
农民团队胜率
上家农民表现
下家农民表现
平均原始得分
平均训练变换得分
得分标准差
P10 / P50 / P90
最大回撤式尾部风险
炸弹使用率
春天/反春率
```

### 模型质量

```text
Policy entropy
Win Brier score
Expected Calibration Error
Score MAE
Score quantile coverage
Belief NLL
Belief count MAE
关键牌概率校准
```

### 工程性能

```text
环境 step/s
合法动作生成/s
GPU actions scored/s
推理 p50/p95/p99
Actor 队列等待时间
Learner GPU 利用率
策略版本滞后
显存与内存峰值
```

## 18.4 Cross-play Matrix

矩阵包含：

```text
DouZero-WP
DouZero-ADP
RLCard
PerfectDou 权重
DouZero+ 可用权重
当前 champion
历史 checkpoint
各类 exploiter
```

分别固定一个模型担任地主或农民，避免只报“整套模型一起替换”的结果。

## 18.5 Exploitability Proxy

斗地主无法方便计算真实 exploitability。可训练：

```text
Landlord best-response proxy
Farmer best-response proxy
```

对冻结 champion 专门训练若干 exploiter，报告 champion 被利用程度。必须称为 proxy，不称为理论 exploitability。

---

# 19. 日志、配置和 checkpoint

## 19.1 配置

建议使用 Hydra/OmegaConf 或纯 YAML + Pydantic，必须支持：

```text
规则配置
特征版本
模型结构
TrainerMode
损失权重
Actor 数量
评测对手
随机种子
路径
```

## 19.2 Checkpoint Manifest

每个 checkpoint 保存：

```json
{
  "git_commit": "...",
  "rules_hash": "...",
  "feature_schema_version": 3,
  "model_arch_version": 2,
  "trainer_mode": "hybrid",
  "frames": 123,
  "episodes": 456,
  "optimizer_state": true,
  "rng_state": true,
  "league_snapshot": "...",
  "metrics": {}
}
```

## 19.3 断点恢复

必须恢复：

- 模型；
- 优化器；
- LR scheduler；
- AMP scaler；
- 随机数状态；
- Actor policy version；
- league；
- 训练阶段和奖励课程参数。

---

# 20. 测试矩阵

## 20.1 规则测试

- 单元测试；
- property-based；
- DouZero differential；
- golden replay；
- serialize/deserialize；
- apply/undo；
- 多规则 profile。

## 20.2 特征测试

- 同状态编码确定性；
- 座位相对化正确；
- 隐藏牌交换不影响 Observation；
- action offsets 正确；
- segment softmax 每段和为 1；
- Pass 编码一致；
- 历史 padding 不改变有效事件表示。

## 20.3 Belief 测试

- DP 对暴力枚举；
- 边缘和为 1；
- 容量守恒；
- 采样频率匹配边缘；
- 极端容量 0 和全部牌；
- Joker 上限；
- FP32 数值稳定。

## 20.4 模型测试

- 任意动作数量；
- 每个状态只有一个动作；
- 超大候选动作集；
- 前向无 NaN；
- 反向梯度有限；
- Teacher/Student 接口隔离；
- 模型保存加载输出一致；
- CPU/GPU 输出误差在容忍范围内。

## 20.5 训练测试

- 100 局 smoke train；
- 单 Actor；
- 多 Actor；
- Actor 意外退出；
- 队列满；
- checkpoint 恢复；
- 旧策略滞后；
- 无合法动作异常；
- 终局奖励符号正确。

---

# 21. 里程碑计划

## M0：规格冻结和仓库骨架

### 要做什么

- 建仓库；
- 写 `RULES.md`；
- 确定两套 RuleConfig；
- 建立 CI；
- 完成许可证审计；
- 固定基线仓库 commit 和权重 manifest；
- 建立配置系统和版本字段。

### 产物

```text
仓库骨架
规则文档
许可证文档
CI
baseline manifest
```

### 验收

- 文档能明确回答任意牌型边界；
- CI 可运行 Rust/Python 基础测试；
- 不存在尚未决定却已写死在代码中的规则。

---

## M1：Rust `douzero_post_bid` 规则引擎

### 要做什么

- 牌和点数结构；
- 所有牌型识别；
- 合法动作生成；
- 状态转移；
- Pass 和轮次重置；
- 胜负与 WP/ADP；
- 序列化；
- apply/undo；
- property-based 测试。

### 验收

- 随机采样大量合法决策状态，与 DouZero 动作集合零非预期差异；
- 随机完整游戏无状态不变量错误；
- golden replay 结果固定；
- apply/undo 完全恢复。

**未通过时禁止写神经网络。**

---

## M2：PyO3、批量环境和统一 Arena

### 要做什么

- Python 绑定；
- `PyBatchDdzEnv`；
- DouZero 模型适配器；
- paired deal 生成器；
- cross-play；
- bootstrap CI；
- 性能 benchmark。

### 验收

- 官方权重可运行；
- 新旧环境在固定 replay 上结果一致；
- 评测可生成地主/农民分角色报告；
- 重复运行相同 seed 结果一致。

---

## M3：精确 DouZero 基线

### 要做什么

- 实现原始 54 维牌表示；
- 最近 15 动作 LSTM；
- 三角色模型；
- DMC Actor-Learner；
- 读取官方 checkpoint；
- 小规模自训练。

### 验收

- 官方 checkpoint 输出与参考实现一致；
- 自训练曲线合理；
- 模型超过 random 和 RLCard；
- 训练可断点恢复。

目的不是创新，而是证明整个链路正确。

---

## M4：结构化 BIRD-Dou 模型，不加入 Belief

### 要做什么

- 新特征 schema；
- Rank Mixer；
- GRU + Transformer 历史门控；
- Action Encoder；
- Role Adapter；
- Policy/Win/Score/MC-Q 输出；
- DMC 训练模式；
- 与原始 DouZero 同环境同预算对比。

### 验收

- 模型能处理任意 ragged action batch；
- 训练无 NaN；
- 至少不显著弱于原始架构；
- 若更强，完成逐模块消融；
- 若更弱，优先简化而不是继续加模块。

---

## M5：约束 Belief CRF

### 要做什么

- 生成带真实隐藏牌标签的数据；
- 两容器 CRF；
- NLL、边缘、采样；
- belief 特征融合；
- 关键牌校准报告；
- 冻结策略网络做离线预训练；
- 再联合微调。

### 验收

- 守恒违规率 0；
- DP 与暴力枚举一致；
- NLL 优于均匀基线；
- 联合策略至少不退化；
- belief 打乱时策略性能下降，说明模型真正使用了该信息。

---

## M6：Privileged Teacher 与信息集一致蒸馏

### 要做什么

- 全牌 Teacher；
- Privileged Critic；
- belief 采样；
- IS-KD；
- Oracle Dropout；
- 泄漏测试。

### 验收

- Teacher 明显强于同规模 Student，证明完整信息有用；
- 蒸馏 Student 优于不蒸馏版本；
- Student 对同一信息集输出严格一致；
- 直接蒸馏与 IS-KD 做消融，证明 IS-KD 的必要性或诚实报告无收益。

---

## M7：DMC / V-trace / Hybrid 对照

### 要做什么

- inference server；
- Actor 版本化；
- V-trace；
- Hybrid loss；
- 策略滞后监控；
- 三种模式公平对比。

### 验收

- 系统可长时间运行无死锁和内存增长；
- V-trace 数值稳定；
- Hybrid 是否优于 DMC 由实验决定；
- 若 V-trace 失败，保留 DMC 主线，不为“新”而强行使用。

---

## M8：农民 Centralized Critic

### 要做什么

- 共享 Farmer Actor；
- 两个 seat adapters；
- full-state team critic；
- counterfactual baseline；
- 少量反事实 rollout 数据；
- 农民专门 exploiter。

### 验收

- 农民团队对强地主基线提升；
- 上家和下家均无严重退化；
- 不依赖手工合作奖励；
- 地主模型性能不因共享改动被意外改变。

---

## M9：完整叫牌和并发多阶段训练

### 要做什么

- `canonical_full` 引擎；
- Bid Head；
- 三容器 belief；
- 叫牌 Monte Carlo 初始化；
- 联合训练；
- 胜率到得分课程；
- 完整计分 Arena。

### 验收

- 叫牌不退化为总叫/总不叫；
- 叫牌概率与实际收益有校准；
- 完整模型优于固定叫牌器组合；
- 角色分布和牌力分布无异常偏移。

---

## M10：Proposal、残局搜索和部署模型

### 要做什么

- Proposal 网络；
- 保护动作；
- 动态 Top-K；
- belief rollout；
- 精确残局求解；
- 搜索蒸馏；
- 小模型蒸馏；
- 推理服务导出。

### 验收

- 剪枝显著提高吞吐且统计强度不下降；
- 搜索只在触发状态工作；
- 搜索版优于纯网络；
- 蒸馏版保留大部分搜索收益；
- 部署接口只需要合法 Observation。

---

# 22. 风险表和回退方案

| 风险 | 现象 | 回退方案 |
|---|---|---|
| Rust 引擎与 DouZero 不一致 | 模型评测不可比较 | 停止模型工作，先做 differential 修复 |
| 结构化模型更慢但不更强 | GPU 利用率低、胜率无提升 | 简化 History Transformer，保留 Rank Mixer；回到纯 DMC |
| Belief NLL 好但策略无提升 | 预测信息没有被正确利用 | 做 belief 打乱消融；增加不确定性特征；检查目标玩家相对座位定义 |
| Belief 过度自信 | 校准差，搜索被误导 | 温度校准、entropy regularization、模型 ensemble |
| Teacher 强但 Student 学不到 | 蒸馏标签冲突 | 使用信息集采样平均、提高温度、只蒸馏优势排序 |
| V-trace 不如 DMC | 策略梯度噪声或策略滞后 | DMC 继续作为主算法；V-trace 只作辅助 Policy Head |
| 农民 Critic 学会错误配合 | 上家或下家退化 | 拆分 seat adapter；减少 critic 权重；加强 frozen-opponent 评测 |
| 完整叫牌污染出牌训练 | 地主牌力分布极端 | 先冻结 Cardplay；叫牌预训练；分阶段解冻 |
| 硬剪枝遗漏关键动作 | 局部出现低级错误 | 关闭剪枝；提高 K；永久保护关键动作；增加完整动作抽检 |
| 搜索出现 strategy fusion | 不同样本选择不同根动作 | 强制 root action consistency；只把样本用于同动作聚合 |
| 训练循环自我博弈过拟合 | 对当前自己强，对旧模型弱 | League、历史 checkpoint、exploiter、cross-play matrix |
| P100 软件兼容问题 | 新 PyTorch 构建缺少 sm_60 | 固定兼容版本；P100 改为 CPU Actor/评测；V100 保持 Learner |

---

# 23. 第一批可直接派发的任务

按顺序派发，禁止并行修改同一核心接口。

## Ticket E001：创建仓库骨架

**输入：** 本文件。  
**输出：** 第 5 节目录、基础 Cargo workspace、Python package、CI。  
**验收：** `cargo test`、`pytest` 均可运行。

## Ticket E002：规则配置和 Rank/Card 基础类型

**输出：** `RuleConfig`、`CardId`、`RankId`、互转、测试。  
**禁止：** 先实现 AI。

## Ticket E003：MoveKind 和规范化 Move

**输出：** 所有牌型数据结构、稳定排序、序列化和测试。

## Ticket E004：牌型检测器

**输出：** `detect_move(RankCounts) -> Result<Move>`。  
**测试：** 每种牌型正反例和边界。

## Ticket E005：自由领出动作生成

**输出：** 从任意手牌生成所有非 Pass 合法动作。  
**测试：** 去重、稳定排序、无越界。

## Ticket E006：跟牌过滤

**输出：** 针对目标动作生成能压过的动作 + 炸弹 + Pass。

## Ticket E007：GameState 和 step

**输出：** post-bid 环境完整运行。

## Ticket E008：Differential Harness

**输出：** Python 脚本对比 DouZero 合法动作集合和随机对局。

## Ticket E009：序列化、apply/undo

**输出：** 搜索友好的可逆状态更新。

## Ticket E010：PyO3 单环境

**输出：** Python 可 reset/legal_actions/step。

## Ticket E011：批量环境

**输出：** 连续 NumPy 缓冲区接口和 benchmark。

## Ticket E012：Arena 和 paired deals

**输出：** 固定牌局、座位轮换、bootstrap CI。

## Ticket E013：官方 DouZero inference adapter

**输出：** 加载地主/上家/下家权重并对新环境出牌。

## Ticket E014：原始特征编码和基线模型

**输出：** 与 DouZero 架构一致的 PyTorch 模型。

## Ticket E015：DMC smoke training

**输出：** 可训练、保存、恢复、打败 random 的最小闭环。

## Ticket E016：新 Feature Schema

**输出：** RaggedBatch、点数、历史、动作编码和 schema 文档。

## Ticket E017：Rank Mixer

**输出：** 独立可测模块和 shape/gradient 测试。

## Ticket E018：History Dual Encoder

**输出：** GRU、Transformer、角色门控及消融开关。

## Ticket E019：Action Encoder 与 segment ops

**输出：** ragged action 前向、segment softmax/mean/max。

## Ticket E020：BIRD-Dou 无 Belief 版本

**输出：** 完整多头模型和 DMC 训练。

后续 Ticket 只在 M4 验收后创建。

---

# 24. 编码 AI 的工作协议

把下面内容作为每次任务的固定前置提示。

```text
你正在实现 BIRD-Dou 斗地主 AI。

仓库中的 docs/IMPLEMENTATION_PLAN.md、docs/RULES.md 和 docs/FEATURE_SCHEMA.md 是唯一规格来源。

工作规则：
1. 本次只实现指定 Ticket，不得顺手重构其他模块。
2. 开始前先阅读相关文档和现有代码，输出：
   - 任务目标
   - 将修改的文件
   - 数据结构/API 变化
   - 测试方案
3. 若规格与代码冲突，停止扩展功能，明确指出冲突；不得静默改变规则。
4. 所有公共接口必须有类型标注、文档和错误处理。
5. 不得提交空实现、伪实现、固定返回值或仅能通过单个示例的代码。
6. 每个功能至少包含单元测试；规则模块优先加入 property-based 测试。
7. 运行项目规定的格式化、静态检查和测试。
8. 性能相关修改必须提供修改前后的 benchmark，不能只凭感觉声称更快。
9. 不允许向 Student Observation 加入真实隐藏手牌。
10. 不允许把新的超参数硬编码在模型中，必须写入配置。
11. 完成后报告：
    - 实现内容
    - 修改文件
    - 测试结果
    - 已知限制
    - 下一 Ticket 的依赖是否满足
12. 不开始下一 Ticket，直到当前 Ticket 验收通过。
```

---

# 25. 最终推荐的研发主线

最稳妥的主线是：

```text
规则完全正确
→ DouZero 基线完全复现
→ 结构化状态和动作模型
→ 约束隐藏牌 CRF
→ 信息集一致特权蒸馏
→ DMC / V-trace / Hybrid 公平对照
→ 农民中央 Critic
→ 完整叫牌并发训练
→ 动作 Proposal
→ 残局 belief search
```

不要倒序。尤其不能从“大 Transformer + PPO + 搜索”开始，因为一旦结果不好，将无法判断是规则、特征、训练、隐藏信息、合作还是搜索出了问题。

这套计划最关键的研究点不是模型尺寸，而是三件事：

1. **隐藏牌分布必须满足严格牌数守恒。**
2. **完整信息教师必须先在同一信息集内聚合，才能教给不可见隐藏牌的学生。**
3. **地主、上家农民、下家农民共享知识，但允许角色和座位专门化。**

只要这三条实现严谨，再用统一 Arena 做消融，就能形成一个真正有研究价值、也有工程竞争力的斗地主 AI 项目。

---

# 26. 参考工作

- Zha et al., *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*, ICML 2021.
- Zhao et al., *DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided Learning*, CoG 2022.
- Yang et al., *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*, NeurIPS 2022.
- Luo et al., *Enhanced DouDiZhu Card Game Strategy Using Oracle Guiding and Adaptive Deep Monte Carlo Method*, IJCAI 2024.
- Chen et al., *DouRN: Improving DouZero by Residual Neural Networks*, 2024.
- Lei and Lei, *AlphaDou: High-Performance End-to-End Doudizhu AI Integrating Bidding*, 2024.
- Wang et al., *Explicit Cooperation Mechanism and Multimodal Fusion Prediction Model Empower DouDizhu Agents*, Applied Soft Computing, 2025.
- Li and Matsuzaki, *Developing Agents for Complete DouDizhu Game Enhanced With Concurrent and Multistage Training Methods*, IEEE Transactions on Games, 2025.
- Sun and Zhang, *DouRD: Enhancing DouDizhu AI with Role-Differentiated Modeling*, 2026.
- Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, 2018.
- Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, 2017.
- Brown et al., *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games*, 2020.
- Schmid et al., *Student of Games: A Unified Learning Algorithm for Both Perfect and Imperfect Information Games*, 2023.

