# BIRD-Dou

BIRD-Dou is a research-oriented DouDizhu AI project. Development follows the
ticket order and acceptance gates in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).
Engineering coverage and the deliberately separate empirical-gate status are
tracked in [`docs/REQUIREMENT_COVERAGE.md`](docs/REQUIREMENT_COVERAGE.md).

The repository is being implemented in the Ticket order defined by the plan.
The Rust foundation includes versioned rule profiles, canonical card and move
types, move detection, free-lead and follow generation, and executable post-bid
and complete bidding/doubling/scoring state machines. A manifest-pinned differential
harness
checks legal actions and synchronized games against official `DouZero`. The first
PyO3 environment exposes deterministic `reset`, `legal_actions`, `step`, `observe`,
`restore`, and `serialize` calls without reimplementing rules in Python. State
serialization and compact LIFO apply/undo are available for replay and search consumers.
The Python stack includes structured actors, constrained belief, privileged
training, distributed learners, farmer coordination, and complete bidding.

## Python single environment

Install the mixed Rust/Python package and drive an authoritative game:

```python
from birddou import PyDdzEnv, load_rule_config

rules = load_rule_config("configs/rules/douzero_post_bid.yaml")
env = PyDdzEnv()
observation = env.reset(seed=7, rule_config=rules)

while not env.terminal:
    action = env.legal_actions()[0]
    result = env.step(action)

state_bytes = env.serialize()
```

The seed contract is `splitmix64_fisher_yates_v1`; changing it requires a new
algorithm identifier and golden-deal fixtures.

For actor-style execution, one native object owns the whole batch and exchanges
only C-contiguous NumPy arrays:

```python
import numpy as np

from birddou import PyBatchDdzEnv, load_rule_config

rules = load_rule_config("configs/rules/douzero_post_bid.yaml")
batch = PyBatchDdzEnv(rules)
observation = batch.reset(np.arange(256, dtype=np.uint64))
actions = batch.legal_actions_packed()
active = actions["offsets"][1:] > actions["offsets"][:-1]
indices = np.where(active, 0, -1).astype(np.int64)
result = batch.step_packed(indices)
```

Action range `i` is `[offsets[i], offsets[i + 1])`; indices passed to
`step_packed` are local to that range. Already-terminal rows use `-1`.

## Paired evaluation

Compare two policies on one fixed deal manifest with symmetric role rotation and
deal-clustered bootstrap confidence intervals:

```bash
python -m birddou.cli.evaluate --candidate longest_move --baseline first_legal \
  --deals 100 --seed 20260722 --output artifacts/eval/report.json
```

The JSON artifact contains separate landlord, upstream/downstream farmer, farmer
team, and overall estimates. See [`docs/EVALUATION.md`](docs/EVALUATION.md) for
the pairing contract, cross-play API, and Final-evaluation precision rule.

Official DouZero ADP and WP policies use the same Arena boundary. Their source and
weights are fetched into the ignored artifact cache and verified against the
tracked manifest:

```bash
python -m pip install -e ".[model]"
python scripts/fetch_douzero_baseline.py --weight-set douzero_ADP
python -m birddou.cli.evaluate --candidate douzero_ADP --baseline first_legal \
  --deals 10 --seed 20260722
```

The adapter loads all three role networks into the native checkpoint-compatible
architecture and uses the native feature encoder by default. See
[`docs/EVALUATION.md`](docs/EVALUATION.md#e013-official-douzero-inference) for the
role mapping, safety checks, and reproducibility contract.

The evaluation registry also includes pinned RLCard 1.0.7 rule-agent and
PerfectDou adapters. RLCard is installed with `.[rlcard]`; PerfectDou is fetched
and checksum-verified by `scripts/fetch_perfectdou_baseline.py` and executed across
a timeout-bounded JSONL boundary because the official encoder is a Python-3.7
Linux binary. Setup and cross-play commands are documented in
[`docs/EVALUATION.md`](docs/EVALUATION.md#rlcard-and-perfectdou-baselines).

Named `--bird-dou-policy NAME=CHECKPOINT` and
`--full-game-policy NAME=CHECKPOINT` registrations add current champions,
historical snapshots, and exploiters to the same matrix without weakening
checkpoint fingerprint checks. Under `canonical_full`, card-play-only baselines
receive the declared fixed bidder and joint checkpoints use their saved Bid Head.
Exact-DouZero learner checkpoints use `--dmc-policy NAME=CHECKPOINT`.

## DMC smoke training

Run the 100-game, single-actor terminal-return training gate and its fixed-deal
random-baseline evaluation:

```bash
bird-dou-train-dmc --config configs/train/dmc_smoke.yaml \
  --report artifacts/train/dmc_smoke/report.json
```

The resumable checkpoint includes model, optimizer, scheduler, scaler, RNG,
policy-version, and training-phase state. See [`docs/TRAINING.md`](docs/TRAINING.md)
for the warm-start disclosure and exact resume contract.

## BIRD-Dou feature batches

The first native model schema packs any number of decision states and legal
actions into validated PyTorch tensors:

```python
from birddou.features import encode_ragged_batch, load_feature_config

feature_config = load_feature_config("configs/model/bird_dou_features_v1.yaml")
batch = encode_ragged_batch(observations, legal_actions, rules, config=feature_config)
scores_per_state = [
    batch.action_meta[batch.action_offsets[i] : batch.action_offsets[i + 1]]
    for i in range(batch.batch_size)
]
```

Rank features are observer-relative; history is complete up to the versioned
bounded-retention rule; action offsets and chosen flat indices are checked at
construction. See [`docs/FEATURE_SCHEMA.md`](docs/FEATURE_SCHEMA.md).

The E017 `RankTokenEncoder` and `RankMixer` preserve the ordered 15-rank axis with
parallel depthwise 3/5 convolutions, pointwise SwiGLU, and periodic relative-bias
attention. Local convolution, global attention, and numeric channels are
independently switchable for ablation; see
[`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md#e017-ordered-rank-mixer).

E018 encodes the complete bounded public history through parallel GRU and causal
Transformer branches. A learned three-seat gate combines them with scalar context;
either branch and the gate can be disabled independently without changing the
feature schema.

E019 keeps all legal actions in flat ragged storage. Action counts, post-action
hands, categorical metadata, and the owning state form each query; cross-attention
is restricted to that state's 15 rank tokens. Stable segment reductions add
legal-set context and normalize policy logits per state without action padding.
See
[`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md#e019-ragged-legal-action-encoder).

E020 assembles those modules into the 21.35M-parameter
`bird_dou_no_belief_v1` shared model. It provides policy, win, conditional score,
MC-Q, turns-to-finish, and score-quantile heads with small landlord/farmer seat
adapters. A resumable one-game structured DMC gate is available with
`bird-dou-train --config configs/train/bird_dou_dmc_smoke.yaml`; see
[`docs/TRAINING.md`](docs/TRAINING.md#e020-bird-dou-shared-model-dmc).

M5 adds an exact two-hidden-player cardinality CRF. It provides differentiable
log-partition/NLL, exact marginals and key-card calibration, capacity-preserving
sampling, mixed-policy label generation, frozen offline pretraining, and joint
policy fine-tuning. The Student continues to accept public `RaggedBatch` only;
see [`docs/BELIEF.md`](docs/BELIEF.md).

M6 adds the training-only full-hand Teacher, centralized privileged critic,
Oracle Dropout, and information-set-consistent KD over exact Belief samples.
Student artifacts retain no full-hand input surface; see
[`docs/DISTILLATION.md`](docs/DISTILLATION.md).

M7 adds bounded, versioned, action-aware inference batching; close-safe Actor and
Learner queues; compact trajectory replay; standard V-trace; independently
switchable Hybrid losses; policy-lag monitoring; and a validator that refuses
unequal DMC/V-trace/Hybrid comparisons. See
[`docs/DISTRIBUTED_TRAINING.md`](docs/DISTRIBUTED_TRAINING.md).

M8 adds a separate full-state Farmer Team Critic, exact ragged counterfactual
baselines, landlord-invariant Farmer specialist updates, bounded Top-N native-state
rollouts, Farmer exploiter schedules, and a confidence-aware promotion gate. No
handcrafted cooperation reward enters training; see
[`docs/FARMER_COORDINATION.md`](docs/FARMER_COORDINATION.md).

M9 adds the `canonical_full` deal-to-score engine, an information-set-safe Bid Head,
exact three-container belief, privileged Monte Carlo initialization, complete-game
joint collection, a metric-gated win-to-score curriculum, distribution monitoring,
and complete scoring Arena execution. See [`docs/BIDDING.md`](docs/BIDDING.md).
No strength improvement is claimed without the required paired research run.

M10 adds a cheap dynamic Top-K Proposal with non-negotiable tactical protection,
public-condition-triggered root-consistent Belief rollout, a native apply/undo
endgame solver, search and compact-model distillation, and a hash-bound deployment
bundle whose runtime input is only `Observation + legal_actions`. The recorded CPU
benchmark and the still-required trained-checkpoint research gates are separated in
[`docs/SEARCH_AND_DEPLOYMENT.md`](docs/SEARCH_AND_DEPLOYMENT.md).

## Development checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
```
