# Model Architecture

## E014 exact DouZero baseline

The `douzero_lstm_mlp_v1` baseline is a native PyTorch implementation whose state
dictionary keys and tensor shapes exactly match the three official checkpoints.
The landlord model consumes 373 flat features; both farmer roles share the same
484-wide architecture but load independent parameters.

```text
history [A, 5, 162]
  -> one-layer LSTM(hidden=128)
  -> final history state
  -> concatenate flat candidate features
  -> Linear(501 or 612, 512) + ReLU
  -> four × Linear(512, 512) + ReLU
  -> Linear(512, 1)
  -> scalar value per legal action
```

Stable first-maximum selection matches the reference inference behavior. The
three networks are eager-loaded, switched to evaluation mode, and run under
`torch.inference_mode()`. Official checkpoints are deserialized with
`weights_only=True`; missing and extra keys are fatal. The model and compatibility
feature versions are locked in
[`../configs/model/douzero_baseline.yaml`](../configs/model/douzero_baseline.yaml).

This baseline deliberately has no new architectural claims. Section 9 of
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) defines the gated BIRD-Dou
architecture work that begins only after exact baseline reproduction and DMC
smoke training.

## E017 ordered Rank Mixer

The first BIRD-Dou module consumes the E016 `[B, 15, 9]` categorical and
`[B, 15, 3]` numeric rank fields. `RankTokenEncoder` uses a dedicated rank
embedding, seven independent count embeddings, and a straight-eligibility
embedding. Optional normalized numeric channels are concatenated before a linear
projection to `d_model`; counts are never treated only as continuous magnitudes.

Each local residual block is:

```text
RMSNorm
  -> depthwise Conv1D(kernel=3, padding=1)
   + depthwise Conv1D(kernel=5, padding=2)
  -> pointwise SwiGLU
  -> dropout / per-state DropPath
  -> residual
```

At the configured cadence (default every two local blocks), multi-head global
attention adds a learned per-head relative bias indexed by signed rank distance
from `-14` through `+14`. This preserves the ordered 3-through-big-joker axis:
local kernels model straights, pair chains, and airplanes, while global attention
can connect bombs, jokers, and distant control ranks.

The default E017 configuration is four blocks, `d_model=256`, eight heads, and
two attention blocks. Convolution, attention, and numeric channels have separate
configuration switches. Disabling both mixer paths is an exact identity fallback,
which makes ablations auditable. DropPath is stochastic only in training and
deterministic in evaluation. The independent module checks input shape and
finiteness, has finite-gradient tests through real E016 features, and survives a
weights-only state-dictionary round trip exactly. Configuration is frozen in
[`../configs/model/bird_dou_v1.yaml`](../configs/model/bird_dou_v1.yaml).

## M10 Proposal and compact deployment actor

The Proposal network is a separate cheap public encoder over state summaries, action
metadata, rank counts, and post-action hand counts. Dynamic Top-K is external to the
main BIRD-Dou model, preserving the full-action path for ablation and safety checks.
Permanent tactical masks are unioned with Top-K before `subset_ragged_batch` creates
the canonical-order heavy-model input.

The compact deployment model reuses this scorer and adds a small aggregate
policy/value outcome head. It is trained by large-model distillation rather than by
exposing privileged inputs. Its production policy boundary is exactly one public
`Observation` plus its native legal-action list. See
[`SEARCH_AND_DEPLOYMENT.md`](SEARCH_AND_DEPLOYMENT.md).

## M5 constrained Belief extension

`bird_dou_belief_v1` inserts an exact two-container cardinality CRF at the
`pre_belief_state` boundary. Its neural scores use public information only. Exact
per-rank moments and key-card probabilities are pooled and fused before the
unchanged role/action/output path. The Student never receives oracle hands; those
exist only as labels in the dataset and NLL function. See
[`BELIEF.md`](BELIEF.md) for DP, sampling, leakage, calibration, and training
contracts, and
[`../configs/model/bird_dou_belief_v1.yaml`](../configs/model/bird_dou_belief_v1.yaml)
for the versioned architecture.

## M6 privileged Teacher

The training-only Teacher adds three exact-hand rank-token grids and a global
45-token interaction Transformer around an E020-compatible public trunk. Oracle
Dropout masks only hidden-player counts. The Student checkpoint never includes
this interface. Information-set KD averages Teacher Q over exact Belief samples
before constructing a policy target; see
[`DISTILLATION.md`](DISTILLATION.md).

## M8 Farmer Team Critic

The training-only Farmer Critic adds a dedicated team-Q head over the privileged
full-state action representation and rejects landlord rows. It is not nested in or
serialized with the shared public Actor. The Actor's existing downstream/upstream
adapters and heads remain distinct; the default specialist update freezes the
shared trunk and landlord path. See
[`FARMER_COORDINATION.md`](FARMER_COORDINATION.md).

## E018 dual full-history encoder

`HistoryEventEncoder` consumes every valid E016 public event rather than only a
short suffix. Per-rank count embeddings retain the 15-rank action shape;
categorical embeddings represent phase, relative actor, event flags, move kind,
main rank, and wing kind. Chain length, card count, cards-left, multiplier, trick,
and within-trick position are normalized numeric channels. A learned positional
embedding is added after projection to `d_model`; padded rows are zeroed by the
valid-prefix mask.

The same event sequence feeds two independent summaries:

```text
2-layer GRU                         causal Transformer (3 layers, 8 heads)
  -> hidden at last valid row         -> state at last valid row
```

The Transformer uses both an upper-triangular causal mask and the right-padding
mask. An all-padding history is made numerically safe internally and returns an
exact zero summary from both branches. Changing future events is tested to leave
every earlier Transformer row bit-identical; changing any padded value leaves the
final summary unchanged.

The scalar vector has its own MLP. With both history branches enabled, a
feature-wise seat-aware gate computes:

```text
g = sigmoid(gate_mlp([gru_h, transformer_h, scalar_h]) + seat_bias[0..2])
history_h = g * gru_h + (1 - g) * transformer_h
```

This permits landlord, downstream farmer, and upstream farmer to learn different
memory mixtures while sharing both encoders. GRU, Transformer, and role gating
have independent configuration switches. A single enabled branch bypasses the
gate exactly; disabling the gate with both branches uses an exact 50/50 mean.
Diagnostics return both branch states, gate values, scalar state, and fused state.
All-zero history, padding, causality, finite gradients, branch ablations, seat
biases, and weights-only restoration are independently tested.

## E019 ragged legal-action encoder

All legal candidates stay in one flat `[M, ...]` allocation. The validated
`action_offsets [B+1]` and its exact inverse `action_state_index [M]` associate
each candidate with one state, so no tensor is padded to a batch-wide maximum
action count.

Each candidate combines four `d_model` representations:

```text
15-rank action counts -> rank/count embeddings -> 2 local RankConvBlocks -> mean/max
15-rank post-hand counts -> the same shared encoder
14 action metadata columns -> independent categorical embeddings -> projection
state_h[action_state_index]
  -> SwiGLU query
```

The action query attends only the corresponding state's 15 Rank Mixer tokens.
It does not repeat or attend the full history sequence; public history has already
been summarized in `state_h`. Attention returns `[M, heads, 15]` normalized
diagnostic weights and a `[M, d_model]` rank context. Query and context then form
the base action representation.

Differentiable segment sum, mean, maximum, log-sum-exp, and softmax operations
work directly over non-empty half-open offset ranges. They use stable max
subtraction for exponentials and support arbitrary trailing feature axes. The
encoder broadcasts per-state segment mean and maximum back to each base action,
then uses a final SwiGLU fusion to produce set-aware actions. Runtime and storage
are linear in the actual candidate count `M`; tests cover singleton segments and
a 4,096-action stress segment without padding.

Post-hand features, rank cross-attention, and legal-set context have independent
ablation switches in
[`../configs/model/bird_dou_v1.yaml`](../configs/model/bird_dou_v1.yaml). Disabled
cross-attention returns exact zero context and diagnostics; disabled set context
returns the base action representation exactly.

## E020 complete no-Belief BIRD-Dou

`bird_dou_no_belief_v1` composes the independently tested E017-E019 modules into
one shared 21.35M-parameter model:

```text
rank categorical/numeric -> RankTokenEncoder -> RankMixer -> mean/max ----+
public event sequence ----> GRU + causal Transformer -> seat gate ---------+-->
scalar vector ------------> ScalarEncoder --------------------------------+   state SwiGLU
                                                                               -> role/seat adapter
flat action/post-hand/meta + state + 15-rank cross-attention -----------------> set-aware action
                                                                               -> seat output head
```

The adapter adds landlord/farmer role embeddings and three relative-seat
embeddings. It then applies a 64-wide landlord/downstream/upstream bottleneck
residual and seat-specific LayerNorm affine parameters. The expensive rank,
history, state, and action trunks are shared; only the compact adapters and final
deep output heads differ by seat. Relative seat is derived from the frozen
`landlord_relative` scalar, so landlord rotation does not change its meaning.

Every flat candidate returns policy logit and per-state segment probability,
win logit, positive conditional win score, negative conditional loss score,
their probability-weighted expected score, terminal Monte Carlo Q, positive
turns-to-finish, and 11 monotone conditional score quantiles. Documented
selection modes are policy, win probability, expected score, MC-Q, and
risk-adjusted expected score. All heads preserve the original action ordering and
shape `[M]` (quantiles `[M, 11]`).

The architecture consumes only `RaggedBatch`, whose opponent information is the
union unknown pool. `belief_enabled: false` is enforced rather than ignored; the
constrained hidden-card model begins in M5. The full config, including output-head
depth, is frozen in
[`../configs/model/bird_dou_v1.yaml`](../configs/model/bird_dou_v1.yaml).
