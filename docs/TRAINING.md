# Training

## E015 DMC smoke loop

The first trainable loop deliberately stays small and exact: one local Actor plays
complete games through `PyDdzEnv`, scores canonical legal actions with the E014
three-role baseline, and records the chosen candidate for every decision. After
terminal, all decisions made by a role receive that seat's final transformed
payoff. Each role network performs one configured MSE or Huber regression update
per episode with finite-loss checks and gradient clipping.

The checked-in gate runs 100 complete games, starts from the checksum-verified
official ADP weights, uses low epsilon-greedy exploration, and then performs a
fixed-deal, role-balanced Arena comparison against `SeededRandomPolicy`:

```bash
python -m pip install -e ".[model]"
python scripts/fetch_douzero_baseline.py --weight-set douzero_ADP
bird-dou-train-dmc --config configs/train/dmc_smoke.yaml \
  --report artifacts/train/dmc_smoke/report.json
```

ADP warm start is explicit in the configuration; it is not presented as evidence
that 100 CPU games can learn DouDizhu from random initialization. The gate proves
that self-play collection, role-correct terminal supervision, backward updates,
checkpointing, restoration, and unified evaluation form a working closed loop.
`initialization: random` remains available for pipeline experiments.

### Reproducibility and resume

Episode seeds are derived from the master seed and persisted episode counter.
Exploration uses a dedicated NumPy generator. A checkpoint restores all three
models and optimizers, LR schedulers, AMP scaler, NumPy/Torch RNG states, training
counters, policy version, training phase, reward curriculum, and minimal league
state. Checkpoint loading uses PyTorch weights-only mode and rejects configuration
or rule-hash drift.

Each atomic save writes:

- `checkpoint.pt`: complete resumable state;
- `manifest.json`: schema/model/feature/rule versions, Git revision, counters,
  SHA-256, and state-presence audit;
- `landlord.ckpt`, `landlord_down.ckpt`, `landlord_up.ckpt`: inference weights;
- `metrics.jsonl`: one deterministic loss/payoff/counter record per episode.

To extend an interrupted run to a new total episode budget, keep the same training
semantics and output directory:

```bash
bird-dou-train-dmc --config configs/train/dmc_smoke.yaml \
  --episodes 200 --resume
```

Output path, total episode budget, and evaluation-only settings are excluded from
the semantic fingerprint; optimizer, exploration, loss, initialization, model,
feature, and rules settings are not. E015 remains the intentionally small local
reference loop; the M7 process supervisor, bounded IPC inference bridge, and
vectorized native self-play worker provide the scaled Actor path described below.

## E020 BIRD-Dou shared-model DMC

The structured model uses one shared actor for all three seats. At every decision,
the actor encodes the complete information-safe observation and its entire legal
action segment, selects by the configured model utility (MC-Q by default), and
caches that one-state `RaggedBatch`, the chosen local index, serialized state,
behavior log-probability, and policy version. After terminal, each transition gets
the acting seat's true objective payoff, raw platform score, win label, and
remaining game-turn target.

One episode is collated without action padding: offsets and chosen indices are
rebased into one `[B states, M actions]` batch. The primary loss regresses the
chosen `mc_q` to the terminal Deep Monte Carlo return. Small configured auxiliary
weights train per-segment policy likelihood, win probability, conditional score,
turn count, and conditional quantiles. No shaped cooperation reward is used.
Losses and gradient norm must remain finite before the shared optimizer updates.

The CPU smoke config disables exact hand-decomposition features explicitly to
keep the one-game architecture gate quick; it uses the reserved E016 sentinel and
does not change action legality or model shape:

```bash
bird-dou-train --config configs/train/bird_dou_dmc_smoke.yaml \
  --report artifacts/train/bird_dou_dmc_smoke/report.json
```

Add `--evaluate` for a role-balanced fixed-deal comparison with seeded random.
The checkpoint contains the shared model, AdamW optimizer, scheduler, AMP scaler,
NumPy/Torch RNGs, counters, policy version, metrics, and exact model/feature/rule
fingerprints. `bird_dou.ckpt` is the weights-only inference artifact. Resume rejects
any semantic, schema, rule, feature, or architecture drift.

## M5 Belief training

Generate mixed-policy labeled information sets with `bird-dou-generate-dataset`,
then run `bird-dou-train-belief`. The first phase freezes the public encoder and
optimizes exact constrained NLL. The joint method unfreezes it and combines
behavior-policy likelihood with independently weighted Belief NLL. Key-card
calibration and uniform-allocation NLL are emitted alongside the checkpoint. The
full leakage and artifact contract is documented in [`BELIEF.md`](BELIEF.md).

## M6 Teacher and IS-KD

The privileged critic can regress chosen full-state MC-Q to terminal team returns.
IS-KD keeps Teacher evaluation under `no_grad`, samples four legal hidden hands by
default, averages action Q only across Belief samples, and trains Student
policy/value heads with one value-loss average per information set. Strict IS-KD
does not consume the true hidden assignment; including it and direct true-state KD
are explicit privileged ablations. Oracle Dropout is supplied per Teacher forward
so a training curriculum can move from full to partially masked hidden hands. See
[`DISTILLATION.md`](DISTILLATION.md).

## M7 DMC, V-trace, and Hybrid

The distributed boundary now provides action-aware versioned microbatch inference,
bounded Actor/Learner queues, role-homogeneous versioned trajectories, standard
clipped V-trace, and a unified three-mode learner step. Actors split each game by
observer so a trajectory's terminal reward and every preceding transition share
one seat perspective; mixed-observer batches are rejected. Raw platform rewards
remain beside their stable training transform. A checked experiment contract requires identical
frames, updates, seeds, model, rules, Actor layout, inference limits, and device
before DMC/V-trace/Hybrid metrics are accepted as comparable. No mode is declared
stronger without those completed runs. See
[`DISTRIBUTED_TRAINING.md`](DISTRIBUTED_TRAINING.md).

The process topology in `configs/train/actor_system.yaml` uses spawn-safe Python
Actors. Each process owns several Rust environments, sends ragged state batches to
one central versioned Inference Server, and writes complete compact trajectories
through a bounded multiprocessing queue. Actor crashes are diagnosed and restarted
only within the configured budget; IPC requests and shutdowns have finite timeouts.

## M8 Farmer Team Critic

Farmer updates can use a separate full-state Team Critic and a per-segment COMA
counterfactual advantage while the executing Actor remains information-set safe.
The default specialist optimizer freezes every landlord execution dependency and
updates the two Farmer adapters/heads plus their packed Farmer embedding rows.
Sparse Top-N alternative rollouts restore exact Rust states and use only true team
terminal returns. Hand-authored cooperation shaping is rejected. See
[`FARMER_COORDINATION.md`](FARMER_COORDINATION.md).

## M9 complete bidding and concurrent stages

The Bid Head is initialized against frozen Cardplay by branching every legal bid
over information-set-consistent opponent/bottom samples and running native games to
terminal. Research runs require a checksum- and version-pinned pretrained
Cardplay checkpoint. MC initialization wraps it with fixed phase-correct bidding
and doubling, while the frozen joint stage executes the same Cardplay component.
The explicit random-cardplay smoke exception uses `LongestMovePolicy` only to test
mechanics. Full-game collection then retains both bidding and card-play decisions
under the same terminal return. Bid collection is reproducible epsilon-greedy
MC-Q, and joint learning regresses selected Q/win/score outcomes without a
REINFORCE term. The curriculum unfreezes Cardplay only after
completed-game, calibration, call-rate, and redeal gates pass. MC pretraining and
the first two stages use a pure win/loss Q target; separate score-head loss and
score-utility coefficients are enabled only in the later gated stage. Q entropy is
zero by default. Score and rob modes use separate non-degeneration statistics.
Configuration, monitoring, and the empirical claim boundary
are documented in [`BIDDING.md`](BIDDING.md).

The executable smoke/resume path is:

```bash
bash scripts/train_full_game.sh
```

It checkpoints Bid Head, Cardplay, optimizer, scheduler, AMP scaler, RNGs, policy
version, curriculum stage, bidding calibration window, MC pretraining progress,
parsed bidding/Cardplay training-config fingerprints, and the League snapshot.
Filesystem locator strings are excluded from semantic fingerprints, so moving a
run does not invalidate it while editing referenced config content does.
`bid_pretraining_metrics.jsonl` records every resumable
privileged-label update separately from complete-game `metrics.jsonl`.

Mixed precision permits FP16/BF16 network activations but retains FP32 CRF dynamic
programs, RMS variance, probability normalization, and loss reductions. CPU BF16
gates run in ordinary CI; the CUDA-only full-game FP16 update runs when a GPU is
available.

## M10 Proposal and search distillation

Proposal training ranks all native legal actions from cheap public features. Hard
pruning remains off until the independent Teacher-recall, tactical-recall,
throughput, paired non-regression, and full-action-control gates all pass. The
configured seeded control fraction always trains on the full action set.

Triggered search produces normalized root visit probabilities, bounded values, and
Belief sample summaries for offline policy/value distillation. A second stage
distills the resulting large no-search network into the compact deployment actor.
Promotion requires paired positive search evidence and retention of the configured
gain fraction. Defaults and the empirical claim boundary are in
[`SEARCH_AND_DEPLOYMENT.md`](SEARCH_AND_DEPLOYMENT.md).
