# Distributed Training

## M7 topology and bounded ownership

Actors keep authoritative Rust environments, encode only public observations, and
submit `RaggedBatch` requests to `InferenceServer`. The server queue has a hard
request limit. A batch closes when either the configured state limit or the sum of
the actual legal-action counts reaches its limit; candidates are never padded to a
batch-wide maximum. Adjacent requests for different policy versions are evaluated
separately against the requested immutable model snapshot.

Every response contains CPU tensors in the original candidate order, the original
action offsets, per-state segment-normalized policy probabilities, MC-Q, win and
score outputs, and the policy version that produced them. The microbatch deadline,
queue capacity, device, state limit, and action limit are versioned in
[`../configs/train/inference_server.yaml`](../configs/train/inference_server.yaml).

Backpressure is an awaited queue insertion rather than an unbounded request list.
Actor cancellation cancels its response future; the worker skips it. `stop()` wakes
blocked submissions, resolves all queued futures, and drains queue accounting. The
threaded `BoundedTrajectoryQueue` provides the same terminal close contract for
Actor-to-Learner episode transfer: a close wakes blocked producers and consumers,
and consumers may either drain or explicitly discard the bounded contents.

`Transition` records the serialized native state, observer, canonical selected
action, behavior log-probability, policy version, transformed reward, terminal
flag, and raw platform score. An Actor splits each completed game into up to three
role-homogeneous trajectories: every transition in one trajectory has the same
observer, and its terminal transition receives that observer's own payoff. This
makes `gamma = 1` recursion run between consecutive decisions made by the same
seat instead of crossing opponents' reward perspectives. Mixed-observer
trajectories are rejected both when constructed and at the Learner boundary.
`EpisodeMeta` records deal seed, rules hash, all three model versions, winner, and
zero-sum raw payoff. `TrajectoryReplay` evicts only whole oldest role trajectories
and samples from an explicit seed. The Learner can reconstruct observations and
stable legal actions from serialized states instead of storing every candidate
feature in replay.

## V-trace and switchable learner

`vtrace_from_log_probabilities` implements the standard backward recursion using
time-major `[T, B]` tensors. The initial terminal-task configuration is checked in
[`../configs/train/vtrace.yaml`](../configs/train/vtrace.yaml):

```text
gamma = 1.0
rho_bar = 1.0
c_bar = 1.0
policy_gradient_rho_bar = 1.0
```

Log importance ratios are bounded before exponentiation. Terminal flags zero the
continuation discount, and all inputs and outputs are checked for finite values.
`PolicyLagMonitor` retains only a fixed-size rolling window and reports mean,
maximum and p95 version lag, stale fraction, and importance-weight range.

`LearnerTrajectoryBatch` deliberately keeps `raw_reward` and `training_reward`
as separate time-major tensors and carries an `observer_seat` tensor that must be
constant down every batch column. `bird_dou_learner_step` selects the recorded flat
action in every ragged segment, computes current log-probabilities, state entropy,
an action-probability-weighted MC-Q state value, V-trace targets, policy lag, and
the selected mode's loss in one typed path. `TrainerMode.DMC` uses terminal MC-Q
and outcome heads; `TrainerMode.VTRACE` uses policy/value plus outcome heads;
`TrainerMode.HYBRID` additionally retains MC-Q, Belief, KD and auxiliary terms.
Each Hybrid coefficient is independently configurable in
[`../configs/train/hybrid.yaml`](../configs/train/hybrid.yaml).

Raw score is never fed to the optimizer without the stable transform:

```text
sign(raw_score) * log2(1 + abs(raw_score))
```

The win/score curriculum blend is metric-gated by its caller; no fixed-step stage
change is hidden inside the loss.

## Fair comparison contract

[`../configs/train/algorithm_comparison.yaml`](../configs/train/algorithm_comparison.yaml)
declares one common model, rules profile, seed set, environment-frame budget,
learner-update budget, Actor/env layout, unroll length, inference limits, and
device for all three modes. `validate_fair_comparison` rejects a missing mode,
unequal budget or input, incomplete run, non-finite metric, or inconsistent metric
set. Its report is a neutral ordered table and intentionally does not select a
winner. V-trace or Hybrid superiority is an experimental result, never an
implementation assumption; DMC remains the stable baseline.

The automated M7 gate covers hand-computed on/off-policy targets, terminal cuts,
extreme ratios and gradients, all mode formulas, bounded lag history, queue-full
backpressure, shutdown while blocked, Actor cancellation, two simultaneous model
versions, action/state batch ceilings, 128-request sustained service, bounded
replay eviction, reconstruction, opposing landlord/farmer terminal-gradient signs,
mixed-role rejection, and unfair-comparison rejection.
