# Constrained Hidden-Card Belief

## M5 two-container CRF

At a card-play decision, the public observation exposes the union unknown pool
`u[15]` and the two hidden players' remaining capacities. Container A is always
the next relative player and B the previous relative player. The model scores
every `x[r] = 0..u[r]` assignment to A; B is fixed as `u[r] - x[r]`.

The CRF admits only allocations satisfying `sum(x) = capacity_a`. A differentiable
forward log-space dynamic program computes `logZ`; the matching backward program
computes all `[B, 15, 5]` rank-count marginals. Invalid counts are masked, not
clipped. The supervised loss is exactly:

```text
NLL = logZ - sum_r score[r, true_x[r]]
```

The implementation also returns expected count, variance, and entropy for both
hidden players. Key summaries are the probabilities of holding a 2, small joker,
big joker, or at least one bomb; the bomb event is computed by a second restricted
partition function rather than an independence approximation. CRF reductions
remain FP32 under model autocast.

Backward-DP ancestral sampling draws complete `[B, samples, 15]` hands. Every
sample preserves each rank's unknown count and the exact total capacity. Tests
compare both DP directions with brute enumeration and compare 20,000 samples with
exact marginals; conservation violations are required to be zero.

## Policy fusion

`bird_dou_belief_v1` retains the E020 shared public encoder. A score network uses
only its pre-Belief state, public rank tokens, unknown counts, capacities, and
relative seat. Marginal moments form six per-rank channels (A/B expectation,
variance, entropy); pooled rank features and eight key-card probabilities become
`belief_pool`. A SwiGLU combines this pool with the public state before the
existing role adapter, legal-action encoder, and output heads.

Fusion is a learned residual `public_state + tanh(scale) * belief_update` whose
scale starts at exactly zero. Loading an E020 base checkpoint therefore gives
bit-identical policy and MC-Q outputs before joint fine-tuning, preventing an
untrained Belief branch from degrading the policy merely by being enabled.

The Student forward interface accepts only `RaggedBatch`. True hidden allocations
are never an argument. Tests hold the public batch fixed while changing external
oracle labels and require bit-identical Student output. Zeroing the learned CRF
score network changes policy output, which verifies that the fusion path is live.

## Supervision data and leakage boundary

The dataset generator is the only component that reads the full serialized
training state. It extracts the next relative player's remaining hand, then
requires both oracle hands to reconstruct the public unknown pool and public card
capacities exactly. Stored `.npz` files contain the public ragged features,
behavior action, training-only label, and policy-source index. Loading disallows
pickled/object arrays. A JSON manifest records the dataset SHA-256, schema,
master seed, state count, and source-policy mix.

The generic generator accepts any `Policy`, allowing random, rule, official
DouZero, current-model, and historical-checkpoint mixtures. The checked-in smoke
command uses deterministic random and longest-move policies without requiring
external weights:

```bash
bird-dou-generate-dataset --games 1 --seed 5005 \
  --output artifacts/datasets/belief_smoke.npz
```

## E020 warm-start, offline pretraining, and behavior anchoring

The CLI refuses to train a randomly initialized public policy. It pins an E020
checkpoint by SHA-256, policy version, base-model fingerprint, feature fingerprint,
and rules hash; loads its weights strictly into `model.base`; requires the residual
scale to be exactly zero; and proves bit-identical `policy_logit` and `mc_q` on the
training dataset before the first NLL update. That identity and parity report are
stored in the Belief checkpoint. The feature fingerprint uses the parsed feature
config plus the explicit `decomposition_features` training override, matching E020.

Offline training then freezes the public rank/history/state encoder and optimizes
exact NLL in shuffled mini-batches. The optional follow-up stage is deliberately
named `behavior_anchored_belief_finetune`: it imitates recorded behavior actions
plus weighted Belief NLL to limit representation drift. It is not DMC, on-policy
RL, Teacher distillation, or evidence of policy improvement. Configuration is
in [`../configs/train/belief_pretrain.yaml`](../configs/train/belief_pretrain.yaml):

```bash
bird-dou-train-belief --config configs/train/belief_pretrain.yaml \
  --report artifacts/train/belief_pretrain/report.json
```

The command reports trained and uniform constrained NLL and writes reliability
bins, Brier score, and expected calibration error for 2, both jokers, and any
bomb. The seed-5005 one-game warm-start smoke produced 54 mixed-policy states and
reduced mean NLL from the uniform `5.1511` baseline to `4.6691` after four updates. This
small smoke verifies mechanics and is not a strength claim; policy improvement,
label-shuffle degradation, and calibration confidence intervals require the
fixed-budget M5 experiment matrix before reporting research conclusions.
