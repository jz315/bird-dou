# Proposal, Endgame Search, and Deployment

## M10 scope and safety boundary

M10 is optional on top of the complete M9 game. It does not change rule legality,
the public `Observation`, or the unpruned Actor. Hard pruning and online search are
separately switchable, so either can be disabled without changing a checkpoint's
information contract.

No hidden assignment is accepted by the deployment policy. Hidden samples are
materialized only inside the search evaluator from a serialized public root and a
cardinality-constrained Belief assignment. Native replay validation reconstructs
the other container by exact deck conservation and rejects incompatible samples.

## Cheap Proposal and protected actions

`ProposalNetwork` scores each legal action from the public own/unknown rank summary,
public scalar state, action structure, post-action hand counts, and the native exact
minimum remaining play-group feature. Its small MLP is
configured by [`proposal_v1.yaml`](../configs/model/proposal_v1.yaml). Dynamic K
interpolates between `min_k` and `max_k` using a normalized uncertainty value.
Selected actions retain their original canonical order.

The protected mask can never be removed by Top-K. It always contains Pass, bombs,
rocket, and direct-finish actions. The caller supplies audited Boolean masks for
actions that block an opponent's immediate finish and Teacher high-value actions,
plus one exploration index per state. These tactical labels are explicit because
they must be computed from the relevant public/Teacher pipeline, not guessed by the
cheap scorer. `should_use_full_action_set` deterministically reserves the configured
fraction of training states for the complete legal set.

Hard pruning is disabled until `evaluate_proposal_gate` accepts independent-set
measurements for Teacher-best recall, 100% direct-finish recall, 100% bomb/rocket
recall, actual wall-clock throughput, paired non-regression, and observed unpruned
controls. NaN or out-of-range measurements are rejected.

Run the reproducible wall-clock harness with:

```bash
python scripts/benchmark_proposal.py --states 8 --iterations 5 --warmup 2 --threads 1
```

On the 2026-07-22 local CPU run with exact decomposition enabled, 553 legal actions
across eight states were reduced to 162 (29.29%). Proposal plus subset construction
plus the full Actor took 1.4708 s for five iterations, versus 3.1919 s for full
scoring, a measured 2.17x speedup.
This is a hardware-specific throughput result with randomly initialized networks;
it is not evidence of policy-strength retention.

## Triggered root-consistent search

[`endgame_search.yaml`](../configs/train/endgame_search.yaml) versions the master
switch, public triggers, rollout budget, exact-solve threshold, and risk coefficient.
The guarded entry point evaluates only these conditions:

- total public remaining cards;
- minimum public per-player remaining count;
- a legal bomb or rocket decision;
- constrained-Belief entropy.

When no condition fires, it returns without validating or consuming hidden samples.
When enabled, every root action is forced through every identical hidden-state
sample. A sample with a different acting player or legal root set is rejected. The
result reports expected bounded score value, win probability, standard deviation,
risk-adjusted value, and exact-solved sample count per root action.

This is an information-set-consistent engineering approximation. It has no claimed
two-player zero-sum public-belief convergence guarantee for three-player DouDizhu.

## Native exact endgame solver

`ddz-search` provides a perfect-information landlord-versus-farmer-team minimax
solver for card-play states under a configurable remaining-card and node budget.
It uses the authoritative native `apply_in_place`/`undo`, a transposition table over
future-relevant state, and deterministic canonical action order. Winners minimize
distance to a forced terminal result; losing sides maximize it. The result contains
forced landlord win/loss, plies to terminal, best action, visited nodes, and cache
hits. The Python binding accepts only a serialized, replay-validated state and the
matching `RuleConfig`.

## Search and compact-model distillation

`SearchDistillationBatch` stores the public RaggedBatch, normalized root visit
distribution, search value, and an explicit Belief-sample summary. Search policy
and value targets train a no-search network. A second loss distills its detached
policy/value targets into `CompactPolicyModel`. The loss weights, temperature, and
minimum retained-gain fraction are versioned in
[`search_distillation.yaml`](../configs/train/search_distillation.yaml).

The retention gate requires an actually positive search gain, the configured
fraction of that gain in the compact model, and a positive paired 95% confidence
interval lower bound. `evaluate_search_acceptance` independently rejects any
evaluation that searched outside trigger states or failed to beat the pure network.

## Observation-only deployment bundle

`export_deployment_bundle` atomically writes compact weights and a JSON manifest.
The manifest binds model/config fingerprints, optional Bid Head identity, rule
configuration hash, weight checksum, schema version, and the exact
`Observation+legal_actions:v1` input contract. Loading uses weights-only mode and
rejects any checksum, rule, architecture, configuration, or Bid Head mismatch.

`DeploymentPolicy.select_action` supports bidding, optional doubling, and card play
using only the acting player's legal `Observation` and canonical legal actions. It
has no API for opponent, teammate, bottom, Teacher, Critic, or sampled hidden hands.

Export trained tensor state dictionaries for an inference service with:

```bash
bird-dou-export-deployment \
  --compact-config configs/model/compact_policy_v1.yaml \
  --compact-weights artifacts/checkpoints/compact-state.pt \
  --rules configs/rules/douzero_post_bid.yaml \
  --output artifacts/deployment/bird-dou-compact.pt
```

Complete-game bundles add both `--bid-config` and `--bid-weights`; supplying only
one is rejected.

## Acceptance status

The implementation, safety, serialization, trigger, exact-solve, distillation, and
deployment contract tests are automated. The local CPU benchmark establishes a
throughput improvement under its recorded conditions. Full M10 research acceptance
still requires trained Proposal/search/compact checkpoints and a predeclared paired
Arena run proving no pruning regression, positive search gain, and retained compact
gain. Until those artifacts exist, the corresponding gates correctly remain
unclaimed rather than being filled with synthetic metrics.
