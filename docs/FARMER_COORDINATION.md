# Farmer Coordination

## Shared Actor and landlord isolation

The E020 Actor already shares rank, history, state, and action encoders across all
seats while retaining `landlord_down` and `landlord_up` embeddings, bottleneck
adapters, normalization rows, and output heads. M8 trains those two Farmer paths
through `FarmerSpecialistOptimizer`.

With the default `protect_landlord: true`, the shared trunk, landlord adapter, and
landlord output head are frozen. The packed role/seat/normalization embeddings are
optimized without weight decay, and any nonzero landlord-row gradient is rejected
before an update. A post-step snapshot guard provides a second exact check. Thus a
Farmer-only update must leave a fixed landlord information set bit-identical. The
unprotected shared-trunk mode exists only as an explicit ablation and requires a
new landlord evaluation.

The Actor checkpoint remains an ordinary `BirdDouModel` state dictionary. It has
no full-hand, Team Critic, or rollout input and never receives a teammate hand.

## Centralized Team Critic and counterfactual advantage

`FarmerTeamCritic` is a separate training-only model. It reuses the privileged
three-hand interaction encoder, forces Oracle Dropout to zero, rejects landlord
decision rows, and emits one `Q_team(s,a)` for every legal farmer action. Its
versioned configuration is
[`../configs/model/farmer_team_critic_v1.yaml`](../configs/model/farmer_team_critic_v1.yaml).

For each ragged legal-action segment, M8 computes:

```text
b(s,I) = sum_a pi_actor(a|I) Q_team(s,a)
A_cf    = Q_team(s,a_taken) - b(s,I)
```

The Actor loss uses the detached counterfactual advantage. The Critic separately
regresses the selected team Q to the true shared farmer terminal return. Optional
sparse alternative-action targets add Huber supervision to the same Critic. There
are no rewards for passing to a teammate, suppressing a teammate, or consuming a
particular landlord card; `handcrafted_cooperation_rewards: true` is rejected by
configuration construction.

## Bounded alternative rollouts

`PyDdzEnv.restore` reconstructs a versioned serialized state through the
authoritative Rust replay validator and an explicit `RuleConfig`. Restore is
transactional and returns the current public observation. It enables independent
branches without duplicating legality or transition logic in Python.

Only a deterministic bounded set of high-priority Farmer states is selected. At
one selected state, only the configured Top-N Actor alternatives are branched.
Every continuation policy sees public observations and legal actions only. The
result stores the source-state digest, canonical action bytes, Actor score, true
farmer-team return, and rollout length. The original environment is never mutated.
State, branch, and action caps are in
[`../configs/train/farmer_coordination.yaml`](../configs/train/farmer_coordination.yaml).

## Exploiter and acceptance gate

`generate_farmer_exploiter_schedule` evaluates the current Farmer champion and a
Farmer-only exploiter on the same deal set against one frozen strong landlord;
both Farmer seats always use the same shared policy ID. Formal promotion still
uses the paired Arena report. `evaluate_farmer_acceptance` requires a predeclared
farmer-team win delta (the lower confidence bound by default), bounds regression
for both downstream and upstream seats independently, and requires exact landlord
parameter isolation.

The gate reports failure instead of claiming that Centralized Critic training is
automatically stronger. A research-scale run against the chosen strong landlord
baseline is required before promotion.
