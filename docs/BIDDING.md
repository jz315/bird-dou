# Complete bidding and staged joint training

## Authoritative full-game flow

`canonical_full` now executes the complete state machine in Rust:

```text
17-card deal → score/rob bidding → landlord + 3 bottom cards
→ optional three-seat doubling → card play → bomb/rocket/spring scoring
```

Before landlord resolution every observation has role `unassigned`, hides the
bottom cards, and exposes a 37-card unknown union. Resolving a bid transfers the
three bottom cards to the landlord. Score bidding, rob bidding, doubling, bomb and
rocket multipliers, spring/anti-spring, score caps, visibility, and all-pass policy
remain explicit `RuleConfig` choices. The engine represents all-pass as a zero-payoff
terminal attempt. When `redeal_on_all_pass` is enabled, `Arena` derives a new seed
with the versioned SplitMix64 mapping, records each attempt, and enforces a bounded
redeal count.

Single and packed environments use the same action protocol. Packed actions and
history rows carry both `phase` and a phase-local `action_code`, so bidding and
doubling never masquerade as empty card-play moves. `reset_complete_deal(...)` is a
privileged training API for information-set-consistent Monte Carlo samples; it is
not passed to an executing policy.

## Three-container belief

During bidding, each unknown rank is allocated among hidden player A, hidden
player B, and the three-card bottom. `three_container_crf.py` scores every valid
per-rank pair `(count_A, count_B)` and computes the exact partition with
`dp[rank][count_A][count_B]`. Bottom count is inferred by conservation.

The implementation provides exact allocation marginals, per-container count
marginals, moments, entropy, supervised NLL, and exact sequential sampling. Tests
compare the partition and marginals with brute-force enumeration and check rank and
capacity conservation at extreme capacities.

## Bid Head

`BidHead` consumes only:

- the acting player's 17-card rank counts;
- public bid history and relative actors;
- absolute seat and explicit rule mode/features;
- the 37-card unknown union through the constrained three-container belief;
- the complete ragged legal bid set.

Each legal action receives an MC-Q value, final-win logit, and expected final
score. A segment softmax over MC-Q is a diagnostic distribution normalized only
within that state's legal actions; it is not treated as an on-policy Actor-Critic
sample. The card-play Ragged encoder explicitly rejects bidding observations,
which keeps the two feature contracts unambiguous.

## Monte Carlo initialization and joint training

`generate_initial_bid_mc_labels` freezes the supplied continuation policy and
branches every legal initial bid across explicit sampled opponent-hand/bottom
allocations. Every branch runs the native rules engine to terminal and records the
acting seat's final win and raw score. Samples must preserve the same acting seat
and 17-card information set.

`sample_initial_bid_deals` constructs those allocations from one native initial
deal: it keeps the first bidder and own 17-card rank counts bit-identical, shuffles
only the 37-card hidden union with a derived seed, then re-applies the exact
17/17/3 capacities. `FullGameTrainer` completes the configured number of these MC
supervised updates before joint episodes. The update counter, sampled seed,
hidden-sample count, loss, optimizer/scheduler/scaler state, and RNG state are
checkpointed, so interruption cannot silently repeat or skip initialization.

A formal run must name a pretrained Cardplay checkpoint, its SHA-256, and its
policy version. The trainer verifies those values plus model and feature
fingerprints before creating any bid label. MC rollout composes a fixed auditable
bidder and doubler with that Cardplay-only `BirdDouPolicy`, so Cardplay never sees
bidding or doubling observations. The frozen joint stage uses the same Cardplay
component. Checkpoints, manifests, and pretraining rows record the composite,
bidding, doubling, and Cardplay identities together with the model hash/version,
architecture, decision mode, and rules hash. The executable gate runs one actual
checksum-pinned warm-start MC update across pass and bids 1/2/3 and observes subsequent
bidding, doubling, and card-play phases. `allow_random_cardplay_smoke: true` is the
only checkpoint-free path; it uses `LongestMovePolicy` solely for a fast mechanical
CPU smoke and carries no playing-strength claim.

`collect_complete_episode` records bidding and card-play decisions from one full
game under a single terminal payoff. `build_joint_bid_batch` attaches that payoff
to every earlier bid, while card-play decisions remain available to the existing
learner. `joint_bid_loss` trains chosen bid value/outcome heads and
`combine_joint_training_loss` includes the external card-play loss only after the
curriculum unfreezes it.

Joint collection is value-based: after fixed warmup, the Bid Head chooses MC-Q
epsilon-greedily with a context-derived reproducible exploration draw. The selected
action's MC-Q, win, and score heads regress the terminal outcome. There is no
REINFORCE term, action-dependent baseline, behavior-policy assumption, or hidden
off-policy correction. MC initialization supervises all legal candidates. Entropy
is computed as a segment sum per information set and then averaged across states.

Under autocast, matrix-heavy network layers may use FP16/BF16 while constrained CRF
dynamic programs, RMS variance accumulation, probability normalization, and loss
reductions remain FP32. CPU BF16 forward/backward gates cover Bid Head, Cardplay,
Belief, Teacher, and IS-KD; a CUDA FP16 full-game update is collected by the test
suite when CUDA is available.

The metric-gated curriculum has three stages:

1. `bid_win_frozen`: Cardplay frozen, win-first Bid Head initialization;
2. `joint_win`: Bid and Cardplay train together, score weight still zero;
3. `joint_score`: configured terminal score loss is enabled.

No stage advances by step count alone. Completed-game count, calibration error,
call rate, and redeal rate must all pass configured gates.

## Evaluation and claim boundary

The windowed monitor reports landlord strength mean/std, bids 1/2/3 ratio, positive
bid rate, redeal rate, and win/mean score conditioned on winning bid. The formal
acceptance function additionally requires:

- non-degenerate bidding;
- calibrated final-win predictions;
- bounded landlord-strength distribution drift;
- a strictly positive paired lower confidence bound against the declared fixed
  bidder plus strong Cardplay baseline.

These are executable gates, not an empirical claim. The repository's smoke tests
prove the full pipeline, exact constraints, and complete scoring. A research-scale
paired run is still required before claiming that a trained bidder is stronger.

Relevant configuration files are
[`configs/model/bid_head_v2.yaml`](../configs/model/bid_head_v2.yaml) and
[`configs/train/bidding.yaml`](../configs/train/bidding.yaml). Full-game run budgets
`bid_pretraining_batches` and `bid_pretraining_hidden_samples` are declared in the
full-game trainer config rather than hard-coded.
