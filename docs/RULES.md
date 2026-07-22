# Rules

This document is normative for implemented rule and representation choices. Rule
behavior not yet implemented remains governed by
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). E002 defines card identity and
the two versioned rule configurations; move recognition and state transitions begin
with E003.

## Card and rank representation

The deck contains 54 unique physical card IDs. Suits are intentionally opaque: they
make deals and replays auditable but never affect move comparison.

| Rank ID | Rank | Physical card IDs |
|---:|---|---|
| 0–11 | 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A | `4*r` through `4*r+3` |
| 12 | 2 | 48–51 |
| 13 | Small joker | 52 |
| 14 | Big joker | 53 |

`RankCounts` is a 15-element array in that order. Expanding counts to physical
cards chooses the lowest IDs for each rank, so conversion is deterministic. Invalid
card IDs, duplicate physical cards, and impossible rank multiplicities are errors;
they are never clipped or normalized silently.

## Canonical move representation

`MoveKind` schema version 1 assigns stable tags in the order shown below. A move
stores rank counts, `main_rank`, `chain_len`, and the redundant but validated
`total_cards` value.

| Tag | Kind | `main_rank` | `chain_len` | Total cards |
|---:|---|---|---:|---:|
| 0 | Pass | sentinel 15 | 0 | 0 |
| 1 | Single | played rank | 1 | 1 |
| 2 | Pair | pair rank | 1 | 2 |
| 3 | Triple | triple rank | 1 | 3 |
| 4 | Triple with single | triple rank | 1 | 4 |
| 5 | Triple with pair | triple rank | 1 | 5 |
| 6 | Straight | lowest body rank | number of body ranks, at least 5 | `chain_len` |
| 7 | Pair straight | lowest body rank | number of body ranks, at least 3 | `2*chain_len` |
| 8 | Triple straight | lowest body rank | number of body ranks, at least 2 | `3*chain_len` |
| 9 | Airplane with singles | lowest triple rank | number of triple ranks, at least 2 | `4*chain_len` |
| 10 | Airplane with pairs | lowest triple rank | number of triple ranks, at least 2 | `5*chain_len` |
| 11 | Four with two singles | four-card rank | 1 | 6 |
| 12 | Four with two pairs | four-card rank | 1 | 8 |
| 13 | Bomb | four-card rank | 1 | 4 |
| 14 | Rocket | big-joker rank 14 | 1 | 2 |

Chains may use only rank IDs 0–11 (3 through A); 2 and jokers never enter a
chain. Every body rank must have the multiplicity implied by the declared kind.
Kinds with pair attachments require even multiplicity at every non-body rank.
Attachment legality that depends on `RuleConfig` is enforced by the E004 detector,
not silently inferred by the data structure.

Moves have the following stable total order:

```text
kind tag → total_cards → chain_len → main_rank → rank counts lexicographically
```

Pass therefore sorts first. Serialization uses the snake-case kind name and all
five fields. Deserialization reconstructs the move through the validating
constructor and rejects unknown fields, impossible counts, noncanonical metadata,
or a forged `total_cards` value.

## Move detection

`detect_move(RankCounts)` classifies the structural union of move forms that can
be expressed by a rule configuration. `detect_move_with_rules(RankCounts,
RuleConfig)` additionally rejects disabled four-with-two forms and attachment-rank
multiplicity forbidden by the selected profile. No default profile is selected
inside the structural detector.

Detection applies these boundaries:

- empty counts are the canonical Pass;
- groups contain exactly one body rank with multiplicity 1, 2, 3, or 4;
- triple attachments exclude the triple body rank; pair attachments have even
  multiplicity;
- straight, pair-straight, and triple-straight ranks are consecutive and confined
  to 3 through A, with minimum lengths 5, 3, and 2 respectively;
- airplane bodies contain consecutive ranks with exactly three cards each;
- airplane single wings contain one physical card per body rank, while pair wings
  contain one pair per body rank before profile multiplicity restrictions;
- four-with-two has one four-card body and exactly two singles or two pairs;
- rocket contains exactly one small joker and one big joker.

If the same airplane card multiset admits more than one body window, the highest
valid `main_rank` is selected. This makes a single card multiset normalize to one
stable action. The E008 differential gate compares card-multiset action sets, so
this metadata normalization does not manufacture duplicate platform actions.

## Free-lead action generation

`generate_lead_moves(RankCounts, RuleConfig)` returns every legal non-Pass move
that can be led from a hand. Generation is template-based rather than an
enumeration of arbitrary hand subsets:

- group templates emit every available single, pair, triple, bomb, rocket, and
  triple attachment;
- straight, pair-straight, and triple-straight templates emit every qualifying
  subinterval of each consecutive run from 3 through A;
- airplane templates enumerate every consecutive triple body and allocate wings
  from non-body capacity;
- four-with-two templates allocate attachments only when that form is enabled.

Attachment allocation operates on rank capacities after reserving the complete
body. Body ranks cannot also supply wings. A `distinct_ranks` rule caps each
attachment rank at one unit; `may_share_rank` permits multiple single or pair
units from the same rank up to the physical hand capacity. Every candidate is
passed through rule-aware detection, so ambiguous shapes receive the same
canonical metadata as directly detected moves.

The returned vector is deduplicated and sorted by `Move`'s stable total order.
It never contains Pass or consumes more cards of a rank than the hand holds.

## Follow action generation

`generate_follow_moves(hand, target, rules)` requires a legal, canonical,
non-Pass target. It returns Pass plus every move from `hand` that satisfies the
following comparison hierarchy:

1. a normal move beats the target only when kind and chain length match and its
   main rank is strictly higher;
2. every bomb beats every non-bomb normal move;
3. a bomb beats another bomb only at a strictly higher main rank;
4. the rocket beats every non-rocket move;
5. nothing beats the rocket, so its response set contains only Pass.

Bomb and rocket overrides do not need to match the target's kind or card count.
The result is deduplicated and uses the same stable total order as free-lead
generation, which places Pass first. A Pass target is rejected because two
consecutive passes transfer control to free-lead generation at the game-state
layer rather than creating a response to Pass.

## Post-bid game state and transitions

`PostBidGame` executes configuration ID 1 from a fully specified deal. Its
constructor requires a 20-card landlord hand that already contains the three
bottom cards, two 17-card farmer hands, and a rank-wise partition of the complete
54-card deck. The landlord acts first. Configuration profiles with bidding or
doubling are rejected by this E007 engine rather than silently skipping phases.

Every successful step appends a sequenced `GameEvent`. Non-Pass play subtracts
the move from the current hand, adds it to that seat's public played cards, updates
cached card counts and the active target, then advances clockwise. Pass does not
change any cards. The first Pass preserves the active target; the second clears
the target and returns free-lead control to its player. Pass is therefore never
legal in a free-lead state.

A player emptying their hand makes the state terminal immediately. No further
action is legal and the winning seat remains `current_player`. Each bomb and the
rocket increments both `bomb_count` and the base-two `multiplier_exp`. Under the
post-bid profile, raw zero-sum payoff is:

```text
landlord win: [+2m, -m, -m]
farmer win:   [-2m, +m, +m]
m = 2 ^ bomb_count
```

The arrays are rotated to the configured landlord seat. Objective payoff uses
seat-wise win/loss signs with magnitude 1 for WP, `2 ^ bomb_count` for ADP, and
`bomb_count + 1` for logADP. This matches the reward and bomb accounting in the
pinned official `DouZero` environment.

`observe(seat)` exposes only the observer's own hand, public bottom and played
cards, hand sizes, targets, counters, and public history. Opponent current hands
are collapsed into one `unknown_pool`; their allocation is not represented.

## State serialization and reversible transitions

Game-state wire schema version 1 is a compact UTF-8 JSON envelope containing an
explicit `schema_version` and the complete `GameState`. Unknown, missing, malformed,
or future-version fields are rejected. `PostBidGame::deserialize_state` does more
than decode fields: it reconstructs the original deal from current and played
cards, replays every sequenced event under the supplied `RuleConfig`, and accepts
the state only when every final field matches the replay result exactly. The
execution-local undo revision is not serialized.

`apply_in_place` shares the transactional transition path used by `step` and
returns an `UndoToken`. A token stores only the acting seat's previous hand,
played cards, affected scalars, history boundary, and revision; it contains no
history vector or complete state copy. `undo` requires last-in-first-out revision
and history matching before truncating the one appended event and restoring the
delta. Each engine instance and cloned search branch has a distinct internal ID,
so tokens cannot cross branches even when their public states coincide. Rejected
transitions and rejected undo attempts leave the state unchanged.

M10 uses the same transition authority for perfect-information endgame solving.
The solver performs coalition minimax for landlord versus the two farmers, restores
every branch through `undo`, and caches only future-relevant state. Python Belief
samples are materialized by `PostBidGame::with_hidden_assignment`: the supplied next
seat hand and inferred third hand must satisfy exact counts and the reconstructed
state must pass normal replay validation. Search never changes move legality.

Because search already selects from `legal_actions`, applying one candidate does
not regenerate the full set. It validates canonical move metadata, per-rank hand
capacity, and target comparison directly in constant rank width before mutation.
The reproducible `benchmark_apply_undo` example compares this path with state
cloning and full-set validation.

[`tests/golden_replays/post_bid_five_bombs.json`](../tests/golden_replays/post_bid_five_bombs.json)
pins a complete replay outcome and the FNV-1a digest of its serialized terminal
state so accidental wire drift is visible in CI.

## Python single-environment reset

`birddou.PyDdzEnv` is a thin PyO3 owner of the authoritative game; Python never duplicates
move detection, action generation, transitions, payoff, or observation masking.
`reset(seed, rule_config)` strictly decodes and validates the supplied dictionary,
then shuffles physical card IDs with the versioned
`splitmix64_fisher_yates_v1` algorithm. `douzero_post_bid` assigns seat 0 as landlord;
`canonical_full` deals 17 cards per seat, retains a hidden three-card bottom, and
selects the first bidder from the seed. `PyDdzEnv` is intentionally a v1 engine:
it rejects a parsed `huanle_classic_v1` v2 dictionary before any legacy state or
replay code can interpret it.

Observations, actions, events, and step results use the same Serde field names as
the Rust protocol and are materialized as ordinary Python dictionaries and lists.
`serialize()` returns the E009 state envelope as `bytes`. Failed configuration,
action, or seat validation is explicit; failed resets and steps do not replace or
partially mutate a valid game. The checked-in seed-7 golden deal prevents silent
shuffle drift across dependency, compiler, or platform updates.

## Configuration schemas

Rule configuration schema version 1 is the legacy schema and records:

- a stable non-zero configuration ID and named profile;
- landlord first-turn policy;
- disabled, score, or rob bidding and an explicit maximum score bid;
- bottom-card visibility and whether the standard binary doubling phase is enabled;
- bomb and rocket multipliers;
- landlord spring and anti-spring toggles plus multiplier;
- airplane and four-with-two attachment multiplicity;
- all-pass redeal policy, optional absolute score cap, and reward representation.

Unknown YAML fields and inconsistent combinations are rejected. This prevents a
misspelled or omitted platform choice from silently becoming a default.

Schema version 2 is a separate Huanle-only wire format. It records the deal,
reveal, calling, robbing, doubling, card-play, pairwise settlement, and reward
subsystems as separate required structures. The Huanle parser requires all 18
reveal-factor entries and every unresolved platform choice, including the caller
reclaim, attachment, spring, and cap policies. A stable SHA-256 rules hash is
computed over the complete typed configuration.

The versioned reader dispatches from `schema_version`; v1 readers reject v2 before
deserializing v2 fields, v2 readers reject v1 before deserializing v1 fields, and
the two engines reject the other schema. The structural fixture in
[`tests/rules/huanle_classic_v1/parser_fixture_v2.yaml`](../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml)
tests this contract but is not a deployable Huanle room profile. The v2 state
machine begins in R003, so no legacy executable path is repurposed as Huanle.

## Huanle match attempts

`HuanleMatchV2` is the v2 lifecycle coordinator, separate from the legacy
single-game engine. A match seed deterministically derives a child seed, full
physical deck, and fallback first-caller candidate for every zero-based deal
attempt. A no-reveal all-pass closes only that attempt: its accepted-action
count and audit data enter `completed_attempts`, then a `Redeal` system event
starts the next child seed under the same match. There is no game-rule retry
limit in this layer.

R004 makes the early attempt lifecycle authoritative. `PreDealReveal` follows
a deterministic six-permutation declaration order derived from the attempt
seed and pinned as `deal_seed_permutation_v1`. After all three declarations,
Rust deals one card to each seat per round
and records only card-count system events, never card identities. In
`DealingReveal`, every still-hidden seat has one reveal-or-continue decision
per round. `RevealStateV2` retains per-seat factors, the event-ordered first
revealer, and the maximum (not product) factor. After 17 rounds, the match
opens `Calling` with the first revealer as `first_caller`, or the seeded
candidate when nobody revealed.

`RevealObservationV2` is an intentionally narrow safe projection for this
stage: it exposes the observer's own partial/full hand and the current hand of
revealed seats only. It has no deck, bottom-card collection, or unrevealed
opponent hand field. It is not the full v2 observation schema, unknown-pool,
serialization, or undo implementation reserved for R009.

R005 adds `CallStateV2` at the Calling boundary. Its only legal actions are
`CallLandlord` and `PassCall`, and only `current_player` may use one. The first
positive call sets `caller` and opens `Robbing` without assigning a final
landlord or manufacturing a rob action. A pass marks both `acted` and
`declined`, allowing R006 to derive eligibility as `!declined`. After all three
passes, a first revealer is deterministically recorded as landlord and the
phase stops at `BottomReveal`; without a revealer, the fully auditable call
state produces the existing same-match `AllPass` redeal. Direct generic action
recording rejects every phase action so no later phase can be simulated before
its ticket owns it.

The coordinator has a replayable decision log and a separately ordered system
log. R004/R005 validate reveal and call actions directly. One accepted call
action may deterministically append its immediate all-pass lifecycle event;
replay compares that entire generated event suffix rather than accepting an
unvalidated shortcut. Rob, bottom, double, play, and settlement remain later
phase state machines. This preserves every attempt's action budget while
keeping phase legality in its owning ticket.

## `douzero_post_bid`

[`configs/rules/douzero_post_bid.yaml`](../configs/rules/douzero_post_bid.yaml)
is pinned as configuration ID 1:

- the landlord is already assigned, holds 20 cards, and acts first;
- the farmers hold 17 cards each and the three bottom cards are public;
- bidding, doubling, spring, anti-spring, all-pass redeal, and score caps are disabled;
- every bomb and the rocket multiply ADP magnitude by two;
- reward may be WP, ADP, or logADP; the checked-in profile selects ADP;
- four-with-two singles and four-with-two pairs are enabled;
- two single attachments may share a rank, while pair attachments use distinct ranks;
- airplane single attachments may share a rank, while pair attachments use distinct ranks.

These compatibility constraints are validated in code. The attachment choices and
reward behavior were checked against official DouZero commit
`718a5c920bf3361e34178a38f3b80458e176b351` (`move_generator.py`, `game.py`, and
`env.py`). The E008 source-pinned harness now verifies legal action sets and
synchronized complete trajectories against that commit on every differential run.

## `canonical_full`

[`configs/rules/canonical_full.yaml`](../configs/rules/canonical_full.yaml) is the
project's explicit research profile, not an alias for an unnamed commercial platform.
Configuration ID 2 selects:

- score bidding with a maximum bid of 3; all players passing causes a redeal;
- public bottom cards, landlord first play, and the standard binary doubling phase;
- two-times bomb, rocket, spring, and anti-spring multipliers;
- both four-with-two forms; two single attachments may share a rank and pair
  attachments must use distinct ranks;
- airplane single and pair attachment ranks must be distinct;
- uncapped raw platform score.

The bidding, doubling, complete payoff, all-pass, privacy, undo, restore, and random
full-game transitions are executable and tested. Any new platform variant must use
a new configuration ID and explicit fields rather than changing this profile in place.
