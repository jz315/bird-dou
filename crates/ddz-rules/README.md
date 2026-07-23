# ddz-rules

Authoritative DouDizhu rules for BIRD-Dou.

This crate is a **breaking replacement** for the current `crates/ddz-rules`. It is
written against the single-domain-model `ddz-core` already present in the repository:
there is one `GameState`, one `Phase`, one `GameAction`, and one `Observation`.
There are no `V1`/`V2` runtime types, no legacy state machine, no score-bidding branch,
and no parallel `match_v2` implementation.

## Responsibilities

`ddz-rules` owns:

- validated rule configuration and stable rule hashes;
- deterministic physical dealing and redeals;
- standard non-laizi move detection and comparison;
- template-based lead generation;
- **shape-directed** follow generation;
- Huanle reveal, call, rob, bottom reveal, double, card-play, and terminal transitions;
- information-set-safe observations;
- spring detection, pairwise landlord–farmer settlement, and learner rewards;
- transactional `step`, snapshot-based undo, rule-owned invariants, and exact event replay on restore.

It intentionally does **not** own models, reinforcement learning, vectorized environments,
Python bindings, UI state, timers, networking, or bean-account persistence.

## Architecture

```text
config/                 immutable, validated rules and canonical hash
moves/
  detect/               structural recognition + profile validation
  generate/             group / chain / four-with-two generators
engine/
  automatic.rs          deterministic phase advancement only
  legal.rs              legal actions for the current decision
  transition/           one module per player-controlled phase
  observe.rs            private state -> information-safe observation
  landlord.rs           landlord resolution and same-match redeal
settlement.rs           spring + pairwise zero-sum payoff
```

No source file is intended to become an all-purpose game engine. Player actions are
handled by small transition modules, while automatic transitions are centralized and
bounded.

## Supported profiles

### `douzero_post_bid`

Starts directly in `Phase::CardPlay` with a fixed landlord. It is intended for exact
post-bid comparison with DouZero and exposes WP, ADP, or logADP terminal rewards.

### `huanle_classic`

Runs:

```text
PreDeal reveal
-> incremental dealing and optional during-deal reveal
-> call landlord
-> rob landlord
-> reveal bottom cards
-> landlord post-bottom reveal decision
-> per-player doubling
-> card play
-> pairwise settlement
```

All-pass without a revealer creates a new deterministic deal attempt **inside the same
match** and preserves prior public events. If at least one player revealed, the first
revealer becomes landlord when all three pass.

## Reveal schedule

The supplied rule text states that during-deal reveal can be ×4 or ×3 but does not state
the exact card-count boundary. This crate does not guess it. The rule config contains:

```rust
pub during_deal_factors: [u32; 18]
```

Index `n` is the factor available after receiving `n` cards; zero disables the decision
at that count. Fill the table from a confirmed platform specification before calling the
profile frozen.

## Pairwise doubling

Doubling is not a single global bit. If `L` is landlord and `F_i` one farmer:

```text
pair_stake_i = common_stake
             * (L doubled ? 2 : 1)
             * (F_i doubled ? 2 : 1)
```

The landlord payoff is the sum of the two pair transfers. Therefore the two farmers may
legitimately receive different raw scores while the total payoff remains zero.

## Incremental-deal observation invariant

Incremental dealing means an observation's unknown pool includes undealt cards. The workspace's
`ddz-core` already validates that invariant; it adds no new runtime type.

## Replace the crate

```bash
rm -rf crates/ddz-rules
cp -R /path/to/this/ddz-rules crates/ddz-rules
```

Then run:

```bash
cargo fmt --all -- --check
cargo test -p ddz-core
cargo test -p ddz-rules
cargo clippy -p ddz-rules --all-targets -- -D warnings
```

## Important rule choices

The code implements the selected interpretation that an original caller may reclaim the
landlord after another player robs, with every eligible seat receiving at most one real
rob decision. Attachment-rank multiplicity remains explicit in `MoveRules` because the
provided natural-language rules do not settle every airplane/four-with-two edge case.

Spring and anti-spring are configurable but disabled in the supplied Huanle template,
because they were not part of the frozen rule text. Enable them explicitly only for a room
whose specification includes those multipliers.
