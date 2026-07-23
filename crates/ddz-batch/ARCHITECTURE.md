# Architecture

## Responsibilities

`ddz-batch` owns only cross-environment concerns:

- a native vector of independent `ddz_rules::Game` values;
- per-slot generation/revision identity;
- revision-aware legal-action caching;
- full and partial transactional reset;
- atomic masked stepping with rollback;
- packed structure-of-arrays protocols;
- exact typed snapshot/restore.

It does not own rules, features, neural-network tensors, RNG algorithms, or Python objects.

## Modules

```text
batch/
  env.rs          owner, read-only access, generation allocation
  reset.rs        full/partial reset
  restore.rs      state codec, exact snapshot, deterministic replay
  legal.rs        revision-aware legal cache access
  observe.rs      current/explicit observations and public history
  step.rs         prevalidation, atomic commit, rollback, reward emission

protocol/
  actions.rs      ragged legal actions
  events.rs       ragged event streams
  step.rs         transition/reward result
  observation/
    status.rs
    cards.rs
    reveal.rs
    landlord.rs
    doubling.rs
    stake.rs
    card_play.rs
    outcome.rs
```

No source file is an all-purpose batch engine.

## Why native but initially sequential

A single Rust call already removes Python object churn and rule duplication. Correct transactional
semantics and profiling come before adding Rayon. The slot representation is independent and can be
parallelized later without changing the public protocol, but parallel execution is not included
without a benchmark proving it helps the actual legal-generation workload.

## Cache contract

Each cache entry is keyed by `SlotVersion`. Successful action commit increments revision and
invalidates only affected slots. Reset/restore changes generation. Failed transactions restore both
state and versions, so old cache entries remain valid.

## Information boundary

The fixed packed observation is built from `Game::observe` for an explicit observer. Public
history also comes from that observation. Raw engine event deltas are clearly named authoritative because unresolved double choices are
temporarily hidden by the rule layer. Consumers either resynchronize through the public-history
API or maintain an equivalent visibility-aware history outside the batch crate.

## Reward contract

`PackedStepResult.reward` is emitted once, only on the transition that becomes terminal. It follows
the configured learner reward mode. `raw_payoff` is the terminal pairwise score snapshot and remains
available on later skipped terminal calls. This avoids repeating terminal learner reward every actor
tick.
