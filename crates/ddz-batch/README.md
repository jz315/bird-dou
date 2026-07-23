# ddz-batch

Native vectorized ownership and packed buffers for BIRD-Dou.

This crate is the batch layer above the single-domain-model `ddz-core` and the rewritten
`ddz-rules`. It owns a `Vec<Game>` inside Rust. Python/PyO3 callers make one call for reset,
observation, legal-action generation, or step; they do not loop over individual environments and
do not reimplement any rule.

## Design

```text
ddz-core      domain values and information-set structures
    ↑
ddz-rules     authoritative transitions and legal actions
    ↑
ddz-batch     native slot ownership, packed SoA buffers, atomic stepping
    ↑
ddz-pyo3      NumPy views / Python API
```

There is no second rule state machine in this crate.

## Main API

```rust
use ddz_core::Seat;
use ddz_rules::{RewardMode, RuleConfig};
use ddz_batch::{BatchEnv, ResetSpec};

let rules = RuleConfig::douzero_post_bid(1, RewardMode::WinPercentage);
let mut batch = BatchEnv::new(rules)?;

let observation = batch.reset_all(&[
    ResetSpec::post_bid(10, Seat::ZERO),
    ResetSpec::post_bid(11, Seat::ONE),
])?;

let actions = batch.legal_actions_packed()?;
let indices = vec![0_i64, 0_i64];
let step = batch.step_packed_checked(
    &indices,
    &actions.generation,
    &actions.revision,
)?;
# Ok::<(), ddz_batch::BatchError>(())
```

Each slot owns the global action range:

```text
[offsets[i], offsets[i + 1])
```

Indices passed into `step_packed` are **local** to that range. `-1` means “do not advance this
slot” and is required for terminal slots.

## Huanle reset

```rust
use ddz_rules::{EconomyContext, RuleConfig};
use ddz_batch::{BatchEnv, ResetSpec};

let rules = RuleConfig::huanle_classic(2, [0; 18]);
let mut batch = BatchEnv::new(rules)?;
batch.reset_all(&[
    ResetSpec::Huanle {
        match_seed: 100,
        economy: EconomyContext::unlimited(),
    },
    ResetSpec::huanle(101),
])?;
# Ok::<(), ddz_batch::BatchError>(())
```

## Stale-response protection

Every slot carries:

```text
generation: changes on reset/restore
revision:   changes on every committed player action
```

`PackedActions` and `PackedObservation` expose both. An asynchronous inference result should be
submitted through `step_packed_checked`; a result generated before a reset or for an older state is
rejected before any slot mutates.

## Transaction contract

`reset_all`, `reset_slots`, `restore_all`, and `step_packed` are transactional:

- all lengths, profiles, versions, and local indices are validated before mutation;
- each rule transition uses `Game::apply_with_undo`;
- a later slot failure rolls back every earlier slot;
- packed-output construction is part of the transaction;
- if rollback itself fails, `BatchError::Rollback` instructs the caller to discard the batch.

## Fixed state versus history

`observations_current` packs fixed current state only. It does **not** repack complete history on
every turn. This avoids the old quadratic history-copy pattern.

Use:

- `public_history_packed()` after reset/restore or whenever a consumer needs an information-safe
  public resynchronization;
- `authoritative_history_packed()` only for replay/debugging;
- `PackedStepResult::authoritative_events` for engine audit, not model input.

A hot-path actor may maintain history from the actions it selected, but it must apply the same
visibility rule as `ddz-rules`: unresolved double choices are buffered until the doubling round
resolves. The batch crate deliberately does not label authoritative deltas as public.

## Partial reset

```rust
use ddz_core::Seat;
use ddz_batch::{SlotReset, ResetSpec};

batch.reset_slots(&[
    SlotReset {
        slot: 3,
        spec: ResetSpec::post_bid(400, Seat::TWO),
    },
])?;
# Ok::<(), ddz_batch::BatchError>(())
```

Only reset slots receive a new generation and lose their legal-action cache.

## Snapshot and restore

`BatchEnv::snapshot()` records:

- immutable rules hash;
- match seed and economy per slot;
- exact slot version;
- versioned `ddz-core` state bytes.

`restore_snapshot()` deterministically replays every state through `ddz-rules::Game::restore` and
preserves versions. `restore_all` and `restore_encoded_all` instead assign fresh generations.

## Replace the crate

```bash
rm -rf crates/ddz-batch
cp -R /path/to/ddz-batch crates/ddz-batch
```

Then run:

```bash
cargo fmt --all -- --check
cargo test -p ddz-core
cargo test -p ddz-rules
cargo test -p ddz-batch
cargo clippy -p ddz-batch --all-targets -- -D warnings
```

The old `ddz-pyo3` binds the previous flat batch protocol and must be migrated next. Do not add a
compatibility state machine or duplicate rule logic back into this crate.

## Built-in history-free observation path

The workspace rule engine already exposes `Game::observe_without_history`. It does not change rule
semantics and allows the fixed batch protocol to remain truly fixed-cost with respect to match
history length.
