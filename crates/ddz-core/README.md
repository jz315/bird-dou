# ddz-core

`ddz-core` is the dependency-free domain layer for BIRD-Dou (apart from Serde).

It intentionally contains **no** legal-action generation, bidding transition logic,
scoring policy, model code, or platform-specific rule decisions. Those belong in
`ddz-rules`.

## Design

- One `GameState`, one `GameAction`, one `Observation`.
- Strong seat/card/rank value objects prevent invalid indexing.
- `SeatMap<T>` replaces raw `[T; 3]` indexing throughout higher crates.
- `DealPlan` preserves the private physical deck order required by during-deal reveal.
- `LandlordSelectionState` represents call/rob progress without embedding one platform's
  transition algorithm.
- `DoublingState` stores per-seat decisions; `StakeState` stores common factors only.
- `Observation` has no seed, deck order, or hidden hands.
- Wire versions exist only in JSON envelopes (`codec.rs`); type names are not versioned.

## Integration boundary

`ddz-rules` should:

1. construct and mutate `GameState`;
2. produce legal `GameAction` values;
3. apply player and system transitions;
4. calculate reveal factors, rob order, doubling eligibility and payoffs;
5. construct `Observation` from `GameState`;
6. call `GameState::validate()` after transitions in debug/tests.

## Replacement

Copy this directory over `crates/ddz-core`. The other crates will need to migrate from
raw `u8` seats and `[u8; 15]` counts to `Seat` and `RankCounts`; that migration is
deliberate and should be done when `ddz-rules` is rewritten.
