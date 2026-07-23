# Migration from the old ddz-batch

This is a breaking replacement for the old single-file post-bid batch.

## Type changes

```text
old: BatchDdzEnv
new: BatchEnv

old: reset(&[u64])
new: reset_all(&[ResetSpec])

old: PostBidGame-only storage
new: ddz_rules::Game for DouZeroPostBid or HuanleClassic

old: one flat PackedObservation
new: nested PackedObservation grouped by domain

old: terminal rows require -1; active rows cannot skip
new: -1 is a general slot mask and is still mandatory for terminal rows

old: BATCH_SCHEMA_VERSION = 1
new: BATCH_SCHEMA_VERSION = 2
```

## Removed behavior

- no rule reimplementation;
- no legacy bid action encoding;
- no continuously repeated `last_objective_payoff`;
- no full-history copy inside every fixed observation;
- no whole-batch legal-cache invalidation after partial reset;
- no unchecked asynchronous action response.

## PyO3 migration

The next crate should expose nested protocol groups as NumPy arrays. Suggested Python dictionary:

```text
observation = {
  "status": {...},
  "cards": {...},
  "reveal": {...},
  "landlord": {...},
  "doubling": {...},
  "stake": {...},
  "card_play": {...},
  "outcome": {...},
}
```

Keep arrays as borrowed/read-only views where the PyO3 lifetime allows; do not rebuild lists of
per-environment Python dictionaries.

## Built-in ddz-rules companion API

The workspace's `ddz-rules` already provides a history-free information-safe observation path, so
fixed batched observations do not clone the entire public event log every turn.
