# Packed protocol v2

All rank axes have length 15 and seat axes have length 3.

## Legal actions

```text
offsets              [B + 1] u64
owner                [M]     u32
generation/revision  [B]     u64
action scalar fields [M]
move_cards           [M, 15] u8
```

## Fixed observation

```text
status fields                  [B]
own_hand / unknown / bottom    [B, 15]
revealed_hands / played_cards  [B, 3, 15]
seat masks/counts              [B, 3]
```

Terminal rows produced by `observations_current` have `status.valid = 0` and observer-dependent
card buffers are zero. Call `observations_for` to request an explicit terminal perspective.

## Events

```text
offsets          [B + 1]
sequence/attempt [E]
event kind       [E]
actor            [E]
player action    [E]
system kind/args [E]
```

Player and system rows share one aligned event axis. Unused fields contain sentinels.

## Sentinels

```text
NO_SEAT          -1
SKIP_ACTION      -1
NO_U8            255
NO_RANK          255
NO_ACTION        255
NO_EVENT_CODE    255
```

## Stable enum codes

The crate exports named constants for every non-`repr(u8)` protocol tag. Consumers must use those
constants rather than copying numeric literals. Binary decision subcodes use:

```text
DECISION_NO   0   continue/pass/decline
DECISION_YES  1   reveal/call/rob/double
```

`MoveKind` is the only move tag encoded from its Rust `repr(u8)` declaration.

## Version checks and masks

`step_packed_checked` validates generation/revision only for rows whose local action index is not
`SKIP_ACTION_INDEX`. This allows an asynchronous caller to commit a ready subset while masking
slots whose older inference response is being discarded.

## History boundary

Fixed observations are built through the workspace-provided `Game::observe_without_history` API.
`public_history_packed` is an explicit full resync. Raw `authoritative_events` are suitable for
replay and auditing, but unresolved double choices must not be fed to a player's model history.
