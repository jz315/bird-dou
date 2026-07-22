# Feature Schema

The normative BIRD-Dou model feature schema is introduced by E016. E014 first
adds the isolated `douzero_54x15_v1` compatibility schema, while E011 defines the
separate version-1 **raw batch transport schema**. Raw transport contains
authoritative observations and canonical actions, not embedded or normalized
model features.

## E014 DouZero compatibility features

`encode_douzero_features` is a native NumPy encoder over the information-safe
`Observation`; it never reads engine state or individual opponent hands. Each
rank 3 through 2 occupies four cumulative count bits, followed by one bit for
each joker, for 54 values total. Pass is the all-zero vector.

The last 15 public actions are left-padded with Pass vectors, then every three
successive 54-vectors are concatenated to produce `[5, 162]`. This history is
repeated once per legal candidate. Candidate rows retain the authoritative legal
action order.

| Position | Flat candidate shape | History shape |
|---|---:|---:|
| `landlord` | `[A, 373]` | `[A, 5, 162]` |
| `landlord_down` | `[A, 484]` | `[A, 5, 162]` |
| `landlord_up` | `[A, 484]` | `[A, 5, 162]` |

The flat landlord vector concatenates own cards, unknown cards, active last move,
both farmers' played cards and remaining-card one-hots, a 15-way bomb-count
one-hot, and the candidate. Farmer vectors additionally represent the landlord
and teammate's most recent actions and use the matching remaining-card one-hots.
The schema version is `1`; any ordering or dimensional change requires a new
version. `--douzero-feature-encoder official_reference` remains an explicit
test-only fallback for differential checks against the pinned upstream source;
`native` is the production default.

## E016 BIRD-Dou RaggedBatch schema v1

`encode_ragged_batch` accepts only current-player `Observation` objects and their
authoritative legal-action lists. It cannot access individual opponent hands.
For `B` states, `M` total actions, and configured history length `T` (default 96),
the output is:

| Tensor | Shape | Dtype |
|---|---:|---|
| `rank_categorical` | `[B, 15, 9]` | `int64` |
| `rank_numeric` | `[B, 15, 3]` | `float32` |
| `history_rank_counts` | `[B, T, 15]` | `int64` |
| `history_meta` | `[B, T, 15]` | `int64` |
| `history_mask` | `[B, T]` | `bool` |
| `scalars` | `[B, 15]` | `float32` |
| `action_rank_counts` | `[M, 15]` | `int64` |
| `post_hand_counts` | `[M, 15]` | `int64` |
| `action_meta` | `[M, 14]` | `int64` |
| `action_state_index` | `[M]` | `int64` |
| `action_offsets` | `[B+1]` | `int64` |
| `chosen_action_flat_index` | `[B]` | `int64` |

Every state owns the non-empty half-open segment
`[action_offsets[i], action_offsets[i+1])`. `action_state_index` must be exactly
the inverse mapping. Training converts a local chosen index to its flat index;
inference uses `-1`. Construction validates all shapes, dtypes, offsets, state
indices, and chosen indices. `RaggedBatch.to(device)` moves the full structure as
one semantic unit.

### Rank tokens and relative seats

The nine categorical columns, in frozen order, are `rank_id`, `own_count`,
`unknown_count`, three public-played counts in self/next/previous relative-seat
order, `last_non_pass_count`, `public_bottom_count`, and
`is_straight_eligible`. Counts remain categorical for embeddings. The three
numeric channels are capacity-normalized own, unknown, and total public-played
counts. Rotating physical seats while preserving the observer-relative
information set leaves these rank features identical.

### Full public history

Valid events are stored from row zero and padding is on the right. The mask is the
only padding authority. Metadata columns, in order, are phase, relative actor,
Pass/play/bid/double flags, move kind, main rank, chain length, wing kind, total
cards, reconstructed cards left after the event, multiplier exponent after the
event, trick index, and position in trick. Rank counts are stored separately.

If public history exceeds `T`, all bidding events are retained, followed by the
configured earliest play context and the latest play events. The number omitted
is recorded in `history_truncated`; padding width never changes a valid row's
encoding. Post-bid v1 has no double events, but the stable column is reserved.

### Candidate actions and hand decomposition

Action metadata freezes the following order: move kind, main rank, chain length,
wing kind, total cards, Pass/bomb/rocket flags, empties-hand and leaves-one-card
flags, broken-bomb and broken-pair counts, `min_groups_after`, and capped optimal
decomposition count. A bomb is “broken” only when partially consumed; a pair is
broken when a rank with at least two cards falls below two after the action.

`min_groups_after` is exact under the active `RuleConfig`. The `ddz-search`
dynamic program uses authoritative Rust lead generation and shares one memo over
all post-action hands in a state. The count records optimal **ordered** play
decompositions and saturates at the configured cap. These are model features only
and never prune, reorder, or replace legal actions. The ablation switch
`decomposition_features: false` emits reserved sentinels 21 and `cap + 1`.
Configuration is versioned in
[`../configs/model/bird_dou_features_v1.yaml`](../configs/model/bird_dou_features_v1.yaml).

Every non-scalar value returned by `PyBatchDdzEnv` is a C-contiguous NumPy array.
Count and tag buffers use `uint8`; event sequences use `uint32`; ragged offsets
and action indices use `int64`; payoffs use `int32`. Masks intentionally use
compact `uint8` values 0/1. No object-dtype array is part of the protocol.

## Batch observation

For batch size `B`, total public history events `H`, and 15 ordered ranks:

| Field | Shape | Dtype |
|---|---:|---|
| `phase`, `observer`, `role`, `current_player` | `[B]` | `uint8` |
| `own_hand`, `public_bottom_cards`, `unknown_pool` | `[B, 15]` | `uint8` |
| `public_played` | `[B, 3, 15]` | `uint8` |
| `cards_left` | `[B, 3]` | `uint8` |
| `landlord` | `[B]` | `int8`, `-1` absent |
| `last_non_pass_valid`, counters, terminal | `[B]` | `uint8` |
| `last_non_pass_rank_counts` | `[B, 15]` | `uint8` |
| `last_non_pass_kind/main_rank/chain_len/total_cards` | `[B]` | `uint8` |
| `history_offsets` | `[B+1]` | `int64` |
| `history_rank_counts` | `[H, 15]` | `uint8` |
| history actor/kind/rank/length/card-count metadata | `[H]` | compact integers |

Phase codes are bidding 0, doubling 1, card play 2, and terminal 3. Role codes
are landlord 0, farmer 1, and unresolved/unassigned 2. Missing move kind and main
rank use `255`.
Observation privacy is unchanged from the Rust engine: only `own_hand` and the
union `unknown_pool` are returned, never the two opponent allocations.

## Packed actions

For `M` total legal actions, environment `i` owns the stable range
`[offsets[i], offsets[i+1])`. `phase`, phase-local `action_code`, `rank_counts`,
kind, main rank, chain length, total cards, and `state_index` are packed without
objects. Non-play actions use zero rank counts and sentinel move metadata. Terminal
rows have empty ranges.
`step_packed` consumes one **local** `int64` index per row; a row that was already
terminal must supply `-1`.

## M10 Proposal and distillation views

Proposal consumes a strict subset of existing public `RaggedBatch` columns; it does
not add a hidden-hand feature. `subset_ragged_batch` preserves state rows and
canonical action order while replacing action tensors and offsets with the protected
Top-K subset. Search distillation pairs the unchanged public batch with normalized
root visits, a search value, and a declared Belief-sample summary. The summary is an
offline target-side feature and is not part of the deployment Actor input.

## Batch step result

Step results use fixed `[B]`, `[B, 15]`, and `[B, 3]` arrays plus the complete
next `BatchObservation`. `acted` distinguishes real transitions from terminal
no-ops. Event sequence/actor, action kind/rank, and next player use `-1` or `255`
sentinels where no event exists. All indices are validated before any row mutates;
unexpected later failures roll back earlier rows in strict reverse order.
