# Architecture notes

## Dependency direction

```text
ddz-core        domain values and invariants
    ^
ddz-rules      authoritative rules and transitions
    ^
ddz-batch      vectorized execution
    ^
ddz-pyo3      Python boundary
```

`ddz-rules` never imports a model, tensor library, Python runtime, web type, or training
concept.

## State ownership

`Game` owns immutable `RuleConfig`, match seed, economy eligibility context, and one
private `GameState`. `Game::step` is transactional:

1. verify current actor;
2. generate and check legal actions;
3. apply one phase-specific transition;
4. perform bounded automatic advancement;
5. validate the complete core state and rule-owned cross-field invariants;
6. restore the snapshot on any error.

## Huanle landlord selection

Calling and robbing use the existing core states:

- `CallingState` records acted and declined seats;
- declined seats are excluded from the rob order;
- the rob order is stable and contains each eligible seat at most once;
- the current candidate is skipped rather than being asked to rob itself;
- the caller is placed last when reclaim is enabled.

This models sequences such as:

```text
A calls -> B robs -> C passes -> A reclaims
```

with two successful rob factors.

## Move generation

Lead generation enumerates rank templates, never all physical-card subsets. Follow
generation requests only:

- the same kind, chain length, and a higher main rank;
- bombs when the target is not a rocket;
- the rocket;
- pass.

The detector remains the canonicalizer for ambiguous rank-count shapes. Generated moves
are deduplicated in `BTreeSet<Move>`.

## Public information

The private deal plan and match seed never enter `Observation`. During deal, `unknown_pool`
contains undealt cards. Revealed current hands are removed from the pool. Double choices
are hidden from public event history until the round resolves, while public acted and
eligible sets remain available.


## Restore and persistence validation

`Game::restore` does not trust a structurally valid snapshot. It reconstructs the game
from the rule profile, match seed, economy context, and the recorded player actions,
then requires the complete authoritative state and event history to match exactly. This
catches stale multipliers, altered doubling choices, forged system events, and prior
redeal drift.

`EconomyContext` remains an external trusted input because bean-account persistence is
not a rule-engine responsibility. A persistence layer should store it next to the core
state and rule hash.

## Rule-owned invariants

The rule layer verifies relationships that `ddz-core` deliberately cannot know:

- reveal factor equals the maximum public reveal factor;
- rob and bomb exponents equal their authoritative counters;
- DouZero never enters reveal/call/rob/double phases;
- Huanle reaches card play only after resolving per-player doubling;
- terminal payoff and spring state equal deterministic pairwise settlement.
