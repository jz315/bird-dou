# Requirement Coverage

This matrix separates software completion from empirical research claims. A unit
test that proves a gate rejects bad input is not counted as evidence that a trained
checkpoint passes that gate.

| Milestone | Implemented software and executable boundary | Current evidence | Research status |
|---|---|---|---|
| M0 | Workspace, CI, two versioned rules profiles, manifests, license audit, typed config/schema contracts | Rust/Python static and unit checks | Engineering complete |
| M1 | Rust cards/moves, exhaustive legal generation, transition/scoring, serialization and apply/undo | Official differential: 100 games, 6,257 states, 6,157 actions, zero drift; property/golden tests | Gate complete |
| M2 | ABI3 PyO3 single/batch environments, packed NumPy transport, paired Arena, bootstrap and cross-play | 5.32x current recorded packed-boundary speedup; deterministic Arena tests | Gate complete |
| M3 | Exact 54-plane/LSTM model, official ADP/WP adapters, three-role DMC, resume, RLCard adapter | Machine-readable M3 gate accepted: zero differential drift, random CI lower 0.6667, RLCard CI lower 0.3333, exact next-update resume | Smoke gate complete; not a from-scratch or Final-strength claim |
| M4 | Versioned ragged features, Rank Mixer, dual history, action encoder, role adapters and multi-head BIRD-Dou DMC | Shape/gradient/NaN tests and one-game train/resume smoke | Same-budget DouZero research comparison pending |
| M5 | Exact two-container CRF, NLL/marginals/sampler, calibration, data generation, checksum/version/fingerprint-pinned E020 warm-start, frozen Belief pretraining and explicitly behavior-anchored fine-tuning | Conservation/brute-force tests; bit-exact zero-gate policy/MC-Q warm-start; smoke NLL 4.6409 vs uniform 5.1511 | Post-training policy non-regression and Belief-shuffle paired research run pending |
| M6 | Full-hand Teacher, privileged critic, Oracle Dropout, direct KD and strict information-set KD with per-state value weighting | Leakage/isomorphism/gradient, hidden-assignment invariance and ragged-weighting tests | Teacher-strength and direct-vs-IS-KD playing-strength ablations pending |
| M7 | Bounded multiprocess Actors, vectorized native envs, central ragged inference, role-homogeneous replay, V-trace, Hybrid learner and fair-budget validator | Queue-full/crash/restart/shutdown, role-split multi-Actor games, mixed-role rejection and hand-computed gradient tests | Long-duration memory/deadlock soak and equal-budget three-mode run pending |
| M8 | Shared Farmer actor interfaces, full-state Team Critic, COMA baseline, bounded counterfactual rollouts, exploiter and promotion gates | Invariance/gradient/rollout/gate tests | Paired improvement against a selected strong landlord pending |
| M9 | Complete bidding/doubling/scoring engine, three-container belief, value-based Bid Head, phase-composed checksum/version-pinned Cardplay continuation, pure-win-first MC/DMC targets, score/rob-specific monitoring, content-fingerprinted curriculum and resumable joint trainer | Checksum-pinned warm-start MC rollout across bidding/doubling/card-play, selected-action DMC with default-zero Q entropy, GRU padding/eval/NaN gates, content-drift resume rejection, CPU BF16 forward/backward gates, random-continuation joint train/load/resume smoke, plus CUDA FP16 gate when available | A strong pretrained Cardplay artifact, calibrated non-degenerate bidder and positive paired Gate C run pending |
| M10 | Proposal/protection/dynamic Top-K, root-consistent rollouts, exact solver, search/compact distillation and hash-bound deployment | CPU Proposal benchmark: 2.17x at 29.29% full-action control; trigger/solver/bundle tests | Recall/non-regression, positive search gain and retained-gain research runs pending |

## Evaluation and checkpoint coverage

- `bird-dou-evaluate` and `bird-dou-crossplay` accept official DouZero ADP/WP,
  pinned RLCard, external PerfectDou, exact-DouZero DMC checkpoints, BIRD-Dou
  current/history/exploiter checkpoints, and joint full-game checkpoints.
- Under `canonical_full`, card-play baselines receive an explicit fixed bidder;
  joint checkpoints restore their Bid Head. DouZero and PerfectDou role selection
  follows the resolved landlord rather than a fixed absolute seat.
- JSON reports include paired confidence intervals, score distributions and tail
  risk, role win rates, transformed score, bomb/spring rates, and flattened
  per-match bidding/rules/terminal audit records.
- DMC, BIRD-Dou DMC, and full-game checkpoints contain model/feature versions,
  fingerprints, rules hash, Git revision, optimizer/scheduler/scaler, RNG,
  policy version, training stage, and League snapshot. Resume tests compare the
  next update, not merely successful deserialization.

## Interpretation

The remaining items are trained-checkpoint experiments, not missing acceptance
logic or hidden placeholder returns. `milestone_gates.py` makes M3-M6 acceptance
explicit; M7-M10 use their algorithm, promotion, bidding, Proposal, search, and
retention gate types. They require predeclared budgets, fixed deal
manifests, and enough compute to reach the specified CI or calibration thresholds.
Until those artifacts exist, no M4-M10 strength improvement is claimed.
