# Evaluation

## E012 fixed-deal Arena

All formal policy comparisons use a versioned `splitmix64_v1` seed manifest. For
each seed, the Arena evaluates the candidate and baseline symmetrically at
`landlord`, `landlord_down`, and `landlord_up`: three paired comparisons and six
matches per deal. The cards are never redrawn between the two sides of a pair.
Because the post-bid game has a 20-card landlord hand and 17-card farmer hands,
"seat rotation" means rotating policy identity through the three fixed role seats,
not moving an incompatible hand to another role.

Policies receive only the current player's `Observation`, canonical legal-action
list, and reproducibility metadata. The Arena records the rule hash, fixed-deal
identity, initial/final deal seeds, redeal count, seat-to-policy assignment,
landlord and bidding record (empty for the post-bid profile), role, payoff, bomb
count, and terminal-state digest. An invalid policy decision terminates evaluation;
there is no fallback action.

Run the built-in smoke policies and write an auditable JSON report:

```bash
bird-dou-evaluate --candidate longest_move --baseline first_legal \
  --deals 100 --seed 20260722 --bootstrap-resamples 10000 \
  --output artifacts/eval/longest-vs-first.json
```

The equivalent source-tree command is:

```bash
python -m birddou.cli.evaluate --deals 100 --bootstrap-resamples 10000
```

The report separates landlord, downstream farmer, upstream farmer, farmer team,
and overall results. Candidate-minus-baseline confidence intervals use a seeded
percentile bootstrap. The independent resampling unit is the whole deal: farmer
and overall role values are averaged inside a deal before bootstrap resampling,
so the six correlated trajectories are never treated as six independent samples.
Use `ArenaReport.meets_precision(...)` for a predeclared Final-evaluation CI
half-width rule; the formal maximum deal manifest and threshold should be fixed
before the run. The reproducible defaults are in
[`../configs/eval/default.yaml`](../configs/eval/default.yaml).

`Arena.evaluate_cross_play(...)` builds the full ordered landlord-policy ×
farmer-team-policy matrix on the same fixed deal set. Its cell CIs likewise
resample complete deals.

## M9 complete scoring Arena

`Arena.play_match(...)` also supports `canonical_full`. Before landlord resolution,
the policy context role is `None`; after resolution it is mapped relative to the
actual landlord rather than fixed seat 0. All-pass attempts are deterministically
redealt when configured, with every bidding attempt retained in the result. The
terminal raw payoff already includes winning bid, doubling, bombs/rocket, spring or
anti-spring, and the optional score cap.

Complete-bidding research comparisons must balance absolute seat assignment over
the same seed manifest and cluster uncertainty by original deal. Post-bid
role-labelled summaries must not be reinterpreted as dynamic-landlord role reports.
The M9 acceptance gate consumes the resulting paired lower confidence bound plus
calibration and distribution reports; see [`BIDDING.md`](BIDDING.md).

## M10 pruning, search, and compact gates

Proposal evaluation records Teacher-best, direct-finish, and bomb/rocket recall on
an independent set; end-to-end wall-clock throughput including Proposal and subset
construction; the observed full-action control fraction; and a fixed-deal paired
strength interval. Search evaluation additionally records trigger counts and any
out-of-trigger invocation. Its candidate must have a positive paired 95% CI lower
bound against the same pure network. Compact evaluation compares against that pure
network and reports the fraction of the independently established search gain that
is retained.

The local CPU throughput command and result are recorded in
[`SEARCH_AND_DEPLOYMENT.md`](SEARCH_AND_DEPLOYMENT.md). Those timings do not replace
the trained-checkpoint Research Gate.

Evaluation scale guidance:

- **Smoke:** very small manifest, contract and termination checks only.
- **Quick Gate:** enough fixed deals to reject clearly weaker candidates.
- **Research Gate:** predeclared fixed set with 95% CI reporting.
- **Final:** extend the predeclared fixed manifest until the selected CI precision
  target is met or the predeclared maximum is reached.

## E013 official DouZero inference

The official ADP and WP checkpoint sets are external artifacts. Install the
optional inference dependency, then fetch the pinned upstream source and the
checksum-declared three-role weights:

```bash
python -m pip install -e ".[model]"
python scripts/fetch_douzero_baseline.py \
  --weight-set douzero_ADP --weight-set douzero_WP
```

Run either set through the same fixed-deal Arena and report format used by every
other policy:

```bash
bird-dou-evaluate --candidate douzero_ADP --baseline douzero_WP \
  --deals 100 --seed 20260722 --device cpu \
  --output artifacts/eval/douzero-adp-vs-wp.json
```

`OfficialDouZeroPolicy` eagerly checks the size and SHA-256 of the landlord,
downstream-farmer, and upstream-farmer checkpoints, loads state dictionaries with
PyTorch's weights-only mode, and rejects missing or unexpected network keys. It
passes only the Arena's current-player `Observation` to the native compatibility
encoder and network. Candidate rows preserve BIRD-Dou's canonical legal action
order before stable first-maximum selection. In post-bid evaluation the seat map
is 0=`landlord`, 1=`landlord_down`, 2=`landlord_up`; in complete games the same
three networks are selected relative to the resolved landlord.
Native inference needs the weights but not an importable upstream checkout. The
explicit `--douzero-feature-encoder official_reference` fallback requires the
pinned source and exists for differential testing.

Both source and weights live under the ignored local baseline cache. Their pinned
commit, mirror revision, byte sizes, and checksums are declared in
[`../artifacts/baselines/douzero/manifest.toml`](../artifacts/baselines/douzero/manifest.toml).
There is no random-weight fallback: absent or altered artifacts are fatal.

E008 introduces the rule-engine differential gate. The official `DouZero` source
is not vendored; fetch and verify the pinned commit, then run synchronized games
(weights are not needed for this command):

```bash
python scripts/fetch_douzero_baseline.py
python scripts/differential_douzero.py --games 100 --seed 20260722 \
  --json-output artifacts/eval/differential-douzero.json
```

## RLCard and PerfectDou baselines

The pinned RLCard 1.0.7 rule agent is an optional post-bid baseline:

```bash
python -m pip install -e ".[rlcard]"
bird-dou-evaluate --candidate longest_move --baseline rlcard_rule_v1 \
  --deals 100 --seed 20260722
```

`RlcardRulePolicy` reconstructs RLCard's documented raw observation from the
current player's hand, the merged unknown-card pool, public trace, and canonical
legal actions. It serializes access to RLCard's process-global NumPy RNG and
restores the caller's state after every decision, so repeated formal runs are
seeded without contaminating other policies.

PerfectDou is pinned by commit and checksum, but its official feature encoder and
left-hand calculator are CPython-3.7 Linux shared objects. Fetch and verify the
release on any host, then run its worker in a compatible Python 3.7 x86-64 Linux
environment with `onnxruntime==1.7.0`:

```bash
python scripts/fetch_perfectdou_baseline.py
python3.7 -m pip install onnxruntime==1.7.0
bird-dou-evaluate --candidate perfectdou --baseline douzero_ADP \
  --perfectdou-command \
  'python3.7 -u scripts/perfectdou_worker.py --source artifacts/baselines/perfectdou/source' \
  --deals 100 --seed 20260722
```

The command may instead begin with a container or WSL launcher, provided it
exposes the same line-delimited JSON protocol. The BIRD-Dou process sends only the
acting player's hand, the merged opponent pool, public history/counts, bottom
cards, and legal actions. It never sends the two opponent hands separately.
Responses are timeout-bounded and must map one-to-one to a canonical legal action.
The adapter acts only during card play, but maps all three model roles relative to
the landlord resolved by either supported rules profile.

Run the full ordered landlord-policy by farmer-team-policy matrix with:

```bash
bird-dou-crossplay \
  --landlord-policies douzero_ADP,douzero_WP,rlcard_rule_v1 \
  --farmer-policies douzero_ADP,douzero_WP,rlcard_rule_v1 \
  --deals 100 --seed 20260722 --output artifacts/eval/crossplay.json
```

If `perfectdou` appears in either list, pass the same
`--perfectdou-command`. Every matrix cell uses the identical fixed deal set and
deal-clustered bootstrap contract.

Current champions, historical checkpoints, and landlord/farmer exploiters are
registered by stable names instead of being mistaken for built-in baselines:

```bash
bird-dou-crossplay \
  --bird-dou-policy champion=artifacts/league/champion/checkpoint.pt \
  --bird-dou-policy history-42=artifacts/league/history-42/checkpoint.pt \
  --bird-dou-policy farmer-exploiter=artifacts/league/farmer-exploiter/checkpoint.pt \
  --landlord-policies champion,history-42,douzero_ADP \
  --farmer-policies champion,farmer-exploiter,douzero_ADP
```

The loader accepts raw card-play weights, resumable BIRD-Dou DMC checkpoints, and
the card-play part of full-game checkpoints. Wrapped checkpoints are checked
against model and feature fingerprints; the decomposition-feature ablation is
recovered from the saved fingerprint.

The earlier exact-DouZero three-role learner uses its own explicit registration:

```bash
bird-dou-evaluate \
  --dmc-policy dmc-smoke=artifacts/train/dmc_smoke/checkpoint.pt \
  --candidate dmc-smoke --baseline rlcard_rule_v1 \
  --deals 20 --seed 15015 --bootstrap-resamples 1000
```

That local warm-start smoke produced a 0.50 overall win-rate delta with a paired
95% CI of `[0.3333, 0.65]` over 20 deals/120 matches. It closes the executable M3
smoke comparison, not a from-scratch or Final-scale strength claim.

For Gate C, select `canonical_full`. Ordinary card-play baselines are then wrapped
in the declared fixed bidder/doubler, while a joint checkpoint supplies its own
Bid Head:

```bash
bird-dou-evaluate --rules configs/rules/canonical_full.yaml \
  --full-game-policy current=artifacts/train/full_game/checkpoint.pt \
  --candidate current --baseline douzero_ADP --fixed-bid-score 1 \
  --deals 100 --seed 20260722 --output artifacts/eval/full-game.json
```

Every CLI report includes the summary/CIs plus flattened per-match audit records:
resolved landlord, complete bidding record, redeals, assignments, per-seat
payoffs/bombs, spring outcome, rules hash, and terminal-state digest.

## E011 packed-interface benchmark

The reproducible benchmark compares the E010 per-environment object/JSON boundary
with the E011 packed NumPy boundary using identical deals and first-maximum-card
decisions. Both paths include legal-action transport, transition, and construction
of the next current-player observation; the command aborts if transition counts
diverge.

```bash
python scripts/benchmark_batch_env.py --batch-size 256 --ticks 100 --repeats 5 \
  --seed 20260722
```

Reference run on 2026-07-22 (`i7-12700F`, Rust 1.96.0, Python 3.13.11,
NumPy 2.4.6):

| Interface | Median seconds | Transitions/s |
|---|---:|---:|
| E010 single Python objects | 8.3624 | 2,258.80 |
| E011 packed Rust batch | 1.5729 | 12,009.18 |

The packed interface was **5.32×** faster for 18,889 matched transitions per
repeat. This measures boundary and packing throughput on one machine; it is not a
claim of parallel rule execution and must be rerun on target actor hardware.

At every decision state the harness compares current player, all hands and played
cards, remaining counts, normalized active target and Pass state, bomb multiplier,
terminal winner, raw/ADP payoff, and the complete legal-action set. A trajectory
action is chosen deterministically from the common normalized set and applied to
both engines. The first discrepancy includes seed, turn, field differences, and
missing/extra actions.

The differential harness and Arena serve different checks: the former proves
rule compatibility against DouZero, while the latter compares policies without
deal or role confounding.
