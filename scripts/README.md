# Scripts

`build_guandan_native.ps1` 构建独立的掼蛋 Python 扩展并放入本地
`python/birddou` 包，供四人 Web 模式使用；生成的 `.pyd` 已被 Git 忽略。

E008 provides the first executable reproduction workflow:

```bash
python scripts/fetch_douzero_baseline.py
python scripts/differential_douzero.py --games 100 --seed 20260722 \
  --json-output artifacts/eval/differential-douzero.json
```

The fetch command checks out the manifest-pinned source into an ignored artifact
cache and refuses to replace a dirty checkout. The differential command compares
every decision state's legal-action set and synchronized state transition. It exits
non-zero on the first mismatch and prints a stable JSON summary on success.

The PerfectDou baseline has a separate pinned fetch/checksum workflow:

```bash
python scripts/fetch_perfectdou_baseline.py
```

Its official encoder is a Python-3.7 Linux binary. `perfectdou_worker.py` is the
JSONL process bridge used inside that compatible environment; it is not a native
replacement for the upstream encoder. See
[`../docs/EVALUATION.md`](../docs/EVALUATION.md#rlcard-and-perfectdou-baselines).

After the DMC random/RLCard paired runs, combine all M3 evidence with the
next-update resume test declaration:

```bash
python scripts/audit_m3_gate.py \
  --differential artifacts/eval/differential-douzero.json \
  --random-evaluation artifacts/train/dmc_smoke/report.json \
  --rlcard-evaluation artifacts/eval/dmc-smoke-vs-rlcard.json \
  --checkpoint-resume-exact --output artifacts/eval/m3-gate.json
```

The command exits non-zero when any required lower bound or exactness condition
fails.

E011 adds a paired interface benchmark. It runs the same seeds and deterministic
maximum-card choices through independent E010 Python objects and the packed E011
batch, and refuses to report if their transition counts diverge:

```bash
python scripts/benchmark_batch_env.py --batch-size 256 --ticks 100 --repeats 5
```

The reported speedup includes legal-action transport, stepping, and the next
current-player observation. It is an interface-throughput comparison, not a claim
that sequential Rust rule execution itself becomes parallel.

E015 adds the installed `bird-dou-train-dmc` command. It executes the versioned
single-actor DMC configuration, atomically writes a complete checkpoint and
per-episode metrics, and evaluates the resulting three-role policy against seeded
random on fixed paired deals. See [`../docs/TRAINING.md`](../docs/TRAINING.md).

M7 adds `benchmark_inference_server.py`. It drives a fixed number of requests from
a bounded number of concurrent synthetic Actors through the real scheduler and
ragged batching path, then reports actual states/actions, batches, queue peak,
throughput, and traced Python-memory peak. The deterministic fake model isolates
queue and batching overhead; it is not a neural-inference speed claim.

```bash
python scripts/benchmark_inference_server.py --requests 10000 --concurrency 32
```

The specification-level reproducibility entrypoints are checked in as fail-fast
Bash wrappers:

```bash
bash scripts/reproduce_douzero.sh
bash scripts/train_belief.sh
bash scripts/train_cardplay.sh
bash scripts/train_full_game.sh
bash scripts/run_crossplay.sh --landlord-policies douzero_ADP,douzero_WP \
  --farmer-policies douzero_ADP,douzero_WP
```

`PYTHON` selects the interpreter. Dataset/game budgets and output paths can be
overridden with the environment variables documented inside each wrapper.
